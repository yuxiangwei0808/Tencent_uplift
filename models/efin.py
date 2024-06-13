import torch
import torch.nn as nn
import torch.nn.functional as F


class EFIN(nn.Module):
    def __init__(self, input_dim, hc_dim, hu_dim, is_self, act_type='elu'):
        super(EFIN, self).__init__()
        self.nums_feature = input_dim
        self.is_self = is_self

        # interaction attention
        self.att_embed_1 = nn.Linear(hu_dim, hu_dim, bias=False)
        self.att_embed_2 = nn.Linear(hu_dim, hu_dim)
        self.att_embed_3 = nn.Linear(hu_dim, 1, bias=False)

        # self-attention
        self.softmax = nn.Softmax(dim=-1)
        self.Q_w = nn.Linear(hu_dim, hu_dim, bias=True)
        self.K_w = nn.Linear(hu_dim, hu_dim, bias=True)
        self.V_w = nn.Linear(hu_dim, hu_dim, bias=True)

        # representation parts for X
        self.x_rep = nn.Embedding(input_dim, hu_dim)

        # representation parts for T
        self.t_rep = nn.Linear(1, hu_dim)

        # control net
        self.c_fc = nn.Sequential(
            nn.Linear(input_dim * hu_dim, hc_dim),
            nn.ReLU(),
            nn.Linear(hc_dim, hc_dim),
            nn.ReLU(),
            nn.Linear(hc_dim, hc_dim // 2),
            nn.ReLU(),
            nn.Linear(hc_dim // 2, hc_dim // 4)
        )

        self.c_logit = nn.Linear(hc_dim // 4, 1)
        self.c_tau = nn.Linear(hc_dim // 4, 1)

        if is_self:
            self.c_fc.add_module('fc5', nn.Linear(hc_dim // 4, hc_dim // 8))
            self.c_logit = nn.Linear(hc_dim // 8, 1)
            self.c_tau = nn.Linear(hc_dim // 8, 1)

        # uplift net
        self.u_fc = nn.Sequential(
            nn.Linear(hu_dim, hu_dim),
            nn.ReLU(),
            nn.Linear(hu_dim, hu_dim // 2),
            nn.ReLU(),
            nn.Linear(hu_dim // 2, hu_dim // 4)
        )

        if is_self:
            self.u_fc.add_module('fc4', nn.Linear(hu_dim // 4, hu_dim // 8))
            self.t_logit = nn.Linear(hu_dim // 8, 1)
            self.u_tau = nn.Linear(hu_dim // 8, 1)
        else:
            self.t_logit = nn.Linear(hu_dim // 4, 1)
            self.u_tau = nn.Linear(hu_dim // 4, 1)

        # activation function
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError('unknown act_type {0}'.format(act_type))

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        attn_weights = self.softmax(torch.sigmoid(attn_weights))
        outputs = torch.matmul(attn_weights, V)
        return outputs, attn_weights

    def interaction_attn(self, t, x):
        t_rep = torch.sigmoid(self.att_embed_1(t)).unsqueeze(1)
        x_rep = torch.sigmoid(self.att_embed_2(x))
        attention = self.att_embed_3(torch.relu(t_rep + x_rep)).squeeze(-1)
        attention = torch.softmax(attention, dim=1)
        outputs = torch.matmul(attention.unsqueeze(1), x).squeeze(1)
        return outputs, attention

    def forward(self, feature_list, is_treat):
        t_true = is_treat.unsqueeze(1)  # B, 1
        # hu_dim looks like channel
        x_rep = feature_list.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)  # B, N, 1 * 1, N, hu_dim -> B, N, hu_dim

        # control net
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True)
        xx, xx_weight = self.self_attn(_x_rep, _x_rep, _x_rep)
        _x_rep = xx.view(xx.size(0), -1)

        c_last = self.c_fc(_x_rep)
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # uplift net
        t_rep = self.t_rep(torch.ones_like(t_true))
        xt, xt_weight = self.interaction_attn(t_rep, x_rep)

        u_last = self.u_fc(xt)
        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit)

        return c_logit, c_prob, c_tau, t_logit, t_prob, u_tau

    def calculate_loss(self, feature_list, is_treat, label_list):
        c_logit, c_prob, c_tau, t_logit, t_prob, u_tau = self.forward(feature_list, is_treat)

        y_true = label_list.unsqueeze(1)
        t_true = is_treat.unsqueeze(1)

        c_logit_fix = c_logit.detach()
        uc = c_logit
        ut = c_logit_fix + u_tau

        loss1 = F.mse_loss((1 - t_true) * uc + t_true * ut, y_true)
        loss2 = F.binary_cross_entropy_with_logits(t_logit, 1 - t_true)
        loss = loss1 + loss2

        return loss
