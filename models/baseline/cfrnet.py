import numpy as np

import torch
import torch.nn as nn


class CFRNET(nn.Module):
    """
    CFRNet class -- a balancing neural network predictive model with two heads.
    """
    def __init__(self, input_dim, h_dim=128, is_self=False, act_type='elu', rep_norm=False):
        super(CFRNET, self).__init__()
        self.is_self = is_self
        self.rep_norm = rep_norm

        # representation parts
        self.R_fc1 = nn.Linear(input_dim, h_dim)

        # y0 estimate
        self.y0_fc1 = nn.Linear(h_dim, h_dim)
        self.y0_fc2 = nn.Linear(h_dim, h_dim // 2)
        self.y0_fc3 = nn.Linear(h_dim // 2, h_dim // 4)
        out_dim = h_dim // 4
        if self.is_self:
            self.y0_fc4 = nn.Linear(h_dim // 4, h_dim // 8)
            out_dim = h_dim // 8

        self.y0_logit = nn.Linear(out_dim, 1)

        # y1 estimate
        self.y1_fc1 = nn.Linear(h_dim, h_dim)
        self.y1_fc2 = nn.Linear(h_dim, h_dim // 2)
        self.y1_fc3 = nn.Linear(h_dim // 2, h_dim // 4)
        out_dim = h_dim // 4
        if self.is_self:
            self.y1_fc4 = nn.Linear(h_dim // 4, h_dim // 8)
            out_dim = h_dim // 8

        self.y1_logit = nn.Linear(out_dim, 1)

        # temporary variable to avoid exceptions in MMD loss
        self.temp = nn.Parameter(data=torch.ones((1, 1), dtype=torch.float), requires_grad=False)

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

    def forward(self, feature_list):
        # representation parts
        R_fc1 = self.act(self.R_fc1(feature_list))

        if self.rep_norm:
            temp = torch.sum(torch.square(R_fc1), dim=1, keepdim=True)
            temp = torch.sqrt(torch.clamp(temp, 1e-10, np.inf))
            B_rep_norm = R_fc1 / temp
        else:
            B_rep_norm = 1.0 * R_fc1

        # y0 estimate
        y0_last = self.act(self.y0_fc3(self.act(self.y0_fc2(self.act(self.y0_fc1(B_rep_norm))))))
        if self.is_self:
            y0_last = self.act(self.y0_fc4(y0_last))

        y0_logit = self.y0_logit(y0_last)

        # y1 estimate
        y1_last = self.act(self.y1_fc3(self.act(self.y1_fc2(self.act(self.y1_fc1(B_rep_norm))))))
        if self.is_self:
            y1_last = self.act(self.y1_fc4(y1_last))

        y1_logit = self.y1_logit(y1_last)

        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)

        return y0_prob, y1_prob, B_rep_norm

    def calculate_loss(self, feature_list, is_treat, label_list, total_treat_prob=0.5, alpha=1.):
        # Model outputs
        y0_pred, y1_pred, B_rep_norm = self.forward(feature_list)

        '''Losses'''
        # regression_loss
        y_true = torch.unsqueeze(label_list, 1)
        t_true = torch.unsqueeze(is_treat, 1)

        loss0_logits = (1 - t_true) * torch.square(y_true - y0_pred)
        loss1_logits = t_true * torch.square(y_true - y1_pred)
        loss0 = torch.mean(loss0_logits)
        loss1 = torch.mean(loss1_logits)
        regression_loss = loss0 + loss1

        # Linear MMD
        Xc = (1. - t_true) * B_rep_norm
        Xt = t_true * B_rep_norm
        stepc = torch.maximum(torch.sum(1. - t_true), self.temp)
        stept = torch.maximum(torch.sum(t_true), self.temp)
        mean_control = torch.sum(Xc, dim=0) / stepc
        mean_treated = torch.sum(Xt, dim=0) / stept
        mmd = torch.sum(
            torch.square(
                2.0 * total_treat_prob * mean_treated - 2.0 * (1.0 - total_treat_prob) * mean_control))
        temp = torch.sqrt(torch.clamp(mmd, 1e-10, np.inf))
        imb_loss = alpha * temp

        loss = regression_loss + imb_loss

        return loss