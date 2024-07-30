import numpy as np

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
"""
References "EUEN：Addressing Exposure Bias in Uplift Modeling for Large-scale Online Advertising
"""
class EUEN(nn.Module):
    """
    EUEN class -- a explicit uplift effect network with two heads.
    """
    def __init__(self, input_dim, hc_dim=64, hu_dim=64, is_self=False, act_type='elu'):
        super(EUEN, self).__init__()
        self.is_self = is_self

        # control net
        self.c_fc1 = nn.Linear(input_dim, hc_dim)
        self.c_fc2 = nn.Linear(hc_dim, hc_dim)
        self.c_fc3 = nn.Linear(hc_dim, hc_dim // 2)
        self.c_fc4 = nn.Linear(hc_dim // 2, hc_dim // 4)
        out_dim = hc_dim // 4
        if self.is_self:
            self.c_fc5 = nn.Linear(hc_dim / 4, hc_dim // 8)
            out_dim = hc_dim // 8

        self.c_logit = nn.Linear(out_dim, 1)
        self.c_tau = nn.Linear(out_dim, 1)

        # uplift net
        self.u_fc1 = nn.Linear(input_dim, hu_dim)
        self.u_fc2 = nn.Linear(hu_dim, hu_dim)
        self.u_fc3 = nn.Linear(hu_dim, hu_dim // 2)
        self.u_fc4 = nn.Linear(hu_dim // 2, hu_dim // 4)
        out_dim = hu_dim // 4
        if self.is_self:
            self.u_fc5 = nn.Linear(hu_dim // 4, hu_dim // 8)
            out_dim = hu_dim // 8
        self.t_logit = nn.Linear(out_dim, 1)
        self.u_tau = nn.Linear(out_dim, 1)

        # temporary variable to avoid exceptions in MMD loss
        self.temp = nn.Parameter(data=torch.ones((1, 1), dtype=torch.float), requires_grad=False)
        self.delta = nn.Parameter(data=0.25 * torch.ones((1, 1), dtype=torch.float), requires_grad=False)

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
        # control net
        c_last = self.act(self.c_fc4(self.act(self.c_fc3(self.act(self.c_fc2(self.act(self.c_fc1(feature_list))))))))
        if self.is_self:
            c_last = self.act(self.c_fc5(c_last))
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # uplift net
        u_last = self.act(self.u_fc4(self.act(self.u_fc3(self.act(self.u_fc2(self.act(self.u_fc1(feature_list))))))))
        if self.is_self:
            u_last = self.act(self.u_fc5(u_last))
        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit)

        return c_logit, c_prob, c_tau, t_logit, t_prob, u_tau

    def calculate_loss(self, feature_list, is_treat, label_list, use_huber=False, use_group_reduce=False):
        # Model outputs
        c_logit, c_prob, c_tau, t_logit, t_prob, u_tau = self.forward(feature_list)

        # regression
        c_logit_fix = c_logit.detach()
        uc = c_logit
        ut = (c_logit_fix + u_tau)
        # ut = c_logit + u_tau

        y_true = torch.unsqueeze(label_list, 1)
        t_true = torch.unsqueeze(is_treat, 1)

        '''Losses'''
        # response loss
        if use_huber:

            residual = torch.abs((1 - t_true) * uc + t_true * ut - y_true)
            large_loss = 0.5 * torch.square(residual)
            small_loss = self.delta * residual - 0.5 * torch.square(self.delta)
            cond = torch.less(residual, self.delta)
            temp = torch.where(cond, large_loss, small_loss)
        else:
            temp = torch.square((1 - t_true) * uc + t_true * ut - y_true)

        if use_group_reduce:
            t_steps = torch.maximum(torch.sum(t_true, dim=0), self.temp)
            t_loss = torch.sum(t_true * temp, dim=0) / t_steps
            c_steps = torch.maximum(torch.sum(1 - t_true, dim=0), self.temp)
            c_loss = torch.sum((1 - t_true) * temp, dim=0) / c_steps
            loss = (t_loss + c_loss) / 2
        else:
            loss = torch.mean(temp)

        return loss