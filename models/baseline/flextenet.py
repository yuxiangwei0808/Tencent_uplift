import torch
import torch.nn as nn
import torch.nn.functional as F

from .tf_utils import *


class threeDenseLayer(nn.Module):
    def __init__(self, input_dim_share, input_dim_c, share_unit, spe_unit, normalize, alpha, is_last):
        super(threeDenseLayer, self).__init__()
        self.share_unit = share_unit
        self.spe_unit = spe_unit
        self.normalize = normalize
        self.alpha = alpha
        self.is_last = is_last

        self.w_share = nn.Parameter(torch.Tensor(input_dim_share, share_unit))
        self.b_share = nn.Parameter(torch.zeros(share_unit))

        self.w_c = nn.Parameter(torch.Tensor(input_dim_c, spe_unit))
        self.b_c = nn.Parameter(torch.zeros(spe_unit))

        self.w_t = nn.Parameter(torch.Tensor(input_dim_c, spe_unit))
        self.b_t = nn.Parameter(torch.zeros(spe_unit))

        nn.init.xavier_uniform_(self.w_share)
        nn.init.xavier_uniform_(self.w_c)
        nn.init.xavier_uniform_(self.w_t)

    def forward(self, inputs):
        share_input, c_input, t_input = inputs

        if not self.is_last:
            share_out = F.elu(torch.matmul(share_input, self.w_share) + self.b_share)
            c_out = F.elu(torch.matmul(c_input, self.w_c) + self.b_c)
            t_out = F.elu(torch.matmul(t_input, self.w_t) + self.b_t)
        else:
            share_out = torch.matmul(share_input, self.w_share) + self.b_share
            c_out = torch.matmul(c_input, self.w_c) + self.b_c
            t_out = torch.matmul(t_input, self.w_t) + self.b_t

        with torch.no_grad():
            penalty = 0
            penalty += ortho_penalty_asymmetric(self.w_share, self.w_t, self.normalize)
            penalty += ortho_penalty_asymmetric(self.w_share, self.w_c, self.normalize)
            if not self.is_last:
                penalty += ortho_penalty_asymmetric(self.w_t, self.w_c, self.normalize)
        
        self.penalty = self.alpha * penalty

        return share_out, c_out, t_out


class FlexTENet(nn.Module):
    def __init__(self, input_dim):
        super(FlexTENet, self).__init__()

        hidden_unit_share = 256
        hidden_unit_y = 256
        dropout = 0.0

        self.batch_norm_1d = True
        self.alpha = 0.0
        self.reg_normalize = False

        self.norm = nn.BatchNorm1d(input_dim)
        
        share_tower_units, share_tower_dropout = get_units_dropout_list(hidden_unit_share, 4, dropout)
        tower_units, tower_dropout = get_units_dropout_list(hidden_unit_y, 4, dropout)

        self.first_layer = threeDenseLayer(input_dim, input_dim,  share_tower_units[0], tower_units[0], self.reg_normalize, self.alpha, is_last=False)
        self.second_layer = threeDenseLayer(share_tower_units[0],  tower_units[0] * 2, share_tower_units[1], tower_units[1], self.reg_normalize, self.alpha, is_last=False)
        self.third_layer = threeDenseLayer(share_tower_units[1],  tower_units[1] * 2, share_tower_units[2], tower_units[2], self.reg_normalize, self.alpha, is_last=False)
        self.fourth_layer = threeDenseLayer(share_tower_units[2],  tower_units[2] * 2, share_tower_units[3], tower_units[3], self.reg_normalize, self.alpha, is_last=False)
        self.last_layer = threeDenseLayer(share_tower_units[3],  tower_units[3] * 2, 1, 1, self.reg_normalize, self.alpha, is_last=True)

    def forward(self, feat_vals):
        if self.batch_norm_1d:
            feat_vals = self.norm(feat_vals)

        first_layer_inputs = (feat_vals, feat_vals, feat_vals)
        share_o0, res_c0, res_t0 = self.first_layer(first_layer_inputs)

        second_layer_inputs = (share_o0, torch.cat([share_o0, res_c0], dim=1), torch.cat([share_o0, res_t0], dim=1))
        share_o1, res_c1, res_t1 = self.second_layer(second_layer_inputs)

        third_layer_inputs = (share_o1, torch.cat([share_o1, res_c1], dim=1), torch.cat([share_o1, res_t1], dim=1))
        share_o2, res_c2, res_t2 = self.third_layer(third_layer_inputs)

        fourth_layer_inputs = (share_o2, torch.cat([share_o2, res_c2], dim=1), torch.cat([share_o2, res_t2], dim=1))
        share_o3, res_c3, res_t3 = self.fourth_layer(fourth_layer_inputs)

        last_layer_inputs = (share_o3, torch.cat([share_o3, res_c3], dim=1), torch.cat([share_o3, res_t3], dim=1))
        logit_o3, logit_c3, logit_t3 = self.last_layer(last_layer_inputs)

        logit_o3 = logit_o3.view(-1, 1)
        logit_c3 = logit_c3.view(-1, 1)
        logit_t3 = logit_t3.view(-1, 1)

        y0_logits = logit_o3 + logit_c3
        y0 = torch.sigmoid(y0_logits)

        y1_logits = logit_o3 + logit_t3
        y1 = torch.sigmoid(y1_logits)

        return y0, y1, None, None
    
    def calculate_loss(self, input, t_true, y_true):
        if t_true.dim() != 2:
            t_true = t_true.unsqueeze(1)
        if y_true.dim() != 2:
            y_true = y_true.unsqueeze(1)
        y0_pred, y1_pred, _, _ = self.forward(input)
        return bce_loss(y_true, t_true, y0_pred, y1_pred)