import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d

from .tf_utils import *


class FiveDenseLayer(nn.Module):
    def __init__(self, common_units_p, common_units_h, normalize, alpha, penalty_disc, orth_type):
        super(FiveDenseLayer, self).__init__()
        self.common_units_p = common_units_p
        self.common_units_h = common_units_h
        self.normalize = normalize
        self.alpha = alpha
        self.penalty_disc = penalty_disc
        self.orth_type = orth_type

        self.rep_y0_w = nn.Parameter(torch.Tensor(common_units_p, common_units_h))
        self.rep_y0_b = nn.Parameter(torch.zeros(common_units_h))

        self.rep_y1_w = nn.Parameter(torch.Tensor(common_units_p, common_units_h))
        self.rep_y1_b = nn.Parameter(torch.zeros(common_units_h))

        self.rep_common_w = nn.Parameter(torch.Tensor(common_units_p, common_units_h))
        self.rep_common_b = nn.Parameter(torch.zeros(common_units_h))

        self.rep_y_common_w = nn.Parameter(torch.Tensor(common_units_p, common_units_h))
        self.rep_y_common_b = nn.Parameter(torch.zeros(common_units_h))

        self.rep_propensity_w = nn.Parameter(torch.Tensor(common_units_p, common_units_h))
        self.rep_propensity_b = nn.Parameter(torch.zeros(common_units_h))

        nn.init.xavier_uniform_(self.rep_y0_w)
        nn.init.xavier_uniform_(self.rep_y1_w)
        nn.init.xavier_uniform_(self.rep_common_w)
        nn.init.xavier_uniform_(self.rep_y_common_w)
        nn.init.xavier_uniform_(self.rep_propensity_w)

    def forward(self, share_input, t_true):
        rep_y0 = F.elu(F.linear(share_input, self.rep_y0_w, self.rep_y0_b))
        rep_y1 = F.elu(F.linear(share_input, self.rep_y1_w, self.rep_y1_b))
        rep_common = F.elu(F.linear(share_input, self.rep_common_w, self.rep_common_b))
        rep_y_common = F.elu(F.linear(share_input, self.rep_y_common_w, self.rep_y_common_b))
        rep_propensity = F.elu(F.linear(share_input, self.rep_propensity_w, self.rep_propensity_b))

        discrepancy = 0.0
        if self.penalty_disc > 0.0:
            n = t_true.size(0)
            n_t = torch.sum(t_true)
            t_true = t_true.unsqueeze(1) if t_true.dim() == 1 else t_true
            rep_y_common = rep_y_common / torch.sqrt(torch.var(rep_y_common, dim=0) + 1e-8)
            mean_control = (n / (n - n_t + 1)) * torch.mean((1 - t_true) * rep_y_common, dim=0)
            mean_treated = (n / n_t + 1) * torch.mean(t_true * rep_y_common, dim=0)
            discrepancy = torch.sum((mean_treated - mean_control) ** 2)

        penalty = 0
        with torch.no_grad():
            if self.orth_type == 'abs':
                col_mu0 = torch.sum(torch.abs(rep_y0), dim=0)
                col_mu1 = torch.sum(torch.abs(rep_y1), dim=0)
                col_c = torch.sum(torch.abs(rep_common), dim=0)
                col_o = torch.sum(torch.abs(rep_y_common), dim=0)
                col_w = torch.sum(torch.abs(rep_propensity), dim=0)

                penalty = torch.sum(col_c * col_o + col_c * col_w + col_c * col_mu1 +
                                    col_c * col_mu0 + col_w * col_o + col_w * col_mu0 +
                                    col_w * col_mu1 + col_o * col_mu0 + col_o * col_mu1 +
                                    col_mu0 * col_mu1)
            else:
                penalty += ortho_penalty_asymmetric(rep_common, rep_y_common, self.normalize)
                penalty += ortho_penalty_asymmetric(rep_common, rep_y0, self.normalize)
                penalty += ortho_penalty_asymmetric(rep_common, rep_y1, self.normalize)
                penalty += ortho_penalty_asymmetric(rep_common, rep_propensity, self.normalize)

                penalty += ortho_penalty_asymmetric(rep_y_common, rep_y0, self.normalize)
                penalty += ortho_penalty_asymmetric(rep_y_common, rep_y1, self.normalize)
                penalty += ortho_penalty_asymmetric(rep_y_common, rep_propensity, self.normalize)

                penalty += ortho_penalty_asymmetric(rep_y0, rep_y1, self.normalize)
                penalty += ortho_penalty_asymmetric(rep_y0, rep_propensity, self.normalize)
                penalty += ortho_penalty_asymmetric(rep_y1, rep_propensity, self.normalize)

        self.penalty_disc = self.penalty_disc * discrepancy + self.alpha * penalty

        return rep_y0, rep_y1, rep_common, rep_y_common, rep_propensity

param_dict = {'l2_reg': 0, 'activation': nn.ELU(), 'hidden_unit_p': 256, 'hidden_unit_h': 256, 'dropout': 0, 'batch_norm': False, 'penalty_disc': 0., 'alpha': 0.0001, 'orth_type': 'fro', 'reg_normalize': False}

class SNet(nn.Module):
    def __init__(self, input_dim, param_dict=param_dict):
        super(SNet, self).__init__()
        l2_reg = param_dict['l2_reg']
        activation = param_dict['activation']
        hidden_unit_p = param_dict['hidden_unit_p']
        hidden_unit_h = param_dict['hidden_unit_h']
        dropout = param_dict['dropout']
        batch_norm = bool(param_dict['batch_norm'])
        self.batch_norm_1d = True

        self.penalty_disc = param_dict['penalty_disc']
        self.alpha = param_dict['alpha']
        self.orth_type = param_dict['orth_type']
        self.reg_normalize = param_dict['reg_normalize']

        self.in_proj = nn.Linear(input_dim, hidden_unit_p)
        self.norm = nn.BatchNorm1d(hidden_unit_p)

        common_units_p, common_dropout_p = get_units_dropout_list(hidden_unit_p, 1, dropout)
        common_units_h, common_dropout_h = get_units_dropout_list(hidden_unit_h, 1, dropout)

        self.share_layer = FiveDenseLayer(common_units_p=common_units_p[0], common_units_h=common_units_h[0], 
                                          normalize=self.reg_normalize, alpha=self.alpha, 
                                          penalty_disc=self.penalty_disc, orth_type=self.orth_type)

        t_units, t_dropout = get_units_dropout_list(hidden_unit_h * 2, 1, dropout)
        self.t_fc = MLP(hidden_units=t_units, activation=activation, dnn_dropout=t_dropout, l2_reg=l2_reg, use_batch_norm=batch_norm)

        tower_units, tower_dropout = get_units_dropout_list(hidden_unit_h * 3, 3, dropout)
        self.y0_fc = MLP(hidden_units=tower_units, activation=activation, dnn_dropout=tower_dropout, l2_reg=l2_reg, use_batch_norm=batch_norm)
        self.y1_fc = MLP(hidden_units=tower_units, activation=activation, dnn_dropout=tower_dropout, l2_reg=l2_reg, use_batch_norm=batch_norm)

        self.t_classifier = nn.Linear(t_units[-1], 1)
        self.y0_classifier = nn.Linear(tower_units[-1], 1)
        self.y1_classifier = nn.Linear(tower_units[-1], 1)

    def forward(self, feat_vals, treatment):
        feat_vals = self.in_proj(feat_vals)
        if self.batch_norm_1d:
            feat_vals = self.norm(feat_vals)

        share_inputs = (feat_vals, treatment)
        rep_y0, rep_y1, rep_common, rep_y_common, rep_propensity = self.share_layer(*share_inputs)

        t_input = torch.cat([rep_common, rep_propensity], dim=1)
        t_output = self.t_fc(t_input)
        t_logits = self.t_classifier(t_output).view(-1, 1)
        t_pred = torch.sigmoid(t_logits)

        y0_input = torch.cat([rep_common, rep_y0, rep_y_common], dim=1)
        y0_output = self.y0_fc(y0_input)
        y0_logits = self.y0_classifier(y0_output).view(-1, 1)
        y0 = torch.sigmoid(y0_logits)

        y1_input = torch.cat([rep_common, rep_y1, rep_y_common], dim=1)
        y1_output = self.y1_fc(y1_input)
        y1_logits = self.y1_classifier(y1_output).view(-1, 1)
        y1 = torch.sigmoid(y1_logits)

        return y0, y1, t_pred, rep_y_common

    def calculate_loss(self, input, t_true, y_true):
        if t_true.dim() != 2:
            t_true = t_true.unsqueeze(1)
        if y_true.dim() != 2:
            y_true = y_true.unsqueeze(1)
        y0_pred, y1_pred, t_pred, _ = self.forward(input, t_true)
        loss_bce = bce_loss(y_true, t_true, y0_pred, y1_pred)
        losst = bce_loss_single(y_true = t_true, y_pred = t_pred)
        return loss_bce + losst