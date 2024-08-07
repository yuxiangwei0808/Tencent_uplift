import torch
import torch.nn as nn
import torch.nn.functional as F


from .tf_utils import *


class TLearner(nn.Module):
    def __init__(self, input_dim):
        super(TLearner, self).__init__()

        l2_reg = 0.
        activation = F.elu
        hidden_unit = 256
        dropout = 0.
        batch_norm = False
        self.batch_norm_1d = False

        self.norm = nn.BatchNorm1d(hidden_unit)
        self.in_proj = nn.Linear(input_dim, hidden_unit)

        common_units, common_dropout = get_units_dropout_list(hidden_unit, 1, dropout)
        self.y0_first_output = MLP(hidden_units=common_units, activation=activation, dnn_dropout=common_dropout, l2_reg=l2_reg, use_batch_norm=batch_norm)
        self.y1_first_output = MLP(hidden_units=common_units, activation=activation, dnn_dropout=common_dropout, l2_reg=l2_reg, use_batch_norm=batch_norm)

        tower_units, tower_dropout = get_units_dropout_list(hidden_unit, 3, dropout)
        self.y0_fc = MLP(hidden_units=tower_units, activation=activation, dnn_dropout=tower_dropout, l2_reg=l2_reg, use_batch_norm=batch_norm)
        self.y1_fc = MLP(hidden_units=tower_units, activation=activation, dnn_dropout=tower_dropout, l2_reg=l2_reg, use_batch_norm=batch_norm)

        self.y0_classifier = nn.Linear(tower_units[-1], 1)
        self.y1_classifier = nn.Linear(tower_units[-1], 1)

    def forward(self, feat_vals):
        feat_vals = self.in_proj(feat_vals)

        if self.batch_norm_1d:
            feat_vals = self.norm(feat_vals)

        y0_first = self.y0_first_output(feat_vals)
        y0_output = self.y0_fc(y0_first)
        y0_logits = self.y0_classifier(y0_output).view(-1, 1)  # (batch_size, 1)
        y0 = torch.sigmoid(y0_logits)

        y1_first = self.y1_first_output(feat_vals)
        y1_output = self.y1_fc(y1_first)
        y1_logits = self.y1_classifier(y1_output).view(-1, 1)  # (batch_size, 1)
        y1 = torch.sigmoid(y1_logits)

        return y0, y1, None, None

    def calculate_loss(self, feature_list, is_treat, label_list):
        if label_list.dim() == 1:
            label_list = label_list.unsqueeze(1)
        if is_treat.dim() == 1:
            is_treat = is_treat.unsqueeze(1)
            
        y0, y1, _, _ = self.forward(feature_list)
        return bce_loss(label_list, is_treat, y0, y1)
