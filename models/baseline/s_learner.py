import torch
import torch.nn as nn
import torch.nn.functional as F

from .tf_utils import *


class SLearner(nn.Module):
    def __init__(self, input_dim, multi_treat=False):
        super(SLearner, self).__init__()

        l2_reg = 0.0
        activation = F.elu
        batch_norm = False
        hidden_unit = 64
        dropout = 0

        self.batch_norm_1d = True
        self.multi_treat = multi_treat

        self.norm = nn.BatchNorm1d(hidden_unit)
        self.in_proj = nn.Linear(input_dim, hidden_unit)

        common_units, common_dropout = get_units_dropout_list(hidden_unit, 1, dropout)
        common_units[0] += 1
        self.first_output = MLP(hidden_units=common_units, activation=activation, dnn_dropout=common_dropout, l2_reg=l2_reg, use_batch_norm=batch_norm)

        tower_units, tower_dropout = get_units_dropout_list(hidden_unit, 3, dropout)
        tower_units[0] += 1
        self.out_fc = MLP(hidden_units=tower_units, activation=activation, dnn_dropout=tower_dropout, l2_reg=l2_reg, use_batch_norm=batch_norm)
        self.classifier = nn.Linear(tower_units[-1], 1)

    def forward(self, feat_vals, treatment, training=None):
        if treatment.dim() == 1:
            treatment = treatment.unsqueeze(1)
        feat_vals = self.in_proj(feat_vals)

        if self.batch_norm_1d:
            feat_vals = self.norm(feat_vals)

        if training:
            tower_inputs = torch.cat([feat_vals, treatment], dim=1)
            y_first_output = self.first_output(tower_inputs)
            y_output = self.out_fc(y_first_output)
            y_logits = self.classifier(y_output).view(-1, 1)  # (batch_size, 1)
            y = torch.sigmoid(y_logits)
            return y
        else:
            pred_control = torch.zeros_like(treatment)
            pred_control_inputs = torch.cat([feat_vals, pred_control], dim=1)
            y0_first_output = self.first_output(pred_control_inputs)
            y0_output = self.out_fc(y0_first_output)
            y0_logits = self.classifier(y0_output).view(-1, 1)  # (batch_size, 1)
            y0 = torch.sigmoid(y0_logits)

            pred_treat = torch.tensor([[1, 0]], device=treatment.device).expand_as(treatment) if self.multi_treat else torch.ones_like(treatment)
            pred_treat_inputs = torch.cat([feat_vals, pred_treat], dim=1)
            y1_first_output = self.first_output(pred_treat_inputs)
            y1_output = self.out_fc(y1_first_output)
            y1_logits = self.classifier(y1_output).view(-1, 1)  # (batch_size, 1)
            y1 = torch.sigmoid(y1_logits)

            if self.multi_treat:
                pred_treat2 = torch.tensor([[1, 1]], device=treatment.device).expand_as(treatment)
                pred_treat_inputs2 = torch.cat([feat_vals, pred_treat2], dim=1)
                y2_first_output = self.first_output(pred_treat_inputs2)
                y2_output = self.out_fc(y2_first_output)
                y2_logits = self.classifier(y2_output).view(-1, 1)  # (batch_size, 1)
                y2 = torch.sigmoid(y2_logits)
                y1 = [y1, y2]

            return y0, y1, None, None
        
    def calculate_loss(self, feature_list, is_treat, label_list):
        if label_list.dim() == 1:
            label_list = label_list.unsqueeze(1)
        y_pred = self.forward(feature_list, is_treat, training=True)
        return bce_loss_single(label_list, y_pred)
