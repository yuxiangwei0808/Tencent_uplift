import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def bce_loss(y_true, t_true, y0_pred, y1_pred):
    loss0 = F.binary_cross_entropy(y0_pred, y_true, reduction='none')
    loss1 = F.binary_cross_entropy(y1_pred, y_true, reduction='none')
    loss = torch.mean((1. - t_true) * loss0 + t_true * loss1)
    return loss


def bce_loss_single(y_true, y_pred):
    loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    loss = loss.mean()
    return loss


def ortho_penalty_asymmetric(params_0, params_1, normalize=False):
    if normalize:
        params_0 = F.normalize(params_0, p=2, dim=0)
        params_1 = F.normalize(params_1, p=2, dim=0)

    x_min = min(params_0.size(0), params_1.size(0))
    y_min = min(params_0.size(1), params_1.size(1))

    penalty = torch.sum((params_0[:x_min, :y_min] * params_1[:x_min, :y_min]) ** 2)
    return penalty
    
    
def get_units_dropout_list(first_units, depth, dropout_rate):
    unit_list = []
    dropout_list = []
    for i in range(depth):
        cur_units = int(first_units // math.pow(2, i))
        unit_list.append(cur_units)
        dropout_list.append(dropout_rate)
    return unit_list, dropout_list


class MLP(nn.Module):
    def __init__(self, hidden_units, activation, dnn_dropout, l2_reg, use_batch_norm=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        self.bn = nn.BatchNorm1d(hidden_units[0])

        for i, unit in enumerate(hidden_units):
            self.layers.append(nn.Linear(hidden_units[i-1] if i > 0 else hidden_units[0], unit))
            self.dropouts.append(nn.Dropout(dnn_dropout[i]))
        
        self.activation = activation
        self.l2_reg = l2_reg

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = self.activation(x)
            if self.use_batch_norm:
                x = self.bn(x)
            x = dropout(x)
        return x


class SingleMLP(nn.Module):
    def __init__(self, hidden_units, out_dim, activation, dnn_dropout, l2_reg, w_initializer=nn.init.xavier_uniform_, b_initializer=nn.init.constant_, name=None):
        super(SingleMLP, self).__init__()
        self.dnn_network = nn.Linear(hidden_units, out_dim)
        w_initializer(self.dnn_network.weight)
        b_initializer(self.dnn_network.bias, 0.)
        self.activation = activation if activation != None else nn.Identity()
        self.dropout = nn.Dropout(dnn_dropout)
        self.dropout_rates = dnn_dropout
        self.l2_reg = l2_reg

    def forward(self, inputs, training=None):
        x = self.dnn_network(inputs)
        x = self.activation(x)
        if self.dropout_rates > 0.0:
            x = self.dropout(x)
        return x
    

def represention_norm(x, is_norm):
    if is_norm:
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        x_norm = torch.sqrt(torch.clamp(x_norm, min=1e-12))
    else:
        x_norm = x * 1.0
    return x_norm
