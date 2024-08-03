import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class HydraNet(nn.Module):
    def __init__(self, input_dim: int, is_regularized: bool, shared_hidden: int = 512, outcome_hidden: int = 256, num_treats=1):
        super(HydraNet, self).__init__()
        self.is_regularized = is_regularized
        self.num_treats = num_treats
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, shared_hidden),
            nn.ReLU()
        )

        self.treat_out = nn.Linear(shared_hidden, 1)

        self.y0_layers = self._make_outcome_layers(shared_hidden, outcome_hidden)
        self.yt_layers = nn.ModuleList([self._make_outcome_layers(shared_hidden, outcome_hidden) for _ in range(num_treats)])

        self.epsilon = nn.Linear(1, 1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)

    def _make_outcome_layers(self, shared_hidden: int, outcome_hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(shared_hidden, outcome_hidden),
            nn.ReLU(),
            nn.Linear(outcome_hidden, outcome_hidden),
            nn.ReLU(),
            nn.Linear(outcome_hidden, 1)
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method to train model.

        Parameters
        ----------
        inputs: torch.Tensor
            covariates

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        z = self.shared_layers(inputs)
        t_pred = torch.sigmoid(self.treat_out(z))

        y0 = self.y0_layers(z)
        yt = [self.yt_layers[i](z) for i in range(self.num_treats)]

        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])

        return y0, yt, t_pred, eps

    def calculate_loss(self, inputs: torch.Tensor, t_true: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y0, yt, t_pred, eps = self.forward(inputs)
        if t_true.dim() > 1 and t_true.shape[1] > 1:
            idx_ctrl, idx_5ai, idx_9ai = (t_true[:, 0] == 0), (t_true[:, 0] == 1) & (t_true[:, 1] == 0), (t_true[:, 1] == 1)
            if self.is_regularized:
                loss = tarreg_loss(y_true[idx_ctrl | idx_5ai], t_true[:, 0][idx_ctrl | idx_5ai], t_pred[idx_ctrl | idx_5ai], y0[idx_ctrl | idx_5ai], yt[0][idx_ctrl | idx_5ai])
                loss += tarreg_loss(y_true[idx_ctrl | idx_9ai], t_true[:, 1][idx_ctrl | idx_9ai], t_pred[idx_ctrl | idx_9ai], y0[idx_ctrl | idx_9ai], yt[1][idx_ctrl | idx_9ai])
            else:
                loss = dragonnet_loss(y_true[idx_ctrl | idx_5ai], t_true[:, 0][idx_ctrl | idx_5ai], t_pred[idx_ctrl | idx_5ai], y0[idx_ctrl | idx_5ai], yt[0][idx_ctrl | idx_5ai])
                loss += dragonnet_loss(y_true[idx_ctrl | idx_9ai], t_true[:, 1][idx_ctrl | idx_9ai], t_pred[idx_ctrl | idx_9ai], y0[idx_ctrl | idx_9ai], yt[1][idx_ctrl | idx_9ai])
        else:
            if self.is_regularized:
                loss = tarreg_loss(y_true, t_true, t_pred, y0, yt[0], eps)
            else:
                loss = dragonnet_loss(y_true, t_true, t_pred, y0, yt[0])
            
        return loss
    

def dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, alpha=1.0):
    """
    Generic loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """
    t_pred = (t_pred + 0.01) / 1.02
    loss_t = F.binary_cross_entropy(t_pred.squeeze(), t_true).sum()

    loss0 = ((1. - t_true) * (y_true - y0_pred).pow(2)).sum()
    loss1 = (t_true * (y_true - y1_pred).pow(2)).sum()
    loss_y = loss0 + loss1

    loss = loss_y + alpha * loss_t

    return loss


def tarreg_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0, beta=1.0):
    """
    Targeted regularisation loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    beta: float
        targeted regularization hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """
    vanilla_loss = dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, alpha)
    t_pred = (t_pred + 0.01) / 1.02

    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    h = (t_true / t_pred) - ((1 - t_true) / (1 - t_pred))

    y_pert = y_pred + eps * h
    targeted_regularization = ((y_true - y_pert) ** 2).sum()

    # final loss
    loss = vanilla_loss + beta * targeted_regularization
    return loss


if __name__ == '__main__':
    feat = torch.randn(16, 622)
    t = torch.randint(0, 2, (16, 2), dtype=torch.float)
    y = torch.ones(16)
    
    model = HydraNet(input_dim=622, num_treats=2, is_regularized=False) 
    print(model(feat))
    print(model.calculate_loss(feat, t, y ))