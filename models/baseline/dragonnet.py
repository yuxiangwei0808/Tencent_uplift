import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class DragonNet(nn.Module):
    """
    Base Dragonnet model.

    Parameters
    ----------
    input_dim: int
        input dimension for covariates
    shared_hidden: int
        layer size for hidden shared representation layers
    outcome_hidden: int
        layer size for conditional outcome layers
    """
    def __init__(self, input_dim: int, is_regularized: bool, shared_hidden: int = 512, outcome_hidden: int = 256):
        super(DragonNet, self).__init__()
        self.is_regularized = is_regularized
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
        self.y1_layers = self._make_outcome_layers(shared_hidden, outcome_hidden)

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
        y1 = self.y1_layers(z)

        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])

        return y0, y1, t_pred, eps

    def calculate_loss(self, inputs: torch.Tensor, t_true: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y0, y1, t_pred, eps = self.forward(inputs)
        if self.is_regularized:
            loss = tarreg_loss(y_true, t_true, t_pred, y0, y1, eps)
        else:
            loss = dragonnet_loss(y_true, t_true, t_pred, y0, y1)
        
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
    t = torch.randn(16)
    y = torch.randn(16)
    
    model = DragonNet(input_dim=622)
    print(model(feat))
    print(model.calculate_loss(feat, y, t, is_regularized=True))
    print(model.calculate_loss(feat, y, t, is_regularized=False))