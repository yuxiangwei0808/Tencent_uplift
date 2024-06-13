import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_chans: int, hidden_chans: list, drop_rate: int=0):
        self.cnns = nn.ModuleList(nn.Sequential(
            nn.Conv1d(in_chans, hidden_chans[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_chans[0]),
            nn.ReLU(),
            nn.Dropout1d(drop_rate),
            nn.Conv1d(hidden_chans[0], hidden_chans[0], kernel_size=1, stride=2),
            nn.BatchNorm1d(hidden_chans[0]),  # TODO delete this
        ))
        
        for i in range(1, len(hidden_chans)):
            self.cnns.append(nn.Sequential(
                nn.Conv1d(hidden_chans[i - 1], hidden_chans[i], kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_chans[i]),
                nn.ReLU(),
                nn.Dropout1d(drop_rate),
                nn.Conv1d(hidden_chans[i], hidden_chans[i], kernel_size=1, stride=2),
                nn.BatchNorm1d(hidden_chans[i]),  # TODO delete this
            ))
        
    def forward(self, x):
        return self.cnns(x)


class CNNBottleneck(nn.Module):
    # takes from convnextv2
    def __init__(self, in_chans: int, hidden_chans: list, drop_rate: int=0):
        self.cnns = nn.ModuleList(nn.Sequential(
            nn.Conv1d(in_chans, in_chans, kernel_size=7, padding=3, groups=in_chans),
            nn.BatchNorm1d(in_chans),
            nn.Conv1d(in_chans, 4 * in_chans),
            nn.ReLU(),
            nn.BatchNorm1d(in_chans * 4),
            nn.Conv1d(4 * in_chans, hidden_chans[0], stride=2), 
            nn.Dropout1d(drop_rate),
        ))
        
        for i in range(1, len(hidden_chans)):
            self.cnns.append(nn.Sequential(
                nn.Conv1d(hidden_chans[i - 1], hidden_chans[i - 1], kernel_size=7, padding=3, groups=in_chans),
                nn.BatchNorm1d(hidden_chans[i - 1]),
                nn.Conv1d(hidden_chans[i - 1], 4 * hidden_chans[i - 1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_chans[i - 1] * 4),
                nn.Conv1d(4 * hidden_chans[i -1], hidden_chans[i], stride=2), 
                nn.Dropout1d(drop_rate),
            ))
        
    def forward(self, x):
        return self.cnns(x)


def cnn_simple():
    return CNN(in_chans=0, hidden_chans=[0,0,0,0], drop_rate=0.1)

def cnn_bottleneck_simple():
    return CNNBottleneck(in_chans=0, hidden_chans=[0, 0, 0, 0], drop_rate=0.1)