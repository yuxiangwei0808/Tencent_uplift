import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate, stride=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout1d(drop_rate),
            nn.Conv1d(out_ch, out_ch, kernel_size=1, stride=stride),
        )
    
    def forward(self, x):
        return self.enc(x)


class CNN(nn.Module):
    def __init__(self, in_chans: int, hidden_chans: list, strides: list, drop_rate: int=0):
        super().__init__()
        assert len(hidden_chans) == len(strides)
        self.cnns = nn.ModuleList([BasicBlock(in_chans, hidden_chans[0], drop_rate, strides[0])])
        
        for i in range(1, len(hidden_chans)):
            self.cnns.append(
                BasicBlock(hidden_chans[i - 1], hidden_chans[i], drop_rate, strides[i])
            )
        self.cnns = nn.Sequential(*self.cnns)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.cnns(x)


class BottleNeck(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.BatchNorm1d(in_ch),
            nn.Conv1d(in_ch, 4 * in_ch, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(in_ch * 4),
            nn.Conv1d(4 * in_ch, out_ch, kernel_size=1, stride=2), 
            nn.Dropout1d(drop_rate),
        )

    def forward(self, x):
        return self.enc(x)


class CNNBottleneck(nn.Module):
    # takes from convnextv2
    def __init__(self, in_chans: int, hidden_chans: list, drop_rate: int=0):
        super().__init__()
        self.cnns = nn.ModuleList([BottleNeck(in_chans, hidden_chans[0], drop_rate)])
        
        for i in range(1, len(hidden_chans)):
            self.cnns.append(BottleNeck(hidden_chans[i - 1], hidden_chans[i], drop_rate))
        self.cnns = nn.Sequential(*self.cnns)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.cnns(x)


def cnn_simple(hidden_chans, in_chans=1, strides=[2, 2, 2, 2]):
    return CNN(in_chans=in_chans, hidden_chans=hidden_chans, strides=strides, drop_rate=0.2)

def cnn_bottleneck_simple(hidden_chans):
    return CNNBottleneck(in_chans=1, hidden_chans=hidden_chans, drop_rate=0.2)