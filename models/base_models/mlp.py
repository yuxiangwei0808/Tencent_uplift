import torch
import torch.nn as nn
import torchvision


class MLP(nn.Module):
    def __init__(self, in_chans: int, hidden_chans: list, drop_rate: int=0, transpose=False):
        super().__init__()
        
        self.mlp = torchvision.ops.MLP(in_channels=in_chans, hidden_channels=hidden_chans, 
                                        norm_layer=nn.LayerNorm, dropout=drop_rate)        
        self.transpose = transpose

    def forward(self, x):
        x = self.mlp(x)
        if self.transpose:
            x = x.permute(0, 2, 1)
        return x
    
    
if __name__ == '__main__':
    x = torch.ones(4, ).to(torch.long)
    model = MLP(in_chans=1, hidden_chans=[16, 32, 64])
    print(model(x).shape)