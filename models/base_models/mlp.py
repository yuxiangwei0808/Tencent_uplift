import torch
import torch.nn as nn
import torchvision


class MLP(nn.Module):
    def __init__(self, in_chans: int, hidden_chans: list, norm_layer: nn.Module, drop_rate: int=0, transpose=False, encoder=None):
        super().__init__()
        
        self.mlp = torchvision.ops.MLP(in_channels=in_chans, hidden_channels=hidden_chans, 
                                        norm_layer=norm_layer, dropout=drop_rate)        
        self.transpose = transpose  # input B C N
        self.encoder = encoder

    def forward(self, x):
        if self.encoder is not None:
            if x.dim() == 2:
                assert self.transpose, "when input dim=2, emb shape is B C N, expect transpose"
            x = self.encoder(x.to(torch.long))
        x = x.permute(0, 2, 1) if self.transpose and x.dim() > 2 else x
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # B N C
            self.transpose = True

        x = self.mlp(x)
        x = x.permute(0, 2, 1) if self.transpose and x.dim() > 2 else x
        return x
    
    
if __name__ == '__main__':
    x = torch.ones(4, ).to(torch.long)
    model = MLP(in_chans=1, hidden_chans=[16, 32, 64])
    print(model(x).shape)