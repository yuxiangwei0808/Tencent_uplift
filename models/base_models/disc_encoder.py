import torch
import torch.nn as nn


class DiscEncoder(nn.Module):
    """Feature encoder for discrete one-hot features using nn.Embedding"""
    def __init__(self, feature_groups, embed_dim, enc: nn.Module = None):
        super().__init__()
        self.groups = feature_groups.groups  # dict
        self.indices = feature_groups.indices  # list

        self.embeders = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=len(ind), embedding_dim=embed_dim) for name, ind in self.groups.items()
        })

        self.enc = enc if enc is not None else nn.Identity()  # subsequent discerete feature encoder
    
    def forward(self, x):
        # takes in the whole batch without padding and grouping
        x = [self.embeders[name](torch.argmax(x[:, self.groups[name]], dim=1)) for name in self.embeders]
        x = torch.cat(x, dim=1)
        
        assert x.dim() == 2
        
        return self.enc(x)
