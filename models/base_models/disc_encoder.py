import torch
import torch.nn as nn

class feature_groups:
    groups = {
        'group_weekday': list(range(0, 7)),
        'group_time': list(range(7, 15)),
        'group_gender': list(range(15, 19)),
        'group_camp': [19, 20],
        'group_grade': list(range(21, 32)),
        'group_lane': list(range(32, 37)),
        # TODO encode district by nn.Embedding
        'group_district': list(range(37, 41)),
        'group_area': list(range(41, 76)),
        'group_r': list(range(76, 85)),
        # 'group_rank5': list(range(85, 148)),
    }
    indices = list(range(0, 85))
    # indices = list(range(0, 148))


class DiscEncoder(nn.Module):
    """Feature encoder for discrete one-hot features using nn.Embedding"""
    def __init__(self, embed_dim, out_shape='1d', enc: nn.Module = None):
        super().__init__()
        self.groups = feature_groups.groups  # dict
        self.indices = feature_groups.indices  # list
        self.out_shape = out_shape

        self.embeders = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=len(ind), embedding_dim=embed_dim) for name, ind in self.groups.items()
        })

        self.enc = enc if enc is not None else nn.Identity()  # subsequent discerete feature encoder
    
    def forward(self, x):
        # takes in the whole batch without padding and grouping
        x = [self.embeders[name](torch.argmax(x[:, self.groups[name]], dim=1)) for name in self.embeders]
        if self.out_shape == '1d':
            x = torch.cat(x, dim=1)
        else:
            x = torch.stack(x, dim=1)


        x = self.enc(x)

        return x
