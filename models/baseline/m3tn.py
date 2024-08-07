import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_models import resnet18


class ClsHead(nn.Module):
    def __init__(self, in_dim, num_cls=1):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_cls)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        return self.fc(self.pool(x).squeeze())


class MMOE(nn.Module):
    def __init__(self, encoder_class, num_experts, num_treats, enc_kwargs, in_feat):
        r"""
        a Multi-gate MOE to encoder features
        Args:
            encoder_class (nn.Module, list): encoder class of each expert. Can be a neural net or another MOE
            num_experts (int): number of experts
            task_names (list): names of tasks
            enc_kwargs (dict): dict of encoder kwargs that corresponds to each task
            in_feat (int): input feature number
        """
        super().__init__()
        self.num_experts = num_experts
        self.num_treats = num_treats

        if len(enc_kwargs) > 1:
            # each expert will have separate kwargs
            assert len(enc_kwargs) == num_experts == len(encoder_class), "each expert should have corresponding args"
            self.expert_shared = nn.ModuleList([enc(**arg) for enc, arg in zip(encoder_class, enc_kwargs.values())])
        else:
            self.expert_shared = nn.ModuleList([encoder_class(**enc_kwargs['all']) for _ in range(num_experts)])

        self.gate_specific = nn.ModuleList([nn.Sequential(nn.Linear(in_feat, self.num_experts), nn.Softmax(dim=-1)) for _ in range(num_treats + 1)])
        
    def forward(self, inputs: torch.tensor):
        experts_shared_rep = torch.stack([e(inputs) for e in self.expert_shared])
            
        out = []
        for treat in range(self.num_treats + 1):
            selector = self.gate_specific[treat](torch.flatten(inputs, start_dim=1))
            gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
            out.append(gate_rep)
        return out
    
    def get_share_params(self):
        return self.expert_shared.parameters()

    def zero_grad_share_params(self):
        self.expert_shared.zero_grad(set_to_none=False)
        

class M3TN(nn.Module):
    def __init__(self, input_dim, num_treats):
        super().__init__()
        self.num_treats = num_treats
        self.mmoe = MMOE(encoder_class=resnet18, num_experts=4, in_feat=input_dim, num_treats=num_treats,
                            enc_kwargs={'all': {'hidden_dim': 16, 'out_dim': None, 'drop': 0.2}})
        self.y0_head = ClsHead(in_dim=128)
        self.heads = nn.ModuleList([ClsHead(in_dim=128) for _ in range(num_treats)])
        
    def forward(self, x):
        x = self.mmoe(x)
        if self.num_treats > 1:
            return self.y0_head(x[0]), [self.heads[i](x[i + 1]) for i in range(len(self.heads))], None, None
        else:
            return self.y0_head(x[0]), self.heads[0](x[1]), None, None

    def calculate_loss(self, x, t_true, y_true):
        y0, yk, _, _ = self.forward(x)
        # only implemneted for dual treatments
        if self.num_treats > 1:
            idx_ctrl, idx_5ai, idx_9ai = (t_true[:, 0] == 0), (t_true[:, 0] == 1) & (t_true[:, 1] == 0), (t_true[:, 1] == 1)
            loss = dragonnet_loss(y_true[idx_ctrl | idx_5ai], t_true[:, 0][idx_ctrl | idx_5ai],  y0[idx_ctrl | idx_5ai], yk[0][idx_ctrl | idx_5ai])
            loss += dragonnet_loss(y_true[idx_ctrl | idx_9ai], t_true[:, 1][idx_ctrl | idx_9ai], y0[idx_ctrl | idx_9ai], yk[1][idx_ctrl | idx_9ai])
        else:
            loss = dragonnet_loss(y_true, t_true,  y0, yk)
        return loss
        
        
def dragonnet_loss(y_true, t_true, y0_pred, y1_pred, alpha=1.0):
    loss0 = ((1. - t_true) * (y_true - y0_pred).pow(2)).sum()
    loss1 = (t_true * (y_true - y1_pred).pow(2)).sum()
    loss_y = loss0 + loss1
    return loss_y