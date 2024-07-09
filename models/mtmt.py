import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp, ClassifierHead

from .base_models import *
from .mmoe import MMOE
from .multi_head_attention import MultiHeadAttn


class ClsHead(nn.Module):
    def __init__(self, in_dim, num_cls=1):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_cls)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        return self.fc(self.pool(x).squeeze())
    

class MTMT(nn.Module):
    def __init__(self,
                 user_feat_enc: nn.Module,
                 treat_feat_enc: nn.Module,
                 task_names: list,
                 t_dim: int,
                 u_dim: int,
                 tu_dim: int,
                 tu_enhance_norm: bool = None,
                 treat_feat_enc_s: nn.Module = None,
                 num_cls: list = [1],
                ):
        r"""
        Args:
            user_feat_enc (nn.Module, list): encoder class to encode features. The encoder can encode all features, or encode feature groups separately via MOE.
            treat_feat_enc (nn.Module): treatment feature encoder
            task_names (list): names of tasks
            num_cls (list): number of classes of each task
            t_in_dim (int): input dimension of treatment feature
            t_dim (int): dimension of the encoded treatment feature
            u_in_dims (int, list): input dimension(s) of user features
            u_dim (int): dimension of the encoded user feature
            tu_dim (int): dimension of the self-attention for treatment-user feature self-attention interaction
            
        """
        super().__init__()
                
        self.task_names = set(task_names)
        self.user_enc = user_feat_enc
        self.treatment_enc = treat_feat_enc
        self.treatment_enc_s = treat_feat_enc_s
        self.assert_success = False
        self.u_tau = nn.ModuleDict({name: ClsHead(u_dim, nc) for name, nc in zip(task_names, num_cls)})  # assume tasks are all binary or regression

        u_dim *= len(self.task_names)  # since we cat the output of different tasks
        
        self.Q_w = nn.Linear(t_dim, tu_dim, bias=True)
        self.K_w = nn.Linear(u_dim, tu_dim, bias=True)
        self.V_w = nn.Linear(u_dim, tu_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        
        self.self_attention = MultiHeadAttn(embed_dim=tu_dim, qdim=t_dim, kdim=u_dim, vdim=u_dim, num_head=4, attn_drop=0.2)
        
        self.tu_enhance = nn.Sequential(
            # TODO try add layers here or use attention
            Mlp(tu_dim, hidden_features=tu_dim // 2, norm_layer=tu_enhance_norm),
            # MLP(tu_dim, [tu_dim * 2, tu_dim * 4, tu_dim * 2, tu_dim], drop_rate=0.2)
        )
        
        self.tu_logit = ClsHead(tu_dim, 1)
        self.tu_tau   = ClsHead(tu_dim, 1)

        if treat_feat_enc_s is not None:
            self.tu_enhance_s = nn.Sequential(
                Mlp(tu_dim, hidden_features=tu_dim // 2, norm_layer=tu_enhance_norm),
            )
            self.tu_logit_s = ClsHead(tu_dim, 1)
            self.tu_tau_s   = ClsHead(tu_dim, 1)


    def forward(self, user_input, treat, treat_s=None):
        if isinstance(self.user_enc, nn.ModuleList):
            # group features and apply different encoders
            assert len(self.user_enc) == len(user_input), "number of groups of input should equal to the number of encoders"
            user_feats = [self.user_enc[i](user_input[i]) for i in range(len(user_input))]
            if not isinstance(user_feats[0], dict):
                user_feats = [{'label_nextday_login': feat} for feat in user_feats]

            if not self.assert_success:
                assert all(set(d.keys()) == self.task_names for d in user_feats), "all encodes features should have the same tasks"
                self.assert_success = True
            
            user_feat = {}
            for task in self.task_names:
                user_feat[task] = torch.cat([d[task] for d in user_feats], dim=-1)
        else:
            user_feat = self.user_enc(user_input)  # dict of tensors
            user_feat = {'label_nextday_login': user_feat} if not isinstance(user_feat, dict) else user_feat

        if isinstance(self.treatment_enc, nn.Embedding):
            treat = treat.to(torch.long)
            treat_s = treat_s.to(torch.long) if treat_s is not None else treat_s
        
        treat_feat = self.treatment_enc(treat)  # B N or B T N
        treat_feat_s = self.treatment_enc_s(treat_s) if treat_s is not None else None
        
        u_logit = {task: self.u_tau[task](user_feat[task]).squeeze() for task in self.task_names}
        
        # TODO try add or weighted add (linear) insteand of cat
        # TODO maybe we need to calculate a tu_logit for each task
        # TODO ablate norm dimension
        user_feat_cat = torch.cat([t for t in user_feat.values()], dim=1)  # B C N
        user_feat_norm  = user_feat_cat / (torch.linalg.norm(user_feat_cat, dim=0, keepdim=True) + 1e-6)

        treat_feat_norm = treat_feat / (torch.linalg.norm(treat_feat, dim=0, keepdim=True) + 1e-6)
        # user_feat_norm = user_feat_cat
        # treat_feat_norm = treat_feat

        treat_feat_norm = treat_feat_norm.unsqueeze(-1) if treat_feat_norm.dim() == 2 else treat_feat_norm.transpose(-1, -2)
        tu_feat, _ = self.self_attention(treat_feat_norm, user_feat_norm.transpose(-1, -2), user_feat_norm.transpose(-1, -2))

        # enhance treatment-user feature
        tu_feat_enhanced = self.tu_enhance(tu_feat).transpose(-1, -2)  # B C N
        
        # regularizer
        tu_tau = self.tu_tau(tu_feat_enhanced).squeeze()
        tu_logit = self.tu_logit(tu_feat_enhanced).squeeze()

        if treat_s is not None:
            # more specific treatments
            treat_feat_norm_s = treat_feat_s / (torch.linalg.norm(treat_feat_s, dim=1, keepdim=True) + 1e-6)
            tu_feat_s, _ = ...
            tu_feat_enhanced_s = self.tu_enhance_s(tu_feat_s).transpose(-1, -2)
            
            tu_tau_s = self.tu_tau_s(tu_feat_enhanced_s)
            tu_logit_s = self.tu_logit_s(tu_feat_enhanced_s)
            return u_logit, tu_tau, tu_logit, tu_tau_s, tu_logit_s
        
        return u_logit, tu_tau, tu_logit, None
        
    def calculate_loss(self, user_input, treat, y_true, target_task):
        # TODO try different weight balancing for MTL
        # TODO multi-treatment
        
        u_logit, tu_tau, tu_logit, tu_logit_s = self.forward(user_input, treat)
        
        # tu_logit += u_logit['nextday_login'].detach()  # EFIN
        # tu_logit_s += u_logit['nextday_login'].detach()

        """For multi-task, there are two design choices: treat each task as regression and use mse, or treat each task as classification and use bce and projects label to 0-n"""

        loss1 = sum([F.mse_loss((1 - treat) * (u_logit[t] + tu_tau).squeeze() + treat * (tu_logit + u_logit[t].detach()).squeeze(), y_true[:, i]) for i, t in enumerate(target_task)])  # binary
        # loss1 = F.mse_loss((1 - treat[:, 0]) * u_logit['nextday_login'].squeeze() + treat[:, 0] * tu_logit.squeeze(), y_true)  # v0
        # loss1 = F.mse_loss((1 - treat[:, 0]) * u_logit['nextday_login'].squeeze() + treat[:, 0] * ((1 - treat[:, 1]) * tu_logit.squeeze() + treat[:, 1] * tu_logit_s.squeeze()), y_true)  # tu_logit_s directly models tau, v1
        # loss1 = F.mse_loss(1 - treat[:, 0]) * u_logit['nextday_login'].squeeze() + treat[:, 0] * (tu_logit.squeeze() + treat[:, 1] * tu_logit_s.squeeze()), y_true)  # tu_logit_s models difference over 5AI
        
        # ESN IPW regularize
        # tu_tau = torch.sigmoid(tu_tau)
        # loss2 = F.binary_cross_entropy_with_logits(tu_logit * tu_tau, y_true * treat) + F.binary_cross_entropy_with_logits(u_logit['nextday_login'] * (1 - tu_tau), y_true * (1 - treat))
        # loss2 = F.mse_loss(tu_logit * torch.sigmoid(tu_tau), y_true * treat) + F.mse_loss(u_logit['nextday_login'] * (1 - torch.sigmoid(tu_tau)), y_true * (1 - treat))
        # loss2 = F.mse_loss((1 - treat) * u_logit['nextday_login'].squeeze() * (1 - tu_tau) + treat * tu_logit.squeeze() * tu_tau, y_true)
        # loss_pr = F.binary_cross_entropy_with_logits(tu_tau, treat)

        
        # interfere the model not to directly classify samples for treatments
        # loss3 = F.binary_cross_entropy_with_logits(tu_tau.squeeze(), 1 - treat)  # binary

        # inverse probability metric loss to mitigate gap between treated and control y0
        # loss4 = F.mse_loss((u_logit['nextday_login'] * treat).mean(), (u_logit['nextday_login'] * (1 - treat)).mean())

        # 
        # loss5 = torch.clamp(F.mse_loss(1 - tu_logit, y_true, reduction='none') * treat, min=0.8**2).mean()
        
        # return loss1 + loss2 * (loss1 / loss2).detach() + loss3 * (loss1 / loss3).detach()
        return loss1