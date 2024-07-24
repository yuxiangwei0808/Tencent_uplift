import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp, ClassifierHead

from .base_models import *
from .mmoe import MMOE
from .multi_head_attention import MultiHeadAttn


def process_treat(treat, version):
    if version == 1:  # 5AI/control -> tau, 9AI/control -> tau_s
        t = torch.clone(treat[:, 0])
        t[treat[:, 1] == 1] = 0
        ts = treat[:, 1]
    elif version == 2:  # treat/control -> tau, 9AI/control -> tau_s
        t = treat[:, 0]
        ts = treat[:, 1]
    elif version == 3:  # control/5AI/9AI -> tau/tau_s
        t = torch.clone(treat[:, 0])
        t[treat[:, 1] == 1] = 2
        ts = torch.clone(treat[:, 0])
        ts[treat[:, 1] == 1] = 2
    elif version == 4:
        t = torch.clone(treat[:, 0])
        t[treat[:, 1] == 1] = 2
        ts = torch.clone(treat[:, 1])
    elif version == 5:
        t = torch.clone(treat[:, 0])
        ts = torch.clone(treat[:, 0])
        ts[treat[:, 1] == 1] = 2

    return t, ts


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
                 log_tensors=False,
                 combined_input=False,
                 combined_output=False,
                 process_treats=False,
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
                
        self.task_names = task_names
        self.user_enc = user_feat_enc
        self.treatment_enc = treat_feat_enc
        self.treatment_enc_s = treat_feat_enc_s
        self.assert_success = False
        self.log_tensors = log_tensors
        self.combined_input = combined_input
        self.combined_output = combined_output
        self.process_treats = process_treats
        self.u_tau = nn.ModuleDict({name: ClsHead(u_dim, nc) for name, nc in zip(task_names, num_cls)})  # assume tasks are all binary or regression

        u_dim *= len(self.task_names)  # since we cat the output of different tasks
        
        self.Q_w = nn.Linear(t_dim, tu_dim, bias=True)
        self.K_w = nn.Linear(u_dim, tu_dim, bias=True)
        self.V_w = nn.Linear(u_dim, tu_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        
        self.self_attention = MultiHeadAttn(embed_dim=tu_dim, qdim=t_dim, kdim=u_dim, vdim=u_dim, num_head=4, attn_drop=0.2)
        
        self.tu_enhance = Mlp(tu_dim, hidden_features=tu_dim // 2, norm_layer=tu_enhance_norm)
        
        self.tu_logit = ClsHead(tu_dim, 1)
        self.tu_tau   = ClsHead(tu_dim, 1)

        if treat_feat_enc_s is not None:
            self.self_attention_s = MultiHeadAttn(embed_dim=tu_dim, qdim=t_dim, kdim=u_dim, vdim=u_dim, num_head=4, attn_drop=0.2)
            self.tu_enhance_s = nn.Sequential(
                Mlp(tu_dim, hidden_features=tu_dim // 2, norm_layer=tu_enhance_norm),
            )
            self.tu_logit_s = ClsHead(tu_dim, 1)

    def forward(self, user_input, treat=None, treat_s=None):
        if self.combined_input:  # combine treat into user_input
            treat = user_input[:, -1]
            user_input = user_input[:, :-1]

        if self.process_treats:
            treat, treat_s = process_treat(treat, version=2)

        if isinstance(self.user_enc, nn.ModuleList):
            # group features and apply different encoders
            assert len(self.user_enc) == len(user_input), "number of groups of input should equal to the number of encoders"
            user_feats = [self.user_enc[i](user_input[i]) for i in range(len(user_input))]
            if not isinstance(user_feats[0], dict):
                user_feats = [{self.task_names[0]: feat} for feat in user_feats]

            if not self.assert_success:
                assert all(set(d.keys()) == set(self.task_names) for d in user_feats), "all encodes features should have the same tasks"
                self.assert_success = True
            
            user_feat = {}
            for task in self.task_names:
                user_feat[task] = torch.cat([d[task] for d in user_feats], dim=-1)
        else:
            user_feat = self.user_enc(user_input)  # dict of tensors
            user_feat = {self.task_names[0]: user_feat} if not isinstance(user_feat, dict) else user_feat

        if isinstance(self.treatment_enc, nn.Embedding):
            treat = treat.to(torch.long)
            treat_s = treat_s.to(torch.long) if treat_s is not None else treat_s
        treat_feat = self.treatment_enc(treat)  # B N or B T N
        treat_feat_s = self.treatment_enc_s(treat_s) if treat_s is not None else None

        u_logit = {task: self.u_tau[task](user_feat[task]).squeeze() for task in self.task_names}
                
        # TODO try add or weighted add (linear) insteand of cat
        # TODO maybe we need to calculate a tu_logit for each task
        user_feat_cat = torch.cat([t for t in user_feat.values()], dim=1)  # B C N
        user_feat_norm  = user_feat_cat / (torch.linalg.norm(user_feat_cat, dim=1, keepdim=True) + 1e-6)

        treat_feat = treat_feat.unsqueeze(-1) if treat_feat.dim() == 2 else treat_feat.transpose(-1, -2)  # B N C, C = 1
        treat_feat_norm = treat_feat / (torch.linalg.norm(treat_feat, dim=-1, keepdim=True) + 1e-6)

        tu_feat, _ = self.self_attention(treat_feat_norm, user_feat_norm.transpose(-1, -2), user_feat_norm.transpose(-1, -2))

        # enhance treatment-user feature
        tu_feat_enhanced = self.tu_enhance(tu_feat).transpose(-1, -2)  # B C N
        
        # regularizer
        tu_tau = self.tu_tau(tu_feat_enhanced).squeeze()
        tu_logit = self.tu_logit(tu_feat_enhanced).squeeze()
        tu_logit_s = None

        if treat_s is not None:
            # more specific treatments
            treat_feat_s = treat_feat_s.unsqueeze(-1) if treat_feat_s.dim() == 2 else treat_feat_s.transpose(-1, -2)  # B N C, C = 1
            treat_feat_s_norm = treat_feat_s / (torch.linalg.norm(treat_feat_s, dim=-1, keepdim=True) + 1e-6)
            tu_feat_s, _ = self.self_attention_s(treat_feat_s_norm, user_feat_norm.transpose(-1, -2), user_feat_norm.transpose(-1, -2))
            tu_feat_enhanced_s = self.tu_enhance_s(tu_feat_s).transpose(-1, -2)

            tu_logit_s = self.tu_logit_s(tu_feat_enhanced_s).squeeze()

        if self.log_tensors:
            return u_logit, tu_tau, tu_logit, tu_logit_s, tu_feat
        if self.combined_output:
            return u_logit['label_nextday_login'] * (1 - treat) + (tu_logit + u_logit['label_nextday_login'].detach()) * treat

        return u_logit, tu_tau, tu_logit, tu_logit_s
        
    def calculate_loss(self, user_input, treat, y_true, target_task):
        # TODO try different weight balancing for MTL
        # TODO multi-treatment        
        u_logit, tu_tau, tu_logit, tu_logit_s = self.forward(user_input, treat)
        
        # tu_logit += u_logit['nextday_login'].detach()  # EFIN
        # tu_logit_s += u_logit['nextday_login'].detach()
        
        """For multi-task, there are two design choices: treat each task as regression and use mse, or treat each task as classification and use bce and projects label to 0-n"""
        if treat.dim() == 1:  # single treatment
            loss1 = sum([F.mse_loss((1 - treat) * u_logit[t].squeeze() + treat * (tu_logit + u_logit[t].detach()).squeeze(), y_true[:, i]) for i, t in enumerate(target_task)])  # binary
        else:
            # Control * y0 + Treatment * (y0 + tau)
            # loss1 = sum([F.mse_loss((1 - treat[:, 0]) * u_logit[t].squeeze() + treat[:, 0] * (tu_logit + u_logit[t].detach()).squeeze(), y_true[:, i]) for i, t in enumerate(target_task)])  # v0
            # Control * y0 + (5AI * (y0 + tau) + 9AI * (y0 + tau_s)).
            # TODO: try 1. 5AI/control -> tau, 9AI/control -> tau_s  2. treat/control -> tau, 9AI/control -> tau_s  3. control/5AI/9AI -> tau/tau_s
            # loss1 = sum([F.mse_loss((1 - treat[:, 0]) * u_logit[t].squeeze() + treat[:, 0] * ((1 - treat[:, 1]) * (tu_logit + u_logit[t].detach()).squeeze() + treat[:, 1] * (tu_logit_s + u_logit[t].detach()).squeeze()), y_true[:, i]) for i, t in enumerate(target_task)])  # tu_logit_s directly models tau of 9AI, v1
            # Control * y0 + Treatment * ((y0 + tau) + 9AI * tau_s).
            # TODO try 2. 5AI/control -> tau, 5AI/9AI -> tau_s -> tau_s 3. control/5AI/9AI -> tau/tau_s  4. control/5AI/9AI -> tau, 5AI/9AI -> tau_s
            loss1 = sum([F.mse_loss((1 - treat[:, 0]) * u_logit[t].squeeze() * (1 - F.sigmoid(tu_tau)) + F.sigmoid(tu_tau) * treat[:, 0] * ((tu_logit + u_logit[t].detach()).squeeze() + treat[:, 1] * tu_logit_s.squeeze()), y_true[:, i]) for i, t in enumerate(target_task)])  # tu_logit_s models difference over 5AI
            # Control * y0 + Treatment * (y0 + tau + tau_s). 5. treat/control -> tau, 5AI/9AI/control -> tau_s
            # loss1 = sum([F.mse_loss((1 - treat[:, 0]) * u_logit[t].squeeze() + treat[:, 0] * ((tu_logit + tu_logit_s + u_logit[t].detach()).squeeze()), y_true[:, i]) for i, t in enumerate(target_task)])  # tu_logit_s models difference over treat
        
        # ESN IPW regularize
        # tu_tau = torch.sigmoid(tu_tau)
        # loss2 = F.binary_cross_entropy_with_logits(tu_logit * tu_tau, y_true * treat) + F.binary_cross_entropy_with_logits(u_logit['nextday_login'] * (1 - tu_tau), y_true * (1 - treat))
        # loss2 = F.mse_loss(tu_logit * torch.sigmoid(tu_tau), y_true * treat) + F.mse_loss(u_logit['nextday_login'] * (1 - torch.sigmoid(tu_tau)), y_true * (1 - treat))
        # loss2 = F.mse_loss((1 - treat) * u_logit['nextday_login'].squeeze() * (1 - tu_tau) + treat * tu_logit.squeeze() * tu_tau, y_true)
        # loss_pr = F.binary_cross_entropy_with_logits(tu_tau, treat)
        
        # interfere the model not to directly classify samples for treatments
        # loss3 = F.binary_cross_entropy_with_logits(tu_tau.squeeze(), 1 - treat)  # binary

        # lower tau for active players but not necessarily vice versa; set pre30 > 25 (0.58) as high active
        treat = treat[:, 0] if treat.dim() > 1 else treat
        loss5 = torch.clamp(F.mse_loss(1 - tu_logit, torch.ones_like(tu_logit, device=tu_logit.device), reduction='none') * (y_true[:, 1] > 0.6) * treat, min=0.2).mean()

        # determine wheter treatment
        # loss6 = F.binary_cross_entropy_with_logits(tu_tau, treat) * 0.1
        
        return loss1 + loss5