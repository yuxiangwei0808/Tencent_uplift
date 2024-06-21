import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from timm.layers import Mlp, ClassifierHead

from .abs_mt_arch import AbsArchitecture
from .base_models import *
from .multi_head_attention import MultiHeadAttentionCustom


class ClsHead(nn.Module):
    def __init__(self, in_dim, num_cls=1):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_cls)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        return self.fc(self.pool(x).squeeze())
    

class feat_encoder(nn.Module):
    def __init__(self, u_in_dims):
        super().__init__()
        self.enc = nn.Identity()
        
        
    def forward(self, x):
        return self.enc(x)


class MMOE(AbsArchitecture):
    def __init__(self, encoder_class, num_experts, task_names, in_feats, rep_dim, rep_grad, multi_input, device, **kwargs):
        r"""
        a Multi-gate MOE to encoder features
        Args:
            encoder_class (nn.Module): the encoder class of each expert. Can be a neural net or another MOE
            num_experts (int): number of experts
            task_names (list): names of tasks
            in_feats (list, int): list of inputs features if each expert only encode a feature group
            rep_dim (int): dimension of the initially encoded representation
        """
        super().__init__(task_names, encoder_class, None, rep_grad, multi_input, device, **kwargs)
        self.num_experts = num_experts
        self.task_names = task_names
        self.rep = nn.ModuleList([nn.Linear(c, rep_dim) for c in in_feats]) if isinstance(in_feats, list) else nn.Linear(in_feats, rep_dim)
        self.expert_shared = nn.ModuleList([encoder_class(f) for f in in_feats])
        if isinstance(in_feats, list):
            assert len(in_feats) == num_experts, "each expert should have a corresponding feature length"
            self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(in_feat, self.num_experts),
                                                      nn.Softmax(dim=-1)) for task, in_feat in zip(self.task_names, in_feats)})
        else:
            self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(in_feats, self.num_experts),
                                                      nn.Softmax(dim=-1)) for task in self.task_names})
        self.decoders = nn.ModuleDict({task: nn.Linear(rep_dim, 1) for task in self.task_names})
        
    def forward(self, inputs: Union[torch.tensor, list], task_name : str =None):
        # TODO use nn.embedding to encode inputs as EFIN
        if isinstance(inputs, list):
            # TODO extend this to arbitary number of experts for each feature group
            assert len(inputs) == len(self.expert_shared), "number of experts should be equal the number of feature group"
            inputs = [r(input) for r, input in zip(self.rep, inputs)]
            experts_shared_rep = torch.stack([e(input) for e, input in zip(self.expert_shared, inputs)])
        else:
            inputs = self.rep(inputs)
            experts_shared_rep = torch.stack([e(inputs) for e in self.expert_shared])
            
        out = {}
        for task in self.task_names:
            if task_name is not None and task != task_name:
                continue
            selector = self.gate_specific[task](torch.flatten(inputs, start_dim=1))
            gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            # out[task] = self.decoders[task](gate_rep)
        return out
    
    def get_share_params(self):
        return self.expert_shared.parameters()

    def zero_grad_share_params(self):
        self.expert_shared.zero_grad(set_to_none=False)


class MTMT(nn.Module):
    def __init__(self,
                 user_feat_enc: nn.Module,
                 treat_feat_enc: nn.Module,
                 task_names: list,
                 num_treats: int,
                 t_dim: int,
                 u_dim: int,
                 tu_dim: int,
                 tu_enhance_norm: bool = None,
                ):
        r"""
        Args:
            user_feat_enc (nn.Module): an encoder class to encode features. The encoder can encode all features, or encode feature groups separately via MOE.
            treat_feat_enc (nn.Module): treatment feature encoder
            task_names (list): names of tasks
            num_treats (int): number of treatments
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
        
        self.Q_w = nn.Linear(t_dim, tu_dim, bias=True)
        self.K_w = nn.Linear(u_dim, tu_dim, bias=True)
        self.V_w = nn.Linear(u_dim, tu_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        
        # self.self_attention = nn.MultiheadAttention(embed_dim=u_dim, num_heads=8, kdim=t_dim, vdim=u_dim, batch_first=True)  # L B C input
        # self.self_attention = nn.MultiheadAttention(embed_dim=tu_dim, num_heads=8)
        self.self_attention = MultiHeadAttentionCustom(embed_dim=tu_dim, num_heads=8, batch_first=True, qdim=t_dim, kdim=u_dim, vdim=u_dim)  # L B C input
        
        self.tu_enhance = nn.Sequential(
            # TODO try add layers here or use attention
            Mlp(tu_dim, hidden_features=tu_dim // 2, norm_layer=tu_enhance_norm),
            # Mlp(16, hidden_features=16 * 2, norm_layer=tu_enhance_norm),
        )
        
        self.tu_logit = ClsHead(tu_dim, num_treats)
        self.tu_tau   = ClsHead(tu_dim, num_treats)
        
        self.u_tau = nn.ModuleDict({name: ClsHead(u_dim, 1) for name in task_names})  # assume tasks are all binary or regression
    
    def forward(self, user_input, treatment_input):        
        user_feat = self.user_enc(user_input.unsqueeze(1))  # dict of tensors
        user_feat = {'nextday_login': user_feat}

        if isinstance(self.treatment_enc, nn.Embedding):
            treatment_input = treatment_input.to(torch.long)
        elif treatment_input.dim() == 1:
            treatment_input = treatment_input.unsqueeze(-1)
        treat_feat = self.treatment_enc(treatment_input)  # B N
        assert treat_feat.dim() == 2
        
        u_logit = [self.u_tau[task](user_feat[task]) for task in self.task_names]
        
        user_feat_cat = torch.cat([t for t in user_feat.values()], dim=1)  # B C N
        user_feat_norm  = user_feat_cat / torch.linalg.norm(user_feat_cat, dim=1, keepdim=True)
        treat_feat_norm = treat_feat / torch.linalg.norm(treat_feat, dim=1, keepdim=True)
        # user_feat_norm = user_feat_cat
        # treat_feat_norm = treat_feat

        # TODO try EFIN's self interaction but change dimension
        # TODO Use Torch's multihead self attention
        # tu_feat, _ = self.self_attn(treat_feat_norm.unsqueeze(-1), user_feat_norm.transpose(-1, -2), user_feat_norm.transpose(-1, -2))  # B N C
        tu_feat, _ = self.self_attention(treat_feat_norm.unsqueeze(-1), user_feat_norm.transpose(-1, -2), user_feat_norm.transpose(-1, -2))

        # enhance treatment-user feature
        tu_feat_enhanced = self.tu_enhance(tu_feat).transpose(-1, -2)  # B C N
        # tu_feat_enhanced = self.tu_enhance(tu_feat.transpose(-1, -2)) # B C N
        
        # regularizer
        tu_tau = self.tu_tau(tu_feat_enhanced)
        tu_logit = self.tu_logit(tu_feat_enhanced)
        
        return u_logit, tu_tau, tu_logit
        
    def calculate_loss(self, user_input, treatment_input, y_true):
        # TODO try different weight balancing for MTL
        # TODO multi-treatment
        
        u_logit, tu_tau, tu_logit = self.forward(user_input, treatment_input)
        
        loss1 = F.mse_loss((1 - treatment_input) * u_logit[0].squeeze() + treatment_input * tu_logit.squeeze(), y_true)  # binary
        
        # ESN IPW regularize
        
        # interfere the model not to directly classify samples for treatments
        loss3 = F.binary_cross_entropy_with_logits(tu_tau.squeeze(), 1 - treatment_input)  # binary
        
        return loss1 + loss3
    
    
    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        # TODO EFIN adds a sigmoid before softmax, looks wierd
        attn_weights = self.softmax(attn_weights)
        outputs = torch.matmul(attn_weights, V)
        return outputs, attn_weights


def mtmt_res_emb_v0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_1():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=128), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_2():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=128)

def mtmt_res_emb_v0_3():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=512)

def mtmt_res_emb_v0_tFeatChangeDim():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=16, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_woNorm():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_MulAttn():
    user_feat_enc_hidden_dim=16
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=32), treat_feat_enc=MLP(in_chans=1, hidden_chans=[16, 32]), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=128)

def mtmt_res_emb_v0_MulAttn0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=38), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=128)

def mtmt_res_emb_v0_MulAttnCus():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_transEnhance():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_normEnhance():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256, tu_enhance_norm=nn.BatchNorm1d)

def mtmt_res_emb_v1():
    user_feat_enc_hidden_dim = 64
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)

def mtmt_res_emb_v1_0():
    user_feat_enc_hidden_dim = 32
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)

def mtmt_res_emb_v2():
    user_feat_enc_hidden_dim = 64
    return MTMT(user_feat_enc=resnet50(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=user_feat_enc_hidden_dim * 32, tu_dim=256)

def mtmt_res_emb_v2_0():
    user_feat_enc_hidden_dim = 8
    return MTMT(user_feat_enc=resnet50(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=user_feat_enc_hidden_dim * 32, tu_dim=256)

def mtmt_res_mlp_v0():
    user_feat_enc_hidden_dim = 64
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=MLP(in_chans=1, hidden_chans=[16, 32, 64]), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)

def mtmt_res_mlp_v0_0():
    user_feat_enc_hidden_dim = 16
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=MLP(in_chans=1, hidden_chans=[16]), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)

def mtmt_res_mlp_v0_1():
    user_feat_enc_hidden_dim = 16
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=MLP(in_chans=1, hidden_chans=[16, 32]), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)