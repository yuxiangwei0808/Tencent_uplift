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
                 num_treats: int = 1,
                 tu_enhance_norm: bool = None,
                 treat_feat_enc_s: nn.Module = None,
                ):
        r"""
        Args:
            user_feat_enc (nn.Module, list): encoder class to encode features. The encoder can encode all features, or encode feature groups separately via MOE.
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
                
        self.task_names = set(task_names)
        self.user_enc = user_feat_enc
        self.treatment_enc = treat_feat_enc
        self.treatment_enc_s = treat_feat_enc_s
        self.assert_success = False
        self.u_tau = nn.ModuleDict({name: ClsHead(u_dim, 1) for name in task_names})  # assume tasks are all binary or regression

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
                user_feats = [{'nextday_login': feat} for feat in user_feats]

            if not self.assert_success:
                assert all(set(d.keys()) == self.task_names for d in user_feats), "all encodes features should have the same tasks"
                self.assert_success = True
            
            user_feat = {}
            for task in self.task_names:
                user_feat[task] = torch.cat([d[task] for d in user_feats], dim=-1)
        else:
            user_feat = self.user_enc(user_input)  # dict of tensors
            user_feat = {'nextday_login': user_feat} if not isinstance(user_feat, dict) else user_feat

        if isinstance(self.treatment_enc, nn.Embedding):
            treat = treat.to(torch.long)
            treat_s = treat_s.to(torch.long) if treat_s is not None else treat_s
        
        treat_feat = self.treatment_enc(treat)  # B N
        treat_feat_s = self.treatment_enc_s(treat_s) if treat_s is not None else ...
        
        u_logit = {task: self.u_tau[task](user_feat[task]).squeeze() for task in self.task_names}
        
        # TODO try add or weighted add (linear) insteand of cat
        # TODO maybe we need to calculate a tu_logit for each task
        # TODO ablate norm dimension
        user_feat_cat = torch.cat([t for t in user_feat.values()], dim=1)  # B C N
        user_feat_norm  = user_feat_cat / (torch.linalg.norm(user_feat_cat, dim=1, keepdim=True) + 1e-6)

        treat_feat_norm = treat_feat / (torch.linalg.norm(treat_feat, dim=1, keepdim=True) + 1e-6)
        # user_feat_norm = user_feat_cat
        # treat_feat_norm = treat_feat

        # TODO try EFIN's self interaction but change dimension
        treat_feat_norm = treat_feat_norm.unsqueeze(-1) if treat_feat_norm.dim() == 2 else treat_feat_norm.transpose(-1, -2)
        tu_feat, _ = self.self_attention(treat_feat_norm, user_feat_norm.transpose(-1, -2), user_feat_norm.transpose(-1, -2))

        # enhance treatment-user feature
        tu_feat_enhanced = self.tu_enhance(tu_feat).transpose(-1, -2)  # B C N
        
        # regularizer
        tu_tau = F.sigmoid(self.tu_tau(tu_feat_enhanced).squeeze())
        tu_logit = self.tu_logit(tu_feat_enhanced).squeeze()

        if treat_s is not None:
            # more specific treatments
            treat_feat_norm_s = treat_feat_s / (torch.linalg.norm(treat_feat_s, dim=1, keepdim=True) + 1e-6)
            tu_feat_s, _ = ...
            tu_feat_enhanced_s = self.tu_enhance_s(tu_feat_s).transpose(-1, -2)
            
            tu_tau_s = self.tu_tau_s(tu_feat_enhanced_s)
            tu_logit_s = self.tu_logit_s(tu_feat_enhanced_s)
            return u_logit, tu_tau, tu_logit, tu_tau_s, tu_logit_s
        
        return u_logit, tu_tau, tu_logit
        
    def calculate_loss(self, user_input, treatment_input, y_true):
        # TODO try different weight balancing for MTL
        # TODO multi-treatment
        
        u_logit, tu_tau, tu_logit = self.forward(user_input, treatment_input)
        
        assert len(u_logit) == 1
        tu_logit += u_logit['nextday_login'].detach()  # EFIN
        loss1 = F.mse_loss((1 - treatment_input) * u_logit['nextday_login'].squeeze() + treatment_input * tu_logit.squeeze(), y_true)  # binary
        # loss1 = F.mse_loss(1 - treatment_input[:, 0]) * u_logit['nextday_login'].squeeze() + treatment_input[:, 0] * tu_logit.squeeze(), y_true)
        # loss1 = F.mse_loss(1 - treatment_input[:, 0]) * u_logit['nextday_login'].squeeze() + treatment_input[:, 0] * ((1 - treatment_input[:, 1]) * tu_logit.squeeze() + treatment_input[:, 1] * tu_logit_s.squeeze()), y_true)  # tu_logit_s directly models y_k
        # loss1 = F.mse_loss(1 - treatment_input[:, 0]) * u_logit['nextday_login'].squeeze() + treatment_input[:, 0] * (tu_logit.squeeze() + treatment_input[:, 1] * tu_logit_s.squeeze()), y_true)  # tu_logit_s models difference over 5AI
        
        # ESN IPW regularize
        # loss2 = F.binary_cross_entropy_with_logits(tu_logit * tu_tau, y_true * treatment_input) + F.binary_cross_entropy_with_logits(u_logit['nextday_login'] * (1 - tu_tau), y_true * (1 - treatment_input))
        
        # interfere the model not to directly classify samples for treatments
        # loss3 = F.binary_cross_entropy_with_logits(tu_tau.squeeze(), 1 - treatment_input)  # binary

        # inverse probability metric loss to mitigate gap between treated and control y0
        loss4 = F.mse_loss((u_logit['nextday_login'] * treatment_input).mean(), (u_logit['nextday_login'] * (1 - treatment_input)).mean())
        
        # return loss1 + loss2 * (loss1 / loss2).detach() + loss3 * (loss1 / loss3).detach()
        return loss1 + loss4 * (loss1 / loss4).detach()
    
    
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

def mtmt_res_emb_v0_4():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_4_0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=None, drop=0.2), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_5():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Identity(), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

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

def mtmt_cnn_emb_v0():
    return MTMT(user_feat_enc=cnn_simple(hidden_chans=[16, 32, 64, 128]), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_cnn_emb_v1():
    return MTMT(user_feat_enc=cnn_bottleneck_simple(hidden_chans=[16, 32, 64, 128]), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_vit_emb_v0():
    return MTMT(user_feat_enc=vit_tiny_patch2_224(img_size=622, in_chans=1), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=192, tu_dim=256)

def mtmt_mmoe_emb_v0():
    return MTMT(user_feat_enc=MMOE(encoder_class=resnet18, num_experts=4, task_names=['nextday_login'], in_feat=622, 
                            enc_kwargs={'all': {'hidden_dim': 16, 'out_dim': None}},
                            rep_grad=False), 
                 treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['nextday_login'],
                 num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_cnn_emb_v0():
    # seperately process discrete feature; zero-encode and pad the discrete feature
    return MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=16), cnn_simple(hidden_chans=[16, 32, 64, 128], in_chans=9, strides=[1, 1, 2, 1])]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['nextday_login'],
                num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_cnn_emb_v1():
    # seperately process discrete feature; zero-encode but not pad the discrete feature
    return MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=16), cnn_simple(hidden_chans=[16, 32, 64, 128], in_chans=1, strides=[1, 1, 2, 1])]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['nextday_login'],
                num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_disc__emb_v0():
    # seperately encode discrete features and add back as the input to user_feat_enc
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, disc_encoder=DiscEncoder(embed_dim=8, out_shape='1d')),
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['nextday_login'],
                num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_disc_mlp__emb_v0():
    return  MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=8), 
                    DiscEncoder(embed_dim=8, out_shape='2d',
                                enc=MLP(hidden_chans=[16, 32, 64], in_chans=9, transpose=True))
                ]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['nextday_login'],
                num_treats=1, t_dim=1, u_dim=64, tu_dim=128)


def mtmt_res_disc_mlp__emb_cnn_v0():
    return  MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=8), 
                    DiscEncoder(embed_dim=8, out_shape='2d',
                                enc=MLP(hidden_chans=[16, 32, 64], in_chans=9, transpose=True))
                ]), 
                treat_feat_enc=cnn_simple(in_chans=1, hidden_chans=[4, 8, 16], strides=[1, 1, 1], encoder=nn.Embedding(num_embeddings=2, embedding_dim=8)),
                task_names=['nextday_login'],
                num_treats=1, t_dim=16, u_dim=64, tu_dim=128)

def mtmt_res_disc_cnn__emb_v0():
    return  MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=4), 
                    DiscEncoder(embed_dim=8, out_shape='2d',
                                enc=cnn_simple(hidden_chans=[12, 16, 32], in_chans=9, strides=[1, 2, 1]))
                ]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['nextday_login'],
                num_treats=1, t_dim=1, u_dim=32, tu_dim=64)

def mtmt_res_disc_cnn__emb_v0_0():
    return  MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=8), 
                    DiscEncoder(embed_dim=8, out_shape='2d',
                                enc=cnn_simple(hidden_chans=[16, 32, 64], in_chans=9, strides=[1, 2, 1]))
                ]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['nextday_login'],
                num_treats=1, t_dim=1, u_dim=64, tu_dim=128)

def mtmt_res_disc_cnn__emb_v1():
    return MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=16), 
                    DiscEncoder(embed_dim=8, out_shape='1d',
                                enc=cnn_simple(hidden_chans=[16, 32, 64, 128], in_chans=1, strides=[1, 1, 2, 1]))
                ]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['nextday_login'],
                num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_cnn_MLP_v0():
    # seperately process discrete feature; zero-encode and pad the discrete feature
    return MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=16), MLP(in_chans=9, hidden_chans=[16, 32, 64, 128], transpose=True)]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['nextday_login'],
                num_treats=1, t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res__emb_MLP_v0_4_0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=None, drop=0.2), 
                treat_feat_enc=MLP(in_chans=1, transpose=True, hidden_chans=[4, 8, 16], drop_rate=0.1, encoder=nn.Embedding(num_embeddings=2, embedding_dim=8)),
                task_names=['nextday_login'], num_treats=1, t_dim=16, u_dim=128, tu_dim=256)

def mtmt_res__emb_cnn_v0_4_0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=None, drop=0.2), 
                treat_feat_enc=cnn_simple(in_chans=1, hidden_chans=[4, 8, 16], strides=[1, 1, 1], encoder=nn.Embedding(num_embeddings=2, embedding_dim=8)),
                task_names=['nextday_login'], num_treats=1, t_dim=16, u_dim=128, tu_dim=256)