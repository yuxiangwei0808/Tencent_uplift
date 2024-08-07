import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from einops import rearrange

class MultiHeadAttn(nn.Module):
    def __init__(self, qdim, kdim, vdim, embed_dim, num_head, attn_drop=0.):
        super().__init__()

        self.Q_w = nn.Linear(qdim, embed_dim, bias=True)
        self.K_w = nn.Linear(kdim, embed_dim, bias=True)
        self.V_w = nn.Linear(vdim, embed_dim, bias=True)

        self.embed_dim = embed_dim
        self.num_head = num_head
        assert embed_dim % num_head == 0, "embedding dim should be multiples of number of"

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.act = nn.Softmax(dim=-1)
        
        self._init_weights()

    def forward(self, q, k, v, return_weight=False):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)

        if self.num_head > 1:
            Q = rearrange(Q, 'b n (h ch) -> b h n ch', h=self.num_head, ch=self.embed_dim // self.num_head)
            K = rearrange(K, 'b n (h ch) -> b h n ch', h=self.num_head, ch=self.embed_dim // self.num_head)
            V = rearrange(V, 'b n (h ch) -> b h n ch', h=self.num_head, ch=self.embed_dim // self.num_head)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        attn_weights = self.act(attn_weights)
        attn_weights = self.attn_drop(attn_weights)
        outputs = torch.matmul(attn_weights, V)

        if self.num_head > 1:
            outputs = rearrange(outputs, 'b h n ch -> b n (h ch)')

        if return_weight:
            return outputs, attn_weights
        return outputs

    def _init_weights(self):
        xavier_uniform_(self.Q_w.weight)
        xavier_uniform_(self.K_w.weight)
        xavier_uniform_(self.V_w.weight)
        constant_(self.Q_w.bias, 0.)
        constant_(self.K_w.bias, 0.)
        constant_(self.K_w.bias, 0.)