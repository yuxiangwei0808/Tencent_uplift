import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.layers import Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, RotaryEmbeddingCat, \
    apply_rot_embed_cat, apply_keep_indices_nlc, trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    to_2tuple, use_fused_attn
from timm.models._features import feature_take_indices

from .transformer import PatchEmbed


class EvaAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class EvaBlockPostNorm(nn.Module):
    """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,  # ignore for post-norm
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            num_reg_tokens: int = 0,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
    ):
        """

        Args:
            img_size:
            patch_size:
            in_chans:
            num_classes:
            global_pool:
            embed_dim:
            depth:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            drop_rate:
            pos_drop_rate:
            proj_drop_rate:
            attn_drop_rate:
            drop_path_rate:
            norm_layer:
            init_values:
            class_token:
            use_abs_pos_emb:
            use_rot_pos_emb:
            use_post_norm:
            ref_feat_shape:
            head_init_scale:
        """
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                num_prefix_tokens=self.num_prefix_tokens,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if pos_embed is not None:
            x = x + pos_embed

        if self.reg_token is not None:
            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
            x = torch.cat(to_cat + [x], dim=1)

        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
        return x, rot_pos_embed

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int], Tuple[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.
        Args:
            x: Input image tensor
            indices: Take last n blocks if an int, if is a sequence, select by matching indices
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format for EVA-ViT features must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x, rope=rot_pos_embed)
            if i in take_indices:
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int], Tuple[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)
        return x