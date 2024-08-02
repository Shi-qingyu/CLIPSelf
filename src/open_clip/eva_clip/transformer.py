import os
import logging
from functools import partial
from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

try:
    from timm.models.layers import trunc_normal_
except:
    from timm.layers import trunc_normal_
    
from .rope import VisionRotaryEmbedding, VisionRotaryEmbeddingFast
from .utils import to_2tuple

if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        import deepspeed
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        print("Please 'pip install deepspeed'")
        deepspeed = None
        from torch.utils.checkpoint import checkpoint
else:
    from torch.utils.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")

class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        output = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        if self.training and os.getenv('RoPE') == '1':
            return x, patch_indices_keep

        return x


def _in_projection_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    ):
    """
    https://github.com/pytorch/pytorch/blob/db2a237763eb8693a20788be94f8c192e762baa8/torch/nn/functional.py#L4726
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
            xattn=False,
            rope=False
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop
        self.rope = rope

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        if self.xattn:
            q = q.contiguous().view(L, N, self.num_heads, -1).transpose(0, 1)
            k = k.contiguous().view(L, N, self.num_heads, -1).transpose(0, 1)
            v = v.contiguous().view(L, N, self.num_heads, -1).transpose(0, 1)

            x = xops.memory_efficient_attention(
                q, k, v,
                p=self.xattn_drop,
                scale=self.scale if self.logit_scale is None else None,
                attn_bias=xops.LowerTriangularMask() if attn_mask is not None else None,
                )
        else:
            q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
            k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
            v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

            if self.logit_scale is not None:
                attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
                logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
                attn = attn.view(N, self.num_heads, L, L) * logit_scale
                attn = attn.view(-1, L, L)
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                    new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                    attn_mask = new_attn_mask
                attn += attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x
    
    def proj_without_attn(self, x, attn_mask=None):
        weight = self.in_proj_weight.data
        bias = self.in_proj_bias.data
        v_proj_weight = torch.chunk(weight, 3, dim=0)[-1]
        v_bias = torch.chunk(bias, 3, dim=0)[-1]
        x = F.linear(input=x, weight=v_proj_weight, bias=v_bias)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x

class CustomAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=True,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
            xattn=False
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        N_q, B_q, C_q = q.shape
        N_k, B_k, C_k = k.shape
        N_v, B_v, C_v = v.shape
        if self.xattn:
            # B, N, C -> B, N, num_heads, C
            q = q.permute(1, 0, 2).reshape(B_q, N_q, self.num_heads, -1)
            k = k.permute(1, 0, 2).reshape(B_k, N_k, self.num_heads, -1)
            v = v.permute(1, 0, 2).reshape(B_v, N_v, self.num_heads, -1)

            x = xops.memory_efficient_attention(
                q, k, v,
                p=self.xattn_drop,
                scale=self.scale if self.logit_scale is None else None,
                attn_bias=xops.LowerTriangularMask() if attn_mask is not None else None
                )
        else:
            # B*H, L, C
            q = q.contiguous().view(N_q, B_q * self.num_heads, -1).transpose(0, 1)
            k = k.contiguous().view(N_k, B_k * self.num_heads, -1).transpose(0, 1)
            v = v.contiguous().view(N_v, B_v * self.num_heads, -1).transpose(0, 1)

            if self.logit_scale is not None:
                # B*H, N_q, N_k
                attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
                logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
                attn = attn.view(B_q, self.num_heads, N_q, N_k) * logit_scale
                attn = attn.view(-1, N_q, N_k)
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                    new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                    attn_mask = new_attn_mask
                attn += attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = torch.bmm(attn, v)
            
        if self.head_scale is not None:
            x = x.view(B_q, self.num_heads, N_q, C_q) * self.head_scale
            x = x.view(-1, N_q, C_q)
        x = x.transpose(0, 1).reshape(N_q, B_q, C_q)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x

class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            cross_attn: bool = False,
            xattn: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.ln_1_k = norm_layer(d_model) if cross_attn else self.ln_1
        self.ln_1_v = norm_layer(d_model) if cross_attn else self.ln_1
        self.attn = CustomAttention(
            d_model, n_head,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            xattn=xattn
        )

        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        q = q + self.ls_1(self.ln_attn(self.attn(self.ln_1(q), self.ln_1_k(k), self.ln_1_v(v), attn_mask=attn_mask)))
        q = q + self.ls_2(self.mlp(self.ln_2(q)))
        return q

class CustomTransformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = True,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            cross_attn: bool = False,
            xattn: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.xattn = xattn

        self.resblocks = nn.ModuleList([
            CustomResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                scale_cosine_attn=scale_cosine_attn,
                scale_heads=scale_heads,
                scale_attn=scale_attn,
                scale_fc=scale_fc,
                cross_attn=cross_attn,
                xattn=xattn)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype 

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None, attn_mask: Optional[torch.Tensor] = None):
        if k is None and v is None:
            k = v = q
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                q = checkpoint(r, q, k, v, attn_mask)
            else:
                q = r(q, k, v, attn_mask=attn_mask)
        return q


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            xattn: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        if xattn:
            self.attn = Attention(d_model, n_head, xattn=True)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.xattn = xattn

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        if self.xattn:
            return self.attn(x, attn_mask=attn_mask)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x
    
    def forward_without_attn(self, x, attn_mask=None):
        x = x + self.ls_1(self.attn.proj_without_attn(self.ln_1(x), attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x
        

class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            xattn: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer, xattn=xattn)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x
    
    def forward_without_last_block(self, x, attn_mask=None):
        for r in self.resblocks[:-1]:
            x = r(x, attn_mask=attn_mask)
        return x
    
    def forward_without_attn(self, x, attn_mask=None):
        x = self.resblocks[-1].forward_without_attn(x)[1:]
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            patch_dropout: float = 0.,
            global_average_pool: bool = False,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            xattn: bool = True,
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.ln_pre = norm_layer(width)
        
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            xattn=xattn
        )

        self.global_average_pool = global_average_pool
        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False
        
        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def get_num_layers(self):
        return self.transformer.layers

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding'}

    def forward(self, x: torch.Tensor, return_all_features: bool=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if not return_all_features:
            if self.global_average_pool:
                x = x.mean(dim=1) #x = x[:,1:,:].mean(dim=1)
            else:
                x = x[:, 0]

            x = self.ln_post(x)

            if self.proj is not None:
                x = x @ self.proj

        return x

    def encode_dense(self, x, keep_shape=True):
        bs, _, h, w = x.shape
        h = h // self.conv1.stride[0]
        w = w // self.conv1.stride[1]
        x = self.conv1(x).permute(0, 2, 3, 1).flatten(1, 2)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.class_embedding[None, None].expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.positional_embedding is not None:
            x = x + self.rescale_positional_embedding(out_size=(h, w)).to(x.dtype)

        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)
        x = self.transformer.forward_without_last_block(x=x)
        x = self.transformer.forward_without_attn(x)    # (L, B, C)
        x = x.permute(1, 0, 2)  # (B, L, C)

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        x = F.normalize(x, dim=-1)   # normalize along last dimension
        if keep_shape:
            x = x.view(bs, h, w, -1).permute(0, 3, 1, 2)
        return x

    def extract_roi_features(self, x, normed_boxes, **kwargs):
        x = self.encode_dense(x, keep_shape=True)

        return roi_align(x, self._denormalize_boxes(normed_boxes, x), (1, 1),
                         1.0, -1, True)[..., 0, 0]

    def rescale_positional_embedding(self, out_size):
        h, w = out_size
        patch_shape = (self.image_size[0] // self.conv1.stride[0], self.image_size[1] // self.conv1.stride[1])
        if (h, w) == patch_shape:
            return self.positional_embedding
        rescaled_positional_embedding = \
            self.positional_embedding.new_zeros(1, 1 + h * w, self.positional_embedding.shape[-1])
        rescaled_positional_embedding[0, 0] = self.positional_embedding[0]
        pe_2d = self.positional_embedding[1:].T.contiguous().view(
            1, -1, *patch_shape)
        pe_2d = F.interpolate(pe_2d, out_size, mode='bicubic', align_corners=False).view(-1, h * w)
        rescaled_positional_embedding[0, 1:] = pe_2d.T.contiguous()

        return rescaled_positional_embedding

    def mask_pool(self, x, masks):
        feature_map = self.encode_dense(x, keep_shape=False)
        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = torch.cat(masks).float().flatten(-2, -1)    # bs, h*w
        feature_map = torch.repeat_interleave(
            feature_map, torch.tensor(num_masks_per_image, device=feature_map.device), dim=0)
        features = (feature_map * masks.unsqueeze(-1)).sum(1) / (masks.sum(1, keepdim=True) + 1e-12)

        return features

    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes

    def encode_rois_and_image(self, x, normed_boxes):
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks[:-1]:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x_image = self.head(
            self.post_attention(
                self.blocks[-1](
                    x, rel_pos_bias=rel_pos_bias)
            )
        )
        x_image = F.normalize(x_image, dim=-1)

        x = self.blocks[-1].forward_without_attn(x)[:, 1:]
        x = self.norm(x)
        x = self.head(x)
        assert self.fc_norm is None
        x = F.normalize(x, dim=-1)   # normalize along last dimension
        x = x.view(bs, h, w, -1).permute(0, 3, 1, 2)
        x_rois = roi_align(x, self._denormalize_boxes(normed_boxes, x),
                           (1, 1), 1.0, -1, True)[..., 0, 0]
        x_rois = F.normalize(x_rois, dim=-1)

        return x_rois, x_image


class TextTransformer(nn.Module):
    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            xattn: bool= False,
            attn_mask: bool = True
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            xattn=xattn
        )
        
        self.xattn = xattn
        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if attn_mask:
            self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)
        else:
            self.attn_mask = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable
    
    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'positional_embedding', 'token_embedding'}
        return {'positional_embedding'}

    def get_num_layers(self):
        return self.transformer.layers

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, return_all_features: bool=False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        # x = self.transformer(x) # no attention mask is applied
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if not return_all_features:
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def lock(self, *args, **kwargs):
        print(f'Freeze the text encoder', flush=True)
        for p in self.parameters():
            p.requires_grad = False
