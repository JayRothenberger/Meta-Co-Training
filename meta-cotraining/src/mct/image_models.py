import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.distributed as dist
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import numpy as np
from math import sqrt

import torchvision
from torchvision.transforms import v2

import torch.distributed as dist


import multiprocessing as mp

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from skimage import io, color
import numpy as np
from scipy.spatial import KDTree
from collections import Counter
from scipy.optimize import linprog

import numpy as np

import torch
import clip
from transformers import CLIPVisionModel, SiglipVisionModel, AlignVisionModel, Blip2VisionModel, Owlv2VisionModel
from functools import lru_cache

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2 Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn_out = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

    @staticmethod
    def compute_macs(module, input, output):
        B, N, C = input[0].shape

        module.__flops__ += module.flops(N) * B


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = input_resolution[0]
        self.W = input_resolution[1]

        self.attn_mask_dict = {} # {self.H: self.create_attn_mask(self.H, self.W)}



    def create_attn_mask(self, H, W):
        # calculate attention mask for SW-MSA

        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1))  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


    def forward(self, x):

        
        B, L, C = x.shape
        H = int(sqrt(L))
        W = H

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            if H is self.attn_mask_dict.keys():
                attn_mask = self.attn_mask_dict[H]
            else:
                self.attn_mask_dict[H] = self.create_attn_mask(H, W).to(x.device)
                attn_mask = self.attn_mask_dict[H]

        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(x_windows, attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size} mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)


    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        H = int(sqrt(L))
        W = H

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x, _ = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def forward_with_features(self, x):
        fea = []
        for blk in self.blocks:
            x, _ = blk(x)
            fea.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, fea

    def forward_with_attention(self, x):
        attns = []
        for blk in self.blocks:
            x, attn = blk(x)
            attns.append(attn)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, attns


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size.
        in_chans (int): Number of input channels.
        num_classes (int): Number of classes for classification head.
        embed_dim (int): Embedding dimension.
        depths (tuple(int)): Depth of Swin Transformer layers.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): normalization layer.
        ape (bool): If True, add absolute position embedding to the patch embedding.
        patch_norm (bool): If True, add normalization after patch embedding.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_dense_prediction=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution




        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Region prediction head
        self.use_dense_prediction = use_dense_prediction
        if self.use_dense_prediction: self.head_dense = None


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # todo: to be implemented
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x_region = self.norm(x)  # B L C
        x = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        if self.use_dense_prediction:
            return x, x_region
        else:
            return x


    def forward_feature_maps(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x_grid = self.norm(x)  # B L C
        x = self.avgpool(x_grid.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        return x, x_grid


    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        # Perform forward pass separately on each resolution input.
        # The inputs corresponding to a single resolution are clubbed and single
        # forward is run on the same resolution inputs. Hence we do several
        # forward passes = number of different resolutions used. We then
        # concatenate all the output features.

        # When region level prediction task is used, the network output four variables:
        # self.head(output_cls):       view-level prob vector
        # self.head_dense(output_fea): regioin-level prob vector
        # output_fea:                  region-level feature map (grid features)
        # npatch:                      number of patches per view
        
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        if self.use_dense_prediction:
            start_idx = 0
            
            for end_idx in idx_crops:
                _out_cls, _out_fea  = self.forward_features(torch.cat(x[start_idx: end_idx]))
                B, N, C = _out_fea.shape

                if start_idx == 0:
                    output_cls = _out_cls
                    output_fea = _out_fea.reshape(B * N, C)
                    npatch = [N]
                else:
                    output_cls = torch.cat((output_cls, _out_cls))
                    output_fea = torch.cat((output_fea, _out_fea.reshape(B * N, C) ))
                    npatch.append(N)
                start_idx = end_idx

            return self.head(output_cls), self.head_dense(output_fea), output_fea, npatch 

        else:
            start_idx = 0
            for end_idx in idx_crops:
                _out = self.forward_features(torch.cat(x[start_idx: end_idx]))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
            # Run the head forward on the concatenated features.
            return output
            # return self.head(output)


    def forward_selfattention(self, x, n=1):
        # n=1 return the last layer attn map; otherwise return attn maps in all layers

        
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        if n==1:
            return self.forward_last_selfattention(x)
        else:
            return self.forward_all_selfattention(x)

    def forward_last_selfattention(self, x):

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer(x)
            else:
                x, attns = layer.forward_with_attention(x)
                return attns[-1]

    def forward_all_selfattention(self, x):
        attn_out = []

        for layer in self.layers:
            x, attns = layer.forward_with_attention(x)
            attn_out += attns

        return attn_out


    def forward_return_n_last_blocks(self, x, n=1, return_patch_avgpool=False, depth=[]):

        num_blks = sum(depth)
        start_idx = num_blks - n

        sum_cur = 0
        for i, d in enumerate(depth):
            sum_cur_new = sum_cur + d
            if start_idx >= sum_cur and start_idx < sum_cur_new:
                start_stage = i
                start_blk = start_idx - sum_cur
            sum_cur = sum_cur_new


        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # we will return the averaged token features from the `n` last blocks
        # note: there is no [CLS] token in Swin Transformer
        output = []
        s = 0
        for i, layer in enumerate(self.layers):
            x, fea = layer.forward_with_features(x)

            if i >= start_stage:
                for x_ in fea[start_blk:]:

                    if i == len(self.layers)-1: # use the norm in the last stage
                        x_ = self.norm(x_)

                    x_avg = torch.flatten(self.avgpool(x_.transpose(1, 2)), 1)  # B C     
                    # print(f'Stage {i},  x_avg {x_avg.shape}')          
                    output.append(x_avg)

                start_blk = 0

        return torch.cat(output, dim=-1)



    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
            if dist.get_rank() == 0:
                print(f"GFLOPs layer_{i}: {layer.flops() / 1e9}")
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                        or 'relative_position_index' not in k
                        or 'attn_mask' not in k
                )

                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')

                    if 'relative_position_bias_table' in k and v.size() != model_dict[k].size():
                        relative_position_bias_table_pretrained = v
                        relative_position_bias_table_current = model_dict[k]
                        L1, nH1 = relative_position_bias_table_pretrained.size()
                        L2, nH2 = relative_position_bias_table_current.size()
                        if nH1 != nH2:
                            logging.info(f"Error in loading {k}, passing")
                        else:
                            if L1 != L2:
                                logging.info(
                                    '=> load_pretrained: resized variant: {} to {}'
                                        .format((L1, nH1), (L2, nH2))
                                )
                                S1 = int(L1 ** 0.5)
                                S2 = int(L2 ** 0.5)
                                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                                    size=(S2, S2),
                                    mode='bicubic')
                                v = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

                    if 'absolute_pos_embed' in k and v.size() != model_dict[k].size():
                        absolute_pos_embed_pretrained = v
                        absolute_pos_embed_current = model_dict[k]
                        _, L1, C1 = absolute_pos_embed_pretrained.size()
                        _, L2, C2 = absolute_pos_embed_current.size()
                        if C1 != C1:
                            logging.info(f"Error in loading {k}, passing")
                        else:
                            if L1 != L2:
                                logging.info(
                                    '=> load_pretrained: resized variant: {} to {}'
                                        .format((1, L1, C1), (1, L2, C2))
                                )
                                S1 = int(L1 ** 0.5)
                                S2 = int(L2 ** 0.5)
                                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                                v = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1).flatten(1, 2)

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    def freeze_pretrained_layers(self, frozen_layers=[]):
        for name, module in self.named_modules():
            if (
                    name.split('.')[0] in frozen_layers
                    or '.'.join(name.split('.')[0:2]) in frozen_layers
                    or (len(frozen_layers) > 0 and frozen_layers[0] is '*')
            ):
                for _name, param in module.named_parameters():
                    param.requires_grad = False
                logging.info(
                    '=> set param {} requires grad to False'
                        .format(name)
                )
        for name, param in self.named_parameters():
            if (
                    name.split('.')[0] in frozen_layers
                    or (len(frozen_layers) > 0 and frozen_layers[0] is '*')
                    and param.requires_grad is True
            ):
                param.requires_grad = False
                logging.info(
                    '=> set param {} requires grad to False'
                        .format(name)
                )
        return self


def swin_b():
    swin = SwinTransformer(
        img_size=224,
        in_chans=3,
        num_classes=1000,
        patch_size=4,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=14,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2, #stochastic depth
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ape=False,
        patch_norm=True,
        use_dense_prediction=False,
    )
    return swin


class SSL_distance():
    @lru_cache(maxsize=16)
    def __call__(self, x, y):
        with torch.no_grad():
            x = x.to(visible_device)
            y = y.to(visible_device)
            return x

    def embed(self, x):
        with torch.no_grad():
            x = x.to(visible_device)
            return self.model(x).cpu()


class MAE_distance(SSL_distance):
    """
    masked autoencoder distance
    """
    def __init__(self, chkpt_path='/home/models/mae_pretrain_vit_large.pth', model=vit_base_patch16):
        model = vit_large_patch16()
        checkpoint = torch.load(chkpt_path)

        checkpoint_model = checkpoint['model']  
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        self.model = model
        # self.model = model.to(visible_device)

            
# @lru_cache(maxsize=4)
def min_max_norm(x):
    x = x - x.min()
    return x / x.max()

class EsViT_distance(SSL_distance):
    """
    EsVit distance
    """
    def __init__(self, chkpt_path='/home/models/esvit_swin_b.pth', model=vit_base_patch16):
        model = swin_b()
        checkpoint = torch.load(chkpt_path)
        checkpoint_model = checkpoint['student']
        
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        # load_pretrained_weights(model, chkpt_path, 'student', None, None)
        self.model = model
        # self.model = model.to(visible_device)

class DINOv2_distance(SSL_distance):
    """
    DINOv2 distance
    """
    def __init__(self, model='dinov2_vitl14_reg'):
        os.environ['TORCH_HOME'] = '/home/models/'
        os.environ['TORCH_HUB'] = '/home/models/'
        self.model = torch.hub.load('facebookresearch/dinov2', model)
        
        # self.model = model.to(visible_device)

class Heira_distance(SSL_distance):
    """
    DINOv2 distance
    """
    def __init__(self, model='dinov2_vitl14_reg'):
        os.environ['TORCH_HOME'] = '/home/models/'
        os.environ['TORCH_HUB'] = '/home/models/'
        self.model = torch.hub.load("facebookresearch/hiera", model="mae_hiera_base_224", pretrained=True, checkpoint="mae_in1k")
        

class CLIP_wrapper(torch.nn.Module):
    def __init__(self):
        super(CLIP_wrapper, self).__init__()
        self.mod_list = torch.nn.ModuleList([CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=os.environ['HF_HOME'])])

    def forward(self, x):
        return self.mod_list[0](x).pooler_output


class CLIP_distance(SSL_distance):
    """
    CLIP distance
    """
    def __init__(self, model='ViT-B/16'):
        os.environ['HF_HOME'] = '/scratch/jroth'
        # model, _ = clip.load(model, device='cpu', download_root='/scratch/jroth/models/')
        self.model = CLIP_wrapper()
        # self.model = model.visual.to(visible_device)

# TODO
class BLIPv2_wrapper(torch.nn.Module):
    def __init__(self):
        super(BLIPv2_wrapper, self).__init__()
        self.mod_list = torch.nn.ModuleList([Blip2VisionModel.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=os.environ['HF_HOME'])])

    def forward(self, x):
        return self.mod_list[0](x).pooler_output


class BLIPv2_distance(SSL_distance):
    """
    BLIPv2 distance
    """
    def __init__(self, model='ViT-B/16'):
        os.environ['HF_HOME'] = '/scratch/jroth'
        # model, _ = clip.load(model, device='cpu', download_root='/scratch/jroth/models/')
        self.model = BLIPv2_wrapper()
        # self.model = model.visual.to(visible_device)

# TODO
class OWLv2_wrapper(torch.nn.Module):
    def __init__(self):
        super(OWLv2_wrapper, self).__init__()
        self.mod_list = torch.nn.ModuleList([Owlv2VisionModel.from_pretrained("google/owlv2-base-patch16", cache_dir=os.environ['HF_HOME'])])

    def forward(self, x):
        return self.mod_list[0](x).pooler_output


class OWLv2_distance(SSL_distance):
    """
    OWLv2 distance
    """
    def __init__(self, model='ViT-B/16'):
        os.environ['HF_HOME'] = '/scratch/jroth'
        # model, _ = clip.load(model, device='cpu', download_root='/scratch/jroth/models/')
        self.model = OWLv2_wrapper()
        # self.model = model.visual.to(visible_device)


# TODO
class ALIGN_wrapper(torch.nn.Module):
    def __init__(self):
        super(ALIGN_wrapper, self).__init__()
        self.mod_list = torch.nn.ModuleList([AlignVisionModel.from_pretrained("kakaobrain/align-base", cache_dir=os.environ['HF_HOME'])])

    def forward(self, x):
        return self.mod_list[0](x).pooler_output


class ALIGN_distance(SSL_distance):
    """
    ALIGN distance
    """
    def __init__(self, model='ViT-B/16'):
        os.environ['HF_HOME'] = '/scratch/jroth'
        # model, _ = clip.load(model, device='cpu', download_root='/scratch/jroth/models/')
        self.model = ALIGN_wrapper()
        # self.model = model.visual.to(visible_device)


class SigLIP_wrapper(torch.nn.Module):
    def __init__(self):
        super(SigLIP_wrapper, self).__init__()
        self.mod_list = torch.nn.ModuleList([SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", cache_dir=os.environ['HF_HOME'])])

    def forward(self, x):
        return self.mod_list[0](x).pooler_output



class SigLIP_distance(SSL_distance):
    """
    CLIP distance
    """
    def __init__(self, model='ViT-B/16'):
        os.environ['HF_HOME'] = '/scratch/jroth'
        # model, _ = clip.load(model, device='cpu', download_root='/scratch/jroth/models/')
        self.model = SigLIP_wrapper()
        # self.model = model.visual.to(visible_device)


class SwAV_distance(SSL_distance):
    def __init__(self, model='resnet50w4'):
        os.environ['TORCH_HOME'] = '/home/models/'
        model = torch.hub.load('facebookresearch/swav:main', model)
        self.model = model
        # self.model = model.to(visible_device)

# depth first search to get the membership of each leaf of the k-d Tree
def DFS_KDTree(kdtree):
    leaves = []
    stack = [kdtree.tree]

    while stack:
        node = stack.pop(-1)
        if not isinstance(node, KDTree.leafnode):
            stack.append(node.greater)
            stack.append(node.less)
        else:
            leaves.append(node.idx)
    return leaves

def make_clusters(x):
    # with balanced false the midpoint is used not the median
    x_tree = KDTree(x, balanced_tree=False, leafsize=x.shape[0] // 100, compact_nodes=False)
    # first kd tree iteration
    leaves = DFS_KDTree(x_tree)
    # compute centroids
    centroids = np.stack([np.mean(x[i], 0) for i in leaves], 0)
    # second iteration
    centroids_tree = KDTree(centroids, balanced_tree=True, leafsize=10, compact_nodes=False)
    # compute centroids of centroids
    leaves = DFS_KDTree(centroids_tree)
    centroids = np.stack([np.mean(centroids[i], 0) for i in leaves], 0)
    # yet another tree for efficient nearest neighbor computation (probably unecessary, but convenient)
    centroids_tree = KDTree(centroids, balanced_tree=True, leafsize=10, compact_nodes=False)
    # find the cluster assignment of each element in x
    labels = centroids_tree.query(x)[1]
    # compute the weight of each centroid
    counter = Counter(labels)
    # return tuples of centroid, weight
    return [(centroids[i], counter[i] / x.shape[0]) for i in counter]

def pairwise_euclidean_dist(x, y):
    x2 = np.sum(x**2, axis=1)
    y2 = np.sum(y**2, axis=1)
    
    xy = np.matmul(x, y.T)

    x2 = x2.reshape(-1, 1)
    dists = np.maximum(x2 - 2*xy + y2, 0)
    dists = np.sqrt(dists)
    # dists[np.isnan(dists)] = 0.0
    return dists

def vstack_2d(x):
    return np.squeeze(np.vstack(np.expand_dims(x, -1)))

def compute_transport(clusters_x, clusters_y, dists):
    m, n = dists.shape
    # each position of dists corresponds to both a distance and a flow variable
    c = vstack_2d(dists)

    weights_x = np.array([weight for _, weight in clusters_x])
    weights_y = np.array([weight for _, weight in clusters_y]).squeeze()
    # now we need a matrix A_ub with a 1 in every n position and a 1 in every m position to satisfy (2) and (3)
    # first m rows are for p
    A_ub_p = []
    for i in range(m):
        const = np.zeros_like(dists)
        # sum of all flows from i
        const[i, :] = 1.0
        A_ub_p.append(vstack_2d(const))
    A_ub_p = np.stack(A_ub_p, 0)
    # remaining n rows are for q
    A_ub_q = []
    for j in range(n):
        const = np.zeros_like(dists)
        # sum of all flows to j
        const[:, j] = 1.0
        A_ub_q.append(vstack_2d(const))
    A_ub_q = np.stack(A_ub_q, 0)
    A_ub = np.concatenate((A_ub_p, A_ub_q), 0)
    # and a b_ub equal to w_p_i and w_q_j
    b_ub = np.expand_dims(np.concatenate((weights_x, weights_y)), 0)
    # we need a matrix A_eq with a 1 in every position to satisfy (4)
    A_eq = np.expand_dims(np.ones_like(c), 0)
    # and a b_eq with a 
    b_eq = np.array([1.0])
    # finally we need a tuple (0, None) to specify decison variable bounds
    bounds = (0, None)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    return res['fun']


def compute_dual_transport(clusters_x, clusters_y, dists):
    m, n = dists.shape
    # each position of dists corresponds to both a distance and a flow variable
    c = vstack_2d(dists)

    weights_x = np.array([weight for _, weight in clusters_x])
    weights_y = np.array([weight for _, weight in clusters_y]).squeeze()
    # now we need a matrix A_ub with a 1 in every n position and a 1 in every m position to satisfy (2) and (3)
    # first m rows are for p
    A_ub_p = []
    for i in range(m):
        const = np.zeros_like(dists)
        # sum of all flows from i
        const[i, :] = 1.0
        A_ub_p.append(vstack_2d(const))
    A_ub_p = np.stack(A_ub_p, 0)
    # remaining n rows are for q
    A_ub_q = []
    for j in range(n):
        const = np.zeros_like(dists)
        # sum of all flows to j
        const[:, j] = 1.0
        A_ub_q.append(vstack_2d(const))
    A_ub_q = np.stack(A_ub_q, 0)
    A_ub = np.concatenate((A_ub_p, A_ub_q), 0)
    # and a b_ub equal to w_p_i and w_q_j
    b_ub = np.expand_dims(np.concatenate((weights_x, weights_y)), 0)
    # we need a matrix A_eq with a 1 in every position to satisfy (4)
    A_eq = np.expand_dims(np.ones_like(c), 0)
    # and a b_eq with a 
    b_eq = np.array([1.0])
    # finally we need a tuple (0, None) to specify decison variable bounds
    bounds = (0, None)

    A_gt = np.concat((A_ub, A_eq)).T
    b_gt = np.concat((b_ub, b_eq))


    res = linprog(b_gt, A_ub=-1*A_gt, b_ub=-1*c, A_eq=None, b_eq=None, bounds=None)
    
    return res['x'], res['fun']

def add_x_y_coords(x):
    return np.concatenate((x, np.expand_dims(np.repeat(np.expand_dims(np.arange(x.shape[0]), -1), x.shape[1], axis=1), -1), np.expand_dims(np.repeat(np.expand_dims(np.arange(x.shape[1]), 0), x.shape[0], axis=0), -1)), -1)

# earth mover's distance for image retrieval
def EMD(x, y):
    # convert to cielab space
    x, y = color.rgb2lab(x), color.rgb2lab(y)

    x, y = add_x_y_coords(x), add_x_y_coords(y)
    # serialize the colors
    # TODO: add spatial coordinates to x and y
    x, y = np.vstack(x), np.vstack(y)
    
    clusters_x, clusters_y = make_clusters(x), make_clusters(y)

    centroids_x, centroids_y = np.stack([centroid for centroid, _ in clusters_x], 0), np.stack([centroid for centroid, _ in clusters_y], 0)
    
    dists = pairwise_euclidean_dist(centroids_x, centroids_y) # rows x, columns y
    # compute the solution to the transport problem
    cost = compute_transport(clusters_x, clusters_y, dists)
    # return the distance
    return cost

def Signature(x):
    x = color.rgb2lab(x)
    x = add_x_y_coords(x)
    x = np.vstack(x.squeeze())
    return make_clusters(x)

# @lru_cache(maxsize=4)
def KL(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.sum(np.where(a*b > 0, a * np.log(a / b), np.where(a > 0, -a*np.log(1.001 - a), 0.0)))

def KLKL(a, b):

    return KL(a, b) + KL(b, a)

def JD(a, b):
    return torch.sum((np.sqrt(np.maximum(a, 0.0)) - np.sqrt(np.maximum(b, 0.0)))**2)


def EMD_from_signatures(sig_x, sig_y):
    centroids_x, centroids_y = np.stack([centroid for centroid, _ in sig_x], 0), np.stack([centroid for centroid, _ in sig_y], 0)
    dists = pairwise_euclidean_dist(centroids_x.squeeze(), centroids_y.squeeze()) # rows x, columns y
    return compute_transport(sig_x, sig_y, dists)

def EMD_primal_dual(sig_x, sig_y):
    centroids_x, centroids_y = np.stack([centroid for centroid, _ in sig_x], 0), np.stack([centroid for centroid, _ in sig_y], 0)
    dists = pairwise_euclidean_dist(centroids_x.squeeze(), centroids_y.squeeze()) # rows x, columns y
    print(compute_transport(sig_x, sig_y, dists))
    print(compute_dual_transport(sig_x, sig_y, dists))
    return compute_dual_transport(sig_x, sig_y, dists)

# depth first search to get the membership of each leaf of the k-d Tree
def cie_hist(x):
    x = color.rgb2lab(x)
    x = np.vstack(x)
    bins = np.histogramdd(x, (np.arange(0, 101, 25), np.arange(-100, 101, 25), np.arange(-100, 101, 25)))[0]
    return bins.flatten() / np.sum(bins)

def KL_image(x, y):
    return KL(cie_hist(x), cie_hist(y))

def KLKL_image(x, y):
    return KLKL(cie_hist(x), cie_hist(y))


class EMD_distance():
    """
    Wrapper class with an embed method so that EMD embedding is convenient to define
    """
    def embed(self, x):
        return Signature(x)


class KL_distance():
    """
    Wrapper class with an embed method so that Histogram embedding is convenient to define
    """
    def embed(self, x):
        return cie_hist(x)

IMAGE_TRANSFORMS = {

# CLIP - she's picky
'CLIP': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=(0.48145466, 0.4578275, 0.40821073),
                                            std=(0.26862954, 0.26130258, 0.27577711)
                                            ),
                    ]
                    ),
'OWLv2': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=(0.48145466, 0.4578275, 0.40821073),
                                            std=(0.26862954, 0.26130258, 0.27577711)
                                            ),
                    ]
                    ),
'SigLIP': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=(0.48145466, 0.4578275, 0.40821073),
                                            std=(0.26862954, 0.26130258, 0.27577711)
                                            ),
                    ]
                    ),

# EsViT / MAE / DINO - https://github.com/microsoft/esvit/blob/main/experiments/imagenet/swin/swin_base_patch4_window14_224.yaml
'EsViT': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    ),

'MAE': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    ),

'DINOv2': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    ),

'ALIGN': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    ),

'Heira': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    ),
'BLIPv2': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    ),
'SwAV': v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    )
}

IMAGE_DISTANCES = {

'EMD': EMD_distance,

'Hist': KL_distance,

'CLIP': CLIP_distance,

'BLIPv2': BLIPv2_distance,

'OWLv2': OWLv2_distance,

'ALIGN': ALIGN_distance,

'Heira': Heira_distance,

'SigLIP': SigLIP_distance,

'EsViT': EsViT_distance,

'MAE': MAE_distance,

'DINOv2': DINOv2_distance,

'SwAV': SwAV_distance,

}

visible_device = 0

import gc
import pickle

def write_embeds_to_file(fp, loader, dist):
    embeds = []
    labels = []
    start = time.time()
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            gc.collect()
            embeds.append(dist.embed(x).cpu().type(torch.float16))
            labels.append(y.cpu())
            print(f'{i} / {len(loader)}: {(i + 1)} it / s', end='       \r')
    pickle.dump((np.concatenate(embeds, 0), np.concatenate(labels, 0)), fp)
    embeds.clear()
    labels.clear()

def write_signatures_to_file(fp, loader):

    embeds = []
    labels = []
    start = time.time()
    for i, (x, y) in enumerate(loader):
        embeds.append(x)
        labels.append(y)
        print(f'{i} / {len(loader)}: {(i + 1)} it / s', end='       \r')

    pickle.dump((embeds, labels), fp)

