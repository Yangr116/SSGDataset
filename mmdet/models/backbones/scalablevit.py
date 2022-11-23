import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
# from mmcv.runner import load_checkpoint

from timm.models.vision_transformer import Block as TimmBlock
from timm.models.vision_transformer import Attention as TimmAttention
# from nas_utils import lg
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)
from mmcv.cnn import normal_init, trunc_normal_init, constant_init
import warnings
import math


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
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


class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        # print('H, W, self.ws:', H, W, self.ws)
        # print('x.shape:', x.shape)
        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)  # [B, h_g, w_g, ws, ws, C]

        # [3, B, h_g*w_g, num_heads, ws*ws, C//num_heads]
        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)

        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupAttention_position_lepe_local(nn.Module):
    """
    LSA: self attention within a group adding a position from v
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention_position_lepe_local, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        print('----- GroupAttention_position_lepe_local -----')
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws
        self.lepe_func = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def get_lepe(self, x, func):
        B, C, H, W = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        # (B, num_wins_h, num_wins_w, C, ws, ws)
        x = x.reshape(B, C, h_group, self.ws, w_group, self.ws).permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous().reshape(-1, C, self.ws, self.ws)  # (B*total_groups, C, win_h, win_w)
        lepe = func(x)
        # (B, h_group, w_group, self.ws, self.ws, C)
        lepe = lepe.contiguous().view(B, h_group, w_group, C, self.ws, self.ws).permute(0, 1, 2, 4, 5, 3)
        return lepe

    def img2win(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        x = x.reshape(B, h_group * w_group, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        # [B, h_g*w_g, num_heads, ws*ws, C//num_heads]
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.img2win(q, H, W)
        k = self.img2win(k, H, W)
        _v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        lepe = self.get_lepe(_v, self.lepe_func)  # (B, h_group, w_group, self.ws, self.ws, C)
        v = self.img2win(v, H, W)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        attn = attn + lepe
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupAttention_position_lepe_all(nn.Module):
    """
    LSA: self attention within a group adding and use the all feature map to get the position encoding from v
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention_position_lepe_all, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        print('------ GroupAttention_position_lepe_all -----')
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws
        self.lepe_func = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def get_lepe(self, x, func):
        # B, C, H, W = x.shape
        lepe = func(x).flatten(2).transpose(1, 2)  # (B, N, C)
        return lepe.contiguous()

    def img2win(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        x = x.reshape(B, h_group * w_group, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        # [B, h_g*w_g, num_heads, ws*ws, C//num_heads]
        return x

    def win2img(self, x, H, W):
        """
        x: (B, h_g*w_g, num_heads, ws*ws, C//num_heads)
        """
        h_group, w_group = H // self.ws, W // self.ws
        B = x.shape[0]
        C = x.shape[-1] * self.num_heads
        # (B, h_g*w_g, ws*ws, num_heads, C//num_heads)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = x.transpose(2, 3).reshape(B, H, W, C)  # (B, H, W, C)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # add padding
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        h_group, w_group = Hp // self.ws, Wp // self.ws

        # h_group, w_group = H // self.ws, W // self.ws

        qkv = self.qkv(x).reshape(B, Hp*Wp, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.img2win(q, Hp, Wp)
        k = self.img2win(k, Hp, Wp)
        v = self.img2win(v, Hp, Wp)

        _v = self.win2img(v, Hp, Wp)  # (B, H, W, C)
        lepe = self.get_lepe(_v, self.lepe_func)  # (B, h_group, w_group, self.ws, self.ws, C)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, Hp*Wp, C).contiguous()
        x = x + lepe # (B, N, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x.reshape(B, Hp, Wp, C)
            x = x[:, :H, :W, :].contiguous()
            x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_light_pooling(nn.Module):
    """
    Revise according to Twins Global Attention and subject to our version
    alternative 'conv1d reduction' with 'average pooling reduction'
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 c_ratio=1, size=56, g=1):
        super(Attention_light_pooling, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        print('----- Global Attention Pooling ------')
        # self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # channel expand ratio range of [1, 1.25, 1.5, 1.75, 2]
        self.c_ratio = c_ratio
        self.sr_ratio = sr_ratio
        print(f'@ dim: {dim}, c_ratio: {c_ratio}, sr_ratio: {sr_ratio}')

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * self.c_ratio) ** -0.5

        # self.method = method  # 使用的降维方法
        self.size = size  # reshape的特征图大小

        # lg.info('C_ratio, dim, N, N_reducted, C, G:  ', self.c_ratio, dim, self.N, int(self.N/self.N_ratio), int(dim/self.c_ratio), self.g)
        self.c_new = int(dim * self.c_ratio)
        if sr_ratio > 1:
            self.q = nn.Linear(dim, self.c_new, bias=qkv_bias)
            self.reduction = nn.Sequential(
                nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio),
                nn.Conv2d(dim, self.c_new, kernel_size=1, stride=1))
            self.norm_act = nn.Sequential(
                nn.LayerNorm(self.c_new),
                nn.GELU())
            self.kv = nn.Linear(self.c_new, self.c_new * 2, bias=qkv_bias)
            self.proj = nn.Linear(self.c_new, dim)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.sr_ratio > 1:
            q = self.q(x).reshape(B, N, self.num_heads, int(C / self.num_heads / self.c_ratio)).permute(0, 2, 1, 3)
            _x = x.permute(0, 2, 1).reshape(B, C, H, W)
            _x = self.reduction(_x).reshape(B, self.c_new, -1).permute(0, 2, 1)  # [B, N', C']
            _x = self.norm_act(_x)
            kv = self.kv(_x).reshape(B, -1, 2, self.num_heads, self.c_new // self.num_heads).permute(2, 0, 3, 1, 4)
            C = self.c_new
        else:
            q = self.q(x).reshape(B, N, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_light_dwconv(nn.Module):
    """
    Revise according to Twins Global Attention and subject to our version
    alternative 'conv1d reduction' with 'average pooling reduction'
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 c_ratio=1.25, size=56, g=1):
        super(Attention_light_dwconv, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        print('----- Global Attention dwconv -----\n')
        # self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # channel expand ratio range of [1, 1.25, 1.5, 1.75, 2]
        self.c_ratio = c_ratio
        self.sr_ratio = sr_ratio
        self.c_new = int(dim * self.c_ratio) # 扩展的通道数量
        print(f'@ dim: {dim}, dim_new: {self.c_new}, c_ratio: {c_ratio}, sr_ratio: {sr_ratio}\n')

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * self.c_ratio) ** -0.5

        # self.method = method  # 使用的降维方法
        self.size = size  # reshape的特征图大小

        # lg.info('C_ratio, dim, N, N_reducted, C, G:  ', self.c_ratio, dim, self.N, int(self.N/self.N_ratio), int(dim/self.c_ratio), self.g)

        if sr_ratio > 1:
            self.q = nn.Linear(dim, self.c_new, bias=qkv_bias)
            self.reduction = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim),
                nn.Conv2d(dim, self.c_new, kernel_size=1, stride=1))
            self.norm_act = nn.Sequential(
                nn.LayerNorm(self.c_new),
                nn.GELU())
            self.kv = nn.Linear(self.c_new, self.c_new * 2, bias=qkv_bias)
            self.proj = nn.Linear(self.c_new, dim)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.sr_ratio > 1:
            q = self.q(x).reshape(B, N, self.num_heads, int(self.c_new / self.num_heads)).permute(0, 2, 1, 3)
            _x = x.permute(0, 2, 1).reshape(B, C, H, W)
            _x = self.reduction(_x).reshape(B, self.c_new, -1).permute(0, 2, 1)  # [B, N', C']
            _x = self.norm_act(_x)
            kv = self.kv(_x).reshape(B, -1, 2, self.num_heads, int(self.c_new / self.num_heads)).permute(2, 0, 3, 1, 4)
            C = self.c_new
        else:
            q = self.q(x).reshape(B, N, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, int(C / self.num_heads)).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_light_dwconv_v3(nn.Module):
    """
    Revise according to Twins Global Attention and subject to our version
    alternative 'conv1d reduction' with 'average pooling reduction'
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 c_ratio=1.25, size=56, g=1):
        super(Attention_light_dwconv_v3, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        print('----- Global Attention dwconv v3 -----\n')
        # self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # channel expand ratio range of [1, 1.25, 1.5, 1.75, 2]
        self.c_ratio = c_ratio
        self.sr_ratio = sr_ratio
        self.c_new = int(dim * self.c_ratio) # 扩展的通道数量
        print(f'@ dim: {dim}, dim_new: {self.c_new}, c_ratio: {c_ratio}, sr_ratio: {sr_ratio}\n')

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * self.c_ratio) ** -0.5

        # self.method = method  # 使用的降维方法
        self.size = size  # reshape的特征图大小

        # lg.info('C_ratio, dim, N, N_reducted, C, G:  ', self.c_ratio, dim, self.N, int(self.N/self.N_ratio), int(dim/self.c_ratio), self.g)

        if sr_ratio > 1:
            # 主要是为了适应Cv和Cr不匹配的问题
            self.q = nn.Linear(dim, self.c_new, bias=qkv_bias)
            self.reduction = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim),
                nn.Conv2d(dim, self.c_new, kernel_size=1, stride=1))
            self.norm_act = nn.Sequential(
                nn.LayerNorm(self.c_new),
                nn.GELU())
            self.k = nn.Linear(self.c_new, self.c_new, bias=qkv_bias)
            self.v = nn.Linear(self.c_new, dim, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.sr_ratio > 1:
            q = self.q(x).reshape(B, N, self.num_heads, int(self.c_new / self.num_heads)).permute(0, 2, 1, 3)
            _x = x.permute(0, 2, 1).reshape(B, C, H, W)
            _x = self.reduction(_x).reshape(B, self.c_new, -1).permute(0, 2, 1)  # [B, N', C']
            _x = self.norm_act(_x)
            k = self.k(_x).reshape(B, -1, self.num_heads, int(self.c_new / self.num_heads)).permute(0, 2, 1, 3)
            v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
            C = C
        else:
            q = self.q(x).reshape(B, N, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, int(C / self.num_heads)).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(SBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                     drop_path, act_layer, norm_layer)

    def forward(self, x, H, W):
        return super(SBlock, self).forward(x)


class GroupBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super(GroupBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                         drop_path, act_layer, norm_layer)
        del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_alt_dwconv_lepe_all(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, c_ratio=1.25, sr_ratio=1, ws=1,
                 last_stage=False):
        super(Block_alt_dwconv_lepe_all, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                                    drop_path, act_layer, norm_layer)
        del self.attn
        last_stage=False
        if ws == 1 or last_stage is True:
            self.attn = Attention_light_dwconv(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                c_ratio=c_ratio)
        else:
            self.attn = GroupAttention_position_lepe_all(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                ws=ws)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_alt_dwconv_lepe_all_v3(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, c_ratio=1.25, sr_ratio=1, ws=1,
                 last_stage=False):
        super(Block_alt_dwconv_lepe_all_v3, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                                    drop_path, act_layer, norm_layer)
        del self.attn
        last_stage=False
        if ws == 1 or last_stage is True:
            self.attn = Attention_light_dwconv_v3(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                c_ratio=c_ratio)
        else:
            self.attn = GroupAttention_position_lepe_all(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                ws=ws)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_alt_dwconv_lepe_local(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, c_ratio=1.25, sr_ratio=1, ws=1,
                 last_stage=False):
        super(Block_alt_dwconv_lepe_local, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                                    drop_path, act_layer, norm_layer)
        del self.attn
        if ws == 1 or last_stage is True:
            self.attn = Attention_light_dwconv(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                c_ratio=c_ratio)
        else:
            self.attn = GroupAttention_position_lepe_local(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                ws=ws)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        print('patch overlapping kernel_size=3, stride=2, padding=1')
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(3, 3), stride=2, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, haloing=3):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        # 增加soft 第一层的时候haloing=3，之后的 =1
        print('patch overlapping kernel_size=7, stride=4, padding=3')
        # kernel_size = patch_size[0] + haloing
        #         # padding = (kernel_size - 1) // 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(7, 7), stride=4, padding=3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s
        print('------------- Position embedding --------------')

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


# borrow from PVT https://github.com/whai362/PVT.git
class PyramidVisionTransformer_expandlast_patchoverlap(BaseModule):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()

        for i in range(len(depths)):
            if i == 0:
                # self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
                self.patch_embeds.append(PatchEmbed_overlap(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(
                    # PatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i]))
                    PatchEmbed(224 // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1],
                               embed_dims[i]))  # cover different img size when i==0, so always 224//patch_size

            patch_num = self.patch_embeds[-1].num_patches + 1 if i == len(embed_dims) - 1 else self.patch_embeds[
                -1].num_patches
            self.pos_embeds.append(nn.Parameter(torch.zeros(1, patch_num, embed_dims[i])))
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        # self.norm = norm_layer(embed_dims[-1])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))

        last_conv_out_channel = 4096

        # self.head = nn.Sequential(
        #     nn.Linear(embed_dims[-1], last_conv_out_channel, bias=True),
        #     norm_layer(last_conv_out_channel),
        #     # nn.ReLU(inplace=True),
        #     nn.GELU(),
        #     # nn.Linear(embed_dims[-1], last_conv_out_channel) if num_classes > 0 else nn.Identity(),
        #     nn.Linear(last_conv_out_channel, num_classes) if num_classes > 0 else nn.Identity(),
        # )

        # init weights
        for pos_emb in self.pos_embeds:
            trunc_normal_(pos_emb, std=.02)

        # self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.blocks[k][i].drop_path.drop_prob = dpr[cur + i]
                if torch.cuda.current_device() == 0:
                    print('set drop_path', dpr[cur + i])
            cur += self.depths[k]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class CPVTV2_expandlast_patchoverlap(PyramidVisionTransformer_expandlast_patchoverlap):
    """
    Use useful results from CPVT. PEG and GAP.
    Therefore, cls token is no longer required.
    PEG is used to encode the absolute position on the fly, which greatly affects the performance when input resolution
    changes during the training (such as segmentation, detection)
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block, init_cfg=None):
        super(CPVTV2_expandlast_patchoverlap, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims,
                                                             num_heads, mlp_ratios,
                                                             qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                                                             drop_path_rate, norm_layer, depths,
                                                             sr_ratios, block_cls, init_cfg=None)
        del self.pos_embeds
        del self.cls_token
        # position
        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims])

    def no_weight_decay(self):
        return set(['cls_token'] + ['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x.mean(dim=1)  # GAP her

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class CPVTV2_expandlast_patchoverlap_noPEG(PyramidVisionTransformer_expandlast_patchoverlap):
    """
    Use useful results from CPVT. PEG and GAP.
    Therefore, cls token is no longer required.
    PEG is used to encode the absolute position on the fly, which greatly affects the performance when input resolution
    changes during the training (such as segmentation, detection)
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block,init_cfg=None):
        super(CPVTV2_expandlast_patchoverlap_noPEG, self).__init__(img_size, patch_size, in_chans, num_classes,
                                                                   embed_dims, num_heads, mlp_ratios,
                                                                   qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                                                                   drop_path_rate, norm_layer, depths,
                                                                   sr_ratios, block_cls, init_cfg=None)
        del self.pos_embeds
        del self.cls_token
        print('------------ no PEG ---------------')
        # position
        # self.pos_block = nn.ModuleList(
        #     [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        # )

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x.mean(dim=1)  # GAP her

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class Alt_Dwconv_Lepe(CPVTV2_expandlast_patchoverlap):
    """
    alias Twins-SVT
    """

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 256],
            num_heads=[1, 2, 4],
            mlp_ratios=[4, 4, 4],
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[4, 4, 4],
            sr_ratios=[4, 2, 1],
            block_cls=Block_alt_dwconv_lepe_all,
            wss=[7, 7, 7],
            c_ratios=[1.25, 1.25, 1.25, 1.25],
            sizes=[56, 28, 14, 7],
            extra_norm=False,
            pretrained=None,
            init_cfg=None):
        super(Alt_Dwconv_Lepe, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                              mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                              norm_layer, depths, sr_ratios, block_cls, init_cfg=None)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        
        # del self.blocks
        self.wss = wss
        
        # ---------- add extra norm ---------------
        self.extra_norm = extra_norm
        if self.extra_norm:
            self.norm_list = nn.ModuleList()
            for dim in embed_dims:
                self.norm_list.append(norm_layer(dim))
        # ---------- add extra norm ---------------

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks = nn.ModuleList()
        last_stage = False
        for k in range(len(depths)):
            # if k == len(depths) - 1:
            #     last_stage = True
            _block = nn.ModuleList([
                block_cls(
                    dim=embed_dims[k],
                    num_heads=num_heads[k],
                    mlp_ratio=mlp_ratios[k],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[k],
                    ws=1 if i % 2 == 1 else wss[k],
                    c_ratio=c_ratios[k],
                    last_stage=last_stage) for i in range(depths[k])])

            self.blocks.append(_block)
            cur += depths[k]
        # self.apply(self._init_weights)


    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, strict=False, logger=logger)

    # @auto_fp16()
    def forward_features(self, x):
        outputs = list()

        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    # We use a very simple implementation of PEG (single Sep at each stage).
                    # In fact, we can insert many PEGs into each block, which
                    # can improve the performance further. The core idea behind PEG is that even simple depthwise
                    # convolution provides sufficient positional information if paddings is given. It works quite well
                    # when global reception field is given. See https://arxiv.org/abs/2102.10882 for more details.
                    x = self.pos_block[i](x, H, W)
            if self.extra_norm:
                x = self.norm_list[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
            outputs.append(x)

        return outputs


    def forward(self, x):
        x = self.forward_features(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@BACKBONES.register_module()
class scalable_vit_small(Alt_Dwconv_Lepe):
    def __init__(self, **kwargs):
        super(scalable_vit_small, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 256, 512],
            num_heads=[2, 4, 8, 16],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 20, 2],
            wss=[7, 7, 7, 7],
            sr_ratios=[8, 4, 2, 1],
            drop_path_rate=0.2,
            extra_norm=True,
            sizes=[200, 100, 50, 25],
            block_cls=Block_alt_dwconv_lepe_all,
            c_ratios=[1.25, 1.25, 1.25, 1],
            **kwargs)