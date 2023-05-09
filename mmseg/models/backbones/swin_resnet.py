import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from SoftPool import SoftPool2d

from mmcv.cnn import build_conv_layer, build_plugin_layer
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout

from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS
from ..utils import ResLayer
from ..utils.embed import PatchEmbed, PatchMerging


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# 可变形卷积
class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Deformable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class BasicBlock(BaseModule):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    If style is "pytorch", the stride-two layer is the 3x3 conv layer, if it is
    "caffe", the stride-two layer is the first 1x1 conv layer.
    """

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        """Forward function for plugins."""
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class WindowMSA(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        x = x
        return x


class FeatureComPreBlock(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        self.embed = embed_dims
        self.dcn = DeformConv2d(embed_dims, embed_dims * 2, 3, padding=1, stride=2, modulation=True)
        self.dconv = nn.Conv2d(embed_dims * 2, embed_dims * 2, kernel_size=3, dilation=2, padding=2)
        self.conv1 = nn.Conv2d(embed_dims, embed_dims * 2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(embed_dims, embed_dims * 2, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(embed_dims * 2, embed_dims * 2, kernel_size=1, stride=2)
        self.bn1 = nn.BatchNorm2d(embed_dims)
        self.bn2 = nn.BatchNorm2d(embed_dims * 2)
        self.relu = nn.ReLU(inplace=True)
        self.gn1 = nn.GroupNorm(embed_dims // 2, embed_dims)
        self.gn2 = nn.GroupNorm(embed_dims, embed_dims * 2)
        self.pool = SoftPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.gelu = nn.GELU()

    def forward(self, x, input_size):
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C).permute([0, 3, 1, 2])
        short = x
        x = self.gelu(self.bn2(self.conv1(x)))

        x = self.bn2(self.dconv(x))

        x = self.gelu(self.bn2(self.conv3(x)))
        short = self.gn1(self.pool(short))

        short = self.gelu(self.bn2(self.conv2(short)))
        x = x + short
        x = x.view(B, -1, 2 * C)  # B H/2*W/2 4*C
        out_h = H // 2
        out_w = W // 2
        output_size = (out_h, out_w)
        return x, output_size


class SpaceInterBlock(nn.Module):
    def __int__(self, embed_dim):
        super().__int__()
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.gelu = nn.GELU()

    def forward(self, sim_x, swin_x):
        s = self.gelu(self.bn1(self.conv1(sim_x)))
        s_h, s_w = self.pool_h(s), self.pool_w(s)
        s = torch.matmul(s_h, s_w)
        s = self.gelu(self.bn2(self.conv2(s)))
        return s + swin_x


class SwinBlockSequence(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample
        self.sim = SpaceInterBlock(embed_dims)

    def forward(self, x, hw_shape):
        sin_x = x
        i = 1
        for block in self.blocks:
            x = block(x, hw_shape)
            if i % 2 == 0:
                sin_x = self.sim(sin_x, x)
                x = sin_x
            i += 1

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


# relative aggregate model
# 在batch中分别展平每一个tensor
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# 注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2, pool_types=None):
        super().__init__()
        if pool_types is None:
            pool_types = ['avg', 'max', 'soft']
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU()
        )
        self.pool_types = pool_types
        self.incr = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        channel_attention_sum = None
        # 每个通道得到1x1的pool
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool1d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_pool_mlp = self.mlp(avg_pool)
        max_pool_mlp = self.mlp(max_pool)
        pool_add = avg_pool_mlp + max_pool_mlp
        # 须引入SoftPool
        soft_pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        soft_pool_mlp = self.mlp(soft_pool)
        weight_pool = soft_pool * pool_add
        channel_attention_sum = self.incr(weight_pool)
        att = torch.sigmoid(channel_attention_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return att


class ResPlusSwinBlock(nn.Module):
    """
    in_channels:SwinTransform块输入维度
    out_channels:该阶段网络输出维度
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = BasicConv(in_planes=in_channels, out_planes=out_channels, kernel_size=1,
                                 stride=1, groups=1, relu=True, bn=True, bias=False)
        self.fuse = ChannelAttention(in_channels=out_channels, reduction_ratio=2)
        self.dcn = DeformConv2d(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=3, stride=1, padding=0, modulation=True)

    def forward(self, res_x, swin_x):
        res_x = self.dcn(res_x)
        s1 = self.conv1x1(swin_x)
        short = self.fuse(s1)
        short = short * res_x
        return self.res_x + short + s1


@MODELS.register_module()
class SwinResNet(BaseModule):
    res_arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 res_depth,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3),
                 swin_pretrain_img_size=224,
                 swin_embed_dims=96,
                 swin_patch_size=4,
                 swin_window_size=7,
                 swin_mlp_ratio=4,
                 swin_depths=(2, 2, 6, 4),
                 swin_num_heads=(3, 6, 12, 24),
                 swin_strides=(4, 2, 2, 2),
                 swin_qkv_bias=True,
                 swin_qk_scale=None,
                 swin_patch_norm=None,
                 swin_drop_rate=0.,
                 swin_attn_drop_rate=0.,
                 swin_drop_path_rate=0.1,
                 swin_use_abs_pos_embed=False,
                 swin_act_cfg=dict(type='GELU'),
                 swin_norm_cfg=dict(type='LN', requires_grad=True),
                 res_stem_channels=64,
                 res_base_channels=64,
                 res_num_stages=4,
                 res_strides=(1, 2, 2, 2),
                 res_dilations=(1, 1, 1, 1),
                 res_style='pytorch',
                 res_deep_stem=False,
                 res_avg_down=False,
                 res_conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 res_norm_eval=False,
                 res_dcn=None,
                 res_stage_with_dcn=(False, False, False, False),
                 res_plugins=None,
                 res_multi_grid=None,
                 res_contract_dilation=False,
                 res_zero_init_residual=True,
                 with_cp=False,
                 swin_pretrained=None,
                 res_pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None):
        super().__init__(init_cfg)

        # ResNet
        self.res_depth = res_depth
        self.frozen_stages = frozen_stages,
        self.swin_pretrained = swin_pretrained,
        self.res_pretrained = res_pretrained,
        self.res_zero_init_residual = res_zero_init_residual,
        res_block_init_cfg = None
        self.res_depth = res_depth,
        self.res_stem_channels = res_stem_channels
        self.res_base_channels = res_base_channels
        self.res_num_stages = res_num_stages
        assert 1 <= res_num_stages <= 4
        self.res_strides = res_strides
        self.res_dilations = res_dilations
        assert len(res_strides) == len(res_dilations) == res_num_stages
        self.out_indices = out_indices
        assert max(out_indices) < res_num_stages
        self.res_style = res_style
        self.res_deep_stem = res_deep_stem
        self.res_avg_down = res_avg_down
        self.with_cp = with_cp,
        self.res_conv_cfg = res_conv_cfg
        self.norm_cfg = norm_cfg
        self.res_norm_eval = res_norm_eval
        self.res_dcn = res_dcn
        self.res_stage_with_dcn = res_stage_with_dcn
        if res_dcn is not None:
            assert len(res_stage_with_dcn) == res_num_stages
        self.res_plugins = res_plugins
        self.res_multi_grid = res_multi_grid
        self.res_contract_dilation = res_contract_dilation
        self.res_block, res_stage_blocks = self.res_arch_settings[res_depth]
        self.res_stage_blocks = res_stage_blocks[:res_num_stages]
        self.res_inplanes = res_stem_channels

        # SwinTransformer
        swin_num_layers = len(swin_depths)
        self.swin_use_abs_pos_embed = swin_use_abs_pos_embed
        assert swin_strides[0] == swin_patch_size, 'Use non-overlapping patch embed.'
        # 判断输入配置是否正确
        # 判断resnet块个数设置是否正确
        if res_depth not in self.res_arch_settings:
            raise KeyError(f'invalid depth{res_depth} for resnet')

        if isinstance(swin_pretrain_img_size, int):
            swin_pretrain_img_size = to_2tuple(swin_pretrain_img_size)
        elif isinstance(swin_pretrain_img_size, tuple):
            if len(swin_pretrain_img_size) == 1:
                swin_pretrain_img_size = to_2tuple(swin_pretrain_img_size[0])
            assert len(swin_pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(swin_pretrain_img_size)}'
        # 判断预训练权重
        assert not (init_cfg and (swin_pretrained or res_pretrained)), \
            'init_cfg and pretrained cannot be setting at the same time'
        # 初始化SwinTransformer权重
        if isinstance(swin_pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.swin_init_cfg = dict(type='Pretrained', checkpoint=swin_pretrained)
        elif swin_pretrained is None:
            swin_init_cfg = init_cfg
        else:
            print(type(swin_pretrained))
            print('swin_pretrained')
            raise TypeError('pretrained must be a str or None')
        # 初始化Resnet
        if isinstance(res_pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.res_init_cfg = dict(type='Pretrained', checkpoint=res_pretrained)
        elif res_pretrained is None:
            self.res_init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm'])
            ]
            res_block = self.res_arch_settings[res_depth][0]
            if self.res_zero_init_residual:
                if res_block is BasicBlock:
                    res_block_init_cfg = dict(
                        type='Constant',
                        val=0,
                        override=dict(name='norm2')
                    )
                elif res_block is Bottleneck:
                    res_block_init_cfg = dict(
                        type='Constant',
                        val=0,
                        override=dict(name='norm3'))
        else:
            print(type(res_pretrained))
            print('res_pretrained')
            raise TypeError('pretrained must be a str or None')

        self.swin_patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=swin_embed_dims,
            conv_type='Conv2d',
            kernel_size=swin_patch_size,
            stride=swin_strides[0],
            padding='corner',
            norm_cfg=swin_norm_cfg if swin_patch_norm else None,
            init_cfg=None
        )

        if self.swin_use_abs_pos_embed:
            swin_patch_row = swin_pretrain_img_size[0] // swin_patch_size
            swin_patch_col = swin_pretrain_img_size[1] // swin_patch_size
            swin_num_patches = swin_patch_row * swin_patch_col
            self.swin_absolute_pos_embed = nn.Parameter(
                torch.zeros((1, swin_num_patches, swin_embed_dims)))
        self.swin_drop_after_pos = nn.Dropout(p=swin_drop_rate)

        swin_total_depth = sum(swin_depths)
        swin_dpr = [x.item() for x in torch.linspace(0, swin_drop_path_rate, swin_total_depth)]
        self.swin_stages = ModuleList()
        swin_in_channels = swin_embed_dims
        for i in range(swin_num_layers):
            if i < swin_num_layers - 1:
                swin_downsample = FeatureComPreBlock(embed_dims=swin_in_channels)
                # swin_downsample = PatchMerging(
                #     in_channels=swin_in_channels,
                #     out_channels=2 * swin_in_channels,
                #     stride=swin_strides[i + 1],
                #     norm_cfg=swin_norm_cfg if swin_patch_norm else None,
                #     init_cfg=None
                # )
            else:
                swin_downsample = None
            swin_stage = SwinBlockSequence(
                embed_dims=swin_in_channels,
                num_heads=swin_num_heads[i],
                feedforward_channels=int(swin_mlp_ratio * swin_in_channels),
                depth=swin_depths[i],
                window_size=swin_window_size,
                qkv_bias=swin_qkv_bias,
                qk_scale=swin_qk_scale,
                drop_rate=swin_drop_rate,
                attn_drop_rate=swin_attn_drop_rate,
                drop_path_rate=swin_dpr[sum(swin_depths[:i]):sum(swin_depths[:i + 1])],
                downsample=swin_downsample,
                act_cfg=swin_act_cfg,
                norm_cfg=swin_norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.swin_stages.append(swin_stage)
            if swin_downsample:
                swin_in_channels = 2 * swin_in_channels
        self.swin_num_features = [int(swin_embed_dims * 2 ** i) for i in range(swin_num_layers)]
        for i in out_indices:
            swin_layer = build_norm_layer(swin_norm_cfg, self.swin_num_features[i])[1]
            swin_layer_name = f'swin_norm{i}'
            self.add_module(swin_layer_name, swin_layer)
            # 构造网络
        self.res_make_stem_layer(in_channels, res_stem_channels)
        self.res_layers = []
        for i, res_num_blocks in enumerate(self.res_stage_blocks):
            res_stride = res_strides[i]
            res_dilation = res_dilations[i]
            res_dcn = self.res_dcn if self.res_stage_with_dcn[i] else None
            if res_plugins is not None:
                res_stage_plugins = self.res_make_stage_plugins(res_plugins, i)
            else:
                res_stage_plugins = None
            # multi grid is applied to last layer only
            res_stage_multi_grid = res_multi_grid if i == len(
                self.res_stage_blocks) - 1 else None
            res_planes = res_base_channels * 2 ** i
            res_layer = self.make_res_layer(
                block=self.res_block,
                inplanes=self.res_inplanes,
                planes=res_planes,
                num_blocks=res_num_blocks,
                stride=res_stride,
                dilation=res_dilation,
                style=self.res_style,
                avg_down=self.res_avg_down,
                with_cp=with_cp,
                conv_cfg=res_conv_cfg,
                norm_cfg=norm_cfg,
                dcn=res_dcn,
                plugins=res_stage_plugins,
                multi_grid=res_stage_multi_grid,
                contract_dilation=res_contract_dilation,
                init_cfg=res_block_init_cfg)
            self.res_inplanes = res_planes * self.res_block.expansion
            res_layer_name = f'res_layer{i + 1}'
            self.add_module(res_layer_name, res_layer)
            self.res_layers.append(res_layer_name)

            self.res_plus_swin_blocks = ModuleList()
            self.res_plus_swin_blocks.append(ResPlusSwinBlock(128, 256))
            self.res_plus_swin_blocks.append(ResPlusSwinBlock(256, 512))
            self.res_plus_swin_blocks.append(ResPlusSwinBlock(512, 1024))
            self.res_plus_swin_blocks.append(ResPlusSwinBlock(1024, 2048))

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def res_norm1(self):
        return getattr(self, self.res_norm1_name)

    def res_make_stage_plugins(self, plugins, stage_idx):
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.res_num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def res_make_stem_layer(self, in_channels, stem_channels):
        """ 构造 ResNet网络stem层"""
        if self.res_deep_stem:
            self.res_stem = nn.Sequential(
                build_conv_layer(
                    self.res_conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.res_conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.res_conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.res_conv1 = build_conv_layer(
                self.res_conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.res_norm1_name, res_norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.res_norm1_name, res_norm1)
            self.res_relu = nn.ReLU(inplace=True)
        self.res_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def res_freeze_stages(self):
        """冻结区块参数和归化层状态"""
        if self.frozen_stages >= 0:
            if self.res_deep_stem:
                self.res_stem.eval()
                for param in self.res_stem.parameters():
                    param.requires_grad = False
            else:
                self.res_norm1.eval()
                for m in [self.res_conv1, self.res_norm1]:
                    for param in m.parameters():
                        param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'res_layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def swin_freeze_stages(self):
        if self.frozen_stages >= 0:
            self.swin_patch_embed.eval()
            for param in self.swin_patch_embed.parameters():
                param.requires_grad = False
            if self.swin_use_abs_pos_embed:
                self.swin_absolute_pos_embed.requires_grad = False
            self.swin_drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                swin_norm_layer = getattr(self, f'swin_norm{i - 1}')
                swin_norm_layer.eval()
                for param in swin_norm_layer.parameters():
                    param.requires_grad = False

            m = self.swin_stagess[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def swin_train(self, mode=True):
        super().swin_train(mode)
        self.swin_freeze_stages()

    def res_train(self, mode=True):
        super().res_train(mode)
        self.res_freeze_stages()
        if mode and self.res_norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    # 加载SwinTransformer预训练权重并初始化
    def swin_init_weights(self):
        if self.swin_init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            if self.swin_use_abs_pos_embed:
                trunc_normal_(self.swin_absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.swin_init_cfg, f'Only support ' \
                                                       f'specify `Pretrained` in ' \
                                                       f'`init_cfg` in ' \
                                                       f'{self.__class__.__name__} '
            swin_ckpt = CheckpointLoader.load_checkpoint(
                self.swin_init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in swin_ckpt:
                _swin_state_dict = swin_ckpt['state_dict']
            elif 'model' in swin_ckpt:
                _swin_state_dict = swin_ckpt['model']
            else:
                _swin_state_dict = swin_ckpt

            swin_state_dict = OrderedDict()
            for k, v in _swin_state_dict.item():
                if k.startswith('backbone.'):
                    swin_state_dict[k[9:]] = v
                else:
                    swin_state_dict[k] = v
            if list(swin_state_dict.keys())[0].startswith('module.'):
                swin_state_dict = {k[7:]: v for k, v in swin_state_dict.items()}

            if swin_state_dict.get('absolute_pos_embed') is not None:
                swin_absolute_pos_embed = swin_state_dict['absolute_pos_embed']
                swin_N1, swin_L, swin_C1 = swin_absolute_pos_embed.size()
                swin_N2, swin_C2, swin_H, swin_W = self.swin_absolute_pos_embed.size()
                if swin_N1 != swin_N2 or swin_C1 != swin_C2 or swin_L != swin_H * swin_W:
                    print_log('Error in loading absolute_pos_embed, pass')
                else:
                    swin_state_dict['absolute_pos_embed'] = swin_absolute_pos_embed.view(
                        swin_N2, swin_H, swin_W, swin_C2).permute(0, 3, 1, 2).contiguous()

            swin_relative_position_bias_table_keys = [
                k for k in swin_state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in swin_relative_position_bias_table_keys:
                table_pretrained = swin_state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    print_log(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    swin_state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(swin_state_dict, strict=False)

    # 融合SwinTransformer模块与ResNet模块输出

    def forward(self, x):
        res_x = x
        swin_x, swin_hw_shape = self.swin_patch_embed(x)
        # ResNet stem
        if self.res_deep_stem:
            res_x = self.res_stem(res_x)
        else:
            res_x = self.res_conv1(res_x)
            res_x = self.res_norm1(res_x)
            res_x = self.res_relu(res_x)
        res_x = self.res_maxpool(res_x)
        # SwinTransformer
        if self.swin_use_abs_pos_embed:
            swin_x = swin_x + self.swin_absolute_pos_embed
        swin_x = self.swin_drop_after_pos(swin_x)
        outs = []
        for enumerate_res_layers, enumerate_swin_stages in zip(enumerate(self.res_layers), enumerate(self.swin_stages)):
            i, res_layer_name = enumerate_res_layers
            _, swin_stage = enumerate_swin_stages
            res_layer = getattr(self, res_layer_name)
            swin_x, swin_hw_shape, swin_out, swin_out_hw_shape = swin_stage(swin_x, swin_hw_shape)
            res_x = res_layer(res_x)
            if i in self.out_indices:
                swin_norm_layer = getattr(self, f'swin_norm{i}')
                swin_out = swin_norm_layer(swin_out)
                swin_out = swin_out.view(-1, *swin_out_hw_shape, self.swin_num_features[i]).permute(0, 3, 1,
                                                                                                    2).contiguous()

                res_x = self.res_plus_swin_blocks[i](res_x, swin_out)
                outs.append(res_x)
        return outs
