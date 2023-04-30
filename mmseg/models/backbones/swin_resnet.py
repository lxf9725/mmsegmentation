import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
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
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

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
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

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
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

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

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

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

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@MODELS.register_module()
class SwinResNet(BaseModule):
    """

    """
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
                swin_downsample = PatchMerging(
                    in_channels=swin_in_channels,
                    out_channels=2 * swin_in_channels,
                    stride=swin_strides[i + 1],
                    norm_cfg=swin_norm_cfg if swin_patch_norm else None,
                    init_cfg=None
                )
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
                swin_in_channels = swin_downsample.out_channels
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

    def res_plus_swin_model(self, res_x, swin_x):
        return res_x + swin_x

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
                res_x = self.res_plus_swin_model(res_x, swin_out)
                outs.append(res_x)
        return outs
