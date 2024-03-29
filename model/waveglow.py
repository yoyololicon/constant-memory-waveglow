import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

from utils import add_weight_norms

from .base import FlowBase
from .efficient_modules import AffineCouplingBlock, InvertibleConv1x1


@torch.jit.script
def fused_gate(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1.tanh() * x2.sigmoid()


class NonCausalLayer(nn.Module):
    def __init__(self,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()
        pad_size = dilation * (radix - 1) // 2
        self.W = nn.Conv1d(residual_channels, dilation_channels * 2, kernel_size=radix,
                           padding=pad_size, dilation=dilation, bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv1d(
                dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv1d(
                dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, y):
        xy = self.W(x) + y
        zw, zf = xy.chunk(2, 1)
        z = fused_gate(zw, zf)
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        return z[0] + x if len(z) else None, skip


class WN(nn.Module):
    def __init__(self,
                 in_channels,
                 aux_channels,
                 dilation_channels=256,
                 residual_channels=256,
                 skip_channels=256,
                 depth=8,
                 radix=3,
                 bias=False,
                 zero_init=True):
        super().__init__()
        dilations = 2 ** torch.arange(depth)
        self.dilations = dilations.tolist()
        self.in_chs = in_channels
        self.res_chs = residual_channels
        self.dil_chs = dilation_channels
        self.skp_chs = skip_channels
        self.rdx = radix
        self.r_field = sum(self.dilations) + 1

        self.V = nn.Conv1d(aux_channels, dilation_channels *
                           2 * depth, 1, bias=bias)
        self.V.apply(add_weight_norms)

        self.start = nn.Conv1d(in_channels, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)

        self.layers = nn.ModuleList(NonCausalLayer(d,
                                                   dilation_channels,
                                                   residual_channels,
                                                   skip_channels,
                                                   radix,
                                                   bias) for d in self.dilations[:-1])
        self.layers.append(NonCausalLayer(self.dilations[-1],
                                          dilation_channels,
                                          residual_channels,
                                          skip_channels,
                                          radix,
                                          bias,
                                          last_layer=True))
        self.layers.apply(add_weight_norms)

        self.end = nn.Conv1d(skip_channels, in_channels * 2, 1, bias=bias)
        if zero_init:
            self.end.weight.data.zero_()
            if bias:
                self.end.bias.data.zero_()

    def forward(self, x, y):
        x = self.start(x)
        y = self.V(y)
        cum_skip = 0
        for layer, v in zip(self.layers, y.chunk(len(self.layers), 1)):
            x, skip = layer(x, v)
            cum_skip = cum_skip + skip
        return self.end(cum_skip).chunk(2, 1)


class WaveGlow(FlowBase):
    def __init__(self,
                 flows,
                 n_group,
                 n_early_every,
                 n_early_size,
                 hop_size,
                 n_mels,
                 memory_efficient,
                 reverse_mode=False,
                 **kwargs):
        super().__init__(hop_size, reverse_mode)
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.n_mels = n_mels
        self.mem_efficient = memory_efficient

        self.upsample_factor = self._hop_length // n_group
        sub_win_size = self.upsample_factor * 2 + 1
        self.upsampler = nn.ConvTranspose1d(n_mels, n_mels, sub_win_size, self.upsample_factor,
                                            padding=sub_win_size // 2 - self.upsample_factor // 2, groups=n_mels)
        self.upsampler.apply(add_weight_norms)

        self.invconv1x1 = nn.ModuleList()
        self.WNs = nn.ModuleList()

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        self.z_split_sizes = []
        for k in range(flows):
            if k % self.n_early_every == 0 and k:
                n_remaining_channels -= n_early_size
                self.z_split_sizes.append(n_early_size)
            self.invconv1x1.append(InvertibleConv1x1(
                n_remaining_channels, memory_efficient=memory_efficient, reverse_mode=reverse_mode))
            self.WNs.append(
                AffineCouplingBlock(WN, memory_efficient=memory_efficient, in_channels=n_remaining_channels // 2,
                                    aux_channels=n_mels, reverse_mode=reverse_mode, **kwargs))
        self.z_split_sizes.append(n_remaining_channels)

    def forward_computation(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        y = self._upsample_h(h)
        batch_dim = x.size(0)
        x = x.view(batch_dim, -1, self.n_group).transpose(1, 2).contiguous()
        # y = y.view(batch_dim, y.size(1), -1, self.n_group).transpose(2, 3)
        # y = y.reshape(batch_dim, -1, y.size(-1))
        assert x.size(2) <= y.size(2)
        y = y[..., :x.size(2)]

        output_audio = []
        split_sections = [self.n_early_size, self.n_group]

        logdet: torch.Tensor = 0
        for k, (invconv, affine_coup) in enumerate(zip(self.invconv1x1, self.WNs)):
            if k % self.n_early_every == 0 and k:
                split_sections[1] -= self.n_early_size
                early_output, x = x.split(split_sections, 1)
                # these 2 lines actually copy tensors, may need optimization in the future
                output_audio.append(early_output)
                if self.mem_efficient:
                    x = x.clone()

            x, log_det_W = invconv(x)
            x, log_s = affine_coup(x, y)

            logdet += log_det_W + log_s.sum((1, 2))

        # assert split_sections[1] == self.z_split_sizes[-1]
        output_audio.append(x)
        return torch.cat(output_audio, 1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet

    def reverse_computation(self, z: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        y = self._upsample_h(h)
        batch_dim = z.size(0)
        z = z.view(batch_dim, -1, self.n_group).transpose(1, 2).contiguous()
        # y = y.view(batch_dim, y.size(1), -1, self.n_group).transpose(2, 3)
        # y = y.reshape(batch_dim, -1, y.size(-1))
        assert z.size(2) <= y.size(2)
        y = y[..., :z.size(2)]

        if self.mem_efficient:
            remained_z = [r.clone() for r in z.split(self.z_split_sizes, 1)]
        else:
            remained_z = z.split(self.z_split_sizes, 1)
        *remained_z, z = remained_z

        logdet: torch.Tensor = 0
        for k, invconv, affine_coup in zip(range(len(self.WNs) - 1, -1, -1), self.invconv1x1[::-1], self.WNs[::-1]):

            z, log_s = affine_coup.reverse(z, y)
            z, log_det_W = invconv.reverse(z)

            logdet += log_det_W + log_s.sum((1, 2))

            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)

        z = z.transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet

    def _upsample_h(self, h):
        # return F.interpolate(h, scale_factor=self.upsample_factor, mode='linear')
        return self.upsampler(h)
