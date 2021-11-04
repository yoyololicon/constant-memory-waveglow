import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F

from utils.util import add_weight_norms
from .base import FlowBase
from .waveglow import fused_gate
from .efficient_modules import AffineCouplingBlock, InvertibleConv1x1


class Predictor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 layers,
                 bias,
                 groups):
        super().__init__()

        self.groups = groups

        self.start = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels * groups, 1, bias=bias),
            nn.BatchNorm1d(hidden_channels * groups),
            nn.Tanh())

        self.end = nn.Conv1d(hidden_channels * groups,
                             out_channels * groups, 1, bias=bias, groups=groups)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels * groups, hidden_channels *
                          groups, 3, padding=1, bias=bias, groups=groups),
                nn.BatchNorm1d(hidden_channels * groups),
                nn.Tanh(),
                nn.Conv1d(hidden_channels * groups, hidden_channels *
                          groups, 3, padding=1, bias=bias, groups=groups),
                nn.BatchNorm1d(hidden_channels * groups),
                nn.Tanh()
            ) for _ in range(layers)
        ])

    def forward(self, x):
        x = self.start(x)
        for block in self.res_blocks:
            x = block(x) + x
        return self.end(x)


class NonCausalLayerLVC(nn.Module):
    def __init__(self,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()

        self.padding = dilation * (radix - 1) // 2
        self.dilation = dilation

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv1d(
                dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv1d(
                dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, weights):
        batch, steps, *kernel_size = weights.shape
        weights = weights.view(-1, *kernel_size[1:])

        offset = x.shape[2] // steps
        padded_x = F.pad(x, (self.padding,) * 2)
        unfolded_x = padded_x.unfold(2, self.padding * 2 + offset, offset).transpose(
            1, 2).contiguous().view(1, -1, self.padding * 2 + offset)

        z = F.conv1d(unfolded_x, weights, dilation=self.dilation,
                     groups=batch * steps)
        zw, zv = z.view(batch, steps, kernel_size[0], -1).transpose(
            1, 2).contiguous().view(batch, kernel_size[0], -1).chunk(2, 1)
        z = fused_gate(zw, zv)
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        return z[0] + x if len(z) else None, skip


class WN_LVC(nn.Module):
    def __init__(self,
                 in_channels,
                 depth,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 radix,
                 bias,
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

        self.start = nn.Conv1d(in_channels, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)

        self.layers = nn.ModuleList(NonCausalLayerLVC(d,
                                                      dilation_channels,
                                                      residual_channels,
                                                      skip_channels,
                                                      radix,
                                                      bias) for d in self.dilations[:-1])
        self.layers.append(NonCausalLayerLVC(self.dilations[-1],
                                             dilation_channels,
                                             residual_channels,
                                             skip_channels,
                                             radix,
                                             bias,
                                             last_layer=True))

        self.end = nn.Conv1d(skip_channels, in_channels * 2, 1, bias=bias)
        if zero_init:
            self.end.weight.data.zero_()
            if bias:
                self.end.bias.data.zero_()

    def forward(self, x, weights):
        x = self.start(x)
        cum_skip = 0
        for layer, w in zip(self.layers, weights.chunk(len(self.dilations), 0)):
            x, skip = layer(
                x, w.view(w.shape[1], w.shape[2], 2 * self.dil_chs, self.res_chs, self.rdx))
            cum_skip = cum_skip + skip
        return self.end(cum_skip).chunk(2, 1)


class MelGlow(FlowBase):
    def __init__(self,
                 flows,
                 n_group,
                 n_early_every,
                 n_early_size,
                 hop_size,
                 n_mels,
                 depth=7,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=32,
                 radix=3,
                 predict_channels=64,
                 predict_layers=3,
                 reverse_mode=False,
                 bias=False):
        super().__init__(hop_size, reverse_mode=reverse_mode)
        self.flows = flows
        self.depth = depth
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.n_mels = n_mels

        self.upsample_factor = self._hop_length // n_group

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
                n_remaining_channels, memory_efficient=False, reverse_mode=reverse_mode))
            self.WNs.append(
                AffineCouplingBlock(WN_LVC, memory_efficient=False, reverse_mode=reverse_mode,
                                    in_channels=n_remaining_channels // 2,
                                    depth=depth,
                                    dilation_channels=dilation_channels,
                                    residual_channels=residual_channels,
                                    skip_channels=skip_channels,
                                    radix=radix,
                                    bias=bias))
        self.z_split_sizes.append(n_remaining_channels)

        self.pred = Predictor(
            n_mels,
            2 * dilation_channels * residual_channels * radix,
            predict_channels,
            predict_layers,
            bias,
            flows * depth
        )

    def forward_computation(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        batch_dim = x.size(0)
        x = x[:, :x.shape[1] // self._hop_length * self._hop_length]
        x = x.view(batch_dim, -1, self.n_group).transpose(1, 2)
        y = h[..., :x.shape[2] // self.upsample_factor]

        weights = self.pred(y)
        weights = weights.view(weights.shape[0], self.flows * self.depth, -1,
                               weights.shape[2]).permute(1, 0, 3, 2).contiguous().chunk(self.flows, 0)

        output_audio = []
        split_sections = [self.n_early_size, self.n_group]

        logdet: Tensor = 0
        for k, (invconv, affine_coup, lvc_weights) in enumerate(zip(self.invconv1x1,
                                                                    self.WNs,
                                                                    weights)):
            if k % self.n_early_every == 0 and k:
                split_sections[1] -= self.n_early_size
                early_output, x = x.split(split_sections, 1)
                # these 2 lines actually copy tensors, may need optimization in the future
                output_audio.append(early_output)
                x = x.clone()

            x, log_det_W = invconv(x)
            x, log_s = affine_coup(x, lvc_weights)

            logdet += log_det_W + log_s.sum((1, 2))

        assert split_sections[1] == self.z_split_sizes[-1]
        output_audio.append(x)
        return torch.cat([o.transpose(1, 2) for o in output_audio], 2).view(batch_dim, -1), logdet

    def reverse_computation(self, z: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        batch_dim = z.size(0)
        z = z[:, :z.shape[1] // self._hop_length * self._hop_length]
        z = z.view(batch_dim, -1, self.n_group).transpose(1, 2)
        y = h[..., :z.shape[2] // self.upsample_factor]

        weights = self.pred(y)
        weights = weights.view(weights.shape[0], self.flows * self.depth, -1,
                               weights.shape[2]).permute(1, 0, 3, 2).contiguous().chunk(self.flows, 0)

        remained_z = []
        for r in z.split(self.z_split_sizes, 1):
            remained_z.append(r.clone())
        *remained_z, z = remained_z

        logdet: Tensor = 0
        for k, invconv, affine_coup, lvc_weights in zip(range(self.flows - 1, -1, -1),
                                                        self.invconv1x1[::-1],
                                                        self.WNs[::-1],
                                                        weights[::-1]):

            z, log_s = affine_coup.reverse(z, lvc_weights)
            z, log_det_W = invconv.reverse(z)

            logdet += log_det_W + log_s.sum((1, 2))

            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)

        z = z.transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet
