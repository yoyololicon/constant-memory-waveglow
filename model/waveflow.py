import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

from utils import add_weight_norms
from .base import FlowBase
from .efficient_modules import InvertibleConv1x1
from .waveglow import fused_gate


class NonCausalLayer2D(nn.Module):
    def __init__(self,
                 h_dilation,
                 dilation,
                 aux_channels,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()
        self.h_pad_size = h_dilation * (radix - 1)
        self.pad_size = dilation * (radix - 1) // 2

        self.V = nn.Conv1d(aux_channels, dilation_channels * 2, radix,
                           dilation=dilation, padding=self.pad_size, bias=bias)

        self.W = nn.Conv2d(residual_channels, dilation_channels * 2,
                           kernel_size=radix,
                           dilation=(h_dilation, dilation), bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv2d(
                dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv2d(
                dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, y):
        tmp = F.pad(x, [self.pad_size] * 2 + [self.h_pad_size, 0])
        xy = self.W(tmp) + self.V(y).unsqueeze(2)
        zw, zf = xy.chunk(2, 1)
        # z = zw.tanh().mul(zf.sigmoid())
        z = fused_gate(zw, zf)
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        if len(z):
            output = z[0]
            return output + x[:, :, -output.size(2):], skip
        else:
            return None, skip

    def reverse_mode_forward(self, x, y, buffer=None):
        if buffer is None:
            buffer = F.pad(x, [0, 0, self.h_pad_size, 0])
        else:
            buffer = torch.cat((buffer[:, :, 1:], x), 2)
        tmp = F.pad(buffer, [self.pad_size] * 2)
        xy = self.W(tmp) + self.V(y).unsqueeze(2)
        zw, zf = xy.chunk(2, 1)
        z = fused_gate(zw, zf)
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        if len(z):
            output = z[0]
            return output + x, skip, buffer
        else:
            return None, skip, buffer


class WN2D(nn.Module):
    def __init__(self,
                 n_group,
                 aux_channels,
                 dilation_channels=256,
                 residual_channels=256,
                 skip_channels=256,
                 bias=False,
                 zero_init=True):
        super().__init__()

        dilation_dict = {
            8: [1] * 8,
            16: [1] * 8,
            32: [1, 2, 4] * 2 + [1, 2],
            64: [1, 2, 4, 8, 16, 1, 2, 4],
            128: [1, 2, 4, 8, 16, 32, 64, 1],
        }

        self.h_dilations = dilation_dict[n_group]
        dilations = 2 ** torch.arange(8)
        self.dilations = dilations.tolist()
        self.n_group = n_group
        self.res_chs = residual_channels
        self.dil_chs = dilation_channels
        self.skp_chs = skip_channels
        self.aux_chs = aux_channels
        self.r_field = sum(self.dilations) * 2 + 1
        self.h_r_field = sum(self.h_dilations) * 2 + 1

        self.start = nn.Conv2d(1, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)

        self.layers = nn.ModuleList(NonCausalLayer2D(hd, d,
                                                     aux_channels,
                                                     dilation_channels,
                                                     residual_channels,
                                                     skip_channels,
                                                     3,
                                                     bias) for hd, d in zip(self.h_dilations[:-1], self.dilations[:-1]))
        self.layers.append(NonCausalLayer2D(self.h_dilations[-1], self.dilations[-1],
                                            aux_channels,
                                            dilation_channels,
                                            residual_channels,
                                            skip_channels,
                                            3,
                                            bias,
                                            last_layer=True))
        self.layers.apply(add_weight_norms)

        self.end = nn.Conv2d(skip_channels, 2, 1, bias=bias)
        if zero_init:
            self.end.weight.data.zero_()
            if bias:
                self.end.bias.data.zero_()

    def forward(self, x, y):
        x = self.start(x)
        cum_skip = 0
        for layer in self.layers:
            x, skip = layer(x, y)
            cum_skip = cum_skip + skip
        return self.end(cum_skip).chunk(2, 1)

    def reverse_mode_forward(self, x, y, buffer_list=None):
        x = self.start(x)
        new_buffer_list = []
        if buffer_list is None:
            buffer_list = [None] * len(self.layers)

        cum_skip = 0
        for layer, buf in zip(self.layers, buffer_list):
            x, skip, buf = layer.reverse_mode_forward(x, y, buf)
            new_buffer_list.append(buf)
            cum_skip = cum_skip + skip

        return self.end(cum_skip).chunk(2, 1) + (new_buffer_list,)


class WaveFlow(FlowBase):
    def __init__(self,
                 flows,
                 n_group,
                 n_mels,
                 use_conv1x1,
                 memory_efficient,
                 reverse_mode=False,
                 **kwargs):
        super().__init__(256, reverse_mode)
        self.flows = flows
        self.n_group = n_group
        self.n_mels = n_mels
        self.sub_sr = self._hop_length // n_group

        self.upsampler = nn.Sequential(
            nn.ReplicationPad1d((0, 1)),
            nn.ConvTranspose1d(n_mels, n_mels, self.sub_sr *
                               2 + 1, self.sub_sr, padding=self.sub_sr),
            nn.LeakyReLU(0.4, True)
        )
        self.upsampler.apply(add_weight_norms)

        self.WNs = nn.ModuleList()

        if use_conv1x1:
            self.invconv1x1 = nn.ModuleList()

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        for k in range(flows):
            self.WNs.append(WN2D(n_group, n_mels, **kwargs))
            if use_conv1x1:
                self.invconv1x1.append(InvertibleConv1x1(
                    n_group, memory_efficient=memory_efficient, reverse_mode=reverse_mode))

    def forward_computation(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        y = self._upsample_h(h)

        batch_dim = x.size(0)
        x = x.view(batch_dim, 1, -1, self.n_group).transpose(2, 3).contiguous()
        y = y[..., :x.size(-1)]

        if hasattr(self, 'invconv1x1'):
            invconv1x1 = self.invconv1x1
        else:
            invconv1x1 = [None] * self.flows

        logdet: Tensor = 0
        for k, (WN, invconv) in enumerate(zip(self.WNs, invconv1x1)):
            x0 = x[:, :, :1]
            log_s, t = WN(x[:, :, :-1], y)
            xout = x[:, :, 1:] * log_s.exp() + t

            logdet += log_s.sum((1, 2, 3))

            if invconv is None:
                x = torch.cat((xout.flip(2), x0), 2)
            else:
                x, log_det_W = invconv(torch.cat((x0, xout), 2).squeeze(1))
                x = x.unsqueeze(1)
                logdet += log_det_W

        return x.squeeze(1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet

    def reverse_computation(self, z: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        y = self._upsample_h(h)

        batch_dim = z.size(0)
        z = z.view(batch_dim, 1, -1, self.n_group).transpose(2, 3).contiguous()
        y = y[..., :z.size(-1)]

        if hasattr(self, 'invconv1x1'):
            invconv1x1 = self.invconv1x1
        else:
            invconv1x1 = [None] * self.flows

        logdet: Tensor = None
        for k, WN, invconv in zip(range(self.flows - 1, -1, -1), self.WNs[::-1], invconv1x1[::-1]):
            if invconv is None:
                z = z.flip(2)
            else:
                z, log_det_W = invconv.reverse(z.squeeze(1))
                z = z.unsqueeze(1)
                if logdet is None:
                    logdet = log_det_W.repeat(z.shape[0])
                else:
                    logdet += log_det_W

            xnew = z[:, :, :1]
            x = [xnew]

            buffer_list = None
            for i in range(1, self.n_group):
                log_s, t, buffer_list = WN.reverse_mode_forward(
                    xnew, y, buffer_list)
                xnew = (z[:, :, i:i+1] - t) / log_s.exp()
                x.append(xnew)

                if logdet is None:
                    logdet = -log_s.sum((1, 2, 3))
                else:
                    logdet -= log_s.sum((1, 2, 3))
            z = torch.cat(x, 2)

        z = z.squeeze(1).transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet

    def _upsample_h(self, h):
        return self.upsampler(h)
