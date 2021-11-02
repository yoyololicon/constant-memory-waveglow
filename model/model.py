import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.util import add_weight_norms
from librosa.filters import mel
import numpy as np
from nnAudio.Spectrogram import MelSpectrogram

from model.efficient_modules import AffineCouplingBlock, InvertibleConv1x1


@torch.jit.script
def fused_gate(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1.tanh() * x2.sigmoid()


class _NonCausalLayer(nn.Module):
    def __init__(self,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 aux_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()
        pad_size = dilation * (radix - 1) // 2
        self.WV = nn.Conv1d(residual_channels + aux_channels, dilation_channels * 2, kernel_size=radix,
                            padding=pad_size, dilation=dilation, bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv1d(dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv1d(dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, y):
        xy = torch.cat((x, y), 1)
        zw, zf = self.WV(xy).chunk(2, 1)
        z = zw.tanh().mul(zf.sigmoid())
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
        self.aux_chs = aux_channels
        self.rdx = radix
        self.r_field = sum(self.dilations) + 1

        self.start = nn.Conv1d(in_channels, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)

        self.layers = nn.ModuleList(_NonCausalLayer(d,
                                                    dilation_channels,
                                                    residual_channels,
                                                    skip_channels,
                                                    aux_channels,
                                                    radix,
                                                    bias) for d in self.dilations[:-1])
        self.layers.append(_NonCausalLayer(self.dilations[-1],
                                           dilation_channels,
                                           residual_channels,
                                           skip_channels,
                                           aux_channels,
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
        cum_skip = None
        for layer in self.layers:
            x, skip = layer(x, y)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip = cum_skip + skip
        return self.end(cum_skip).chunk(2, 1)


class WaveGlow(BaseModel):
    def __init__(self,
                 flows,
                 n_group,
                 n_early_every,
                 n_early_size,
                 sr,
                 window_size,
                 hop_size,
                 n_mels,
                 memory_efficient,
                 **kwargs):
        super().__init__()
        self.flows = flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.win_size = window_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.sr = sr

        self.upsample_factor = hop_size // n_group
        sub_win_size = window_size // n_group
        # self.upsampler = nn.ConvTranspose1d(n_mels, n_mels, sub_win_size, self.upsample_factor,
        #                                    padding=sub_win_size // 2, bias=False)

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
            self.invconv1x1.append(InvertibleConv1x1(n_remaining_channels, memory_efficient=memory_efficient))
            self.WNs.append(
                AffineCouplingBlock(WN, memory_efficient=memory_efficient, in_channels=n_remaining_channels // 2,
                                    aux_channels=n_mels, **kwargs))
        self.z_split_sizes.append(n_remaining_channels)

        self.mel = nn.Sequential(
                nn.ReflectionPad1d((window_size // 2 - hop_size // 2, window_size // 2 + hop_size // 2)),
                MelSpectrogram(sr, window_size, n_mels, hop_size, center=False, fmax=8000)
            )

    def get_mel(self, x):
        return self.mel(x.unsqueeze(1)).add_(1e-7).log_()

    def forward(self, x, h=None):
        if h is None:
            h = self.get_mel(x)
        y = self._upsample_h(h)

        batch_dim, n_mels, group_steps = y.shape
        x = x.view(batch_dim, -1, self.n_group).transpose(1, 2)
        assert x.size(2) <= y.size(2)
        y = y[..., :x.size(2)]

        output_audio = []
        split_sections = [self.n_early_size, self.n_group]

        for k, (invconv, affine_coup) in enumerate(zip(self.invconv1x1, self.WNs)):
            if k % self.n_early_every == 0 and k:
                split_sections[1] -= self.n_early_size
                early_output, x = x.split(split_sections, 1)
                # these 2 lines actually copy tensors, may need optimization in the future
                output_audio.append(early_output)
                x = x.clone()

            x, log_det_W = invconv(x)
            x, log_s = affine_coup(x, y)
            if k:
                logdet += log_det_W + log_s.sum((1, 2))
            else:
                logdet = log_det_W + log_s.sum((1, 2))

        assert split_sections[1] == self.z_split_sizes[-1]
        output_audio.append(x)
        return torch.cat(output_audio, 1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet, h

    def _upsample_h(self, h):
        return F.interpolate(h, scale_factor=self.upsample_factor, mode='linear')

    def inverse(self, z, h):
        y = self._upsample_h(h)
        batch_dim, n_mels, group_steps = y.shape
        z = z.view(batch_dim, -1, self.n_group).transpose(1, 2)
        assert z.size(2) <= y.size(2)
        y = y[..., :z.size(2)]

        remained_z = []
        for r in z.split(self.z_split_sizes, 1):
            remained_z.append(r.clone())
        *remained_z, z = remained_z

        for k, invconv, affine_coup in zip(range(self.flows - 1, -1, -1), self.invconv1x1[::-1], self.WNs[::-1]):

            z, log_s = affine_coup.inverse(z, y)
            z, log_det_W = invconv.inverse(z)

            if k == self.flows - 1:
                logdet = log_det_W + log_s.sum((1, 2))
            else:
                logdet += log_det_W + log_s.sum((1, 2))

            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)

        z = z.transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet

    @torch.no_grad()
    def infer(self, h, sigma=1.):
        if len(h.shape) == 2:
            h = h[None, ...]

        batch_dim, n_mels, steps = h.shape
        samples = steps * self.hop_size

        z = h.new_empty((batch_dim, samples)).normal_(std=sigma)
        # z = torch.randn(batch_dim, self.n_group, group_steps, dtype=y.dtype, device=y.device).mul_(sigma)
        x, _ = self.inverse(z, h)
        return x.squeeze(), _

    

class _NonCausalLayer2D(nn.Module):
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


        self.V = nn.Conv1d(aux_channels, dilation_channels * 2, radix, dilation=dilation, padding=self.pad_size, bias=bias)

        self.W = nn.Conv2d(residual_channels, dilation_channels * 2, 
                            kernel_size=radix,
                            dilation=(h_dilation, dilation), bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv2d(dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv2d(dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, y):
        tmp = F.pad(x, [self.pad_size] * 2 + [self.h_pad_size, 0])
        xy = self.W(tmp) + self.V(y).unsqueeze(2)
        zw, zf = xy.chunk(2, 1)
        z = zw.tanh().mul(zf.sigmoid())
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        if len(z):
            output = z[0]
            return output + x[:, :, -output.size(2):], skip
        else:
            return None, skip

    def inverse_forward(self, x, y, buffer=None):
        if buffer is None:
            buffer = F.pad(x, [0, 0, self.h_pad_size, 0])
        else:
            buffer = torch.cat((buffer[:, :, 1:], x), 2)
        tmp = F.pad(buffer, [self.pad_size] * 2)
        xy = self.W(tmp) + self.V(y).unsqueeze(2)
        zw, zf = xy.chunk(2, 1)
        z = zw.tanh().mul(zf.sigmoid())
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
            16 : [1] * 8,
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

        self.layers = nn.ModuleList(_NonCausalLayer2D(hd, d,
                                                    aux_channels,
                                                    dilation_channels,
                                                    residual_channels,
                                                    skip_channels,
                                                    3,
                                                    bias) for hd, d in zip(self.h_dilations[:-1], self.dilations[:-1]))
        self.layers.append(_NonCausalLayer2D(self.h_dilations[-1], self.dilations[-1],
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
        cum_skip = None
        for layer in self.layers:
            x, skip = layer(x, y)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip = cum_skip + skip
        return self.end(cum_skip).chunk(2, 1)
    
    def inverse_forward(self, x, y, buffer_list=None):
        x = self.start(x)
        new_buffer_list = []
        if buffer_list is None:
            buffer_list = [None] * len(self.layers)
        
        cum_skip = None
        for layer, buf in zip(self.layers, buffer_list):
            x, skip, buf = layer.inverse_forward(x, y, buf)
            new_buffer_list.append(buf)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip = cum_skip + skip
        
        return self.end(cum_skip).chunk(2, 1) + (new_buffer_list,)


class WaveFlow(BaseModel):
    def __init__(self,
                 flows,
                 n_group,
                 sr,
                 window_size,
                 n_mels,
                 use_conv1x1,
                 memory_efficient,
                 **kwargs):
        super().__init__()
        self.flows = flows
        self.n_group = n_group
        self.win_size = window_size
        self.hop_size = 256
        self.n_mels = n_mels
        self.sr = sr
        self.sub_sr = self.hop_size // n_group

        self.upsampler = nn.Sequential(
            nn.ReplicationPad1d((0, 1)),
            nn.ConvTranspose1d(n_mels, n_mels, self.sub_sr * 2 + 1, self.sub_sr, padding=self.sub_sr),
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
                self.invconv1x1.append(InvertibleConv1x1(n_group, memory_efficient=memory_efficient))
        
        self.mel = nn.Sequential(
                nn.ReflectionPad1d((window_size // 2 - self.hop_size // 2, window_size // 2 + self.hop_size // 2)),   
                MelSpectrogram(sr, window_size, n_mels, self.hop_size, center=False, fmax=8000)
            )

    def get_mel(self, x):
        return self.mel(x.unsqueeze(1)).add_(1e-7).log_()

    def forward(self, x, h=None):
        if h is None:
            h = self.get_mel(x)
        y = self._upsample_h(h)

        batch_dim, n_mels, times = y.shape
        x = x.view(batch_dim, 1, -1, self.n_group).transpose(2, 3).contiguous()
        y = y[..., :x.size(-1)]

        if hasattr(self, 'invconv1x1'):
            invconv1x1 = self.invconv1x1
        else:
            invconv1x1 = [None] * self.flows

        for k, (WN, invconv) in enumerate(zip(self.WNs, invconv1x1)):
            x0 = x[:, :, :1]
            log_s, t = WN(x[:, :, :-1], y)
            xout = x[:, :, 1:] * log_s.exp() + t

            if k:
                logdet += log_s.sum((1, 2, 3))
            else:
                logdet = log_s.sum((1, 2, 3))
            
            if invconv is None:
                x = torch.cat((xout.flip(2), x0), 2)
            else:
                x, log_det_W = invconv(torch.cat((x0, xout), 2).squeeze(1))
                x = x.unsqueeze(1)
                logdet += log_det_W
            
        return x.squeeze(1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet, h

    def _upsample_h(self, h):
        return self.upsampler(h)

    def inverse(self, z, h):
        y = self._upsample_h(h)

        batch_dim, n_mels, times = y.shape
        z = z.view(batch_dim, 1, -1, self.n_group).transpose(2, 3).contiguous()
        y = y[..., :z.size(-1)]

        if hasattr(self, 'invconv1x1'):
            invconv1x1 = self.invconv1x1
        else:
            invconv1x1 = [None] * self.flows

        logdet = None
        for k, WN, invconv in zip(range(self.flows - 1, -1, -1), self.WNs[::-1], invconv1x1[::-1]):
            if invconv is None:
                z = z.flip(2)
            else:
                z, log_det_W = invconv.inverse(z.squeeze(1))
                z = z.unsqueeze(1)
                if logdet is None:
                    logdet = log_det_W.repeat(z.shape[0])
                else:
                    logdet += log_det_W
            
            xnew = z[:, :, :1]
            x = [xnew]
            
            buffer_list = None
            for i in range(1, self.n_group):
                log_s, t, buffer_list = WN.inverse_forward(xnew, y, buffer_list)
                xnew = (z[:, :, i:i+1] - t) / log_s.exp()
                x.append(xnew)

                if logdet is None:
                    logdet = log_s.sum((1, 2, 3))
                else:
                    logdet += log_s.sum((1, 2, 3))
            z = torch.cat(x, 2)

        z = z.squeeze(1).transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, -logdet

    @torch.no_grad()
    def infer(self, h, sigma=1.):
        if h.dim() == 2:
            h = h[None, ...]

        batch_dim, n_mels, steps = h.shape
        samples = steps * self.hop_size

        z = h.new_empty((batch_dim, samples)).normal_(std=sigma)
        x, _ = self.inverse(z, h)
        return x.squeeze(), _


class _predictor(nn.Module):
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
        
        self.end = nn.Conv1d(hidden_channels * groups, out_channels * groups, 1, bias=bias, groups=groups)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels * groups, hidden_channels * groups, 3, padding=1, bias=bias, groups=groups),
                nn.BatchNorm1d(hidden_channels * groups),
                nn.Tanh(),
                nn.Conv1d(hidden_channels * groups, hidden_channels * groups, 3, padding=1, bias=bias, groups=groups),
                nn.BatchNorm1d(hidden_channels * groups),
                nn.Tanh()
            ) for _ in range(layers)
        ])
    
    def forward(self, x):
        x = self.start(x)
        for block in self.res_blocks:
            x = block(x) + x
        return self.end(x)
    

class _NonCausalLayer_LVC(nn.Module):
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
            self.W_o = nn.Conv1d(dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv1d(dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, weights):
        batch, steps, *kernel_size = weights.shape
        weights = weights.view(-1, *kernel_size[1:])
        
        offset = x.shape[2] // steps
        padded_x = F.pad(x, (self.padding,) * 2)
        unfolded_x = padded_x.unfold(2, self.padding * 2 + offset, offset).transpose(1, 2).contiguous().view(1, -1, self.padding * 2 + offset)

        z = F.conv1d(unfolded_x, weights, dilation=self.dilation, groups=batch * steps)
        zw, zv = z.view(batch, steps, kernel_size[0], -1).transpose(1, 2).contiguous().view(batch, kernel_size[0], -1).chunk(2, 1)
        z = zw.tanh().mul(zv.sigmoid())
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

        self.layers = nn.ModuleList(_NonCausalLayer_LVC(d,
                                                        dilation_channels,
                                                        residual_channels,
                                                        skip_channels,
                                                        radix,
                                                        bias) for d in self.dilations[:-1])
        self.layers.append(_NonCausalLayer_LVC(self.dilations[-1],
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
        cum_skip = None
        for layer, w in zip(self.layers, weights.chunk(len(self.dilations), 0)):
            x, skip = layer(x, w.view(w.shape[1], w.shape[2], 2 * self.dil_chs, self.res_chs, self.rdx))
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip = cum_skip + skip
        return self.end(cum_skip).chunk(2, 1)
        
        
class MelGlow(BaseModel):
    def __init__(self,
                 flows,
                 n_group,
                 n_early_every,
                 n_early_size,
                 sr,
                 window_size,
                 hop_size,
                 n_mels,
                 depth=7,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=32,
                 radix=3,
                 predict_channels=64,
                 predict_layers=3,
                 bias=False):
        super().__init__()
        self.flows = flows
        self.depth = depth
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.win_size = window_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.sr = sr

        self.upsample_factor = hop_size // n_group
        sub_win_size = window_size // n_group

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
            self.invconv1x1.append(InvertibleConv1x1(n_remaining_channels, memory_efficient=False))
            self.WNs.append(
                AffineCouplingBlock(WN_LVC, memory_efficient=False, 
                                    in_channels=n_remaining_channels // 2,
                                    depth=depth,
                                    dilation_channels=dilation_channels,
                                    residual_channels=residual_channels,
                                    skip_channels=skip_channels, 
                                    radix=radix,
                                    bias=bias))
        self.z_split_sizes.append(n_remaining_channels)

        self.mel = nn.Sequential(
                nn.ReflectionPad1d((window_size // 2 - hop_size // 2, window_size // 2 + hop_size // 2)),
                MelSpectrogram(sr, window_size, n_mels, hop_size, center=False, fmin=60, fmax=7600)
            )
            
        self.pred = _predictor(
            n_mels, 
            2 * dilation_channels * residual_channels * radix,
            predict_channels,
            predict_layers,
            bias,
            flows * depth
        )

    def get_mel(self, x):
        return self.mel(x.unsqueeze(1)).add_(1e-7).log_()

    def forward(self, x, h=None):
        if h is None:
            h = self.get_mel(x)
        
        batch_dim, n_mels, steps = h.shape
        x = x[:, :x.shape[1] // self.hop_size * self.hop_size]
        x = x.view(batch_dim, -1, self.n_group).transpose(1, 2)
        y = h[..., :x.shape[2] // self.upsample_factor]

        weights = self.pred(y)
        weights = weights.view(weights.shape[0], self.flows * self.depth, -1, weights.shape[2]).permute(1, 0, 3, 2).contiguous().chunk(self.flows, 0)
        
        output_audio = []
        split_sections = [self.n_early_size, self.n_group]

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
            if k:
                logdet += log_det_W + log_s.sum((1, 2))
            else:
                logdet = log_det_W + log_s.sum((1, 2))

        assert split_sections[1] == self.z_split_sizes[-1]
        output_audio.append(x)
        return torch.cat([o.transpose(1, 2) for o in output_audio], 2).view(batch_dim, -1), logdet, h


    def inverse(self, z, h):
        batch_dim, n_mels, steps = h.shape
        z = z[:, :z.shape[1] // self.hop_size * self.hop_size]
        z = z.view(batch_dim, -1, self.n_group).transpose(1, 2)
        y = h[..., :z.shape[2] // self.upsample_factor]
        
        weights = self.pred(y)
        weights = weights.view(weights.shape[0], self.flows * self.depth, -1, weights.shape[2]).permute(1, 0, 3, 2).contiguous().chunk(self.flows, 0)
        
        remained_z = []
        for r in z.split(self.z_split_sizes, 1):
            remained_z.append(r.clone())
        *remained_z, z = remained_z

        for k, invconv, affine_coup, lvc_weights in zip(range(self.flows - 1, -1, -1), 
                                                              self.invconv1x1[::-1], 
                                                              self.WNs[::-1], 
                                                              weights[::-1]):

            z, log_s = affine_coup.inverse(z, lvc_weights)
            z, log_det_W = invconv.inverse(z)

            if k == self.flows - 1:
                logdet = log_det_W + log_s.sum((1, 2))
            else:
                logdet += log_det_W + log_s.sum((1, 2))

            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)

        z = z.transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet

    @torch.no_grad()
    def infer(self, h, sigma=1.):
        if len(h.shape) == 2:
            h = h[None, ...]

        batch_dim, n_mels, steps = h.shape
        samples = steps * self.hop_size

        z = h.new_empty((batch_dim, samples)).normal_(std=sigma)
        x, _ = self.inverse(z, h)
        return x.squeeze(), _


