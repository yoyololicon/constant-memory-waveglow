import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

from utils import add_weight_norms

from .base import FlowBase
from .efficient_modules import AffineCouplingBlock, InvertibleConv1x1
from .waveglow import WN, fused_gate, NonCausalLayer


class MRWaveGlow(FlowBase):
    def __init__(self,
                 prior_flows,
                 n_group,
                 hop_size,
                 n_mels,
                 memory_efficient,
                 levels=3,
                 flows=4,
                 super_resolution=False,
                 reverse_mode=False,
                 **kwargs):
        super().__init__(hop_size, reverse_mode)
        self.flows = flows
        self.prior_flows = prior_flows
        self.n_group = n_group
        self.n_mels = n_mels
        self.super_resolution = super_resolution
        self.levels = levels

        self.upsample_factor = hop_size // n_group

        self.prior_invconv1x1 = nn.ModuleList()
        self.prior_WNs = nn.ModuleList()

        self.invconv1x1_list = nn.ModuleList()
        self.WNs_list = nn.ModuleList()

        in_channels = n_group
        for i in range(levels - 1):
            in_channels = in_channels // 2
            self.invconv1x1_list.append(
                nn.ModuleList([InvertibleConv1x1(in_channels, in_channels) for _ in range(flows)]))

            self.WNs_list.append(
                nn.ModuleList([
                    AffineCouplingBlock(WN, memory_efficient=memory_efficient,  reverse_mode=reverse_mode, 
                    in_channels=in_channels // 2, aux_channels=in_channels + (0 if super_resolution else n_mels), **kwargs) for _ in range(flows)]))

        for k in range(prior_flows):
            self.prior_invconv1x1.append(InvertibleConv1x1(
                in_channels, memory_efficient=memory_efficient, reverse_mode=reverse_mode))
            self.prior_WNs.append(
                AffineCouplingBlock(WN, memory_efficient=memory_efficient, in_channels=in_channels // 2,
                                    aux_channels=n_mels, reverse_mode=reverse_mode, **kwargs))

    def forward_computation(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        y = self._upsample_h(h)

        batch_dim = x.size(0)
        x = x.view(batch_dim, -1, self.n_group).transpose(1, 2)
        assert x.size(2) <= y.size(2)
        y = y[..., :x.size(2)]

        output_audio = []

        logdet: torch.Tensor = 0

        for level in range(self.levels - 1):
            x0, x1 = x[:, ::2], x[:, 1::2]
            x_diff, x = x1 - x0, (x0 + x1) * 0.5
            if self.super_resolution:
                cond = x
            else:
                cond = torch.cat([x, y], 1)
            
            for invconv, affine_coup in zip(self.invconv1x1_list[level], self.WNs_list[level]):
                x_diff, log_det_W = invconv(x_diff)
                x_diff, log_s = affine_coup(x_diff, cond)
                logdet += log_det_W + log_s.sum((1, 2))

            output_audio.append(x_diff)
        
        for invconv, affine_coup in zip(self.prior_invconv1x1, self.prior_WNs):
            x, log_det_W = invconv(x)
            x, log_s = affine_coup(x, y)
            logdet += log_det_W + log_s.sum((1, 2))
        
        output_audio.append(x)
        return torch.cat(output_audio, 1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet

    def reverse_computation(self, z: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        y = self._upsample_h(h)
        batch_dim = z.size(0)
        z = z.view(batch_dim, -1, self.n_group).transpose(1, 2)
        assert z.size(2) <= y.size(2)
        y = y[..., :z.size(2)]

        remained_z = []
        for _ in range(self.levels - 1):
            r, z = z.chunk(2, 1)
            remained_z.append(r.clone())
        z = z.contiguous()

        logdet: torch.Tensor = 0
        for invconv, affine_coup in zip(self.prior_invconv1x1[::-1], self.prior_WNs[::-1]):
            z, log_s = affine_coup.reverse(z, y)
            z, log_det_W = invconv.reverse(z)
            logdet += log_det_W + log_s.sum((1, 2))
        
        for level in range(self.levels - 2, -1, -1):
            z_diff = remained_z.pop()
            if self.super_resolution:
                cond = z
            else:
                cond = torch.cat([z, y], 1)
            
            for invconv, affine_coup in zip(self.invconv1x1_list[level][::-1], self.WNs_list[level][::-1]):
                z_diff, log_s = affine_coup.reverse(z_diff, cond)
                z_diff, log_det_W = invconv.reverse(z_diff)
                logdet += log_det_W + log_s.sum((1, 2))
            
            z_0, z_1 = z - z_diff * 0.5 , z + z_diff * 0.5
            z = torch.stack([z_0, z_1], 2).view(batch_dim, -1, z_0.size(2))


        z = z.transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet

    def _upsample_h(self, h):
        return F.interpolate(h, scale_factor=self.upsample_factor, mode='linear')




