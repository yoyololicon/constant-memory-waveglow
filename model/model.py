import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.util import add_weight_norms
from librosa.filters import mel
import numpy as np


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
        self.d_size = dilation_channels
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
        zw, zf = self.WV(xy).split(self.d_size, 1)
        z = torch.tanh(zw) * torch.sigmoid(zf)
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        return z[0] + x if len(z) else None, skip


class WN(nn.Module):
    def __init__(self,
                 in_channels,
                 aux_channels,
                 dilation_channels=512,
                 residual_channels=512,
                 skip_channels=256,
                 depth=8,
                 radix=3,
                 bias=False):
        super().__init__()
        dilations = radix ** torch.arange(depth)
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
        self.end.weight.data.zero_()
        if bias:
            self.end.bias.data.zero_()

    def forward(self, x, y):
        x = self.start(x)
        cum_skip = 0
        for layer in self.layers:
            x, skip = layer(x, y)
            cum_skip = cum_skip + skip
        return self.end(cum_skip).split(self.in_chs, 1)


class InvertibleConv1x1(nn.Conv1d):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super().__init__(c, c, 1, bias=False)
        W = torch.randn(c, c).qr()[0]
        self.weight.data = W[..., None]

    def inverse(self, z):
        if not hasattr(self, 'inv_weight'):
            if 'HalfTensor' in z.type():
                self.inv_weight = self.weight.float().squeeze().inverse()[..., None].half()
            else:
                self.inv_weight = self.weight.squeeze().inverse()[..., None]
        z = F.conv1d(z, self.inv_weight)
        return z

    def forward(self, z):
        batch_size, group_size, n_of_groups = z.size()
        log_det_W = n_of_groups * self.weight.squeeze().det().abs().log()  # should fix nan logdet
        z = F.conv1d(z, self.weight)
        return z, log_det_W


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
        self.upsampler = nn.ConvTranspose1d(n_mels, n_mels, sub_win_size, self.upsample_factor,
                                            padding=sub_win_size // 2, bias=False)

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
            self.invconv1x1.append(InvertibleConv1x1(n_remaining_channels))
            self.WNs.append(WN(n_remaining_channels // 2, n_mels, **kwargs))
        self.z_split_sizes.append(n_remaining_channels)

        filters = mel(sr, window_size, n_mels, fmax=8000)
        self.filter_idx = np.nonzero(filters)
        self.filter_value = nn.Parameter(torch.Tensor(filters[self.filter_idx]), requires_grad=False)
        self.filter_size = torch.Size(filters.shape)
        self.window = nn.Parameter(torch.hann_window(window_size), requires_grad=False)

    def get_mel(self, x):
        """
        Get mel-spectrogram from raw wave.

        :param x:
        :return:
        """
        batch_size = x.size(0)
        S = torch.stft(x, self.win_size, self.hop_size, window=self.window, pad_mode='constant').pow(2).sum(3)
        mel_filt = torch.sparse_coo_tensor(self.filter_idx, self.filter_value, self.filter_size)
        N = S.size(1)
        mel_S = mel_filt @ S.transpose(0, 1).contiguous().view(N, -1)
        # compress
        mel_S.add_(1e-7).log_()
        return mel_S.view(self.n_mels, batch_size, -1).transpose(0, 1)

    def forward(self, x, h=None):
        if h is None:
            h = self.get_mel(x)
        h = F.pad(h, (0, 1))
        # y = F.interpolate(h, size=((h.size(2) - 1) * self.upsample_factor + 1,), mode='linear')
        y = self.upsampler(h)

        batch_dim, n_mels, group_steps = y.size()
        # y = y.view(batch_dim, n_mels, -1, self.n_group).transpose(2, 3).contiguous().view(batch_dim,
        #                                                                                  n_mels * self.n_group, -1)
        x = x.view(batch_dim, -1, self.n_group).transpose(1, 2)
        assert x.size(2) <= y.size(2)
        y = y[..., :x.size(2)]

        output_audio = []
        logdet = 0

        split_sections = [self.n_early_size, self.n_group]

        for k, (invconv, WN) in enumerate(zip(self.invconv1x1, self.WNs)):
            if k % self.n_early_every == 0 and k:
                split_sections[1] -= self.n_early_size
                early_output, x = x.split(split_sections, 1)
                output_audio.append(early_output)

            x, log_det_W = invconv(x)

            xa, xb = x.split(x.size(1) // 2, 1)
            log_s, t = WN(xa, y)
            xb = xb * log_s.exp() + t
            x = torch.cat((xa, xb), 1)

            logdet = logdet + log_det_W + log_s.sum((1, 2))

        assert split_sections[1] == self.z_split_sizes[-1]
        output_audio.append(x)
        return torch.cat(output_audio, 1), logdet

    @torch.no_grad()
    def infer(self, h, sigma=1.):
        if len(h.shape) == 2:
            h = h[None, ...]
        # y = F.interpolate(h, size=((h.size(2) - 1) * self.upsample_factor + 1,), mode='linear')
        y = self.upsampler(h)
        # y = y[..., :y.size(2) // self.n_group * self.n_group]
        batch_dim, n_mels, group_steps = y.size()
        # y = y.view(batch_dim, n_mels, -1, self.n_group).transpose(2, 3).contiguous().view(batch_dim,
        #                                                                                  n_mels * self.n_group, -1)
        z = torch.randn(batch_dim, self.n_group, group_steps, dtype=y.dtype, device=y.device).mul_(sigma)
        *remained_z, z = z.split(self.z_split_sizes, 1)

        for k, invconv, WN in zip(np.arange(self.flows)[::-1], self.invconv1x1[::-1], self.WNs[::-1]):
            n_half = z.size(1) // 2
            log_s, t = WN(z[:, :n_half], y)
            z[:, n_half:] -= t
            z[:, n_half:] /= log_s.exp()

            z[:] = invconv.inverse(z)

            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)

        x = z.transpose(1, 2).contiguous().view(batch_dim, -1).squeeze()
        return x


if __name__ == '__main__':
    import librosa
    import matplotlib.pyplot as plt

    y, sr = librosa.load(librosa.util.example_audio_file())
    # h = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    # print(h.shape, h.max())
    # plt.imshow(h ** 0.1, aspect='auto', origin='lower')
    # plt.show()

    y = torch.Tensor(y)
    net = WaveGlow(12, 8, 4, 2, sr, 1024, 256, 80, depth=5, residual_channels=64, dilation_channels=64,
                   skip_channels=64, bias=True)
    # print(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad), "of parameters.")

    h = net.get_mel(y[None, ...])[0]
    print(h.shape, h.max())
    plt.imshow(h.numpy(), aspect='auto', origin='lower')
    plt.show()

    x = torch.rand(2, 16000) * 2 - 1
    z, *_ = net(x)
    print(z.shape)

    x = net.infer(h[:, :10])
    print(x.shape)
