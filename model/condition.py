import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram


class MelSpec(nn.Module):
    def __init__(self, sr, n_fft, hop_length, **kwargs) -> None:
        super().__init__()

        self.mel = nn.Sequential(
            nn.ReflectionPad1d((n_fft // 2 - hop_length // 2,
                                n_fft // 2 + hop_length // 2)),
            MelSpectrogram(sample_rate=sr, n_fft=n_fft,
                           hop_length=hop_length, center=False, **kwargs)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mel(x).add_(1e-7).log_()


class LowPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1 / 6, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6,
                        1 / 1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, int((nfft//2+1) * r):] = 0.
        self.register_buffer('filters', f, False)

    # x: [B,T], r: [B], int
    def forward(self, x, r):
        origin_shape = x.shape
        T = origin_shape[-1]
        x = x.view(-1, T)

        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )  # return_complex=False)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2], 1, 1)
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        )  # return_complex=False)
        x = x[:, :T]
        return x.view(*origin_shape)


class STFTDecimate(LowPass):
    def __init__(self, r, *args, **kwargs):
        super().__init__(*args, ratio=[1 / r], **kwargs)
        self.r = r

    def forward(self, x):
        return super().forward(x, 0)[..., ::self.r]
