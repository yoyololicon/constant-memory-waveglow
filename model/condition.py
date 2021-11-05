from torch import nn, Tensor
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
