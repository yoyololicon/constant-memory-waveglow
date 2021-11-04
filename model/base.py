from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class Conditioner(nn.Module):
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


class Reversible(nn.Module):
    _reverse_mode: bool

    def __init__(self, reverse_mode, **kwargs) -> None:
        super().__init__(**kwargs)
        self._reverse_mode = reverse_mode

    def forward_computation(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def reverse_computation(self, z: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        if self._reverse_mode:
            return self.reverse_computation(x, *args, **kwargs)
        return self.forward_computation(x, *args, **kwargs)

    def reverse(self, z: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        if self._reverse_mode:
            return self.forward_computation(z, *args, **kwargs)
        return self.reverse_computation(z, *args, **kwargs)


class FlowBase(Reversible):
    def __init__(self, condition_hop_length: int, reverse_mode=False) -> None:
        super().__init__(reverse_mode=reverse_mode)
        self._hop_length = condition_hop_length

    def forward_computation(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def reverse_computation(self, z: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def infer(self, h: Tensor, sigma: float = 1.) -> Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)

        batch_dim, _, steps = h.shape
        samples = steps * self._hop_length

        z = h.new_empty((batch_dim, samples)).normal_(std=sigma)
        if self._reverse_mode:
            x, _ = self.forward_computation(z, h)
        else:
            x, _ = self.reverse_computation(z, h)
        return x.squeeze()
