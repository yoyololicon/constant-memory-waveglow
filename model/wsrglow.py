import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MuLawEncoding
from .waveglow import WaveGlow


class AngleEmbedding(nn.Module):
    def __init__(self, embed_num, hidden_dim):
        super(AngleEmbedding, self).__init__()
        self.embed_num = embed_num
        self.embed = nn.Embedding(num_embeddings=embed_num,
                                  embedding_dim=hidden_dim)

    def forward(self, index):
        embed_num = self.embed_num
        index = ((index / torch.pi + 1) * (embed_num // 2)).floor().long()
        index = (index < 0) * 0 + (index >= 0) * (index < embed_num) * \
            index + (index >= embed_num) * (embed_num - 1)
        assert torch.min(index).item() >= 0 and torch.max(
            index).item() < embed_num
        return self.embed(index)


class WSRGlow(WaveGlow):
    def __init__(self, upsample_rate: int = 2, memory_efficient: bool = False, **kwargs) -> None:
        super().__init__(
            12, 8 * upsample_rate, 4, 2, 8 * upsample_rate, 8 * 400 + 51 * 9,
            memory_efficient=memory_efficient, **kwargs
        )
        self.mu_enc = nn.Sequential(
            MuLawEncoding(256),
            nn.Embedding(256, 400)
        )
        self.angle_embed = AngleEmbedding(embed_num=120, hidden_dim=50)

        self.n_fft = 16
        self.hop_length = 8
        self.register_buffer('window', torch.hann_window(self.n_fft))

    def _get_cond(self, c):
        c_emb = self.mu_enc(c).view(c.shape[0], -1, 8 * 400).transpose(1, 2)
        spec = torch.stft(
            F.pad(c.unsqueeze(1), (4, 4), mode='reflect').squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False, return_complex=True
        )
        mag = spec.abs()
        phase_emb = self.angle_embed(spec.angle()).permute(
            0, 1, 3, 2).reshape(spec.shape[0], 50 * 9, -1)
        return torch.cat([c_emb, mag, phase_emb], dim=1)

    def forward_computation(self, x, h):
        return super().forward_computation(x, self._get_cond(h))

    def reverse_computation(self, z, h):
        return super().reverse_computation(z, self._get_cond(h))
