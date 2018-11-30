import torch
from operator import mul
from functools import reduce


class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1., elementwise_mean=True):
        super().__init__()
        self.sigma2 = sigma ** 2
        self.mean = elementwise_mean

    def forward(self, z, logdet):
        loss = 0.5 * z.pow(2).sum() / self.sigma2 - logdet
        if self.mean:
            loss = loss / reduce(mul, z.shape)
        return loss
