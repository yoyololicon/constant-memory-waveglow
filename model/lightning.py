import torch
from torch import nn
import pytorch_lightning as pl
from torch.functional import Tensor

from .base import FlowBase, Conditioner
from .loss import WaveGlowLoss


class LightModel(pl.LightningModule):
    def __init__(self, model: FlowBase, conditioner: Conditioner, criterion: nn.Module,
                 config: dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        self.conditioner = conditioner
        self.criterion = criterion

        self.config = config

    def training_step(self, batch, batch_idx):
        x = batch
        cond = self.conditioner(x)
        z, logdet = self.model(x, cond)
        loss = self.criterion(z, logdet)

        values = {
            'loss': loss,
            'log_determinant': logdet.mean(),
            'z_mean': z.mean(),
            'z_std': z.std()
        }
        self.log_dict(values, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

    def forward(self, *args, **kwargs):
        return self.model.infer(*args, **kwargs)
