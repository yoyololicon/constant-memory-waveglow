from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl


import model as module_arch
from .base import FlowBase
import model.condition as module_condition
import model.loss as module_loss
import datasets as module_data
from utils import get_instance


class LightModel(pl.LightningModule):
    model: FlowBase
    conditioner: nn.Module
    criterion: nn.Module

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group('Lightning')
        # parser.add_argument('config', type=str, help='Path to config file')
        return parent_parser

    def __init__(self, config: dict = None, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(config)
        self.save_hyperparameters(kwargs)

        model = get_instance(module_arch, self.hparams.arch)
        conditioner = get_instance(module_condition, self.hparams.conditioner)
        criterion = get_instance(module_loss, self.hparams.loss)

        self.model = model
        self.conditioner = conditioner
        self.criterion = criterion

    def configure_optimizers(self):
        optimizer = get_instance(
            torch.optim, self.hparams.optimizer, self.parameters())
        return optimizer

    def train_dataloader(self):
        train_data = get_instance(module_data, self.hparams.dataset)
        train_loader = DataLoader(
            train_data, **self.hparams.data_loader)
        return train_loader

    def training_step(self, batch, batch_idx):
        x = batch
        cond = self.conditioner(x)
        z, logdet = self.model(x, cond)
        loss = self.criterion(z, logdet)

        values = {
            'loss': loss,
            'logdet': logdet.mean(),
            'z_mean': z.mean(),
            'z_std': z.std()
        }
        self.log_dict(values, prog_bar=True, sync_dist=True)
        return loss

    def forward(self, *args, **kwargs):
        return self.model.infer(*args, **kwargs)
