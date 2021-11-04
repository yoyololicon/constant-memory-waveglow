import os
import json
import argparse

import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model as module_arch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torchaudio


from model import Conditioner, LightModel


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class TestFileCallBack(pl.Callback):
    def __init__(self, test_file: str) -> None:
        super().__init__()

        y, sr = torchaudio.load(test_file)
        self.test_y = y.mean(0)
        self.sr = sr

        self._std = 0
        self._n = 0

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, unused=0) -> None:
        self._n += 1
        self._std += (outputs['z_std'].item() - self._std) / self._n

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LightModel) -> None:
        if self._n == 0:
            return
        y = self.test_y.to(pl_module.device)
        cond = pl_module.conditioner(y)
        pred = pl_module(cond, self._std).cpu()

        trainer.logger.experiment.add_audio(
            'reconstruct_audio', pred[:, None], sample_rate=self.sr)

        self._std = 0
        self._n = 0


def main(args, config):
    pl.seed_everything(args.seed)

    train_data = get_instance(module_data, config['dataset'])
    train_loader = DataLoader(train_data, **config['data_loader'])
    model = get_instance(module_arch, 'arch', config)
    conditioner = Conditioner(**config['conditioner'])
    loss = get_instance(module_loss, 'loss', config)
    lit_model = LightModel(model, conditioner, loss, config['lightning'])

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[TestFileCallBack(
            args.test_file)] if args.test_file else None,
        benchmark=True, detect_anomaly=True)
    trainer.fit(lit_model, train_loader, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch WaveGlow')
    parser.add_argument('config', type=str,
                        help='config file path (default: None)')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(args, config)
