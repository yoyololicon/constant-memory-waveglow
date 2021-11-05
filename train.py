import os
import json
import argparse

import torch


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, DeviceStatsMonitor
from pytorch_lightning.plugins import DDPPlugin
import torchaudio


from model import LightModel


class TestFileCallBack(pl.Callback):
    def __init__(self, test_file: str) -> None:
        super().__init__()

        y, sr = torchaudio.load(test_file)
        self.test_y = y.mean(0)
        self.sr = sr

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LightModel) -> None:
        if not trainer.is_global_zero:
            return
        y = self.test_y.to(pl_module.device)
        cond = pl_module.conditioner(y)
        pred = pl_module(cond, 0.7).cpu()

        trainer.logger.experiment.add_audio(
            'reconstruct_audio', pred[:, None], sample_rate=self.sr, global_step=trainer.global_step)


def main(args, config):
    pl.seed_everything(args.seed)

    gpus = torch.cuda.device_count()
    if config is not None:
        config['data_loader']['batch_size'] //= gpus

    callbacks = [
        ModelSummary(max_depth=-1),
        # DeviceStatsMonitor()
    ]
    if args.test_file:
        callbacks.append(TestFileCallBack(args.test_file))

    lit_model = LightModel(config)

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, log_every_n_steps=1,
        benchmark=True, detect_anomaly=True, gpus=gpus, strategy=DDPPlugin(find_unused_parameters=False))
    trainer.fit(lit_model, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch WaveGlow')
    parser = LightModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--config', type=str,
                        help='config file path (default: None)')
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    config = json.load(open(args.config)) if args.config else None
    main(args, config)
