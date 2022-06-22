import os
import json
import argparse

import torch


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, DeviceStatsMonitor, LearningRateMonitor
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
        y = self.test_y.to(pl_module.device).unsqueeze(0)
        with torch.no_grad():
            cond = pl_module.conditioner(y)
            pred = pl_module(cond, 0.7).cpu()

        trainer.logger.experiment.add_audio(
            'reconstruct_audio', pred[:, None], sample_rate=self.sr, global_step=trainer.global_step)


class ChangeLRCallback(pl.Callback):
    def __init__(self, lr: float) -> None:
        super().__init__()
        self.lr = lr

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr


def main(args, config):
    pl.seed_everything(args.seed)

    gpus = torch.cuda.device_count()
    if config is not None:
        config['data_loader']['batch_size'] //= gpus

    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor('epoch')
        # DeviceStatsMonitor()
    ]
    if args.test_file:
        callbacks.append(TestFileCallBack(args.test_file))
    if args.lr:
        callbacks.append(ChangeLRCallback(args.lr))

    if args.ckpt_path:
        kwargs = {}
        if config is not None:
            kwargs['config'] = config
        lit_model = LightModel.load_from_checkpoint(args.ckpt_path, **kwargs)
    else:
        lit_model = LightModel(config)

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, log_every_n_steps=1,
        benchmark=True, detect_anomaly=True, gpus=gpus,
        strategy=DDPPlugin(find_unused_parameters=False) if gpus > 1 else None)
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
    parser.add_argument('--lr', type=float, default=None,
                        help='force learning rate')
    parser.add_argument('--no-tf32', action='store_true')
    args = parser.parse_args()

    if args.no_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    config = json.load(open(args.config)) if args.config else None
    main(args, config)
