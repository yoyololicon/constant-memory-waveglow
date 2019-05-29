import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, optimizer, resume, config, data_loader, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, optimizer, resume, config)
        self.config = config
        self.data_loader = data_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def train(self):
        """
        Full training logic
        """
        for step, data in enumerate(self.data_loader):
            step += 1 + self.start_step
            self.model.train()

            data = data.to(self.device)

            self.optimizer.zero_grad()
            *output, _ = self.model(data)
            loss = self.loss(*output)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step(step)
            self.writer.add_scalar('loss', loss.item())

            if self.verbosity >= 2 and step % self.log_step == 0:
                self.logger.info('Train Step: [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    step,
                    self.steps,
                    100.0 * step / self.steps,
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if step % self.save_freq == 0:
                self._save_checkpoint(step)

            if step >= self.steps:
                break
