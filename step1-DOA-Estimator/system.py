import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from argparse import Namespace
from typing import Callable, Optional
from torch.optim.optimizer import Optimizer
from asteroid.utils import flatten_dict

from torch.nn.modules.loss import _Loss
from asteroid.utils.deprecation_utils import DeprecationMixin
EPS = 1e-8

class System(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        config = {} if config is None else config
        self.config = config
        self.hparams = Namespace(**self.config_to_hparams(config))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
      
    def common_step(self, batch, batch_nb, train=True):
        # mixture [b m n] doa [b 2 s]
        mixture, _, single_speaker, tgt_doa = batch
        est_doa = self(mixture)
        loss, loss_dict = self.loss_func(tgt_doa, est_doa, single_speaker)
        return loss, loss_dict

    def training_step(self, batch, batch_nb):
        loss, loss_dict = self.common_step(batch, batch_nb, train=True)
        tensorboard_logs = loss_dict
        return {"loss": loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, loss_dict = self.common_step(batch, batch_nb, train=False)
        tensorboard_logs = loss_dict
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def optimizer_step(self, *args, **kwargs) -> None:
        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            for sched in self.scheduler:
                if isinstance(sched, dict) and sched["interval"] == "batch":
                    sched["scheduler"].step()  # call step on each batch scheduler
            super().optimizer_step(*args, **kwargs)
            
    def configure_optimizers(self):
        """ Required by pytorch-lightning. """

        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            epoch_schedulers = []
            for sched in self.scheduler:
                if not isinstance(sched, dict):
                    epoch_schedulers.append(sched)
                else:
                    assert sched["interval"] in [
                        "batch",
                        "epoch",
                    ], "Scheduler interval should be either batch or epoch"
                    if sched["interval"] == "epoch":
                        epoch_schedulers.append(sched)
            return [self.optimizer], epoch_schedulers
        return self.optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """ Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint
      
    @staticmethod
    def config_to_hparams(dic):
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.Tensor(v)
        return dic


class doa_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, tgt_doa, est_doa, single_speaker): # tgt_doa [b 2 s] est_doa [b 2]

        B = tgt_doa.size(0)
        doa_loss, mae = 0, 0

        est_azimuth = torch.atan2(est_doa[:,1], est_doa[:,0]) / np.pi *180
        tgt_azimuth = torch.atan2(tgt_doa[:,1,0], tgt_doa[:,0,0]) / np.pi * 180
        err_azimuth = torch.abs(tgt_azimuth - est_azimuth)

        for batch in range (B):
            doa_loss += self.mse_loss(est_doa, tgt_doa[:,:,0])
            if err_azimuth[batch] > 180:
                mae = mae + 360 - err_azimuth[batch]
            else:
                mae = mae + err_azimuth[batch]

        mae, doa_loss = mae / B, doa_loss / B

        loss_dict = dict(sig_loss=doa_loss.mean(), mae=mae)

        return doa_loss.mean(), loss_dict
