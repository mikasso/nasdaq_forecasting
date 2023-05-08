from pytorch_lightning.callbacks import Callback

import pytorch_lightning as pl


class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self._val_loss = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._val_loss.append(float(trainer.callback_metrics["val_loss"]))

    @property
    def val_loss(self):
        return self._val_loss[1:]
