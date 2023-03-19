import logging
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError
import const as CONST

import torch

import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.metrics import mape
from pytorch_lightning.callbacks import EarlyStopping, BatchSizeFinder, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger
from eval import eval_model

from utils import SeqDataset

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="lstm")
MODEL_NAME = "LSTM_DIFF"

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    LOGGER.info("Initializing dataset")
    dataset = SeqDataset(sanity_check=False, diff=False)

    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=20,
        min_delta=0.0000001,
        mode="min",
    )
    my_model = RNNModel(
        model="LSTM",
        hidden_dim=1020,
        n_rnn_layers=4,
        dropout=0.2,
        batch_size=64,
        n_epochs=10000,
        optimizer_kwargs={"lr": 1e-3},
        model_name=MODEL_NAME,
        log_tensorboard=True,
        random_state=42,
        input_chunk_length=68 * 15,
        output_chunk_length=5,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs={
            "callbacks": [my_stopper, LearningRateFinder()],
            "accelerator": "gpu",
            "devices": [0],
        },
        loss_fn=MeanSquaredError(),
        add_encoders={
            "cyclic": {"future": ["month"]},
            "datetime_attribute": {"future": ["hour", "dayofweek"]},
            "position": {"future": ["relative"]},
            "transformer": Scaler(),
        },
        show_warnings=True,
    )

    LOGGER.info("Starting training")
    my_model.fit(
        dataset.train_transformed,
        val_series=dataset.val_transformed,
        verbose=True,
    )

    eval_model(my_model, dataset)

    best_model = RNNModel.load_from_checkpoint(model_name=MODEL_NAME, best=True)
    eval_model(best_model, dataset)
