import numpy as np
import pandas as pd
import torch
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError
from LossLogger import LossLogger
from utils import read_csv_ts, visualize_history
import const as CONST
from const import FEATURES
import darts

import logging
import torch
from torchmetrics import MeanSquaredError
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
from darts.models import BlockRNNModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import const as CONST
from train import train_model
from utils import assert_pytorch_is_using_gpu
from const import ModelConfig


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="rnn_models")


def main(config: ModelConfig):
    loss_logger = LossLogger()
    model = TFTModel(
        batch_size=CONST.SHARED_CONFIG.BATCH_SIZE,
        n_epochs=CONST.SHARED_CONFIG.EPOCHS,
        input_chunk_length=CONST.SHARED_CONFIG.INPUT_LEN,
        pl_trainer_kwargs=CONST.SHARED_CONFIG.get_pl_trainer_kwargs([loss_logger]),
        optimizer_kwargs=CONST.SHARED_CONFIG.OPTIMIZER_KWARGS,
        dropout=CONST.SHARED_CONFIG.DROPOUT,
        lr_scheduler_kwargs=CONST.SHARED_CONFIG.LR_SCHEDULER_KWARGS,
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        model_name=config.model_name,
        output_chunk_length=config.output_len,
        hidden_size=config.hidden_state,
        lstm_layers=3,
        loss_fn=MeanSquaredError(),
        log_tensorboard=True,
        force_reset=True,
        save_checkpoints=True,
        add_relative_index=True,
        show_warnings=True,
    )
    trained_model = train_model(model)
    visualize_history(config, loss_logger.train_loss, loss_logger.val_loss)
    return trained_model


if __name__ == "__main__":
    main(CONST.MODEL_CONFIG)
