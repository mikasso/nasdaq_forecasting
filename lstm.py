from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError
import const as CONST

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, SunspotsDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from pytorch_lightning.callbacks import EarlyStopping
import warnings

from utils import load_dataset, read_csv_ts

torch.set_float32_matmul_precision("medium")
df = read_csv_ts(f"{CONST.PATHS.MERGED}/AEM.csv")
series = TimeSeries.from_dataframe(df)
train, val = series.split_before(CONST.TRAIN_DATE_SPLIT)
test_input, test = val.split_before(CONST.VAL_DATE_SPLIT)

transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
test_transformed = transformer.transform(test)
test_input_transformed = transformer.transform(test_input)
series_transformed = transformer.transform(series)

my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0.001,
    mode="min",
)
my_model = RNNModel(
    model="LSTM",
    hidden_dim=40,
    n_rnn_layers=3,
    dropout=0.2,
    batch_size=128,
    n_epochs=400,
    optimizer_kwargs={"lr": 1e-3},
    model_name="AEM_RNN",
    log_tensorboard=True,
    random_state=42,
    training_length=15 * 22,
    input_chunk_length=15 * 20,
    force_reset=True,
    save_checkpoints=True,
    pl_trainer_kwargs={"callbacks": [my_stopper], "accelerator": "gpu", "devices": [0]},
    loss_fn=MeanSquaredError(),
)

my_model.fit(
    train_transformed,
    val_series=val_transformed,
    verbose=True,
)


def eval_model(model, expected_series):
    pred_series = model.predict(n=len(expected_series), series=test_input_transformed)
    plt.figure(figsize=(8, 5))
    transformer.inverse_transform(series_transformed)["price"].plot(label="historical")
    transformer.inverse_transform(pred_series)["price"].plot(label="forecast")
    plt.title("MAPE: {:.2f}%".format(mape(pred_series["price"], expected_series["price"])))
    plt.legend()


eval_model(my_model, test_transformed)

best_model = RNNModel.load_from_checkpoint(model_name="AEM_RNN", best=True)
eval_model(best_model, test_transformed)
