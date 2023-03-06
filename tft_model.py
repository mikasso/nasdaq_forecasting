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
from utils import read_csv_ts


def build_tft_model(window, horizon):
    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.01,
        mode="min",
    )
    return TFTModel(
        input_chunk_length=window,
        output_chunk_length=horizon,
        hidden_size=4,
        lstm_layers=2,
        batch_size=64,
        n_epochs=10,  # TODO change it
        dropout=0.2,
        save_checkpoints=True,
        show_warnings=True,
        model_name="run-tft",
        work_dir="tft",
        log_tensorboard="tft_logs",
        torch_metrics=MeanAbsolutePercentageError(),
        loss_fn=MeanSquaredError(),
        pl_trainer_kwargs={"callbacks": [my_stopper], "accelerator": "gpu", "devices": [0]},
        add_encoders={
            "cyclic": {"future": ["month"]},
            "datetime_attribute": {"future": ["hour", "dayofweek"]},
            "position": {"past": ["relative"], "future": ["relative"]},
            "transformer": Scaler(),
        },
        add_relative_index=False,
        optimizer_kwargs={"lr": 1e-3},
        random_state=42,
        force_reset=True,  # replace
    )


from const import DATA_PATH, BASE_MERGED_PATH, PREDICT_ORDER
import darts

FREQ = "H"
SPLIT_TRAIN = 0.8
SPLIT_VAL = 0.1
SPLIT_TEST = 0.1


def eval_model(result, test, future):
    pred_series = result.predict(n=len(test))
    plt.figure(figsize=(8, 5))
    series_transformed[future].plot(label="actual")
    pred_series[future].plot(label="forecast")
    plt.title("MAPE: {:.2f}%".format(mape(pred_series, test)))
    plt.legend()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    df = read_csv_ts(f"data/experimental/QQQM.csv", index_col="timestamp")  # use int?
    df = df.resample(rule=FREQ).mean().fillna(method="backfill")  # TODO should be weighted mean abg
    series = TimeSeries.from_dataframe(df, fill_missing_dates=True, freq=FREQ)
    series = darts.utils.missing_values.fill_missing_values(series, fill="auto")
    series = series.astype(np.float32)

    # Create training and validation sets:
    training_cutoff = pd.Timestamp(df.iloc[int(SPLIT_TRAIN * len(df))].name)
    val_cutoff = pd.Timestamp(df.iloc[int((SPLIT_TRAIN + SPLIT_VAL) * len(df))].name)
    train, val = series.split_before(training_cutoff)
    val, test = val.split_before(val_cutoff)
    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)
    test_transformed = transformer.transform(test)
    series_transformed = transformer.transform(series)

    model = build_tft_model(60, 1)
    result = model.fit(series=train_transformed, val_series=val_transformed, num_loader_workers=4, verbose=True)

    pred_series = result.predict(n=len(test_transformed) * 2)
    plt.figure(figsize=(8, 5))
    series_transformed["price"].plot(label="actual")
    pred_series["price"].plot(label="forecast")
    plt.title("MAPE: {:.2f}%".format(mape(pred_series, test_transformed)))
    plt.legend()
    plt.show()
