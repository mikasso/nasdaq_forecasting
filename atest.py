from functools import cached_property
from darts import TimeSeries
from darts.dataprocessing.transformers.scaler import Scaler
from typing import List, Set
import pandas as pd
import pandas_market_calendars as mcal
from os import listdir
from os.path import join
import joblib
from typing import List
import numpy as np
import pandas as pd
from darts.dataprocessing.transformers import (
    Scaler,
)
from darts import TimeSeries
from joblib import Parallel, delayed
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts import concatenate
import torch
from torchmetrics import MeanSquaredError
from darts.models import BlockRNNModel
from LossLogger import LossLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelSummary

from const import ModelConfig, ModelTypes
from utils import visualize_history

WINDOW = 30
HORIZON = 1
MIN_LEN = WINDOW + HORIZON
VAL_DATE, TEST_DATE = pd.to_datetime("2018-01-01"), pd.to_datetime("2020-01-01")


class Dataset:
    def __init__(
        self,
        series: List[TimeSeries],
    ) -> None:
        series32 = [s.astype(np.float32) for s in series]
        self.series = series32
        self._verify_sequence(self.series, "series")

    def _verify_sequence(self, seq, seq_name):
        if None in seq:
            raise Exception(f"Dataset contains None as series entry, in sequence {seq_name}")

    def __len__(self):
        return len(self.series)

    @cached_property
    def train(self) -> List[TimeSeries]:
        return slice_for_train_sequence(self.series)

    @cached_property
    def val(self) -> List[TimeSeries]:
        return slice_for_sequence(self.series, VAL_DATE, TEST_DATE)

    @cached_property
    def test(self) -> List[TimeSeries]:
        return slice_for_sequence(self.series, TEST_DATE, None)

    @property
    def train_len(self) -> int:
        return self.len_of_sequence(self.train)

    @property
    def val_len(self) -> int:
        return self.len_of_sequence(self.val)

    @property
    def test_len(self) -> int:
        return self.len_of_sequence(self.test)

    def len_of_sequence(self, sequence):
        return sum([len(series) for series in sequence])


def slice_series_by_date(s: TimeSeries, start: pd.Timestamp, end: pd.Timestamp):
    if start is None or start < s.start_time():
        start = s.start_time()
    if end is None or end > s.end_time():
        end = s.end_time()
    if end < start:
        return None
    if s.start_time() <= start and end <= s.end_time():
        sliced = s[start:end]
        if len(sliced) > MIN_LEN:
            return sliced


def slice_for_sequence(series, start, end):
    return [slice_series_by_date(s, start, end) for s in series]


def slice_for_train_sequence(series):
    return slice_for_sequence(series, None, VAL_DATE)


def get_delete_idx(
    series: List[TimeSeries],
    val_date: pd.Timestamp,
    test_date: pd.Timestamp,
) -> Set[int]:
    def get_delete_idx(sequence):
        return set([idx for idx, x in enumerate(sequence) if x == None])

    sliced_train = slice_for_sequence(series, None, val_date)
    sliced_val = slice_for_sequence(series, val_date, test_date)
    sliced_test = slice_for_sequence(series, test_date, None)
    sliced_sequences = [sliced_train, sliced_val, sliced_test]
    delete_index = set.union(*[get_delete_idx(x) for x in sliced_sequences])
    return delete_index


def filter_by_delete_idx(sequence, delete_idx):
    return [series for idx, series in enumerate(sequence) if idx not in delete_idx]


def build_timeseries():
    dataset_path = "data\csv"
    csv_paths = [join(dataset_path, f) for f in listdir(dataset_path)]
    targets = []
    covariates = []
    nasdaq_calendar = mcal.get_calendar("NASDAQ")
    holidays = nasdaq_calendar.holidays()
    nasdaq_freedays = holidays.holidays
    nasdaq_freq = pd.offsets.CustomBusinessDay(calendar=nasdaq_calendar, holidays=nasdaq_freedays)

    for idx, csv_path in enumerate(csv_paths):
        print(f"Progress {csv_path} {idx}/{len(csv_paths)}")
        df = pd.read_csv(csv_path, index_col="Date")
        df.index = pd.to_datetime(df.index, format="%d-%m-%Y")
        df = df.asfreq(nasdaq_freq)
        if df.isnull().values.any():
            print(f"Skipping {csv_path} - nans")
            continue
        if len(df) < MIN_LEN:
            print(f"Skipping {csv_path} - too small file")
            continue
        target = TimeSeries.from_series(df["Adjusted Close"])
        targets.append(target)
        cov_df = df.drop(columns=["Adjusted Close"])
        covariate = TimeSeries.from_dataframe(cov_df)
        covariates.append(covariate)

    joblib.dump(targets, "temp/targets.pkl")
    joblib.dump(covariates, "temp/covariates.pkl")
    return targets, covariates


def build_target_dataset():
    targets = joblib.load("temp/targets.pkl")
    delete_idx = get_delete_idx(targets, VAL_DATE, TEST_DATE)
    targets = filter_by_delete_idx(targets, delete_idx)
    original_ds = Dataset(targets)
    scaler = Scaler(n_jobs=-1, verbose=True)
    scaler.fit(original_ds.train)
    transformed_targets = scaler.transform(targets)
    ds = Dataset(transformed_targets)
    joblib.dump(ds, "temp/ds.pkl")
    joblib.dump(delete_idx, "temp/delete_idx.pkl")
    joblib.dump(scaler, "temp/ds_scaler.pkl")
    joblib.dump(original_ds, "temp/original.pkl")
    return ds, original_ds


def extend_with_date_covariates(covariates: List[TimeSeries]) -> List[TimeSeries]:
    """Returns future covariates build from date attributes cyclic - month, weekday and hours scaled by MinMaxScaler"""

    def process(series: TimeSeries):
        month_series = datetime_attribute_timeseries(
            series.time_index, attribute="month", dtype=np.float32, cyclic=True
        )
        weekday_series = datetime_attribute_timeseries(
            series.time_index, attribute="weekday", dtype=np.float32, cyclic=True
        )
        year_series = datetime_attribute_timeseries(series.time_index, attribute="year", dtype=np.float32)
        # The series have the same freq so it's allowed to use the same scaler for train/val/test
        year_series = Scaler().fit_transform(year_series)
        date_cov = concatenate([series, year_series, month_series, weekday_series], axis=1)
        return date_cov

    return Parallel(n_jobs=-1)(delayed(process)(series) for series in covariates)


def build_covariate_dataset():
    covariates = joblib.load("temp/covariates.pkl")
    delete_idx = joblib.load("temp/delete_idx.pkl")
    covariates = filter_by_delete_idx(covariates, delete_idx)
    train_covariates = slice_for_train_sequence(covariates)
    scaler = Scaler(n_jobs=-1, verbose=True)
    scaler.fit(train_covariates)
    transformed_covariates = scaler.transform(covariates)
    transformed_covariates_with_date = extend_with_date_covariates(transformed_covariates)
    ds_cov = Dataset(transformed_covariates_with_date)
    joblib.dump(ds_cov, "temp/ds_cov.pkl")
    return ds_cov


def rebuild_datasets():
    targets, covariates = build_timeseries()
    ds, original_ds = build_target_dataset()
    ds_cov = build_covariate_dataset()


if __name__ == "__main__":
    # rebuild_datasets()
    ds = joblib.load("temp/ds.pkl")
    print("loaded ds1")
    ds_cov = joblib.load("temp/ds_cov.pkl")
    print("loaded ds2")
    loss_logger = LossLogger()
    torch.set_float32_matmul_precision("medium")
    model = BlockRNNModel(
        batch_size=128,
        n_epochs=10000,
        input_chunk_length=WINDOW,
        pl_trainer_kwargs={
            "callbacks": [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    min_delta=1e-4,
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="epoch"),
                loss_logger,
            ],
            "accelerator": "gpu",
            "devices": [0],
        },
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-5},
        dropout=0.3,
        model="LSTM",
        model_name="LSTM_other_dataset",
        output_chunk_length=HORIZON,
        hidden_dim=100,
        n_rnn_layers=3,
        hidden_fc_sizes=[512, 64],
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        loss_fn=MeanSquaredError(),
        log_tensorboard=True,
        force_reset=True,
        save_checkpoints=True,
        show_warnings=True,
    )
    model.fit(
        ds.train,
        past_covariates=ds_cov.train,
        val_series=ds.val,
        val_past_covariates=ds_cov.val,
        verbose=True,
        num_loader_workers=2,
    )
    visualize_history(
        ModelConfig(ModelTypes.lstm, output_len=1, model_name=model.model_name, hidden_state=100),
        loss_logger.train_loss,
        loss_logger.val_loss,
    )
