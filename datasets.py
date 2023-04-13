import logging
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, RangeIndex, Timestamp
from sklearn.preprocessing import MinMaxScaler
import const as CONST
from darts.dataprocessing.transformers import (
    Scaler,
)
from darts import TimeSeries
from joblib import Parallel, delayed
from smoothing import apply_differencing, inverse_differencing, inverse_smooth_seq, smooth_seq
from utils import concatanete_seq, read_csv_ts, robust_pct
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts import concatenate
from darts.metrics import mape
import matplotlib.pyplot as plt
from joblib import dump, load

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="Dataset")


class DatasetAccesor:
    """Wraps series and gives acceses to train, val, and test set based on series and indexes, which are start idx in timeseries for given set"""

    def __init__(self, series: List[TimeSeries], val_idx: int, test_idx: int) -> None:
        self._series = series
        self._val_idx = val_idx
        self._test_idx = test_idx

    @property
    def val_idx(self) -> int:
        return self._val_idx

    @property
    def test_idx(self) -> int:
        return self._test_idx

    @property
    def series(self) -> List[TimeSeries]:
        return self._series

    @property
    def train(self) -> List[TimeSeries]:
        return list(map(lambda s: s[0 : self.val_idx], self.series))

    @property
    def val(self) -> List[TimeSeries]:
        return list(map(lambda s: s[self.val_idx : self.test_idx], self.series))

    @property
    def test(self) -> List[TimeSeries]:
        return list(map(lambda s: s[self.test_idx :], self.series))

    def slice(self, start=0, end=None) -> List[TimeSeries]:
        end = len(self.series[0]) if end == None else end
        return list(map(lambda s: s[start:end], self.series))

    @property
    def test_input(self) -> List[TimeSeries]:
        """Train concataneted with validation timeseries"""
        return list(map(lambda s: s[0 : self.test_idx], self.series))


class DatasetTransformer:
    def __init__(
        self,
        darts_scaler: Scaler = Scaler(n_jobs=-1, verbose=True),
        use_diff: bool = True,
        use_smoothing=True,
        verbose=True,
        alpha=0.5,
        n_jobs=-1,
    ) -> None:
        self._scaler = darts_scaler
        self._used_diff = use_diff
        self._used_smoothing = use_smoothing
        self._logger = logging.getLogger(name="DatasetTransformer")
        self._logger.setLevel(level=logging.INFO if verbose else logging.WARN)
        self._alpha = alpha
        self.before_smoothed = None
        self._before_diff = None
        self.n_jobs = n_jobs

    def transform(self, dataset: DatasetAccesor) -> DatasetAccesor:
        self._logger.info(f"Starting transforming {len(dataset.series)} series.")
        series_seq = dataset.series
        self.seq_len = len(series_seq)
        if self.used_smoothing:
            self._logger.info(f"Applying exponential smoothing")
            series_seq = smooth_seq(series_seq, self._alpha)
            self.before_smoothed = DatasetAccesor(series_seq, dataset.val_idx, dataset.test_idx)

        if self.used_diff:
            self._logger.info(f"Applying differencing")
            self._before_diff = DatasetAccesor(series_seq, dataset.val_idx, dataset.test_idx)
            series_seq = apply_differencing(series_seq)

        if self.scaler != None:
            self._logger.info(f"Applying darts scaler {self.scaler.name}")
            temp_dataset = DatasetAccesor(
                series_seq, dataset.val_idx, dataset.test_idx
            )  # might be changed than orignal dataset after diff or smoothing
            self._scaler = self.scaler.fit(temp_dataset.train)
            series_seq = self.scaler.transform(temp_dataset.series)

        return DatasetAccesor(series_seq, dataset.val_idx, dataset.test_idx)

    def inverse(self, transformed_seq: List[TimeSeries], n_jobs=-1) -> List[TimeSeries]:
        """Expects forecast series as transformed_seq"""
        series_seq = transformed_seq
        if self.scaler != None:
            self.scaler.set_n_jobs(n_jobs)
            self._logger.info(f"Inversing darts scaler: {self.scaler.name}")
            series_seq = self.scaler.inverse_transform(series_seq)

        if self.used_diff:
            self._logger.info(f"Inversing differencing")
            last_values = self.get_last_historical_value_seq(series_seq, self._before_diff.series)
            series_seq = inverse_differencing(last_values, series_seq, n_jobs=n_jobs)

        if self.used_smoothing:
            self._logger.info(f"Inversing smoothing")
            last_values = self.get_last_historical_value_seq(series_seq, self.before_smoothed.series)
            series_seq = inverse_smooth_seq(last_values, series_seq, n_jobs=n_jobs)

        return series_seq

    def get_last_historical_value_seq(
        self, relative_series_seq: List[TimeSeries], saved_inversed: List[TimeSeries]
    ) -> List[np.number]:
        last_value_seq = []
        for (transformed, saved) in zip(relative_series_seq, saved_inversed):
            index = saved.get_index_at_point(transformed.start_time()) - 1
            last_value = saved[index].last_value()
            last_value_seq.append(last_value)
        return last_value_seq

    @property
    def scaler(self) -> Scaler:
        return self._scaler

    @property
    def used_diff(self) -> Scaler:
        return self._used_diff

    @property
    def used_smoothing(self) -> Scaler:
        return self._used_smoothing


class SeqDataset(DatasetAccesor):
    def __init__(
        self,
        series: List[TimeSeries],
        val_idx: int,
        test_idx: int,
        target_feature: str,
        used_tickers: List[str],
    ) -> None:
        super().__init__(series, val_idx, test_idx)
        self.target_feature = target_feature
        self.used_tickers = used_tickers

    def __len__(self):
        return len(self.series)

    @staticmethod
    def load(
        sanity_check=False,
        target_feature=CONST.FEATURES.PRICE,
        use_tickers=CONST.TICKERS,
    ):
        if sanity_check == True:
            LOGGER.info("Sanity check dataset")
            length = 10000
            used_tickers = [use_tickers[0]]
        else:
            LOGGER.info(f"Loading full data - assuming length f{use_tickers[0]}.csv")
            length = len(read_csv_ts(f"{CONST.PATHS.MERGED}/{use_tickers[0]}.csv"))
            used_tickers = use_tickers
        LOGGER.info(f"Dataset for following feature { target_feature } for tickers { ' '.join(used_tickers) }")
        val_idx = int(length * CONST.TRAIN_VAL_SPLIT_START)
        test_idx = int(length * CONST.TRAINVAL_TEST_SPLIT_START)
        load_up_to = length

        def process(ticker: str) -> DatasetAccesor:
            LOGGER.info(f"Loading {ticker} timeseries")
            df = read_csv_ts(f"{CONST.PATHS.MERGED}/{ticker}.csv")[:load_up_to][[target_feature]]
            series = TimeSeries.from_dataframe(df).astype(np.float32)
            LOGGER.info(f"Finished loading {ticker} timeseries")
            return series

        all_series = Parallel(n_jobs=-1)(delayed(process)(ticker) for ticker in used_tickers)
        LOGGER.info(f"Completed loading all of timeseries")
        return SeqDataset(all_series, val_idx, test_idx, target_feature, used_tickers)


class Datasets:
    def __init__(
        self,
        original: SeqDataset,
        trasformer: DatasetTransformer,
        transformed: DatasetAccesor,
        covariates: DatasetAccesor,
    ) -> None:
        self.original = original
        self.transformer = trasformer
        self.transformed = transformed
        self.covariates = covariates

    @staticmethod
    def get_datasets_path(sanity_check) -> str:
        return f"{CONST.PATHS.DATA}/preprocessed/datasets{'_sanity' if sanity_check else ''}.pkl"

    @staticmethod
    def build_and_save(sanity_check=CONST.SANITY_CHECK):
        LOGGER.info("Building a new dataset")
        path = Datasets.get_datasets_path(sanity_check)
        datasets = Datasets.build_datasets(sanity_check)
        LOGGER.info(f"Saving a new dataset to {path}")
        dump(datasets, path)

    @staticmethod
    def get_datecovs(dataset: DatasetAccesor) -> DatasetAccesor:
        """Returns future covariates build from date attributes cyclic - month, weekday and hours scaled by MinMaxScaler"""

        def process(series):
            month_series = datetime_attribute_timeseries(
                series.time_index, attribute="month", dtype=np.float32, cyclic=True
            )
            weekday_series = datetime_attribute_timeseries(
                series.time_index, attribute="weekday", dtype=np.float32, cyclic=True
            )
            hour_series = datetime_attribute_timeseries(series.time_index, attribute="hour", dtype=np.float32)
            # The series have the same freq so it's allowed to use the same scaler for train/val/test
            hour_series = Scaler().fit_transform(hour_series)
            date_cov = concatenate([month_series, weekday_series, hour_series], axis=1)
            return date_cov

        all_date_covs = Parallel(n_jobs=-1)(delayed(process)(series) for series in dataset.series)
        return DatasetAccesor(all_date_covs, val_idx=dataset.val_idx, test_idx=dataset.test_idx)

    @staticmethod
    def build_datasets(
        sanity_check=CONST.SANITY_CHECK,
        use_diff=CONST.USE_DIFF,
        use_smoothing=CONST.USE_SMOOTHING,
        use_scaler=CONST.USE_SCALER,
    ):
        verbose = True
        get_scaler = lambda: Scaler(n_jobs=-1, verbose=verbose) if use_scaler else None
        # Load prices dataset from tickers
        dataset = SeqDataset.load(sanity_check, target_feature=CONST.FEATURES.PRICE)
        transformer = DatasetTransformer(
            darts_scaler=get_scaler(), use_diff=use_diff, use_smoothing=use_smoothing, verbose=verbose
        )
        transformed = transformer.transform(dataset)
        # Load shares dataset from tickers
        shares_dataset = SeqDataset.load(sanity_check, target_feature=CONST.FEATURES.SHARES)
        shares_transformer = DatasetTransformer(
            darts_scaler=get_scaler(), use_diff=use_diff, use_smoothing=use_smoothing, verbose=verbose
        )
        shares_transformed = shares_transformer.transform(shares_dataset)
        # Load gold dataset
        gold_dataset = SeqDataset.load(sanity_check, target_feature=CONST.FEATURES.GOLD_PRICE, use_tickers=["gold"])
        gold_transformer = DatasetTransformer(
            darts_scaler=get_scaler(), use_diff=use_diff, use_smoothing=use_smoothing, verbose=verbose
        )
        gold_transformed = gold_transformer.transform(gold_dataset)  # only one series
        # Merge all covariates
        dates_covariates = Datasets.get_datecovs(dataset)
        covariates = [
            concatenate([shares, gold_transformed.series[0], dates], axis=1)
            for dates, shares in zip(dates_covariates.series, shares_transformed.series)
        ]
        cov_dataset = DatasetAccesor(covariates, dataset.val_idx, dataset.test_idx)

        return Datasets(dataset, transformer, transformed, cov_dataset)


from darts.utils.missing_values import missing_values_ratio


def load_datasets(
    sanity_check=CONST.SANITY_CHECK,
) -> Datasets:
    path = Datasets.get_datasets_path(sanity_check)
    LOGGER.info(f"Loading dataset from {path}")
    datasets = load(path)
    return datasets


def assert_error(error: List[float] | float, name: str, eps=0.00001):
    error = error if hasattr(error, "__len__") else [error]
    LOGGER.info(f"{name} assert inversed test is equal to orignal test series")
    print(error)
    for e in error:
        assert e < eps


def test_transforming(ds: Datasets):
    LOGGER.info("testing transforming")
    transformed = ds.transformer.transform(ds.original)
    inversed_test = ds.transformer.inverse(transformed.test)

    error = mape(inversed_test, ds.original.test)
    assert_error(error, "transforming")
    inversed_test[0].plot()
    ds.original.test[0].plot()

    plt.show()


def test_datasets(ds: Datasets):
    LOGGER.info("Checking nans in original dataset")
    for series in ds.original.series:
        assert missing_values_ratio(series) == 0
    LOGGER.info("Checking nans in transformed")
    for series in ds.transformed.series:
        assert missing_values_ratio(series) == 0
    LOGGER.info("Checking nans in covariates")
    for series in ds.covariates.series:
        assert missing_values_ratio(series) == 0


def test_diff(ds: Datasets):
    transformer = DatasetTransformer(None, use_diff=True, use_smoothing=False)
    diff = transformer.transform(ds.original)
    inversed = transformer.inverse(diff.test)
    error = mape(ds.original.test, inversed)
    assert_error(error, "diff")


def test_smoothing(ds: Datasets):
    transformer = DatasetTransformer(None, use_diff=False, use_smoothing=True, n_jobs=1)
    smoothed = transformer.transform(ds.original)
    inversed = transformer.inverse(smoothed.test, n_jobs=1)
    error = mape(ds.original.test, inversed)
    assert_error(error, "smoothing")


def test_smoothing_on_series(series: TimeSeries):
    smoothed = smooth_seq([series])[0]
    original = series[-1000:]
    test = smoothed[-1000:]
    last_val = smoothed[-1001].first_value()
    print(last_val)
    inversed = inverse_smooth_seq([last_val], [test])
    error = mape(inversed, original)
    assert_error(error, "smoothing on series")


if __name__ == "__main__":
    Datasets.build_and_save()
    ds = load_datasets()
    test_datasets(ds)
    test_diff(ds)
    test_smoothing_on_series(ds.original.series[0])
    test_smoothing(ds)
    test_transforming(ds)
