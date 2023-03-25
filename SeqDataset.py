import logging
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, RangeIndex
from sklearn.preprocessing import MinMaxScaler
import const as CONST
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from joblib import Parallel, delayed
from utils import concatanete_seq, read_csv_ts, robust_pct
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts import concatenate

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

    @property
    def test_input(self) -> List[TimeSeries]:
        """Train concataneted with validation timeseries"""
        return list(map(lambda s: s[0 : self.test_idx], self.series))


def get_datecovs(dataset: DatasetAccesor) -> DatasetAccesor:
    """Returns future covariates build from date attributes cyclic - month, weekday and hours scaled by MinMaxScaler"""
    date_covs = []
    for series in dataset.series:
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
        date_covs.append(date_cov)

    return DatasetAccesor(date_covs, val_idx=dataset.val_idx, test_idx=dataset.test_idx)


class TransformedDataset(DatasetAccesor):
    def __init__(self, series: List[TimeSeries], val_idx: int, test_idx: int, scaler: Scaler) -> None:
        super().__init__(series, val_idx, test_idx)
        self._scaler = scaler

    @staticmethod
    def build_from_dataset(dataset, inner_scaler=MinMaxScaler(feature_range=(0, 1))):
        LOGGER.info(f"Transforming {len(dataset.used_tickers)} timeseries")
        scaler = Scaler(inner_scaler, n_jobs=-1)
        scaler = scaler.fit(dataset.train)
        series_transformed = scaler.transform(dataset.series)
        return TransformedDataset(series_transformed, dataset.val_idx, dataset.test_idx, scaler)

    @property
    def scaler(self) -> Scaler:
        return self._scaler


class SeqDataset(DatasetAccesor):
    def __init__(
        self,
        series: List[TimeSeries],
        val_idx: int,
        test_idx: int,
        use_pct: bool,
        target_features: List[str],
        used_tickers: List[str],
    ) -> None:
        super().__init__(series, val_idx, test_idx)
        self.use_pct = use_pct
        self.target_features = target_features
        self.used_tickers = used_tickers

    def __len__(self):
        return len(self.series)

    @staticmethod
    def load(
        sanity_check=False,
        use_pct=False,
        target_features=[CONST.FEATURES.PRICE],
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
        LOGGER.info(
            f"Dataset for following features: { ' '.join(target_features) } for tickers { ' '.join(used_tickers) }"
        )
        val_idx = int(length * CONST.TRAIN_VAL_SPLIT_START)
        test_idx = int(length * CONST.TRAINVAL_TEST_SPLIT_START)
        load_up_to = length

        def process(ticker: str) -> DatasetAccesor:
            LOGGER.info(f"Loading {ticker} timeseries")
            df = read_csv_ts(f"{CONST.PATHS.MERGED}/{ticker}.csv")[:load_up_to][target_features]
            if use_pct:
                df[CONST.FEATURES.PRICE] = robust_pct(df[CONST.FEATURES.PRICE])
            series = TimeSeries.from_dataframe(df).astype(np.float32)
            LOGGER.info(f"Finished loading {ticker} timeseries")
            return series

        all_series = Parallel(n_jobs=-1)(delayed(process)(ticker) for ticker in used_tickers)
        LOGGER.info(f"Completed loading all of timeseries")
        return SeqDataset(all_series, val_idx, test_idx, use_pct, target_features, used_tickers)
