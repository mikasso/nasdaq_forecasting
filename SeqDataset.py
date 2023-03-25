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


def get_datecovs(timeseries_seq: List[TimeSeries]) -> List[TimeSeries]:
    date_covs = []
    for series in timeseries_seq:
        month_series = datetime_attribute_timeseries(
            series.time_index, attribute="month", dtype=np.float32, cyclic=True
        )
        weekday_series = datetime_attribute_timeseries(
            series.time_index, attribute="weekday", dtype=np.float32, cyclic=True
        )
        hour_series = datetime_attribute_timeseries(series.time_index, attribute="hour", one_hot=True)
        non_zero_hours = [f"hour_{h}" for h in range(CONST.BHOURS_US.start[0].hour, CONST.BHOURS_US.end[0].hour)]
        hour_series = hour_series[non_zero_hours]
        date_cov = concatenate([month_series, weekday_series, hour_series], axis=1)
        date_covs.append(date_cov)

    return date_covs


class TransformedDataset(DatasetAccesor):
    def __init__(self, series: List[TimeSeries], val_idx: int, test_idx: int, scaler: Scaler) -> None:
        super().__init__(series, val_idx, test_idx)
        self._scaler = scaler
        series_date_covs = get_datecovs(self.series)
        self._date_cov = DatasetAccesor(series_date_covs, val_idx, test_idx)

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

    @property
    def date_cov(self) -> DatasetAccesor:
        """Returns future covariates that include date attributes"""
        return self._date_cov

    @property
    def features_cov(self) -> DatasetAccesor:
        """Returns future covariates that include shares or other value series"""
        pass


class SeqDataset(DatasetAccesor):
    def __init__(
        self,
        series: List[TimeSeries],
        val_idx: int,
        test_idx: int,
        use_pct: bool,
        target_features: List[str],
        dfs: Dict[str, DataFrame],
        used_tickers: List[str],
    ) -> None:
        super().__init__(series, val_idx, test_idx)
        self.use_pct = use_pct
        self.target_features = target_features
        self.dfs = dfs
        self.used_tickers = used_tickers

    def __len__(self):
        return len(self.series)

    @staticmethod
    def load(
        sanity_check=False,
        use_pct=False,
        target_features=[CONST.FEATURES.PRICE, CONST.FEATURES.SHARES],
    ):
        dfs = {}
        used_tickers = CONST.TICKERS
        if sanity_check == True:
            LOGGER.info("Loading data for sanity check")
            length = 10000
            used_tickers = [CONST.TICKERS[0]]
        else:
            LOGGER.info("Loading full data, assuming length from AEM.csv")
            length = len(read_csv_ts(f"{CONST.PATHS.MERGED}/AEM.csv"))

        val_idx = int(length * CONST.TRAIN_VAL_SPLIT_START)
        test_idx = int(length * CONST.TRAINVAL_TEST_SPLIT_START)
        load_up_to = length

        def process(ticker: str) -> DatasetAccesor:
            LOGGER.info(f"Loading {ticker} timeseries")
            df = read_csv_ts(f"{CONST.PATHS.MERGED}/{ticker}.csv")[:load_up_to][target_features]
            dfs[ticker] = df.copy()
            if use_pct:
                df[CONST.FEATURES.PRICE] = robust_pct(df[CONST.FEATURES.PRICE])
            series = TimeSeries.from_dataframe(df).astype(np.float32)
            LOGGER.info(f"Finished loading {ticker} timeseries")
            return series

        all_series = Parallel(n_jobs=-1)(delayed(process)(ticker) for ticker in used_tickers)
        LOGGER.info(f"Completed loading all of timeseries")
        return SeqDataset(all_series, val_idx, test_idx, use_pct, target_features, dfs, used_tickers)
