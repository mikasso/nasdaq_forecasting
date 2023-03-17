from typing import List, Tuple
import numpy as np
import pandas as pd
import const as CONST
from const import FEATURES
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries


def read_csv_ts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[FEATURES.TIMESTAMP] = pd.DatetimeIndex(df[FEATURES.TIMESTAMP])
    df.set_index(FEATURES.TIMESTAMP, inplace=True)
    return df.asfreq(CONST.BHOURS_US)


def robust_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


class SeqDataset:
    def __init__(self, sanity_check=False) -> None:
        self._series_seq, self._train_seq, self._val_seq, self._test_seq = [], [], [], []
        if sanity_check == True:
            print("Loading data for sanity check")

        for ticker in CONST.TICKERS:
            if sanity_check == False:
                df = read_csv_ts(f"{CONST.PATHS.MERGED}/{ticker}.csv")
                series = TimeSeries.from_dataframe(df)
                train, rest = series.split_before(CONST.TRAIN_DATE_SPLIT)
                val, test = rest.split_before(CONST.VAL_DATE_SPLIT)
            else:
                df = read_csv_ts(f"{CONST.PATHS.MERGED}/{ticker}.csv")[:1000]
                series = TimeSeries.from_dataframe(df)
                train, rest = series.split_before(800)
                val, test = rest.split_before(100)

            self._series_seq.append(series)
            self._train_seq.append(train)
            self._val_seq.append(val)
            self._test_seq.append(test)

        self._transformer = Scaler()
        self._train_transformed = self._transformer.fit_transform(self._train_seq)
        self._val_transformed = self._transformer.transform(self._val_seq)
        self._test_transformed = self._transformer.transform(self._test_seq)
        self._series_transformed = self._transformer.transform(self._series_seq)

    @property
    def series_seq(self) -> List[TimeSeries]:
        return self._series_seq

    @property
    def train_seq(self) -> List[TimeSeries]:
        return self._train_seq

    @property
    def val_seq(self) -> List[TimeSeries]:
        return self._val_seq

    @property
    def test_seq(self) -> List[TimeSeries]:
        return self._test_seq

    @property
    def test_input(self) -> List[TimeSeries]:
        return self._concatanete_seq(self.train_seq, self.val_seq)

    # TRANSFORMED SETS

    @property
    def series_transformed(self) -> List[TimeSeries]:
        return self._series_transformed

    @property
    def train_transformed(self) -> List[TimeSeries]:
        return self._train_transformed

    @property
    def val_transformed(self) -> List[TimeSeries]:
        return self._val_transformed

    @property
    def test_transformed(self) -> List[TimeSeries]:
        return self._test_transformed

    @property
    def test_input_transformed(self) -> List[TimeSeries]:
        return self._concatanete_seq(self.train_transformed, self.val_transformed)

    @property
    def transformer(self) -> Scaler:
        return self._transformer

    def _concatanete_seq(self, a: List[TimeSeries], b: List[TimeSeries]) -> List[TimeSeries]:
        return list(map(lambda xy: xy[0].concatenate(xy[1]), zip(a, b)))


class SeqDatasetWithCov:
    def __init__(self, sanity_check=False) -> None:
        self.dataset = SeqDataset(sanity_check)

    def _map_feature(self, seq: List[TimeSeries], feature: str) -> List[TimeSeries]:
        return list(map(lambda s: s[feature], seq))

    def _concatanete_seq(self, a: List[TimeSeries], b: List[TimeSeries]) -> List[TimeSeries]:
        return list(map(lambda xy: xy[0].concatenate(xy[1]), zip(a, b)))

    @property
    def series_seq(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.series_seq, CONST.FEATURES.PRICE)

    @property
    def train_seq(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.train_seq, CONST.FEATURES.PRICE)

    @property
    def val_seq(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.val_seq, CONST.FEATURES.PRICE)

    @property
    def test_seq(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.test_seq, CONST.FEATURES.PRICE)

    @property
    def test_input(self) -> List[TimeSeries]:
        return self._concatanete_seq(self.train_seq, self.val_seq)

    # COVARIATES

    @property
    def series_seq_cov(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.series_seq, CONST.FEATURES.SHARES)

    @property
    def train_seq_cov(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.train_seq, CONST.FEATURES.SHARES)

    @property
    def val_seq_cov(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.val_seq, CONST.FEATURES.SHARES)

    @property
    def test_seq_cov(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.test_seq, CONST.FEATURES.SHARES)

    @property
    def test_input_cov(self) -> List[TimeSeries]:
        return self._concatanete_seq(self.train_seq_cov, self.val_seq_cov)

    # TRANSFORMED SETS

    @property
    def series_transformed(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.series_transformed, CONST.FEATURES.PRICE)

    @property
    def train_transformed(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.train_transformed, CONST.FEATURES.PRICE)

    @property
    def val_transformed(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.val_transformed, CONST.FEATURES.PRICE)

    @property
    def test_transformed(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.test_transformed, CONST.FEATURES.PRICE)

    @property
    def test_input_transformed(self) -> List[TimeSeries]:
        return self._concatanete_seq(self.train_transformed, self.val_transformed)

    # TRANSFORMED COVARIATES

    @property
    def series_transformed_cov(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.series_transformed, CONST.FEATURES.SHARES)

    @property
    def train_transformed_cov(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.train_transformed, CONST.FEATURES.SHARES)

    @property
    def val_transformed_cov(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.val_transformed, CONST.FEATURES.SHARES)

    @property
    def test_transformed_cov(self) -> List[TimeSeries]:
        return self._map_feature(self.dataset.test_transformed, CONST.FEATURES.SHARES)

    @property
    def test_input_transformed_cov(self) -> List[TimeSeries]:
        return self._concatanete_seq(self.train_transformed_cov, self.val_transformed_cov)
