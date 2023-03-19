import logging
from typing import List, Tuple
import numpy as np
import const as CONST
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries

from utils import read_csv_ts

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="SeqDataset")


class SeqDataset:
    def __init__(
        self,
        sanity_check=False,
        scaler=Scaler(),
        diff=False,
        target_features=[CONST.FEATURES.PRICE, CONST.FEATURES.SHARES],
    ) -> None:
        self._series_seq, self._train_seq, self._val_seq, self._test_seq = [], [], [], []

        if sanity_check == True:
            LOGGER.info("Loading data for sanity check")
            length = 10000
        else:
            LOGGER.info("Loading full data, assuming length from AEM.csv")
            length = len(read_csv_ts(f"{CONST.PATHS.MERGED}/AEM.csv"))

        train_split = int(length * 0.8)
        val_split = int(length * 0.1)
        load_up_to = load_up_to

        for idx, ticker in enumerate(CONST.TICKERS):
            LOGGER.info(f"Loading {ticker} timeseries")
            df = read_csv_ts(f"{CONST.PATHS.MERGED}/{ticker}.csv")[:load_up_to][target_features]
            data = TimeSeries.from_dataframe(df).astype(np.float32)

            series = data.diff() if diff else data
            train, rest = series.split_before(train_split)
            val, test = rest.split_before(val_split)

            self._series_seq.append(series)
            self._train_seq.append(train)
            self._val_seq.append(val)
            self._test_seq.append(test)
            LOGGER.info(f"Loaded {idx+1}/{len(CONST.TICKERS)} of timeseries")

        LOGGER.info(f"Transforming {len(CONST.TICKERS)} timeseries")
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
