import logging
from typing import Callable, Dict, List, Tuple
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import const as CONST
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from joblib import Parallel, delayed
from utils import concatanete_seq, read_csv_ts, robust_pct


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="Dataset")


class DatasetAccesor:
    def __init__(
        self, series: List[TimeSeries], train: List[TimeSeries], val: List[TimeSeries], test: List[TimeSeries]
    ) -> None:
        self._series = series
        self._train = train
        self._val = val
        self._test = test

    @property
    def series(self) -> List[TimeSeries]:
        return self._series

    @property
    def train(self) -> List[TimeSeries]:
        return self._train

    @property
    def val(self) -> List[TimeSeries]:
        return self._val

    @property
    def test(self) -> List[TimeSeries]:
        return self._test

    @property
    def test_input(self) -> List[TimeSeries]:
        """Train concataneted with validation timeseries"""
        return concatanete_seq(self.train, self.val)


def build_covs_selector(cov_features: List[TimeSeries]) -> Callable[[List[TimeSeries]], List[TimeSeries]]:
    selector = lambda seq: list(map(lambda s: s[cov_features], seq))
    return selector


class TransformedDataset(DatasetAccesor):
    def __init__(self, series, train, val, test, scaler: Scaler, cov_features: List[str] = []) -> None:
        super().__init__(series, train, val, test)
        self._scaler = scaler
        self.cov_features = cov_features
        covs_selector = build_covs_selector(self.cov_features)
        self.cov = DatasetAccesor(
            covs_selector(self.series),
            covs_selector(self.train),
            covs_selector(self.val),
            covs_selector(self.test),
        )

    @staticmethod
    def build_from_dataset(dataset, inner_scaler=MinMaxScaler(feature_range=(0, 1))):
        LOGGER.info(f"Transforming {len(dataset.used_tickers)} timeseries")
        scaler = Scaler(inner_scaler, n_jobs=-1)
        train_transformed = scaler.fit_transform(dataset.train)
        val_transformed = scaler.transform(dataset.val)
        test_transformed = scaler.transform(dataset.test)
        series_transformed = scaler.transform(dataset.series)
        return TransformedDataset(
            series_transformed, train_transformed, val_transformed, test_transformed, scaler, dataset.cov_features
        )

    @property
    def scaler(self) -> Scaler:
        return self._scaler


class SeqDataset(DatasetAccesor):
    def __init__(
        self,
        series,
        train,
        val,
        test,
        use_pct: bool,
        target_features: List[str],
        dfs: Dict[str, DataFrame],
        used_tickers: List[str],
        cov_features: List[str] = [],
    ) -> None:
        super().__init__(series, train, val, test)
        self.use_pct = use_pct
        self.target_features = target_features
        self.dfs = dfs
        self.used_tickers = used_tickers
        self.cov_features = cov_features
        covs_selector = build_covs_selector(self.cov_features)
        self.cov = DatasetAccesor(
            covs_selector(self.series),
            covs_selector(self.train),
            covs_selector(self.val),
            covs_selector(self.test),
        )

    def __len__(self):
        return len(self.series)

    @staticmethod
    def load(
        sanity_check=False,
        use_pct=False,
        target_features=[CONST.FEATURES.PRICE, CONST.FEATURES.SHARES],
        cov_features: List[str] = [],
    ):
        all_series, train, val, test = [], [], [], []
        dfs = {}
        used_tickers = CONST.TICKERS
        if sanity_check == True:
            LOGGER.info("Loading data for sanity check")
            length = 10000
            used_tickers = [CONST.TICKERS[0]]
        else:
            LOGGER.info("Loading full data, assuming length from AEM.csv")
            length = len(read_csv_ts(f"{CONST.PATHS.MERGED}/AEM.csv"))

        train_split = int(length * CONST.TRAIN_SPLIT)
        val_split = int(length * CONST.TEST_SPLIT)
        load_up_to = length

        def process(ticker: str) -> DatasetAccesor:
            LOGGER.info(f"Loading {ticker} timeseries")
            df = read_csv_ts(f"{CONST.PATHS.MERGED}/{ticker}.csv")[:load_up_to][target_features]
            dfs[ticker] = df.copy()
            if use_pct:
                df[CONST.FEATURES.PRICE] = robust_pct(df[CONST.FEATURES.PRICE])
            series = TimeSeries.from_dataframe(df).astype(np.float32)
            train_series, rest_series = series.split_before(train_split)
            val_series, test_series = rest_series.split_before(val_split)
            LOGGER.info(f"Finished loading {ticker} timeseries")
            res = DatasetAccesor([series], [train_series], [val_series], [test_series])
            res._inner_name = ticker
            return res

        results = Parallel(n_jobs=-1)(delayed(process)(ticker) for ticker in used_tickers)

        LOGGER.info(f"Building seq dataset of timeseries")
        for result in results:
            all_series.extend(result.series)
            train.extend(result.train)
            val.extend(result.val)
            test.extend(result.test)

        return SeqDataset(all_series, train, val, test, use_pct, target_features, dfs, used_tickers, cov_features)
