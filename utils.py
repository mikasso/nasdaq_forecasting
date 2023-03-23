import logging
from typing import List
import numpy as np
import pandas as pd
import const as CONST
from const import FEATURES
from darts import TimeSeries

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="utils")


def read_csv_ts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[FEATURES.TIMESTAMP] = pd.DatetimeIndex(df[FEATURES.TIMESTAMP])
    df.set_index(FEATURES.TIMESTAMP, inplace=True)
    return df.asfreq(CONST.BHOURS_US)


def robust_pct(series: pd.Series) -> pd.DataFrame:
    """Perform pct_change and asserts there's no inf or nan in data"""
    pd.options.mode.use_inf_as_na = True
    result = series.pct_change().fillna(0)
    assert result.isnull().values.any() == False
    return result


def concatanete_seq(a: List[TimeSeries], b: List[TimeSeries]) -> List[TimeSeries]:
    return list(map(lambda xy: xy[0].concatenate(xy[1]), zip(a, b)))
