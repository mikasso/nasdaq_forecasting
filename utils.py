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


def robust_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def concatanete_seq(a: List[TimeSeries], b: List[TimeSeries]) -> List[TimeSeries]:
    return list(map(lambda xy: xy[0].concatenate(xy[1]), zip(a, b)))
