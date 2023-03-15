import numpy as np
import pandas as pd
import const as CONST
from const import FEATURES


def read_csv_ts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[FEATURES.TIMESTAMP] = pd.DatetimeIndex(df[FEATURES.TIMESTAMP])
    df.set_index(FEATURES.TIMESTAMP, inplace=True)
    return df.asfreq(CONST.BHOURS_US)


def robust_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def load_dataset() -> pd.DataFrame:
    dfs = []
    for ticker in CONST.TICKERS:
        dfs.append(pd.read_csv(f"{CONST.PATHS.MERGED}/{ticker}.csv"))
    return pd.concat(dfs, axis="column")
