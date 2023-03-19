import os
import pandas as pd
from typing import List
import logging
import const as CONST
from const import FEATURES
import matplotlib.pyplot as plt

from utils import read_csv_ts

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="merge")

BATCH_SIZE = 400


def weight_average_sampling(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Resample data
    shares = df[FEATURES.SHARES].resample(rule).sum()
    volume_value = (df[FEATURES.PRICE] * df[FEATURES.SHARES]).resample(rule).sum()
    avg_weight_price = volume_value / shares

    # Creating new DataFrame by passing Dictionary
    df = pd.DataFrame({FEATURES.PRICE: avg_weight_price, FEATURES.SHARES: shares})
    return df


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["canceled"] == False]
    df = df.drop(labels=["canceled"], axis="columns")
    df[FEATURES.TIMESTAMP] = pd.to_datetime(df[FEATURES.TIMESTAMP])
    df = df.set_index(FEATURES.TIMESTAMP)
    df = weight_average_sampling(df, CONST.FREQ)
    return df


def merge_parquets_to_csv(symbol: str, dir_list: List[str]):
    dfs = []
    output_path = f"{CONST.PATHS.MERGED}/{symbol}.csv"
    header, mode = True, "w"

    def batch_save():
        LOGGER.info(f"Saving batch")
        df = pd.concat(dfs)
        df.to_csv(output_path, mode=mode, header=header)
        dfs.clear()

    for index, dir in enumerate(dir_list):
        if index % 100 == 0:
            LOGGER.info(
                f"Merging parquets files from folder. Completed: {'{:.4f}'.format(100 * index/len(dir_list))} %"
            )
        parquet_files = list(filter(lambda file: file.endswith(".parquet"), os.listdir(dir)))
        if len(parquet_files) == 0:
            LOGGER.warning(f"Missing parquet files for {dir}, skipping this date")
            continue
        elif len(parquet_files) > 1:
            LOGGER.warning(f"In {dir} founded more than one parquet file, using the first one")

        parquet_file = parquet_files[0]
        df = pd.read_parquet(os.path.join(dir, parquet_file))
        df = preprocessing(df)
        dfs.append(df)

        if index % BATCH_SIZE == 0 and index != 0:
            batch_save()
            header, mode = False, "a"

    if len(dfs) > 0:
        batch_save()
    return output_path


def get_date_dirs(symbol):
    symbol_path = f"{CONST.PATHS.PARQUET}/{symbol}"
    date_dirs = os.listdir(symbol_path)
    date_dirs = list(filter(lambda name: name > f"date={CONST.START_DATE}", date_dirs))
    date_dirs.sort()
    return list(map(lambda x: f"{CONST.PATHS.PARQUET}/{symbol}/{x}", date_dirs))


def present_df(df: pd.DataFrame):
    print(df)
    df[FEATURES.PRICE].plot(use_index=False)
    plt.show()


def setup_frequency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.asfreq(CONST.BHOURS_US)
    df[CONST.FEATURES.PRICE] = df[CONST.FEATURES.PRICE].fillna(method="ffill")
    df[CONST.FEATURES.SHARES] = df[CONST.FEATURES.SHARES].fillna(value=0)
    return df


for symbol in CONST.TICKERS:
    date_dirs = get_date_dirs(symbol)
    csv_path = merge_parquets_to_csv(symbol, date_dirs)
    df = read_csv_ts(csv_path)
    df = setup_frequency(df)
    df.to_csv(csv_path)
    LOGGER.info(f"Saved merged parquet for {symbol} succesfully.")
    # present_df(df)
