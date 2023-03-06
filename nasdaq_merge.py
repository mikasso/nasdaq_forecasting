import os
import pandas as pd
from typing import List
import logging
import consts

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="preprocessing")


def weight_average_sampling(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Resample data
    shares_sum = df["shares"].resample(rule).sum()
    volume = (df["price"] * df["shares"]).resample(rule).sum()
    avg_weight_price = volume / shares_sum

    # Creating new DataFrame by passing Dictionary
    df = pd.DataFrame({"price": avg_weight_price, "volume": volume})
    df.drop(df.loc[df["volume"] == 0].index, inplace=True)
    return df


def merged_parquets(dir_list: List[str]) -> pd.DataFrame:
    dfs = []
    for index, dir in enumerate(dir_list):
        if index % 10 == 0:
            LOGGER.info(
                f"Merging parquets files from folder. Completed: {'{:.4f}'.format(100 * index/len(dir_list))} %"
            )
        parquet_files = filter(lambda file: file.endswith(".parquet"), os.listdir(dir))
        for parquet_file in parquet_files:
            df = pd.read_parquet(os.path.join(dir, parquet_file))
            dfs.append(df)

    return pd.concat(dfs)


def get_date_dirs(symbol):
    symbol_path = os.path.join(consts.DATA_PATH, symbol)
    date_dirs = os.listdir(symbol_path)
    date_dirs = list(filter(lambda name: name > f"date={consts.START_DATE}", date_dirs))
    date_dirs.sort()
    return list(map(lambda x: f"{consts.DATA_PATH}/{symbol}/{x}", date_dirs))


for symbol in ["QQQM"]:
    date_dirs = get_date_dirs(symbol)
    df = merged_parquets(date_dirs)
    LOGGER.info(f"Loaded and merged all parquet for {symbol} succesfully.")
    df.to_parquet(f"{consts.MERGED_PATH}/{symbol}.parquet")
