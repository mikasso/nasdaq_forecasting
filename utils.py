import logging
import os
import shutil
from typing import List
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import const as CONST
from const import FEATURES, ModelConfig
from darts import TimeSeries

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="utils")


def visualize_history(model_config: ModelConfig, train_loss: List[int], val_loss: List[int], block=False):
    plt.figure(f"{model_config.model_name} loss")
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title(f"model {model_config.model_name} loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper right")
    fig1 = plt.gcf()
    fig1.savefig(f"{model_config.result_path}/loss.svg", format="svg")
    joblib.dump(fig1, f"{model_config.result_path}/loss.pkl")


def assert_pytorch_is_using_gpu():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count()
    LOGGER.info(f"Pytorch using devince no. {torch.cuda.current_device()}")


def read_csv_ts(csv_path: str, time_key=FEATURES.TIMESTAMP) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[time_key] = pd.DatetimeIndex(df[time_key])
    df.set_index(time_key, inplace=True)
    df = df[pd.Timestamp(CONST.START_DATE) : pd.Timestamp(CONST.END_DATE)]
    return df.asfreq(CONST.BHOURS_US)


def robust_pct(series: pd.Series) -> pd.DataFrame:
    """Perform pct_change and asserts there's no inf or nan in data"""
    pd.options.mode.use_inf_as_na = True
    result = series.pct_change().fillna(0)
    assert result.isnull().values.any() == False
    return result


def concatanete_seq(a: List[TimeSeries], b: List[TimeSeries]) -> List[TimeSeries]:
    return list(map(lambda xy: xy[0].concatenate(xy[1]), zip(a, b)))


def create_folder(path: str, delete_if_exists=False):
    try:
        if delete_if_exists:
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
    except OSError:
        LOGGER.error(f"Couldn't create folder {path}")
