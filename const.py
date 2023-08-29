from typing import List
import pandas as pd
import pandas_market_calendars as mcal
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import torch
import os


INTERVAL = "H" if os.getenv("INTERVAL") == None else os.environ["INTERVAL"]
assert INTERVAL == "D" or INTERVAL == "H"

FREQ = "B" if INTERVAL == "D" else "1H"
WORK_DIR = "darts_logs/daily" if INTERVAL == "D" else "darts_logs/hourly"


class PATHS:
    DATA = "data"
    MERGED = "data/daily" if INTERVAL == "D" else "data/merged"
    PARQUET = "data/parquet"
    META = "data/meta"
    CSV = "data/csv"
    RESULTS = "results/daily" if INTERVAL == "D" else "results/hourly"


class FEATURES:
    PRICE = "adjclose" if INTERVAL == "D" else "price"
    SHARES = "volume" if INTERVAL == "D" else "shares"
    TIMESTAMP = "date" if INTERVAL == "D" else "timestamp"


def set_calendar():
    nyse = mcal.get_calendar("NYSE")
    holidays = nyse.holidays()
    nyse_holidays = holidays.holidays
    if INTERVAL == "D":
        nyse_us = pd.offsets.CustomBusinessDay(calendar=nyse, holidays=nyse_holidays)
    else:
        nyse_us = pd.offsets.CustomBusinessHour(start="9:00", end="17:00", calendar=nyse, holidays=nyse_holidays)
    return nyse_us


BHOURS_US = set_calendar()

START_DATE = "20080101"
END_DATE = "20230310"
TICKERS = ["AEM", "AU", "GFI", "HMY", "KGC", "NEM", "PAAS"]

TRAIN_VAL_SPLIT_START = 0.8
TRAINVAL_TEST_SPLIT_START = 0.9
SANITY_CHECK = False
USE_DIFF = True
USE_SMOOTHING = True
USE_SCALER = True
EXTRA_SERIES = ["INFLATION", "^GSPC", "ES=F", "GC=F", "GOLD", "SI=F", "SILVER", "XLF"] if INTERVAL == "D" else ["gold"]


class SHARED_CONFIG:
    INPUT_LEN = 128
    DROPOUT = 0.1
    EPOCHS = 2
    BATCH_SIZE = 256
    SHOW_WARNINGS = True
    OPTIMIZER_KWARGS = {"lr": 1e-3}
    LR_SCHEDULER_KWARGS = {
        "optimizer": torch.optim.Adam,
        "threshold": 0.5,
        "patience": 8,
        "min_lr": 1e-8,
        "verbose": True,
    }

    @staticmethod
    def get_pl_trainer_kwargs(additional_callbacks: List):
        return {
            "callbacks": [
                EarlyStopping(
                    monitor="val_loss",
                    patience=(17 if not SANITY_CHECK else 100),
                    min_delta=0.0001,
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="epoch"),
                *additional_callbacks,
            ],
            "accelerator": "gpu",
            "devices": [0],
        }
