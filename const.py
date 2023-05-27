from enum import Enum
from typing import List
import pandas as pd
import pandas_market_calendars as mcal
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelSummary
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar

from LossLogger import LossLogger


class ModelTypes(Enum):
    rnn = "RNN"
    lstm = "LSTM"
    gru = "GRU"
    transformer = "Transformer"
    tft = "TFT"


RNN_NETWORKS = [ModelTypes.rnn, ModelTypes.lstm, ModelTypes.gru]


class ModelConfig:
    def __init__(self, model_type: ModelTypes, output_len: int, model_name=None, hidden_state=256) -> None:
        self.model_name = f"{model_type}_out_{output_len}" if model_name == None else model_name
        self.model_type = model_type
        self.output_len = output_len
        self.hidden_state = hidden_state

    @property
    def result_path(self) -> str:
        return f"{PATHS.RESULTS}/{self.model_name}"


class PATHS:
    DATA = "data"
    MERGED = "data/merged"
    PARQUET = "data/parquet"
    META = "data/meta"
    CSV = "data/csv"
    RESULTS = "results"


class FEATURES:
    PRICE = "price"
    SHARES = "shares"
    TIMESTAMP = "timestamp"
    GOLD_PRICE = "gold_price"


class SHARED_CONFIG:
    INPUT_LEN = 256
    DROPOUT = 0.2
    EPOCHS = 2500
    BATCH_SIZE = 512
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
                    patience=20,
                    min_delta=0.0001,
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="epoch"),
                *additional_callbacks,
            ],
            "accelerator": "gpu",
            "devices": [0],
        }


FREQ = "1H"
calendar = USFederalHolidayCalendar()


def set_calendar():
    nyse = mcal.get_calendar("NYSE")
    holidays = nyse.holidays()
    nyse_holidays = holidays.holidays
    nyse_us = pd.offsets.CustomBusinessHour(start="9:00", end="17:00", calendar=nyse, holidays=nyse_holidays)
    return nyse_us


BHOURS_US = set_calendar()

READ_COLUMNS = ["timestamp", "price", "shares", "canceled"]
START_DATE = "20080101"
END_DATE = "20230310"
TICKERS = ["AEM", "AUY", "GFI", "HMY", "IAG", "KGC", "NEM", "PAAS"]

TRAIN_VAL_SPLIT_START = 0.8
TRAINVAL_TEST_SPLIT_START = 0.9
SANITY_CHECK = False
USE_DIFF = True
USE_SMOOTHING = True
USE_SCALER = True


MODEL_CONFIG = ModelConfig(
    ModelTypes.lstm, 1, hidden_state=32
)  #  ModelConfig(ModelTypes.gru, 1, "TestModel", hidden_state=110)


saved_model_names = ["BlockRNNModel_LSTM_O5", "BlockRNNModel_LSTM_O1", "BlockRNNModel_LSTM_O5_2"]
