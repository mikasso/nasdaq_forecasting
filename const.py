from enum import Enum
import pandas as pd
import pandas_market_calendars as mcal
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelSummary

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
    def __init__(self, model_type: ModelTypes, output_len: int, model_name=None) -> None:
        self.model_name = f"{model_type}_out_{output_len}" if model_name == None else model_name
        self.model_type = model_type
        self.output_len = output_len

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


MODEL_CONFIG = ModelConfig(ModelTypes.lstm, 5, "BlockRNNModel_LSTM_O5_2")

saved_model_names = ["BlockRNNModel_LSTM_O5", "BlockRNNModel_LSTM_O1", "BlockRNNModel_LSTM_O5_2"]


# BlockRNNModel_LSTM_O5_2
#  model = BlockRNNModel(
#         model="LSTM",
#         hidden_dim=128,
#         n_rnn_layers=2,
#         batch_size=64,
#         n_epochs=1 if CONST.SANITY_CHECK else 2500,
#         optimizer_kwargs={"lr": 1e-4},
#         model_name=CONST.MODEL_NAME,
#         log_tensorboard=True,
#         random_state=42,
#         input_chunk_length=512,
#         output_chunk_length=5,
#         force_reset=True,
#         save_checkpoints=True,
#         lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
#         pl_trainer_kwargs={
#             "callbacks": [
#                 EarlyStopping(
#                     monitor="val_loss",
#                     patience=20,
#                     min_delta=0.000001,
#                     mode="min",
#                 ),
#                 LearningRateMonitor(logging_interval="epoch"),
#             ],
#             "accelerator": "gpu",
#             "devices": [0],
#         },
#         loss_fn=MeanSquaredError(),
#         show_warnings=True,
#     )
