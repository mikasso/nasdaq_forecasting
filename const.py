import pandas as pd
import pandas_market_calendars as mcal

from pandas.tseries.holiday import USFederalHolidayCalendar


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
MODEL_NAME = "BlockRNNModel_LSTM_O5_2"


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
