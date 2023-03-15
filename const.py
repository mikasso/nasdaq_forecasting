import pandas as pd


from pandas.tseries.holiday import USFederalHolidayCalendar


class PATHS:
    DATA = "data"
    MERGED = "data/merged"
    PARQUET = "data/parquet"
    META = "data/meta"
    CSV = "data/csv"


class FEATURES:
    PRICE = "price"
    SHARES = "shares"
    TIMESTAMP = "timestamp"


FREQ = "1H"
BHOURS_US = pd.offsets.CustomBusinessHour(start="4:00", end="19:00", calendar=USFederalHolidayCalendar())

READ_COLUMNS = ["timestamp", "price", "shares", "canceled"]
START_DATE = "20080101"
END_DATE = "20230310"
TICKERS = ["AEM", "AUY", "HMY", "KGC", "NEM"]

TRAIN_DATE_SPLIT = pd.to_datetime("2021-01-01")
VAL_DATE_SPLIT = pd.to_datetime("2022-01-01")
END_DATE = pd.to_datetime("2023-01-01")
