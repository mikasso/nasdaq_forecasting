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
BHOURS_US = pd.offsets.CustomBusinessHour(start="9:00", end="17:00", calendar=USFederalHolidayCalendar())

READ_COLUMNS = ["timestamp", "price", "shares", "canceled"]
START_DATE = "20080101"
END_DATE = "20230310"
TICKERS = ["AEM", "AUY", "GFI", "HMY", "IAG", "KGC", "NEM", "PAAS"]

TRAIN_VAL_SPLIT_START = 0.8
TRAINVAL_TEST_SPLIT_START = 0.9
