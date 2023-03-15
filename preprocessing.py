import pandas as pd
import const as CONST
from utils import read_csv_ts
import matplotlib.pyplot as plt


for i, ticker in enumerate(["AEM"]):
    df = read_csv_ts(f"{CONST.PATHS.MERGED}/{ticker}.csv")
    print(f"{ticker} probes: {len(df)}")
    plt.figure(i)
    plt.plot(df[CONST.FEATURES.PRICE])
    plt.title(ticker)


plt.show()
