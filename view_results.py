import logging
from typing import List
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
import const as CONST
import matplotlib.pyplot as plt
from darts.metrics import mape
from darts import TimeSeries
import pandas as pd
import const as CONST

from utils import read_csv_ts

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="view_results")


def load_predictions(model_name: str, tickers: List[str]) -> List[TimeSeries]:
    ts_seq = []
    for ticker in tickers:
        df = read_csv_ts(f"{CONST.PATHS.RESULTS}/{model_name}/predicted_{ticker}.csv", time_key="time")
        ts = TimeSeries.from_dataframe(df)
        ts_seq.append(ts)
    logging.info(f"Loaded predictions for {len(tickers)}.")
    return ts_seq


def plot_series(series: TimeSeries, predicted: TimeSeries, offset=0):
    y = series.univariate_values()[offset:]

    index_len = series.time_index.size
    x = range(offset, index_len)
    plt.plot(x, y, label="original")

    y = predicted.univariate_values()
    x = range(index_len - predicted.time_index.size, index_len)
    plt.plot(x, y, label="predicted")


def visualize_predictions(original: SeqDataset, predictions: List[TimeSeries], model_name: str, save=True):
    for idx, ticker in enumerate(original.used_tickers):
        predicted = predictions[idx]
        expected = original.test[idx]
        series = original.series[idx]
        mape_error = mape(predicted, expected)
        plt.figure(ticker, figsize=(8, 5))
        series.plot(label="original")
        predicted.plot(label="predicted")
        # plot_series(series, predicted, offset=27400)
        title = ticker + " - MAPE: {:.2f}%".format(mape_error)
        plt.title(title)
        plt.legend()
        fig1 = plt.gcf()
        if save:
            fig1.savefig(f"results/{model_name}/{ticker}.svg", format="svg")
        plt.show()


def get_mape_scores(original: SeqDataset, predictions: List[TimeSeries]) -> pd.Series:
    scores = {}
    for idx, ticker in enumerate(original.used_tickers):
        predicted = predictions[idx]
        expected = original.test[idx]
        score = mape(predicted, expected)
        scores[ticker] = score

    df_score = pd.Series(scores)
    average = df_score.sum() / len(df_score)
    df_score["avg"] = average
    return df_score


def main():
    model_name = "BlockRNNModel_LSTM_O1"
    ds = load_datasets()
    predictions = load_predictions(model_name=model_name, tickers=ds.original.used_tickers)
    model_scores = get_mape_scores(ds.original, predictions)
    baseline_scores = pd.read_csv(f"{CONST.PATHS.RESULTS}/baseline/score.csv")
    compare_df = {model_name: model_scores, "baseline": baseline_scores}
    print(compare_df)
    model_scores.to_csv(f"{CONST.PATHS.RESULTS}/{model_name}/score.csv")
    visualize_predictions(ds.original, predictions, model_name=model_name, save=False)


if __name__ == "__main__":
    main()
