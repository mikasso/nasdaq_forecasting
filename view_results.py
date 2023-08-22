import logging
from typing import List

from joblib import Parallel, delayed
import joblib
import numpy as np
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
import const as CONST
import matplotlib.pyplot as plt
from darts.metrics import mape
from darts import TimeSeries
import pandas as pd
import const as CONST
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)

from darts import concatenate

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="view_results")


def main(config: CONST.ModelConfig = CONST.MODEL_CONFIG, show=True):
    ds = load_datasets()
    predictions = load_results(config)
    for prediction, ticker in zip(predictions, ds.original.used_tickers):
        if config.output_len == 1:
            result = concatenate(prediction, 0)
            result.to_csv(f"{config.result_path}/{ticker}.csv")

    LOGGER.info("Calculating mape errors")
    model_scores = Parallel(n_jobs=-1)(
        delayed(calculate_mapes)(prediction, original) for prediction, original in zip(predictions, ds.original.test)
    )

    LOGGER.info("Saving mape errors")
    df = pd.DataFrame.from_dict(dict(zip(ds.original.used_tickers, model_scores)))
    df.to_csv(f"{config.result_path}/mape.csv")
    df.describe().to_csv(f"{config.result_path}/described_mape.csv")

    LOGGER.info("Rendering charts")
    for prediction, original, ticker in zip(predictions, ds.original.series, ds.original.used_tickers):
        display_predictions(prediction, original, ticker, save=True, path=config.result_path)
        plt.show(block=show)
        plt.cla()


def display_predictions(
    predicted_timeseries: List[TimeSeries], original: TimeSeries, ticker: str, save: bool, path: str
):
    """Display predictions for single original timeseries"""
    plt.figure(ticker)
    plt.plot(range(len(original)), original.values(), color="black", label="original")

    if len(predicted_timeseries[0]) == 1:
        single = concatenate(predicted_timeseries, 0)
        plt.plot(range(len(original) - len(single), len(original)), single.values(), color="orange", label="predicted")
    else:
        colors = ["blue", "green", "red", "pink", "orange", "brown", "gray"]
        for idx, predicted in enumerate(predicted_timeseries):
            previous_point_idx = original.get_index_at_point(predicted.start_time()) - 1
            previous_point = original[previous_point_idx]
            y = np.insert(predicted.values(), 0, previous_point.values())
            x = range(previous_point_idx, previous_point_idx + len(predicted) + 1)
            plt.plot(x, y, color=colors[idx % len(colors)])

    plt.title(ticker)
    plt.xlabel("Timestep")
    plt.ylabel("Price [$]")
    plt.legend(loc="upper left")
    fig1 = plt.gcf()
    if save:
        fig1.savefig(f"{path}/plot_{ticker}.svg", format="svg")
        joblib.dump(fig1, f"{path}/plot_{ticker}.pkl")


def calculate_mapes(predicted_timeseries: List[TimeSeries], original: TimeSeries) -> np.ndarray:
    mapes = []
    original_values = original.values()
    for idx, predicted in enumerate(predicted_timeseries):
        # err = mape(original, predicted)
        y_true = original_values[idx : idx + len(predicted)]
        y_hat = predicted.values()
        err = 100.0 * np.mean(np.abs((y_true - y_hat) / y_true))
        mapes.append(err)
        print(f"Progress {idx}/{len(predicted_timeseries)}")
    return np.array(mapes)


def load_results(config: CONST.ModelConfig) -> List[List[TimeSeries]]:
    return joblib.load(f"{config.result_path}/{config.model_name}.pkl")


def open_charts(path):
    chart_files = filter(lambda x: x.endswith(".svg"), os.listdir(path))
    chart_files = map(lambda x: x.replace(".svg", ".pkl"), chart_files)
    charts = list(map(lambda x: joblib.load(f"{path}/{x}"), chart_files))
    plt.show()


def open_single(path):
    joblib.load(path)
    plt.show()


if __name__ == "__main__":
    open_single("results/ModelTypes.tft_out_1/plot_AEM.pkl")
