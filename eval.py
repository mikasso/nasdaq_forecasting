import logging
import numpy as np
from pandas import DataFrame
import torch
from SeqDataset import SeqDataset, TransformedDataset
import const as CONST
import matplotlib.pyplot as plt
from darts.metrics import mape
from darts.models import RNNModel
import pandas as pd
from utils import concatanete_seq
from darts import TimeSeries

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="eval")


def reverse_pct(pct_ts: TimeSeries, df: DataFrame):
    pd_series = pct_ts.pd_series()
    first_date = pct_ts.time_index[0]
    first_value = df[CONST.FEATURES.PRICE][first_date]
    result = pd_series.add(1, fill_value=0).cumprod() * first_value  # errors get bigger change to float64?
    return TimeSeries.from_series(result)


def eval_model(model, dataset: SeqDataset, transformed: TransformedDataset):
    """Performs auto-regresive predictions and evaluates the model"""
    scores = {}
    LOGGER.info(f"Evaluation of model {model.model_name}")
    LOGGER.info(f"Predicting series")
    transformed_outputs = model.predict(n=len(dataset.test[0]), series=transformed.test_input)
    LOGGER.info(f"Inverse transform of all series")
    outputs = transformed.scaler.inverse_transform(transformed_outputs)

    for idx, ticker in enumerate(dataset.used_tickers):
        output = outputs[idx][CONST.FEATURES.PRICE]
        expected = dataset.test[idx][CONST.FEATURES.PRICE]
        original = dataset.series[idx][CONST.FEATURES.PRICE]
        if dataset.use_pct == True:
            source = dataset.dfs[ticker]
            output = reverse_pct(output, source)
            expected = reverse_pct(expected, source)
            original = reverse_pct(original, source)

        scores[ticker] = mape(output, expected)
        plt.figure(figsize=(8, 5))
        original.plot(label="original")
        output.plot(label="forecast")
        plt.title(ticker + " - MAPE: {:.2f}%".format(scores[ticker]))
        plt.legend()
        plt.show()

    return scores


def eval_by_step(model: RNNModel, dataset: SeqDataset, transformed: TransformedDataset):
    scores = {}
    first_date = len(dataset.train[0]) + len(dataset.val[0])
    transformed_outputs = model.historical_forecasts(
        transformed.series,
        retrain=False,
        start=first_date,
        forecast_horizon=1,
        stride=1,
        last_points_only=True,
        verbose=True,
    )
    # TODO copied from upper
    outputs = transformed.scaler.inverse_transform(transformed_outputs)  # TODO should it be in [] ?
    for idx, ticker in enumerate(dataset.used_tickers):
        output = outputs[idx][CONST.FEATURES.PRICE]
        expected = dataset.test[idx][CONST.FEATURES.PRICE]
        original = dataset.series[idx][CONST.FEATURES.PRICE]
        if dataset.use_pct == True:
            source = dataset.dfs[ticker]
            output = reverse_pct(output, source)
            expected = reverse_pct(expected, source)
            original = reverse_pct(original, source)

        scores[ticker] = mape(output, expected)
        plt.figure(figsize=(8, 5))
        original.plot(label="original")
        output.plot(label="forecast")
        plt.title(ticker + " - MAPE: {:.2f}%".format(scores[ticker]))
        plt.legend()
        plt.show()

    return scores


def eval_model_in_parts(model, dataset: SeqDataset, transformed: TransformedDataset, horizon=75):
    """Performs multi-step predictions and evaluates the model"""
    # TODO doesnt work yet
    scores = {}
    LOGGER.info(f"Evaluation of model {model.model_name}")
    outputs = []

    for offset in range(0, len(dataset.test[0]), 75):
        if offset > 0:
            transformed_next_input = concatanete_seq(
                transformed.test_input, list(map(lambda x: x[:offset], dataset.test))
            )
        else:
            transformed_next_input = transformed.test_input

        transformed_output = model.predict(n=horizon, series=transformed_next_input)
        output = transformed.scaler.inverse_transform(transformed_output)
        outputs.append(output)

    for idx, ticker in enumerate(CONST.TICKERS):
        plt.figure(figsize=(8, 5))
        original = dataset.series[idx][CONST.FEATURES.PRICE]
        original.plot(label="original")
        for output in outputs:
            output = output[idx][CONST.FEATURES.PRICE]
            output.plot(label="forecast")
        plt.title(ticker)
        plt.legend()
        plt.show()

    return scores


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    MODEL_NAME = "LSTM"
    dataset = SeqDataset.load(sanity_check=True, use_pct=False)
    transformed = TransformedDataset.build_from_dataset(dataset)
    model = RNNModel.load_from_checkpoint(model_name=MODEL_NAME, best=True)
    scores = eval_by_step(model, dataset, transformed)
    print(scores)
