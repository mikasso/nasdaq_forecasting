import logging
import numpy as np
import torch
import const as CONST
import matplotlib.pyplot as plt
from darts.metrics import mape
from darts.models import RNNModel
from utils import SeqDataset, concatanete_seq

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="eval")


def eval_model(model, dataset: SeqDataset):
    scores = {}
    expected_output_series = dataset.test_seq
    original_series = dataset.series_seq
    input_series_transformed = dataset.test_input_transformed
    LOGGER.info(f"Evaluation of model {model.model_name}")
    LOGGER.info(f"Predicting series")
    output_series_tranformed = model.predict(n=len(expected_output_series[0]), series=input_series_transformed)
    LOGGER.info(f"Inverse transform of all series")
    output_series = dataset.transformer.inverse_transform(output_series_tranformed)

    for idx, ticker in enumerate(CONST.TICKERS):
        output = output_series[idx][CONST.FEATURES.PRICE]
        expected = expected_output_series[idx][CONST.FEATURES.PRICE]
        original = original_series[idx][CONST.FEATURES.PRICE]
        scores[ticker] = mape(output, expected)
        plt.figure(figsize=(8, 5))
        original.plot(label="original")
        output.plot(label="forecast")
        plt.title(ticker + " - MAPE: {:.2f}%".format(scores[ticker]))
        plt.legend()

        plt.show()

    return scores


def eval_model_in_parts(model, dataset: SeqDataset):
    # TODO doesn't work
    scores = {}
    base_expected_output_series = dataset.test_seq
    original_series = dataset.series_seq
    base_input_series_transformed = dataset.test_input_transformed
    LOGGER.info(f"Evaluation of model {model.model_name}")
    all_output_series_arr = []
    horizon = 75

    for offset in range(0, len(base_expected_output_series[0]), 75):
        if offset > 0:
            input_series_transformed = concatanete_seq(
                base_input_series_transformed, list(map(lambda x: x[:offset], base_expected_output_series))
            )
        else:
            input_series_transformed = base_input_series_transformed
        # expected_output_series = list(map(lambda x: x[offset : offset + horizon], base_expected_output_series))
        output_series_tranformed = model.predict(n=horizon, series=input_series_transformed)
        output_series = dataset.transformer.inverse_transform(output_series_tranformed)
        all_output_series_arr.append(output_series)

    for idx, ticker in enumerate(CONST.TICKERS):
        plt.figure(figsize=(8, 5))
        original = original_series[idx][CONST.FEATURES.PRICE]
        original.plot(label="original")
        for output_series in all_output_series_arr:
            output = output_series[idx][CONST.FEATURES.PRICE]
            output.plot(label="forecast")
        plt.title(ticker)
        plt.legend()
        plt.show()

    return scores


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    MODEL_NAME = "LSTM"
    dataset = SeqDataset(sanity_check=True)
    model = RNNModel.load_from_checkpoint(model_name=MODEL_NAME, best=True)
    scores = eval_model_in_parts(model, dataset)
    print(scores)
