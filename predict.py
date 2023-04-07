import logging
from typing import List
import numpy as np
from pandas import DataFrame
import torch
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
import const as CONST
import matplotlib.pyplot as plt
from darts.metrics import mape
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
import pandas as pd
from darts import TimeSeries
import logging
from const import ModelConfig
from utils import create_folder


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="predict")


def get_multistep_predictions(
    model: TorchForecastingModel,
    ds: Datasets,
) -> List[List[TimeSeries]]:
    """Performs multi-step predictions and evaluates the model using model.historical_forecasts()"""
    predictions = []
    start = ds.original.test[0].start_time()
    model.predict_from_dataset
    transformed_outputs = model.historical_forecasts(
        series=ds.transformed.slice(0, len(ds.transformed.test_input[0]) + 2),  # TODO dont use slice
        past_covariates=ds.covariances.slice(0, len(ds.transformed.test_input[0]) + 2),
        retrain=False,
        start=start,
        forecast_horizon=model.output_chunk_length,
        stride=1,
        last_points_only=False,
        verbose=True,
    )
    transformed_outputs = [transformed_outputs] if CONST.SANITY_CHECK else transformed_outputs
    predictions_len = len(transformed_outputs[0])
    for step in range(0, predictions_len):
        transformed_series = [transformed[step] for transformed in transformed_outputs]
        prediction = ds.transformer.inverse(transformed_series)
        predictions.append(prediction)
    return predictions


def save_predictions(predictions: List[List[TimeSeries]], tickers: List[str], model_name: str):
    create_folder(f"results/{model_name}/", delete_if_exists=True)

    for idx, ticker in enumerate(tickers):
        with open(f"results/{model_name}/predicted_{ticker}.csv", "w") as file:
            for step in range(0, len(predictions)):
                step_df = predictions[step][idx].pd_dataframe()
                step_df.to_csv(file, mode="a", header=True if step == 0 else False)


def main(config: ModelConfig):
    torch.set_float32_matmul_precision("medium")
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    ds = load_datasets()
    model = TorchForecastingModel.load_from_checkpoint(model_name=config.model_name, best=True)
    predictions = get_multistep_predictions(model, ds)
    save_predictions(predictions, ds.original.used_tickers, config.model_name)


if __name__ == "__main__":
    main(CONST.MODEL_CONFIG)
