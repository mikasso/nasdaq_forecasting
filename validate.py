import logging
import types
from typing import List
import joblib
import numpy as np
from pandas import DataFrame
from pytorch_lightning import Trainer
import torch
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
import const as CONST
import matplotlib.pyplot as plt
from darts.metrics import mape
from darts.models import RNNModel
import pandas as pd
from darts import TimeSeries
import logging
from darts.models import BlockRNNModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.utils.data.inference_dataset import (
    PastCovariatesInferenceDataset,
)
from darts import concatenate
from torch.utils.data import Dataset, DataLoader
import warnings

from utils import create_folder


warnings.simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="predict")


def _predict_step(self, val_batch, batch_idx) -> torch.Tensor:
    """performs predict step hacky"""
    output = self._produce_train_output(val_batch[:-1])  # why [:-1]
    return output


def main(config=CONST.MODEL_CONFIG):
    torch.set_float32_matmul_precision("medium")
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

    model = TorchForecastingModel.load_from_checkpoint(model_name=config.model_name, best=True)
    model.model.set_predict_parameters(1, 1, 1, CONST.SHARED_CONFIG.BATCH_SIZE, 4)
    model.trainer = model._init_trainer(model.trainer_params)

    LOGGER.info("Preparing test set")
    ds = load_datasets()
    test_input_start = ds.transformed.test_idx - model.input_chunk_length
    test_data = ds.transformed.slice(start=test_input_start)
    test_covariates = ds.covariates.slice(start=test_input_start)

    # it returns reversed dataset
    test_dataset = model._build_train_dataset(
        target=test_data,
        past_covariates=test_covariates,
        future_covariates=None,
        max_samples_per_ts=None,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=model.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=model._batch_collate_fn,
    )

    LOGGER.info("Running model for prediction")
    model.model.predict_step = types.MethodType(_predict_step, model.model)
    predict = model.trainer.predict(model=model.model, dataloaders=test_dataloader)

    LOGGER.info("Preparing output data for conversion")

    if config.model_type == CONST.ModelTypes.tcn:
        flatten_tensors = [torch.flatten(tensor[:, -config.output_len :, :]) for tensor in predict]
    else:
        flatten_tensors = [torch.flatten(tensor) for tensor in predict]
    flatten_tensor = torch.cat(flatten_tensors)

    LOGGER.info("Spliting all predicted tensor by series idx")
    tensors_by_series = []
    series_count = len(ds.original.test)
    steps = len(flatten_tensor) // series_count
    for idx, series in enumerate(ds.original.test):
        start = idx * steps
        end = (idx + 1) * steps
        values = flatten_tensor[start:end].flip([-1])
        tensors_by_series.append(values)

    LOGGER.info("Building timeseries predictions for each series idx")
    horizon = model.output_chunk_length
    predictions_by_series = []
    for tensor, series in zip(tensors_by_series, ds.original.test):
        grouped_tensor = [tensor[i : i + horizon] for i in range(0, len(tensor), horizon)]
        predictions = []
        for idx, values in enumerate(grouped_tensor):
            start_date = series[idx].start_time()
            times = pd.date_range(start=start_date, periods=horizon, freq=series.freq)
            result = TimeSeries.from_times_and_values(times, values)
            predictions.append(result)
        predictions_by_series.append(predictions)

    LOGGER.info("inversing transformation")
    predictions_len = len(predictions_by_series[0])
    inversed_prediction_by_step = []
    for idx in range(predictions_len):  # could be done in parallel but ds.transformer has to be shared somehow
        input_series = [predictions_by_series[series_idx][idx] for series_idx in range(0, series_count)]
        inversed_prediction_by_step.append(ds.transformer.inverse(input_series, n_jobs=1, verbose=False))

    LOGGER.info("Saving results")
    inversed_by_series_idx = [[series[i] for series in inversed_prediction_by_step] for i in range(series_count)]

    joblib.dump(inversed_by_series_idx, f"{config.result_path}/{config.model_name}.pkl")


if __name__ == "__main__":
    main()
