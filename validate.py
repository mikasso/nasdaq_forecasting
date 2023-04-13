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
from darts.utils.data.inference_dataset import (
    PastCovariatesInferenceDataset,
)
from darts import concatenate

from view_results import visualize_predictions

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="eval")

from torch.utils.data import Dataset, DataLoader


def _predict_step(self, val_batch, batch_idx) -> torch.Tensor:
    """performs predict step hacky"""
    output = self._produce_train_output(val_batch[:-1])
    return output


def save_predictions(predictions: List[TimeSeries], tickers: List[str], model_name: str):
    for predicted, ticker in zip(predictions, tickers):
        df = predicted.pd_dataframe()
        df.to_csv(f"results/{model_name}/predicted_{ticker}.csv")


def main(config=CONST.MODEL_CONFIG):
    torch.set_float32_matmul_precision("medium")
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    ds = load_datasets()

    model = BlockRNNModel.load_from_checkpoint(model_name="GRU_O7", best=True)
    model.model.set_predict_parameters(1, 1, 1, 128, 4)
    model.trainer = model._init_trainer(model.trainer_params)

    test_input_start = ds.transformed.test_idx - model.input_chunk_length - model.output_chunk_length + 1
    test_data = ds.transformed.slice(test_input_start)
    test_covariates = ds.covariates.slice(test_input_start)
    val_dataset = model._build_train_dataset(
        target=test_data,
        past_covariates=test_covariates,
        future_covariates=None,
        max_samples_per_ts=None,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=model.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=model._batch_collate_fn,
    )

    # Hack model so it can predict as its suppose to ;)
    model.model.predict_step = types.MethodType(_predict_step, model.model)

    predict = model.trainer.predict(model=model.model, dataloaders=val_dataloader)
    flatten_tensors = [torch.flatten(tensor) for tensor in predict]
    output_tensor = torch.cat(flatten_tensors)

    tensors_by_horizon = []
    for i in range(0, model.output_chunk_length):
        tensor = output_tensor[i :: model.output_chunk_length]
        tensors_by_horizon.append(tensor)
    # DEBUG this shit and then refactor, should work for both o1, o3, o5
    timeseries_seq_by_horizon = []
    for horizon, tensor in enumerate(tensors_by_horizon):
        results = []
        for idx, series in enumerate(ds.original.test):
            steps = len(series)
            start = idx * steps
            end = start + steps
            values = tensor[start:end].numpy()
            start_date = series[horizon].start_time()
            times = pd.date_range(start=start_date, periods=steps, freq=series.freq)
            result = TimeSeries.from_times_and_values(times, values)
            results.append(result)

        inversed_results = ds.transformer.inverse(results)
        timeseries_seq_by_horizon.append(inversed_results)

    joblib.dump(timeseries_seq_by_horizon, "results.temp")


def display(timeseries_seq_by_horizon: List[List[TimeSeries]], ds: Datasets):
    for horizon, series_seq in enumerate(timeseries_seq_by_horizon):
        visualize_predictions(ds.original, series_seq, model_name="GRU_O7", title_prefix=f"h={horizon}")


if __name__ == "__main__":
    saved = False
    if not saved:
        main()
    ds = load_datasets()
    timeseries_seq_by_horizon = joblib.load("results.temp")
    display(timeseries_seq_by_horizon, ds)


# jesli trzeba policzyc tylko blad to
# 1 2 3 4
#   1 2 3 4
#     1 2 3 4
#       1 2 3 4
# to mozna wziac 4 series skladajace sie z tylko 1 lub tylko 2 lub tylko 3 ..
# i policzyc blad
