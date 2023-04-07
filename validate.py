import logging
import types
from typing import List
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

    model = BlockRNNModel.load_from_checkpoint(model_name=config.model_name, best=True)
    model.model.set_predict_parameters(1, 1, 1, 128, 4)
    model.trainer = model._init_trainer(model.trainer_params)

    test_input_start = ds.transformed.test_idx - model.input_chunk_length
    test_data = ds.transformed.slice(test_input_start)
    test_covariates = ds.covariances.slice(test_input_start)
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

    model.model.predict_step = types.MethodType(_predict_step, model.model)

    predict = model.trainer.predict(model=model.model, dataloaders=val_dataloader)
    flatten_tensors = [torch.flatten(tensor) for tensor in predict]
    result_tensor = torch.cat(flatten_tensors)

    results = []
    for idx, series in enumerate(ds.original.test):
        start = idx * len(series)
        end = start + len(series)
        values = result_tensor[start:end].numpy()
        result = TimeSeries.from_times_and_values(series.time_index, values)
        results.append(result)

    inversed_results = ds.transformer.inverse(results)


if __name__ == "__main__":
    main(CONST.MODEL_CONFIG)
