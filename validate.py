import logging
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

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="eval")


def get_multistep_predictions(
    model: RNNModel,
    ds: Datasets,
    last_points_only=False,
    forecast_horizon=5,
) -> List[TimeSeries]:
    """Performs multi-step predictions and evaluates the model using model.historical_forecasts()"""
    start = ds.original.series[0].get_index_at_point(ds.original.test[0].start_time())
    transformed_outputs = model.historical_forecasts(
        ds.transformed.series,
        past_covariates=ds.covariances.series,
        retrain=False,
        start=start,
        forecast_horizon=forecast_horizon,
        stride=1,
        last_points_only=last_points_only,
        verbose=True,
    )
    transformed_outputs = [transformed_outputs] if CONST.SANITY_CHECK else transformed_outputs
    predictions = ds.transformer.inverse(transformed_outputs)
    return predictions


def save_predictions(predictions: List[TimeSeries], tickers: List[str], model_name: str):
    for predicted, ticker in zip(predictions, tickers):
        df = predicted.pd_dataframe()
        df.to_csv(f"results/{model_name}/predicted_{ticker}.csv")


def main():
    torch.set_float32_matmul_precision("medium")
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    ds = load_datasets()
    model_name = CONST.MODEL_NAME
    model = BlockRNNModel.load_from_checkpoint(model_name=model_name, best=True)
    test_input_dataset = model._build_train_dataset(
        target=ds.transformed.series,
        past_covariates=ds.covariances.series,
        future_covariates=None,
        max_samples_per_ts=None,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_input_dataset,
        batch_size=model.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=model._batch_collate_fn,
    )
    trainer = Trainer(accelerator="gpu")
    features = next(iter(test_dataloader))
    # val_metrics = trainer.validate(model=model.model, dataloaders=val_dataloader)
    # print(val_metrics)
    predict = trainer.predict(model=model.model, dataloaders=test_dataloader, return_predictions=True)
    print(predict)
    print("end")


if __name__ == "__main__":
    main()
