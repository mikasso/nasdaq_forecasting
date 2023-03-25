import logging
from typing import Tuple
import torch
from torchmetrics import MeanSquaredError
from SeqDataset import DatasetAccesor, SeqDataset, TransformedDataset, get_datecovs
from darts.models import RNNModel, BlockRNNModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder, LearningRateMonitor
import const as CONST
from darts import concatenate


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_start(self, trainer, pl_module):
        self.lr_find(trainer, pl_module)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="lstm")
MODEL_NAME = "BlockRNNModel_LSTM_O1"
SANITY_CHECK = False
USE_PCT = False


def get_datasets() -> Tuple[SeqDataset, TransformedDataset, DatasetAccesor]:
    dataset = SeqDataset.load(sanity_check=SANITY_CHECK, use_pct=USE_PCT, target_features=[CONST.FEATURES.PRICE])
    transformed = TransformedDataset.build_from_dataset(dataset)
    shares_dataset = SeqDataset.load(
        sanity_check=SANITY_CHECK, use_pct=USE_PCT, target_features=[CONST.FEATURES.SHARES]
    )
    shares_transformed = TransformedDataset.build_from_dataset(shares_dataset)
    dates_covariates = get_datecovs(dataset)
    covariates = [
        concatenate([dates, shares], axis=1)
        for dates, shares in zip(dates_covariates.series, shares_transformed.series)
    ]
    cov_dataset = DatasetAccesor(covariates, dataset.val_idx, dataset.test_idx)
    return dataset, transformed, cov_dataset


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    LOGGER.info("Initializing dataset")

    dataset, transformed, cov_dataset = get_datasets()

    model = BlockRNNModel(
        model="LSTM",
        hidden_dim=256,
        n_rnn_layers=2,
        dropout=0.3,
        batch_size=64,
        n_epochs=1 if SANITY_CHECK else 2500,
        optimizer_kwargs={"lr": 1e-4},
        model_name=MODEL_NAME,
        log_tensorboard=True,
        random_state=42,
        input_chunk_length=512,
        output_chunk_length=1,
        force_reset=True,
        save_checkpoints=True,
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        pl_trainer_kwargs={
            "callbacks": [
                EarlyStopping(
                    monitor="val_loss",
                    patience=20,
                    min_delta=0.000001,
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            "accelerator": "gpu",
            "devices": [0],
        },
        loss_fn=MeanSquaredError(),
        show_warnings=True,
    )

    LOGGER.info("Starting training")
    model.fit(
        transformed.train,
        past_covariates=cov_dataset.train,
        val_series=transformed.val,
        val_past_covariates=cov_dataset.val,
        verbose=True,
        num_loader_workers=4,
    )
    LOGGER.info("Finished training")
