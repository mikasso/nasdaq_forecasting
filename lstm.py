import logging
import torch
from torchmetrics import MeanSquaredError
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
from darts.models import BlockRNNModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import const as CONST
from utils import assert_pytorch_is_using_gpu


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="lstm")


def main():
    torch.set_float32_matmul_precision("medium")
    assert_pytorch_is_using_gpu()
    LOGGER.info("Loading dataset")
    ds = load_datasets()
    model = BlockRNNModel(
        model="LSTM",
        hidden_dim=32,
        n_rnn_layers=2,
        batch_size=128,
        n_epochs=1,  # 1 if CONST.SANITY_CHECK else 2500,
        optimizer_kwargs={"lr": 1e-4},
        model_name=CONST.MODEL_NAME,
        log_tensorboard=True,
        random_state=42,
        input_chunk_length=1024,
        output_chunk_length=5,
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
    model = model.fit(
        ds.transformed.train,
        past_covariates=ds.covariances.train,
        val_series=ds.transformed.val,
        val_past_covariates=ds.covariances.val,
        verbose=True,
        num_loader_workers=4,
    )
    LOGGER.info("Finished training")


if __name__ == "__main__":
    main()
