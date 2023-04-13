import logging
import torch
from torchmetrics import MeanSquaredError
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
from darts.models import BlockRNNModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import const as CONST
from train import train_model
from utils import assert_pytorch_is_using_gpu
from const import ModelConfig


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="rnn_models")


def main(config: ModelConfig):
    model = BlockRNNModel(
        model=config.model_type.value,
        hidden_dim=128,
        n_rnn_layers=2,
        batch_size=64,
        n_epochs=2500,
        optimizer_kwargs={"lr": 1e-4},
        model_name=config.model_name,
        log_tensorboard=True,
        random_state=42,
        input_chunk_length=512,
        output_chunk_length=config.output_len,
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
    trained_model = train_model(model)
    return trained_model


if __name__ == "__main__":
    main(CONST.MODEL_CONFIG)
