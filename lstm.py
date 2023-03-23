import logging
import torch
from torchmetrics import MeanSquaredError
from SeqDataset import SeqDataset, TransformedDataset
from darts.models import RNNModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="lstm")
MODEL_NAME = "LSTM"

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    LOGGER.info("Initializing dataset")
    dataset = SeqDataset.load(sanity_check=False, use_pct=False)
    transformed = TransformedDataset.build_from_dataset(dataset)
    model = RNNModel(
        model="LSTM",
        hidden_dim=256,
        n_rnn_layers=1,
        dropout=0.3,
        batch_size=64,
        n_epochs=2500,
        optimizer_kwargs={"lr": 1e-4},
        model_name=MODEL_NAME,
        log_tensorboard=True,
        random_state=42,
        input_chunk_length=512,
        training_length=512 + 1,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs={
            "callbacks": [
                EarlyStopping(
                    monitor="val_loss",
                    patience=20,
                    min_delta=0.000001,
                    mode="min",
                ),
                LearningRateFinder(),
            ],
            "accelerator": "gpu",
            "devices": [0],
        },
        loss_fn=MeanSquaredError(),
        show_warnings=True,
        # TODO This crashes forecast_history -> move time to covariates
        # add_encoders={
        #     "cyclic": {"future": ["month"]},
        #     "datetime_attribute": {"future": ["hour", "dayofweek"]},
        #     "position": {"future": ["relative"]},
        #     "transformer": Scaler(),
        # },
    )

    LOGGER.info("Starting training")
    model.fit(transformed.train, val_series=transformed.val, verbose=True, num_loader_workers=4)
    LOGGER.info("Finished training")
