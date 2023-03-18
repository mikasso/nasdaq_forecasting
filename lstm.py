from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError
import const as CONST

import torch

import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.metrics import mape
from pytorch_lightning.callbacks import EarlyStopping, BatchSizeFinder, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger

from utils import SeqDataset

MODEL_NAME = "GRU1"
if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    dataset = SeqDataset(sanity_check=True)

    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=20,
        min_delta=0.01,
        mode="min",
    )
    my_model = RNNModel(
        model="GRU",
        hidden_dim=150,
        n_rnn_layers=2,
        dropout=0.2,
        batch_size=256,
        n_epochs=400,
        optimizer_kwargs={"lr": 1e-3},
        model_name=MODEL_NAME,
        log_tensorboard=True,
        random_state=42,
        input_chunk_length=15 * 30,
        output_chunk_length=15 * 1,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs={
            "callbacks": [my_stopper, LearningRateFinder()],
            "accelerator": "gpu",
            "devices": [0],
        },
        loss_fn=MeanSquaredError(),
        add_encoders={
            "cyclic": {"future": ["month"]},
            "datetime_attribute": {"future": ["hour", "dayofweek"]},
            "position": {"future": ["relative"]},
            "transformer": Scaler(),
        },
    )

    my_model.fit(
        dataset.train_transformed,
        val_series=dataset.val_transformed,
        verbose=True,
        num_loader_workers=4,
    )

    def eval_model(model, dataset):
        scores = {}
        for idx, ticker in enumerate(CONST.TICKERS):
            expected_output_series = dataset.test_seq[idx]["price"]
            original_series = dataset.series_seq[idx]["price"]

            input_series_transformed = dataset.test_input_transformed[idx]
            output_series_tranformed = model.predict(n=len(expected_output_series), series=input_series_transformed)
            output_series = dataset.transformer.inverse_transform(output_series_tranformed)["price"]
            scores[ticker] = mape(output_series, expected_output_series)

            plt.figure(figsize=(8, 5))
            original_series.plot(label="original")
            output_series.plot(label="forecast")
            plt.title(ticker + " - MAPE: {:.2f}%".format(scores[ticker]))
            plt.legend()

            plt.show()

        return scores

    eval_model(my_model, dataset)

    best_model = RNNModel.load_from_checkpoint(model_name="LSTM", best=True)
    eval_model(best_model, dataset)
