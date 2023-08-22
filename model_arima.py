import logging
import torch
from torchmetrics import MeanSquaredError
from LossLogger import LossLogger
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
from darts.models import BlockRNNModel, ARIMA, AutoARIMA
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import const as CONST
from train import train_model
from utils import assert_pytorch_is_using_gpu, visualize_history
from const import ModelConfig
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="rnn_models")


def main(config: ModelConfig):
    model = AutoARIMA()
    model.fit()
    # visualize_history(config, loss_logger.train_loss, loss_logger.val_loss)
    return trained_model


if __name__ == "__main__":
    model = main(CONST.MODEL_CONFIG)
