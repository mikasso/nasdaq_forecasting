import logging
import torch
from torchmetrics import MeanSquaredError
from LossLogger import LossLogger
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
from darts.models import BlockRNNModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import const as CONST
from train import train_model
from utils import assert_pytorch_is_using_gpu, visualize_history
from const import ModelConfig
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="rnn_models")


def main(config: ModelConfig):
    loss_logger = LossLogger()
    model = BlockRNNModel(
        batch_size=CONST.SHARED_CONFIG.BATCH_SIZE,
        n_epochs=CONST.SHARED_CONFIG.EPOCHS,
        input_chunk_length=CONST.SHARED_CONFIG.INPUT_LEN,
        pl_trainer_kwargs=CONST.SHARED_CONFIG.get_pl_trainer_kwargs([loss_logger]),
        optimizer_kwargs=CONST.SHARED_CONFIG.OPTIMIZER_KWARGS,
        dropout=CONST.SHARED_CONFIG.DROPOUT,
        lr_scheduler_kwargs=CONST.SHARED_CONFIG.LR_SCHEDULER_KWARGS,
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        model=config.model_type.value,
        model_name=config.model_name,
        output_chunk_length=config.output_len,
        hidden_dim=config.hidden_state,
        n_rnn_layers=3,
        loss_fn=MeanSquaredError(),
        log_tensorboard=True,
        force_reset=True,
        save_checkpoints=True,
        show_warnings=True,
    )
    trained_model = train_model(model)
    visualize_history(config, loss_logger.train_loss, loss_logger.val_loss)
    return trained_model


if __name__ == "__main__":
    model = main(CONST.MODEL_CONFIG)
