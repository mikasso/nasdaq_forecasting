import logging
import torch
from torchmetrics import MeanSquaredError
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
from darts.models import BlockRNNModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import const as CONST
from utils import assert_pytorch_is_using_gpu


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="train")


def train_model(model):
    torch.set_float32_matmul_precision("medium")
    assert_pytorch_is_using_gpu()
    LOGGER.info("Loading dataset")
    ds = load_datasets()
    LOGGER.info(f"Starting training {model.model_name}")
    model = model.fit(
        ds.transformed.train,
        past_covariates=ds.covariates.train,
        val_series=ds.transformed.val,
        val_past_covariates=ds.covariates.val,
        verbose=True,
        num_loader_workers=4,
    )
    LOGGER.info(f"Finished training of {model.model_name}")
    return model
