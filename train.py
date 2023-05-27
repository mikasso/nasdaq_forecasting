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


def use_half_precision(model: BlockRNNModel, target, past_cov):
    # TODO  Not working
    intit_dataset = model._build_train_dataset(
        target=target, past_covariates=past_cov, future_covariates=None, max_samples_per_ts=None
    )
    train_sample = intit_dataset[0]
    if model.model is None:
        # Build model, based on the dimensions of the first series in the train set.
        model.train_sample, model.output_dim = train_sample, train_sample[-1].shape[1]
        model._init_model(None)
    model.model.half()


def train_model(model: BlockRNNModel):
    torch.set_float32_matmul_precision("medium")
    assert_pytorch_is_using_gpu()
    LOGGER.info("Loading dataset")
    ds = load_datasets(CONST.SANITY_CHECK)
    LOGGER.info(f"Starting training {model.model_name}")
    model = model.fit(
        ds.transformed.train,
        past_covariates=ds.covariates.train,
        val_series=ds.transformed.val,
        val_past_covariates=ds.covariates.val,
        verbose=True,
        num_loader_workers=1 if CONST.SANITY_CHECK else 4,
    )
    LOGGER.info(f"Finished training of {model.model_name}")
    return model
