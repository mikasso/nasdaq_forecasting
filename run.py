import enum
import logging
import model_rnn
import model_tcn
import model_tft
import model_transformer
from model_configs import ModelConfig, ModelTypes, MODEL_CONFIGS
import const as CONST
from utils import create_folder
import test
import view_results
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
import compare
import evaluate

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="Main runner")
RNN_NETWORKS = [ModelTypes.rnn, ModelTypes.lstm, ModelTypes.gru]


def dispatch_model_training(config: ModelConfig):
    if config.model_type in RNN_NETWORKS:
        model_rnn.main(config)
    elif config.model_type == ModelTypes.tft:
        model_tft.main(config)
    elif config.model_type == ModelTypes.tcn:
        model_tcn.main(config)
    elif config.model_type == ModelTypes.transformer:
        model_transformer.main(config)
    else:
        LOGGER.warning(f"Unrecognized model type {config.model_type}")


if __name__ == "__main__":
    LOGGER.info(f"Starting models run, interval = {CONST.INTERVAL}")

    for config in MODEL_CONFIGS:
        LOGGER.info(f"Running model: {config.model_name}")
        create_folder(config.result_path, delete_if_exists=True)
        dispatch_model_training(config)
        test.main(config)
        view_results.main(config, show=False)
        LOGGER.info(f"Evaluating model: {config.model_name}")

    LOGGER.info("Creating comparision heatmap")
    compare.main()
    LOGGER.info("Creating accuracy and weights table for TFT models")
    evaluate.main()
