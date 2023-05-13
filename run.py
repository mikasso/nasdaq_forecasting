import enum
import logging
import model_rnn
import model_tft
import model_transformer
import _predict
from const import ModelConfig, RNN_NETWORKS, ModelTypes
from utils import create_folder
import validate
import view_results
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="Main runner")


def dispatch_model_training(config: ModelConfig):
    if config.model_type in RNN_NETWORKS:
        model_rnn.main(config)
    elif config.model_type == ModelTypes.tft:
        model_tft.main(config)
    elif config.model_type == ModelTypes.transformer:
        model_transformer.main(config)
    else:
        LOGGER.warning(f"Unrecognized model type {config.model_type}")


if __name__ == "__main__":
    LOGGER.info("Starting models run")
    models = [
        ModelConfig(ModelTypes.rnn, 1, 256),
        ModelConfig(ModelTypes.gru, 1, 256),
        ModelConfig(ModelTypes.lstm, 1, 222),
        ModelConfig(ModelTypes.transformer, 1, 64),
        ModelConfig(ModelTypes.tft, 1, 110),
        ModelConfig(ModelTypes.rnn, 7, 256),
        ModelConfig(ModelTypes.gru, 7, 256),
        ModelConfig(ModelTypes.lstm, 7, 222),
        ModelConfig(ModelTypes.transformer, 7, 64),
        ModelConfig(ModelTypes.tft, 7, 110),
    ]

    for model_config in models:
        LOGGER.info(f"Running model: {model_config.model_name}")
        create_folder(model_config.result_path, delete_if_exists=True)
        dispatch_model_training(model_config)
        validate.main(model_config)
        LOGGER.info(f"Finished running model: {model_config.model_name}")
