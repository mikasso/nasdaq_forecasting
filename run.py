import enum
import logging
import model_rnn
import model_tcn
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

MODEL_CONFIGS = [
    ModelConfig(ModelTypes.rnn, 1, hidden_state=170),  # 147
    ModelConfig(ModelTypes.gru, 1, hidden_state=100),  # 154
    ModelConfig(ModelTypes.lstm, 1, hidden_state=85),  #
    ModelConfig(
        ModelTypes.tft,
        1,
        hidden_state=40,
    ),
    ModelConfig(ModelTypes.transformer, 1, hidden_state=32),
    ModelConfig(ModelTypes.rnn, 7, hidden_state=170),
    ModelConfig(ModelTypes.gru, 7, hidden_state=100),
    ModelConfig(ModelTypes.lstm, 7, hidden_state=85),
    ModelConfig(
        ModelTypes.tft,
        7,
        hidden_state=40,
    ),
    ModelConfig(ModelTypes.transformer, 7, hidden_state=32),
    ModelConfig(ModelTypes.tcn, 1),
    ModelConfig(ModelTypes.tcn, 7),
]


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
    LOGGER.info("Starting models run")

    for config in MODEL_CONFIGS:
        LOGGER.info(f"Running model: {config.model_name}")
        # create_folder(config.result_path, delete_if_exists=True)
        # dispatch_model_training(config)
        validate.main(config)
        view_results.main(config, show=False)
        LOGGER.info(f"Finished running model: {config.model_name}")
