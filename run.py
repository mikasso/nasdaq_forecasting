import enum
import logging
import model_rnn
import model_tcn
import model_tft
import model_transformer
from const import ModelConfig, RNN_NETWORKS, ModelTypes
from utils import create_folder
import validate
import view_results
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
import compare
import evaluate

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name="Main runner")

MODEL_CONFIGS = [
    ModelConfig(ModelTypes.rnn, 1, hidden_state=80),
    ModelConfig(ModelTypes.gru, 1, hidden_state=45),
    ModelConfig(ModelTypes.lstm, 1, hidden_state=38),
    ModelConfig(
        ModelTypes.tft,
        1,
        hidden_state=14,
    ),
    ModelConfig(ModelTypes.transformer, 1, hidden_state=8),
    ModelConfig(ModelTypes.tcn, 1),
    ModelConfig(ModelTypes.rnn, 8, hidden_state=80),
    ModelConfig(ModelTypes.gru, 8, hidden_state=45),
    ModelConfig(ModelTypes.lstm, 8, hidden_state=38),
    ModelConfig(
        ModelTypes.tft,
        8,
        hidden_state=14,
    ),
    ModelConfig(ModelTypes.transformer, 8, hidden_state=8),
    ModelConfig(ModelTypes.tcn, 8),
    ModelConfig(ModelTypes.rnn, 40, hidden_state=80),
    ModelConfig(ModelTypes.gru, 40, hidden_state=45),
    ModelConfig(ModelTypes.lstm, 40, hidden_state=38),
    ModelConfig(
        ModelTypes.tft,
        40,
        hidden_state=14,
    ),
    ModelConfig(ModelTypes.transformer, 40, hidden_state=8),
    ModelConfig(ModelTypes.tcn, 40),
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
        create_folder(config.result_path, delete_if_exists=True)
        dispatch_model_training(config)
        validate.main(config)
        view_results.main(config, show=False)
        LOGGER.info(f"Evaluating model: {config.model_name}")

    LOGGER.info("Creating comparision heatmap")
    compare.main()
    LOGGER.info("Creating accuracy and weights table for TFT models")
    evaluate.main()
