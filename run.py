import enum
import logging
import model_rnn
import model_tft
import model_transformer
import predict
from const import ModelConfig, RNN_NETWORKS, ModelTypes
import view_results

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
    models = [
        ModelConfig(ModelTypes.rnn, 1),
        ModelConfig(ModelTypes.gru, 1),
        ModelConfig(ModelTypes.lstm, 1),
        ModelConfig(ModelTypes.transformer, 1),
        ModelConfig(ModelTypes.tft, 1),
        ModelConfig(ModelTypes.rnn, 7),
        ModelConfig(ModelTypes.gru, 7),
        ModelConfig(ModelTypes.lstm, 7),
        ModelConfig(ModelTypes.transformer, 7),
        ModelConfig(ModelTypes.tft, 7),
    ]

    for model_config in models:
        LOGGER.info(f"Running model: {model_config.model_name}")
        dispatch_model_training(model_config)
        predict.main(model_config.model_name)
        LOGGER.info(f"Finished running model: {model_config.model_name}")
