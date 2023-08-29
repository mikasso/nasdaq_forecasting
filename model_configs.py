from enum import Enum
from const import PATHS


class ModelTypes(Enum):
    rnn = "RNN"
    lstm = "LSTM"
    gru = "GRU"
    transformer = "Transformer"
    tft = "TFT"
    tcn = "TCN"


class ModelConfig:
    def __init__(self, model_type: ModelTypes, output_len: int, model_name=None, hidden_state=256) -> None:
        self.model_name = f"{model_type}_out_{output_len}" if model_name == None else model_name
        self.model_type = model_type
        self.output_len = output_len
        self.hidden_state = hidden_state

    @property
    def result_path(self) -> str:
        return f"{PATHS.RESULTS}/{self.model_name}"


MODEL_CONFIGS = [
    ModelConfig(ModelTypes.rnn, 1, hidden_state=80, model_name="TEST_RNN"),
    # ModelConfig(ModelTypes.gru, 1, hidden_state=45),
    # ModelConfig(ModelTypes.lstm, 1, hidden_state=38),
    # ModelConfig(
    #     ModelTypes.tft,
    #     1,
    #     hidden_state=14,
    # ),
    # ModelConfig(ModelTypes.transformer, 1, hidden_state=8),
    # ModelConfig(ModelTypes.tcn, 1),
    # ModelConfig(ModelTypes.rnn, 8, hidden_state=80),
    # ModelConfig(ModelTypes.gru, 8, hidden_state=45),
    # ModelConfig(ModelTypes.lstm, 8, hidden_state=38),
    # ModelConfig(
    #     ModelTypes.tft,
    #     8,
    #     hidden_state=14,
    # ),
    # ModelConfig(ModelTypes.transformer, 8, hidden_state=8),
    # ModelConfig(ModelTypes.tcn, 8),
    # ModelConfig(ModelTypes.rnn, 40, hidden_state=80),
    # ModelConfig(ModelTypes.gru, 40, hidden_state=45),
    # ModelConfig(ModelTypes.lstm, 40, hidden_state=38),
    # ModelConfig(
    #     ModelTypes.tft,
    #     40,
    #     hidden_state=14,
    # ),
    # ModelConfig(ModelTypes.transformer, 40, hidden_state=8),
    # ModelConfig(ModelTypes.tcn, 40),
]

DEFAULT_MODEL = ModelConfig(ModelTypes.lstm, 1, hidden_state=32, model_name="default_lstm")
""" Default Model config """
