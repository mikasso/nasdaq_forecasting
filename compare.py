import pandas as pd
from model_baseline import BASELINE_CONFIG
from run import MODEL_CONFIGS
import const as CONST

if __name__ == "__main__":
    dict = {}
    stocks_dict = {}

    def add_result(config: CONST.ModelConfig):
        df = pd.read_csv(f"{config.result_path}/described_mape.csv", header=0, usecols=range(1, 9))
        dict[config.model_name] = df.mean(axis=1)

    def add_stock_result(config: CONST.ModelConfig):
        df = pd.read_csv(f"{config.result_path}/mape.csv", header=0, usecols=range(1, 9))
        df.mean()
        stocks_dict[config.model_name] = df.mean()

    add_result(BASELINE_CONFIG)
    for config in MODEL_CONFIGS:
        add_result(config)
        add_stock_result(config)

    pd.DataFrame(dict).to_csv(f"{CONST.PATHS.RESULTS}/general/comaprision_7.csv")
    pd.DataFrame(stocks_dict).to_csv(f"{CONST.PATHS.RESULTS}/general/stocks_comaprision_7.csv")
