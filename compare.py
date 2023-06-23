import pandas as pd
from model_baseline import BASELINE_CONFIG
from run import MODEL_CONFIGS
import const as CONST
import seaborn as sns
import matplotlib.pyplot as plt


def extract_name(config):
    return f"{config.model_type.value} {config.output_len}"


if __name__ == "__main__":
    dict = {}
    stocks_dict = {}

    def add_result(config: CONST.ModelConfig):
        df = pd.read_csv(f"{config.result_path}/described_mape.csv", header=0, usecols=range(1, 9))
        dict[extract_name(config)] = df.mean(axis=1)

    def add_stock_result(config: CONST.ModelConfig):
        df = pd.read_csv(f"{config.result_path}/mape.csv", header=0, usecols=range(1, 9))
        stocks_dict[extract_name(config)] = df.mean()

    add_result(BASELINE_CONFIG)
    for config in MODEL_CONFIGS:
        add_result(config)
        add_stock_result(config)

    map_names = {0: "count", 1: "mean", 2: "std", 3: "min", 4: "25%", 5: "50%", 6: "75%", 7: "max"}
    pd.DataFrame(dict).rename(map_names).to_csv(f"{CONST.PATHS.RESULTS}/general/comaprision.csv")

    stocks_comaprison = pd.DataFrame(stocks_dict)
    stocks_comaprison.loc["μ modelu"] = stocks_comaprison.mean(axis="rows")
    stocks_comaprison.to_csv(f"{CONST.PATHS.RESULTS}/general/stocks_comaprision.csv")

    def process_result_map(max, out_len):
        df = stocks_comaprison.filter(like=f"{out_len}", axis=1)
        df["μ spółki"] = df.mean(axis="columns")
        sns.heatmap(df.T, annot=True, vmax=max, cmap="Blues", fmt=".2f")
        plt.savefig(f"{CONST.PATHS.RESULTS}/general/results_{out_len}.svg", format="svg", dpi=300, bbox_inches="tight")
        plt.show()

    process_result_map(2, 1)
    process_result_map(3, 7)


    def average_error(config, window =100):
        df = pd.read_csv(config.result_path+"/mape.csv",index_col=0)
        df.mean(axis="columns").rolling(window).mean().plot()
        plt.show()
