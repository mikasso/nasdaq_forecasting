import pandas as pd
from model_baseline import BASELINE_CONFIG
from run import MODEL_CONFIGS
import const as CONST
import seaborn as sns
import matplotlib.pyplot as plt


def extract_name(config):
    return f"{config.model_type.value} {config.output_len}"


def average_error(config, window=100):
    df = pd.read_csv(config.result_path + "/mape.csv", index_col=0)
    df.mean(axis="columns").rolling(window).mean().plot()
    plt.show()


def main():
    dict = {}
    stocks_dict = {}

    def add_result(config: CONST.ModelConfig):
        df = pd.read_csv(f"{config.result_path}/described_mape.csv", header=0, usecols=range(1, len(CONST.TICKERS) + 1))
        dict[extract_name(config)] = df.mean(axis=1)

    def add_stock_result(config: CONST.ModelConfig):
        df = pd.read_csv(f"{config.result_path}/mape.csv", header=0, usecols=range(1, len(CONST.TICKERS) + 1))
        stocks_dict[extract_name(config)] = df.mean()

    for config in MODEL_CONFIGS:
        add_result(config)
        add_stock_result(config)

    map_names = {0: "count", 1: "mean", 2: "std", 3: "min", 4: "25%", 5: "50%", 6: "75%", 7: "max"}
    pd.DataFrame(dict).rename(map_names).to_csv(f"{CONST.PATHS.RESULTS}/general/comaprision.csv")

    stocks_comaprison = pd.DataFrame(stocks_dict)
    stocks_comaprison.loc["średnia"] = stocks_comaprison.mean(axis="rows")
    stocks_comaprison.to_csv(f"{CONST.PATHS.RESULTS}/general/stocks_comaprision.csv")

    def process_result_map(out_len):
        df = stocks_comaprison.filter(like=f"{out_len}", axis=1)
        df["średnia"] = df.mean(axis="columns")
        sns.heatmap(df.T, annot=True, vmax=df.max().max(), vmin=df.min().min(), cmap="Blues", fmt=".3f")
        plt.tick_params(
            axis="both", which="major", labelsize=10, labelbottom=False, bottom=False, top=False, labeltop=True
        )
        plt.savefig(f"{CONST.PATHS.RESULTS}/general/results_{out_len}.png", format="png", dpi=300, bbox_inches="tight")
        plt.show()

    process_result_map(1)
    process_result_map(8)
    process_result_map(40)


if __name__ == "__main__":
    main()
