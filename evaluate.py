import const as CONST
import matplotlib.pyplot as plt
from darts.metrics import mape
from darts import TimeSeries
import pandas as pd
import const as CONST
import warnings
import os
from run import MODEL_CONFIGS

# plt.style.use("ggplot")
import matplotlib

# matplotlib.rcParams["figure.figsize"] = (20, 10)
warnings.simplefilter(action="ignore", category=FutureWarning)
import darts
from validate import load_model
from view_results import load_results
from darts.datasets import AirPassengersDataset
from darts.explainability.tft_explainer import TFTExplainer
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
from const import ModelConfig, ModelTypes
from darts.models import TFTModel
import numpy as np
from darts import concatenate
from datasets import load_datasets
import joblib
import numpy as np


def plot_losses():
    plt.ioff()
    loss_dict = {}
    for config in MODEL_CONFIGS:
        figure = joblib.load(f"{config.result_path}/loss.pkl")
        axes = figure.get_axes()[0]
        train = axes.get_lines()[0].get_ydata()
        val = axes.get_lines()[1].get_ydata()
        loss_dict[f"{config.model_type.value} h={config.output_len}"] = (train, val)
    plt.close()
    plt.ion()
    figure, axis = plt.subplots(6, 3)
    figure.tight_layout(pad=2.0)
    figure.set_figheight(20)

    min_val_losses = pd.DataFrame(
        index=["h=1", "h=8", "h=40"], columns=["RNN", "LSTM", "GRU", "TCN", "Transformer", "TFT"]
    )

    for idx, (key, value) in enumerate(loss_dict.items()):
        ax = axis[idx // 3, idx % 3]
        ax.plot(value[0])
        ax.plot(value[1])
        k = key.split(" ")
        min_val_losses[k[0]][k[1]] = min(value[1])
        ax.set_yticks(np.arange(0, 0.005, 0.001))
        ax.set_ylim(0, 0.005)
        ax.set_title(key)

    labels = ["Zbiór uczący", "Zbiór walidacyjny"]
    figure.legend(labels, loc="lower right", ncol=len(labels), prop={"size": 16}, bbox_transform=figure.transFigure)
    figure.savefig(CONST.PATHS.RESULTS + "/general/loss.png", format="png")
    plt.show()


def save_tft_weights(ds):
    encoder_importances = []
    statics_importances = []
    for model_name in ["ModelTypes.tft_out_1", "ModelTypes.tft_out_8", "ModelTypes.tft_out_40"]:
        model = load_model(model_name=model_name, map_location="cpu")
        model.to_cpu()
        # create the explainer and generate explanations
        explainer = TFTExplainer(model, ds.transformed.train, ds.covariates.train)
        results = explainer.explain()
        encoder_importances.append(pd.concat(results.get_encoder_importance()))
        statics_importances.append(pd.concat(results.get_static_covariates_importance()))
    df = pd.DataFrame([x.mean() for x in encoder_importances], index=["h=1", "h=8", "h=40"])
    month = (df["month_sin_pastcov"] + df["month_cos_pastcov"]) / 2
    weekday = (df["weekday_sin_pastcov"] + df["weekday_cos_pastcov"]) / 2
    df = df.drop(
        columns=[
            "month_cos_pastcov",
            "weekday_cos_pastcov",
            "month_sin_pastcov",
            "weekday_sin_pastcov",
            "add_relative_index_futcov",
        ]
    )
    df["Miesiąc"] = month
    df["Dzień tygodnia"] = weekday
    df.to_csv(CONST.PATHS.RESULTS + "/general/tft_weights.csv")


def save_accuracy(ds):
    predicted_end_value = None

    def key(config):
        return f"{config.model_type.value} {config.output_len}"

    df = pd.DataFrame(index=[key(x) for x in MODEL_CONFIGS], columns=CONST.TICKERS)
    for config in MODEL_CONFIGS:
        predictions = load_results(config)
        for idx, series_predictions in enumerate(predictions):
            original_series = ds.original.series[idx]
            trend = []
            predicted_trend = []
            for prediction in series_predictions:
                init_value_index = original_series.get_index_at_point(prediction.start_time()) - 1
                init_value_timestamp = original_series.get_timestamp_at_point(init_value_index)
                original_init_value = original_series[init_value_timestamp].first_value()
                original_end_value = original_series[prediction.end_time()].first_value()
                predicted_end_value = prediction[prediction.end_time()].first_value()
                trend.append(original_init_value < original_end_value)
                predicted_trend.append(original_init_value < predicted_end_value)
            acc = np.sum(np.array(trend) == np.array(predicted_trend)) / len(trend)
            df[CONST.TICKERS[idx]][key(config)] = acc

    df.to_csv(CONST.PATHS.RESULTS + "/general/accuracy.csv")


def main():
    ds = load_datasets()
    plot_losses()
    save_tft_weights(ds)
    save_accuracy(ds)


if __name__ == "__main__":
    main()
