import joblib
import pandas as pd
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
import const as CONST

baseline_config = CONST.ModelConfig(CONST.ModelTypes.rnn, output_len=1, model_name="baseline", hidden_state=0)


def get_baseline_predictions(original: SeqDataset):
    predictions = []
    for series, val_series in zip(original.series, original.val):
        from_idx = series.get_index_at_point(val_series[-1].time_index[0])
        predicted = series[from_idx:-1].shift(1)
        series_predictions = []
        for idx in range(len(predicted)):
            series_predictions.append(predicted[idx][0])
        predictions.append(series_predictions)
    return predictions


if __name__ == "__main__":
    model_name = "baseline"
    datasets = load_datasets()
    predictions = get_baseline_predictions(datasets.original)
    joblib.dump(predictions, f"{CONST.PATHS.RESULTS}/{model_name}/{model_name}.pkl")
