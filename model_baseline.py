import pandas as pd
from datasets import SeqDataset, Datasets, DatasetAccesor, DatasetTransformer, load_datasets
from predict import save_predictions
import const as CONST


def get_baseline_predictions(original: SeqDataset):
    predictions = []
    for series, val_series in zip(original.series, original.val):
        from_idx = series.get_index_at_point(val_series[-1].time_index[0])
        predicted = series[from_idx:-1].shift(1)
        predictions.append(predicted)
    return predictions


if __name__ == "__main__":
    model_name = "baseline"
    datasets = load_datasets()
    predictions = get_baseline_predictions(datasets.original)
    save_predictions(predictions, datasets.original.used_tickers, model_name)
