from typing import List
from darts import TimeSeries
from joblib import Parallel, delayed


ALPHA = 0.5


def smooth_seq(series_seq: List[TimeSeries], alpha=ALPHA):
    return Parallel(n_jobs=-1)(delayed(smooth)(series, alpha) for series in series_seq)


def inverse_smooth_seq(
    transformed_past_seq: List[TimeSeries], forecast_transformed_seq: List[TimeSeries], alpha=ALPHA, n_jobs=-1
) -> List[TimeSeries]:
    return Parallel(n_jobs=n_jobs)(
        delayed(inverse_smooth)(past, forecast, alpha)
        for past, forecast in zip(transformed_past_seq, forecast_transformed_seq)
    )


def smooth(series: TimeSeries, alpha=ALPHA):
    """Executes exponential smoothing for univariate timeseries"""
    values = series.values(copy=False)
    results = series.values(copy=True)
    for idx, [value] in enumerate(values):  # only readonly operations
        if idx == 0:
            results[0, 0] = value
        else:
            results[idx, 0] = alpha * value + (1 - alpha) * results[idx - 1, 0]

    return TimeSeries.from_times_and_values(series.time_index, results, freq=series.freq, columns=series.components)


def inverse_smooth(transformed_past: TimeSeries, forecast_transformed: TimeSeries, alpha=ALPHA):
    """inverses exponentialSmoothing, original must contain first date of transformed"""
    first_value = transformed_past.last_value()
    results = forecast_transformed.values(copy=True)
    values = forecast_transformed.values(copy=False)

    def calculate(transformed_past, transformed_future):
        return 1 / alpha * (transformed_future + (alpha - 1) * transformed_past)

    for idx, [value] in enumerate(values):  # only readonly operations
        if idx == 0:
            results[0, 0] = calculate(first_value, value)
        else:
            results[idx, 0] = calculate(values[idx - 1, 0], value)

    return TimeSeries.from_times_and_values(forecast_transformed.time_index, results, freq=forecast_transformed.freq)
