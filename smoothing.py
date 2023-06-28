from typing import List
from darts import TimeSeries
from joblib import Parallel, delayed
import numpy as np


ALPHA = 0.5


def smooth_seq(series_seq: List[TimeSeries], alpha=ALPHA):
    return Parallel(n_jobs=-1)(delayed(smooth)(series, alpha) for series in series_seq)


def inverse_smooth_seq(
    inital_value_seq: List[np.number], forecast_transformed_seq: List[TimeSeries], alpha=ALPHA, n_jobs=-1
) -> List[TimeSeries]:
    return Parallel(n_jobs=n_jobs)(
        delayed(inverse_smooth)(inital_value, forecast, alpha)
        for inital_value, forecast in zip(inital_value_seq, forecast_transformed_seq)
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

    return TimeSeries.from_times_and_values(
        series.time_index,
        results,
        freq=series.freq,
        columns=series.components,
        static_covariates=series.static_covariates,
    )


def inverse_smooth(inital_value: np.number, forecast_transformed: TimeSeries, alpha=ALPHA):
    """inverses exponentialSmoothing, original must contain first date of transformed"""
    results = forecast_transformed.values(copy=True)
    values = forecast_transformed.values(copy=False)

    def calculate(transformed_past, transformed_future):
        return 1 / alpha * (transformed_future + (alpha - 1) * transformed_past)

    for idx, [value] in enumerate(values):  # only readonly operations
        if idx == 0:
            results[0, 0] = calculate(inital_value, value)
        else:
            results[idx, 0] = calculate(values[idx - 1, 0], value)

    output = TimeSeries.from_times_and_values(forecast_transformed.time_index, results, freq=forecast_transformed.freq)

    return output


def apply_differencing(series_seq: List[TimeSeries]):
    def process(s: TimeSeries):
        s = s.diff(dropna=False)  # keep length of timeseries
        s.values(copy=False)[0, 0] = 0.0
        return s

    return Parallel(n_jobs=-1)(delayed(process)(s) for s in series_seq)


def apply_pct_change(series_seq: List[TimeSeries]):
    def process(s: TimeSeries):
        df = s.pd_dataframe().pct_change() * 100
        df.iloc[0][0] = 0
        return TimeSeries.from_dataframe(df)

    return Parallel(n_jobs=-1)(delayed(process)(s) for s in series_seq)


def apply_log(series_seq: List[TimeSeries]):
    def process(s: TimeSeries):
        df = s.pd_dataframe()
        df[df.columns[0]] = np.log(df[df.columns[0]] + 1)
        return TimeSeries.from_dataframe(df)

    return Parallel(n_jobs=-1)(delayed(process)(s) for s in series_seq)


def inverse_differencing(initial_values_seq: List[np.number], series_seq: List[TimeSeries], n_jobs=-1):
    def process(transformed: TimeSeries, initial_value: np.number):
        series_values = transformed.values(copy=False)
        series_values[0, 0] = series_values[0, 0] + initial_value
        for idx in range(1, len(series_values)):
            series_values[idx, 0] = series_values[idx, 0] + series_values[idx - 1, 0]
        return transformed

    return Parallel(n_jobs=n_jobs)(
        delayed(process)(transformed, last_value) for (transformed, last_value) in zip(series_seq, initial_values_seq)
    )


def inverse_pct_change(initial_values_seq: List[np.number], series_seq: List[TimeSeries], n_jobs=-1):
    def process(transformed: TimeSeries, initial_value: np.number):
        df_series = (transformed.pd_series() / 100).add(1, fill_value=0).cumprod() * initial_value
        return TimeSeries.from_times_and_values(transformed.time_index, df_series.values, freq=transformed.freq)

    return Parallel(n_jobs=n_jobs)(
        delayed(process)(transformed, last_value) for (transformed, last_value) in zip(series_seq, initial_values_seq)
    )
