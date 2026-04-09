import pandas as pd
import numpy as np
from prefect import flow, task


@task
def create_series(arr):
    return pd.Series(arr, name="values")


@task
def clean_data(series):
    return series.dropna()


@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }


@flow
def pipeline_flow():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


if __name__ == "__main__":
    pipeline_flow()


# Prefect is more overhead here because the pipeline is tiny and plain functions
# are enough for a small example like this.
#
# Prefect becomes more useful for larger real workflows that need retries,
# logging, scheduling, monitoring, and repeated runs on new data.