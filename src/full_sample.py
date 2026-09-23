from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_full_sample_forward_returns(
    df: pd.DataFrame,
    holding_periods: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Return forward returns for every eligible date; no event filter is applied."""
    data = df.copy().sort_values("Date").reset_index(drop=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data["return_t"] = pd.to_numeric(data["Close"], errors="coerce").pct_change()
    opens = pd.to_numeric(data["Open"], errors="coerce").to_numpy(float)
    closes = pd.to_numeric(data["Close"], errors="coerce").to_numpy(float)
    for h in holding_periods:
        entry = np.arange(len(data)) + 1
        exit_ = np.arange(len(data)) + h
        valid = (entry < len(data)) & (exit_ < len(data))
        values = np.full(len(data), np.nan)
        values[valid] = closes[exit_[valid]] / opens[entry[valid]] - 1
        data[f"forward_return_{h}"] = values
    return data


def add_return_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    bins = [-np.inf, -0.03, -0.02, -0.01, 0.0, np.inf]
    labels = ["return <= -3%", "-3% < return <= -2%", "-2% < return <= -1%", "-1% < return <= 0%", "return > 0%"]
    result["return_bucket"] = pd.cut(result["return_t"], bins=bins, labels=labels, right=True)
    return result


def summarize_return_buckets(frame: pd.DataFrame, holding_periods=(1, 3, 5, 10)) -> pd.DataFrame:
    rows = []
    for bucket, group in frame.dropna(subset=["return_bucket"]).groupby("return_bucket", observed=False):
        row = {"return_bucket": str(bucket), "n": len(group), "mean_return": group["return_t"].mean()}
        for h in holding_periods:
            values = pd.to_numeric(group[f"forward_return_{h}"], errors="coerce").dropna()
            row[f"mean_forward_return_{h}"] = values.mean()
            row[f"median_forward_return_{h}"] = values.median()
            row[f"win_rate_{h}"] = (values > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_populations(full_sample, events, holding_periods=(1, 3, 5, 10)) -> pd.DataFrame:
    event_dates = set(pd.to_datetime(events["Date"]))
    populations = {
        "All observations": full_sample,
        "Extreme negative events <= -3%": full_sample[full_sample["Date"].isin(event_dates)],
        "Non-event observations": full_sample[~full_sample["Date"].isin(event_dates)],
    }
    rows = []
    for name, group in populations.items():
        row = {"population": name, "n": len(group), "mean_return": group["return_t"].mean()}
        for h in holding_periods:
            row[f"mean_forward_return_{h}"] = group[f"forward_return_{h}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)
