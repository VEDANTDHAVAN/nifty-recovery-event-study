import numpy as np
import pandas as pd


def calculate_baseline_returns(
    df: pd.DataFrame,
    holding_periods: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """
    Calculate forward returns for every eligible trading day.

    Entry:
        Next trading day's Open

    Exit:
        Close h trading observations after the reference day
    """
    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    open_prices = pd.to_numeric(
        data["Open"],
        errors="coerce",
    ).to_numpy(dtype=float)

    close_prices = pd.to_numeric(
        data["Close"],
        errors="coerce",
    ).to_numpy(dtype=float)

    dates = data["Date"].to_numpy()

    result = pd.DataFrame({
        "Date": data["Date"],
    })

    event_positions = np.arange(len(data))

    entry_positions = event_positions + 1

    for h in holding_periods:

        exit_positions = event_positions + h

        valid = (
            (entry_positions < len(data)) &
            (exit_positions < len(data))
        )

        returns = np.full(
            len(data),
            np.nan,
            dtype=float,
        )

        returns[valid] = (
            close_prices[exit_positions[valid]]
            / open_prices[entry_positions[valid]]
        ) - 1.0

        result[f"baseline_return_{h}"] = returns

    return result

def create_strict_baseline(
    df: pd.DataFrame, events: pd.DataFrame,
    baseline: pd.DataFrame, window: int = 5,
) -> pd.DataFrame:
    """
    Remove event dates and the following `window`
    trading observations from the baseline population.

    Event:
        t

    Excluded:
        t, t+1, ..., t+window
    """
    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    baseline = baseline.copy()

    baseline["Date"] = pd.to_datetime(
        baseline["Date"]
    )

    # Convert event dates to a set for fast lookup
    event_dates = set(
        pd.to_datetime(events["Date"])
    )

    # Locate event positions in the original dataframe
    event_positions = data.index[
        data["Date"].isin(event_dates)
    ].to_numpy()

    excluded_positions = set()

    for event_position in event_positions:

        for offset in range(
            0,
            window + 1,
        ):

            position = event_position + offset

            if position < len(data):
                excluded_positions.add(position)

    # Convert excluded dataframe positions into dates
    excluded_dates = set(
        data.loc[
            sorted(excluded_positions),
            "Date"
        ]
    )

    strict_baseline = baseline[
        ~baseline["Date"].isin(excluded_dates)
    ].copy()

    strict_baseline = (
        strict_baseline
        .reset_index(drop=True)
    )

    return strict_baseline