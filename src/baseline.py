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