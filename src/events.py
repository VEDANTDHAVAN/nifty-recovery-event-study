import numpy as np
import pandas as pd

def calculate_daily_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate close-to-close returns."""
    result = df.copy()

    result["Close"] = pd.to_numeric(
        result["Close"], errors="coerce",
    )

    result["event_return"] = result["Close"].pct_change()

    return result


def detect_events(
    df: pd.DataFrame,
    threshold: float = -0.03,
    exclude_overlapping: bool = True,
    overlap_window: int = 5,
) -> pd.DataFrame:
    candidate_events = df[
        df["event_return"] <= threshold
    ].copy()

    data = calculate_daily_returns(df)

    event_indices = data.index[
        data["event_return"] <= threshold
    ].tolist()

    if not exclude_overlapping:
        return data.loc[event_indices].copy()

    selected = []
    last_event_index = -np.inf

    for index in event_indices:
        if index > last_event_index + overlap_window:
            selected.append(index)
            last_event_index = index

    return data.loc[selected].copy()


def calculate_forward_returns(
    df: pd.DataFrame,
    events: pd.DataFrame,
    holding_periods: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:

    result = events.copy()

    # Explicitly guarantee numeric OHLC columns
    open_prices = pd.to_numeric(
        df["Open"],
        errors="coerce",
    ).to_numpy(dtype=float)

    close_prices = pd.to_numeric(
        df["Close"],
        errors="coerce",
    ).to_numpy(dtype=float)

    dates = df["Date"].to_numpy()

    event_positions = result.index.to_numpy(dtype=int)

    # All trades enter at the next trading day's Open
    entry_positions = event_positions + 1

    valid_entry = entry_positions < len(df)

    result["entry_date"] = pd.NaT
    result["entry_price"] = np.nan

    valid_events = event_positions[valid_entry]
    valid_entry_positions = entry_positions[valid_entry]

    result.loc[
        valid_events,
        "entry_date"
    ] = dates[valid_entry_positions]

    result.loc[
        valid_events,
        "entry_price"
    ] = open_prices[valid_entry_positions]

    # Calculate each holding period
    for h in holding_periods:

        exit_positions = event_positions + h

        valid = (
            (entry_positions < len(df)) &
            (exit_positions < len(df))
        )

        valid_events = event_positions[valid]
        valid_entry_positions = entry_positions[valid]
        valid_exit_positions = exit_positions[valid]

        result[f"exit_date_{h}"] = pd.NaT
        result[f"exit_price_{h}"] = np.nan
        result[f"forward_return_{h}"] = np.nan

        result.loc[
            valid_events,
            f"exit_date_{h}"
        ] = dates[valid_exit_positions]

        result.loc[
            valid_events,
            f"exit_price_{h}"
        ] = close_prices[valid_exit_positions]

        forward_returns = (
            close_prices[valid_exit_positions]
            / open_prices[valid_entry_positions]
        ) - 1.0

        result.loc[
            valid_events,
            f"forward_return_{h}"
        ] = forward_returns

    return result

def run_event_study(
    df: pd.DataFrame,
    threshold: float = -0.03,
    holding_periods: tuple[int, ...] = (1, 3, 5, 10),
    exclude_overlapping: bool = True,
    overlap_window: int = 5,
) -> pd.DataFrame:

    events = detect_events(
        df=df,
        threshold=threshold,
        exclude_overlapping=exclude_overlapping,
        overlap_window=overlap_window,
    )

    return calculate_forward_returns(
        df=df,
        events=events,
        holding_periods=holding_periods,
    )