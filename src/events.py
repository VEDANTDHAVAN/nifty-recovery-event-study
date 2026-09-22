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
    overlap_window: int = 5,
) -> pd.DataFrame:

    data = df.copy()

    # Ensure chronological ordering
    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Ensure numeric prices
    for col in ["Open", "High", "Low", "Close"]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    # Calculate close-to-close return
    data["event_return"] = (
        data["Close"]
        / data["Close"].shift(1)
        - 1
    )

    # Candidate events
    candidate_indices = data.index[
        data["event_return"] <= threshold
    ].tolist()

    # Remove overlapping events
    selected_indices = []

    last_event_index = -(
        overlap_window + 1
    )

    for idx in candidate_indices:

        if (
            idx - last_event_index
            > overlap_window
        ):
            selected_indices.append(idx)
            last_event_index = idx

    events = data.loc[
        selected_indices
    ].copy()

    return events.reset_index(drop=True)


def calculate_forward_returns(
    df: pd.DataFrame,
    events: pd.DataFrame,
    holding_periods=(1, 3, 5, 10),
) -> pd.DataFrame:

    data = df.copy()
    result = events.copy()

    # -----------------------------
    # Normalize dataframe
    # -----------------------------

    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    for col in ["Open", "High", "Low", "Close"]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    # Explicit NumPy arrays.
    # This removes the pandas Scalar typing issue.
    dates = data["Date"].to_numpy()

    opens = data["Open"].to_numpy(
        dtype=float
    )

    closes = data["Close"].to_numpy(
        dtype=float
    )

    # Map each date to its row position.
    position_map = {
        date: idx
        for idx, date in enumerate(dates)
    }

    result = (
        result
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Make sure event dates have the same type
    result["Date"] = pd.to_datetime(
        result["Date"]
    )

    event_indices = result["Date"].map(
        position_map
    )

    # -----------------------------
    # Forward returns
    # -----------------------------

    for h in holding_periods:

        result[f"entry_date_{h}"] = pd.NaT

        result[f"entry_price_{h}"] = np.nan

        result[f"forward_return_{h}"] = np.nan

        for i, event_idx in enumerate(
            event_indices
        ):
            # Event date not found
            if pd.isna(event_idx):
                continue

            event_idx_int = int(event_idx)

            entry_idx = event_idx_int + 1
            exit_idx = event_idx_int + h

            # Not enough future data
            if (
                entry_idx >= len(data)
                or exit_idx >= len(data)
            ):
                continue

            entry_price = opens[entry_idx]
            exit_price = closes[exit_idx]

            # Invalid/missing price
            if (
                not np.isfinite(entry_price)
                or not np.isfinite(exit_price)
                or entry_price == 0
            ):
                continue

            result.loc[
                i, f"entry_date_{h}",
            ] = dates[entry_idx]

            result.loc[
                i, f"entry_price_{h}",
            ] = entry_price

            result.loc[
                i, f"forward_return_{h}",
            ] = (exit_price / entry_price) - 1

    return result

def run_event_study(
    df: pd.DataFrame,
    threshold: float = -0.03,
    holding_periods: tuple[int, ...] = (1, 3, 5, 10),
    exclude_overlapping: bool = True,
    overlap_window: int = 5,
) -> pd.DataFrame:
    df = df.copy()

    # Ensure chronological order
    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    # Ensure price columns are numeric
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce",
        )

    # Calculate close-to-close daily return
    df["event_return"] = (
        df["Close"] / df["Close"].shift(1) - 1
    )

    # Detect candidate events
    candidate_indices = df.index[
        df["event_return"] <= threshold
    ].tolist()

    # Remove overlapping events
    if exclude_overlapping:
        selected_indices = []

        last_event_index = -(
            overlap_window + 1
        )

        for idx in candidate_indices:
            if (
                idx - last_event_index
                > overlap_window
            ):
                selected_indices.append(idx)
                last_event_index = idx

    else:
        selected_indices = candidate_indices

    events = df.loc[
        selected_indices
    ].copy()

    # Calculate forward returns
    for h in holding_periods:
        entry_position = events.index + 1
        exit_position = events.index + h

        valid = (
            entry_position < len(df)
        ) & (
            exit_position < len(df)
        )

        events[
            f"entry_date_{h}"
        ] = pd.NaT

        events[
            f"entry_price_{h}"
        ] = np.nan

        events[
            f"forward_return_{h}"
        ] = np.nan

        valid_events = events.index[valid]

        valid_entry = entry_position[valid]
        valid_exit = exit_position[valid]

        events.loc[
            valid_events,
            f"entry_date_{h}"
        ] = df.loc[
            valid_entry,
            "Date"
        ].to_numpy()

        events.loc[
            valid_events,
            f"entry_price_{h}"
        ] = df.loc[
            valid_entry,
            "Open"
        ].to_numpy()

        events.loc[
            valid_events,
            f"forward_return_{h}"
        ] = (
            df.loc[
                valid_exit,
                "Close"
            ].to_numpy(dtype=float)
            /
            df.loc[
                valid_entry,
                "Open"
            ].to_numpy(dtype=float)
            - 1
        )

    return events.reset_index(drop=True)