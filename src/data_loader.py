from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close"]


def load_nifty_data(path: str | Path) -> pd.DataFrame:
    """
    Load and normalize NIFTY 50 daily OHLC data.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [
        str(column).strip().title()
        for column in df.columns
    ]

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Parse date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Convert OHLC to numeric
    for column in ["Open", "High", "Low", "Close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df