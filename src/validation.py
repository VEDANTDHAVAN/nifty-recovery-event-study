import pandas as pd

def validate_dates(df: pd.DataFrame) -> dict:
    dates = df["Date"]

    return {
        "missing_dates": int(dates.isna().sum()),
        "duplicate_dates": int(dates.duplicated().sum()),
        "sorted": bool(dates.is_monotonic_increasing),
        "min_date": dates.min(),
        "max_date": dates.max(),
    }


def validate_ohlc(df: pd.DataFrame) -> dict:
    ohlc = ["Open", "High", "Low", "Close"]

    missing_values = {
        column: int(df[column].isna().sum())
        for column in ohlc
    }

    non_positive = {
        column: int((df[column] <= 0).sum())
        for column in ohlc
    }

    invalid_high_low = int(
        (df["High"] < df["Low"]).sum()
    )

    invalid_open_range = int(
        (
            (df["Open"] > df["High"]) |
            (df["Open"] < df["Low"])
        ).sum()
    )

    invalid_close_range = int(
        (
            (df["Close"] > df["High"]) |
            (df["Close"] < df["Low"])
        ).sum()
    )

    return {
        "missing_values": missing_values,
        "non_positive_values": non_positive,
        "high_less_than_low": invalid_high_low,
        "open_outside_range": invalid_open_range,
        "close_outside_range": invalid_close_range,
    }


def validate_data(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "date_validation": validate_dates(df),
        "ohlc_validation": validate_ohlc(df),
    }

def identify_extreme_returns(
    df: pd.DataFrame,
    threshold: float = 0.05
) -> pd.DataFrame:
    result = df.copy()

    result["DailyReturn"] = (
        result["Close"]
        .pct_change()
    )

    extreme = result[
        result["DailyReturn"].abs() >= threshold
    ].copy()

    return extreme[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "DailyReturn",
        ]
    ]