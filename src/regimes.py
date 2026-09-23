from __future__ import annotations

import numpy as np
import pandas as pd

def calculate_rolling_volatility(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Calculate annualized rolling volatility from daily returns.

    Volatility at date t uses returns available through date t.
    """

    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    returns = data["Close"].pct_change()

    volatility = (
        returns
        .rolling(
            window=window,
            min_periods=window,
        )
        .std()
        * np.sqrt(252)
    )

    volatility.index = data.index

    return volatility

def classify_volatility_regime(
    volatility: pd.Series,
    lookback: int = 252,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> pd.Series:
    """
    Classify volatility into low / medium / high regimes.

    Thresholds are calculated using only historical volatility
    observations available before the current observation.
    """

    regimes = pd.Series(
        index=volatility.index,
        dtype="object",
    )

    for i in range(len(volatility)):

        if i < lookback:
            regimes.iloc[i] = np.nan
            continue

        historical = volatility.iloc[
            max(0, i - lookback):i
        ].dropna()

        current = volatility.iloc[i]

        if pd.isna(current) or len(historical) < 50:
            regimes.iloc[i] = np.nan
            continue

        low_threshold = historical.quantile(
            low_quantile
        )

        high_threshold = historical.quantile(
            high_quantile
        )

        if current <= low_threshold:
            regimes.iloc[i] = "low"

        elif current >= high_threshold:
            regimes.iloc[i] = "high"

        else:
            regimes.iloc[i] = "medium"

    return regimes

def add_volatility_regime(
    df: pd.DataFrame,
    volatility_window: int = 20,
    regime_lookback: int = 252,
) -> pd.DataFrame:

    data = df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    data["rolling_volatility"] = (
        calculate_rolling_volatility(
            data,
            window=volatility_window,
        )
    )

    data["volatility_regime"] = (
        classify_volatility_regime(
            data["rolling_volatility"],
            lookback=regime_lookback,
        )
    )

    return data


def add_trend_regime(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    data = df.copy().sort_values("Date").reset_index(drop=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["ma_20"] = data["Close"].rolling(20, min_periods=20).mean()
    data["ma_50"] = data["Close"].rolling(50, min_periods=50).mean()
    data["trend_ma"] = data["Close"].rolling(window, min_periods=window).mean()
    data["trend_regime"] = pd.Series(pd.NA, index=data.index, dtype="string")
    valid = data["trend_ma"].notna()
    data.loc[valid & (data["Close"] >= data["trend_ma"]), "trend_regime"] = "uptrend"
    data.loc[valid & (data["Close"] < data["trend_ma"]), "trend_regime"] = "downtrend"
    data["trend_volatility_regime"] = data["trend_regime"].astype("string") + "_" + data["volatility_regime"].astype("string")
    data.loc[data[["trend_regime", "volatility_regime"]].isna().any(axis=1), "trend_volatility_regime"] = pd.NA
    return data
