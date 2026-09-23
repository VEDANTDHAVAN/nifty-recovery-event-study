from __future__ import annotations

import pandas as pd

CANONICAL_DF = ["Date", "Open", "High", "Low", "Close", "period", "rolling_volatility", "volatility_regime", "ma_20", "ma_50", "trend_ma", "trend_regime", "trend_volatility_regime"]
CANONICAL_BASELINE = ["Date", "baseline_return_1", "baseline_return_3", "baseline_return_5", "baseline_return_10", "period", "trend_regime", "volatility_regime", "trend_volatility_regime"]
CANONICAL_EVENTS = ["Date", "Open", "High", "Low", "Close", "event_return", "entry_date_1", "entry_price_1", "forward_return_1", "entry_date_3", "entry_price_3", "forward_return_3", "entry_date_5", "entry_price_5", "forward_return_5", "entry_date_10", "entry_price_10", "forward_return_10", "period", "trend_regime", "volatility_regime", "trend_volatility_regime"]

def enforce_schema(frame: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")
    extras = [c for c in frame.columns if c not in expected]
    if any(c.endswith(("_x", "_y")) for c in extras):
        raise ValueError(f"Schema contains merge artifacts: {extras}")
    return frame.loc[:, expected].copy()

def validate_schema(frame: pd.DataFrame, expected: list[str]) -> None:
    if list(frame.columns) != expected:
        raise AssertionError(f"Expected columns {expected}, got {list(frame.columns)}")
