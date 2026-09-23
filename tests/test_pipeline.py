import numpy as np
import pandas as pd

from src.events import detect_events, calculate_forward_returns
from src.baseline import calculate_baseline_returns
from src.regimes import add_volatility_regime, add_trend_regime
from src.statistics import bootstrap_mean_difference
from src.schemas import CANONICAL_EVENTS, enforce_schema


def prices(values):
    d = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Date": d, "Open": values, "High": np.array(values)+1, "Low": np.array(values)-1, "Close": values})


def test_event_threshold_and_overlap():
    d = prices([100, 96, 95, 90, 89, 88, 80, 81])
    e = detect_events(d, threshold=-0.04, overlap_window=2)
    assert e.Date.tolist() == [d.Date.iloc[1], d.Date.iloc[6]]


def test_forward_returns_and_baseline_use_same_convention():
    d = prices([100, 90, 95, 100, 110])
    e = detect_events(d, threshold=-0.05, overlap_window=5)
    ev = calculate_forward_returns(d, e, (1, 3))
    b = calculate_baseline_returns(d, (1, 3))
    assert ev.loc[0, "entry_price_1"] == 95
    assert np.isclose(ev.loc[0, "forward_return_1"], 0.0)
    assert np.isclose(b.loc[1, "baseline_return_1"], ev.loc[0, "forward_return_1"])


def test_regimes_are_present_and_schema_artifacts_rejected():
    d = prices(np.linspace(100, 130, 80))
    r = add_trend_regime(add_volatility_regime(d, regime_lookback=20))
    assert "trend_regime" in r and "volatility_regime" in r
    with np.testing.assert_raises(ValueError):
        enforce_schema(pd.DataFrame({"Date": [], "foo_x": []}), ["Date"])


def test_bootstrap_difference_matches_means():
    e = np.array([0.01, 0.02, -0.01])
    b = np.array([0.0, 0.01, 0.0])
    result = bootstrap_mean_difference(e, b, n_bootstrap=500)
    assert np.isclose(result["observed_difference"], e.mean() - b.mean())
