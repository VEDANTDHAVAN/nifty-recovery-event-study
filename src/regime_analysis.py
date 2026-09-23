from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_regime_analysis(
    events: pd.DataFrame, baseline: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    min_events: int = 5,
) -> pd.DataFrame:
    results = []

    valid_events = events[
        events["volatility_regime"].notna()
    ].copy()

    valid_baseline = baseline[
        baseline["volatility_regime"].notna()
    ].copy()

    for period in ["development", "oos"]:

        period_events = valid_events[
            valid_events["period"] == period
        ]

        period_baseline = valid_baseline[
            valid_baseline["period"] == period
        ]

        for regime in [
            "low", "medium", "high",
        ]:
            regime_events = period_events[
                period_events["volatility_regime"] == regime
            ]

            regime_baseline = period_baseline[
                period_baseline["volatility_regime"] == regime
            ]

            event_n = len(regime_events)
            baseline_n = len(regime_baseline)

            for h in horizons:
                event_col = (f"forward_return_{h}")

                baseline_col = (f"baseline_return_{h}")

                event_returns = pd.to_numeric(
                    regime_events[event_col],
                    errors="coerce",
                ).dropna()

                baseline_returns = pd.to_numeric(
                    regime_baseline[baseline_col],
                    errors="coerce",
                ).dropna()

                valid_event_n = len(event_returns)

                valid_baseline_n = len(baseline_returns)

                if valid_event_n == 0:
                    event_mean = np.nan
                    event_median = np.nan
                    event_win_rate = np.nan
                else:
                    event_mean = (event_returns.mean())

                    event_median = (event_returns.median())

                    event_win_rate = (event_returns.gt(0).mean())

                if valid_baseline_n == 0:
                    baseline_mean = np.nan
                    baseline_median = np.nan
                    baseline_win_rate = np.nan
                else:
                    baseline_mean = (baseline_returns.mean())

                    baseline_median = (baseline_returns.median())

                    baseline_win_rate = (baseline_returns.gt(0).mean())

                if (
                    pd.notna(event_mean)
                    and pd.notna(baseline_mean)
                ):
                    difference = (
                        event_mean - baseline_mean
                    )
                else:
                    difference = np.nan

                results.append({
                    "period": period,
                    "regime": regime,
                    "horizon": h,
                    "event_n": event_n,
                    "valid_event_n": valid_event_n,
                    "baseline_n": baseline_n,
                    "valid_baseline_n": valid_baseline_n,
                    "event_mean": event_mean,
                    "event_median": event_median,
                    "event_win_rate": event_win_rate,
                    "baseline_mean": baseline_mean,
                    "baseline_median": baseline_median,
                    "baseline_win_rate": baseline_win_rate,
                    "mean_difference": difference,
                    "sufficient_events": valid_event_n >= min_events,
                })

    return pd.DataFrame(results)

def calculate_regime_bootstrap(
    events: pd.DataFrame, baseline: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    regimes: tuple[str, ...] = (
        "low", "medium", "high",
    ), min_events: int = 5, n_bootstrap: int = 10_000,
    block_length: int = 5, random_state: int = 42,
) -> pd.DataFrame:
    from src.statistics import (
        block_bootstrap_mean_difference,
    )

    results = []

    valid_events = events[
        events["volatility_regime"].notna()
    ].copy()

    valid_baseline = baseline[
        baseline["volatility_regime"].notna()
    ].copy()

    for period in ["development", "oos"]:
        period_events = valid_events[
            valid_events["period"] == period
        ]

        period_baseline = valid_baseline[
            valid_baseline["period"] == period
        ]

        for regime in regimes:
            regime_events = period_events[
                period_events[
                    "volatility_regime"
                ] == regime
            ]

            regime_baseline = period_baseline[
                period_baseline[
                    "volatility_regime"
                ] == regime
            ]

            for h in horizons:
                event_returns = pd.to_numeric(
                    regime_events[
                        f"forward_return_{h}"
                    ],
                    errors="coerce",
                ).dropna()

                baseline_returns = pd.to_numeric(
                    regime_baseline[
                        f"baseline_return_{h}"
                    ],
                    errors="coerce",
                ).dropna()

                event_n = len(event_returns)

                if event_n < min_events:
                    results.append({
                        "period": period,
                        "regime": regime,
                        "horizon": h,
                        "event_n": event_n,
                        "baseline_n":
                            len(baseline_returns),
                        "sufficient_events": False,
                        "observed_difference": np.nan,
                        "bootstrap_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                    })

                    continue

                result = (
                    block_bootstrap_mean_difference(
                        event_returns=event_returns,
                        baseline_returns=baseline_returns,
                        n_bootstrap=n_bootstrap,
                        block_length=block_length,
                        random_state=random_state,
                    )
                )

                results.append({
                    "period": period,
                    "regime": regime,
                    "horizon": h,
                    "event_n": int(event_n),
                    "baseline_n": int(len(baseline_returns)),
                    "sufficient_events": True,
                    "observed_difference": result["observed_difference"],
                    "bootstrap_se": result["bootstrap_se"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                })

    return pd.DataFrame(results)