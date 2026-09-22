from __future__ import annotations

import numpy as np
import pandas as pd

def bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    rng = np.random.default_rng(random_state)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means[i] = sample.mean()

    alpha = 1 - confidence

    lower = np.quantile(
        bootstrap_means,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2,
    )

    return {
        "mean": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n": len(values),
    }

def bootstrap_mean_difference_ci(
    event_values: pd.Series | np.ndarray,
    baseline_values: pd.Series | np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    event_values = np.asarray(
        event_values,
        dtype=float,
    )

    baseline_values = np.asarray(
        baseline_values,
        dtype=float,
    )

    event_values = event_values[
        ~np.isnan(event_values)
    ]

    baseline_values = baseline_values[
        ~np.isnan(baseline_values)
    ]

    rng = np.random.default_rng(
        random_state
    )

    differences = np.empty(
        n_bootstrap
    )

    for i in range(n_bootstrap):

        event_sample = rng.choice(
            event_values,
            size=len(event_values),
            replace=True,
        )

        baseline_sample = rng.choice(
            baseline_values,
            size=len(baseline_values),
            replace=True,
        )

        differences[i] = (
            event_sample.mean()
            - baseline_sample.mean()
        )

    alpha = 1 - confidence

    lower = np.quantile(
        differences,
        alpha / 2,
    )

    upper = np.quantile(
        differences,
        1 - alpha / 2,
    )

    return {
        "difference": float(
            event_values.mean()
            - baseline_values.mean()
        ),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }

def bootstrap_mean_difference(
    event_returns: pd.Series | np.ndarray,
    baseline_returns: pd.Series | np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Bootstrap the difference between event and baseline means.

    Difference:
        mean(event) - mean(baseline)

    The event and baseline populations are resampled
    independently with replacement.
    """
    event = np.asarray(
        event_returns,
        dtype=float,
    )

    baseline = np.asarray(
        baseline_returns,
        dtype=float,
    )

    event = event[np.isfinite(event)]
    baseline = baseline[np.isfinite(baseline)]

    if len(event) == 0:
        raise ValueError(
            "event_returns contains no valid observations."
        )

    if len(baseline) == 0:
        raise ValueError(
            "baseline_returns contains no valid observations."
        )

    rng = np.random.default_rng(
        random_state
    )

    observed_difference = (
        np.mean(event)
        - np.mean(baseline)
    )

    bootstrap_differences = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):

        event_sample = rng.choice(
            event,
            size=len(event),
            replace=True,
        )

        baseline_sample = rng.choice(
            baseline,
            size=len(baseline),
            replace=True,
        )

        bootstrap_differences[i] = (
            np.mean(event_sample)
            - np.mean(baseline_sample)
        )

    alpha = 1.0 - confidence

    lower = np.quantile(
        bootstrap_differences,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_differences,
        1 - alpha / 2,
    )

    bootstrap_se = np.std(
        bootstrap_differences,
        ddof=1,
    )

    # Two-sided empirical bootstrap p-value.
    # This tests whether the bootstrap distribution
    # is concentrated around zero.
    p_value = (
        np.mean(
            np.abs(bootstrap_differences)
            >= abs(observed_difference)
        )
    )

    return {
        "event_n": float(len(event)),
        "baseline_n": float(len(baseline)),
        "observed_difference": float(
            observed_difference
        ),
        "bootstrap_se": float(
            bootstrap_se
        ),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(p_value),
    }

def moving_block_bootstrap_mean(
    values: pd.Series | np.ndarray,
    n_bootstrap: int = 10_000,
    block_length: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Moving block bootstrap for a time-ordered series.

    Samples contiguous blocks with replacement
    and reconstructs bootstrap samples of the
    original length.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    n = len(values)

    if n == 0:
        raise ValueError(
            "values contains no valid observations."
        )

    if block_length < 1:
        raise ValueError(
            "block_length must be >= 1."
        )

    if block_length > n:
        raise ValueError(
            "block_length cannot exceed sample size."
        )

    rng = np.random.default_rng(
        random_state
    )

    # Circular blocks allow blocks to wrap around
    # the end of the time series.
    extended = np.concatenate(
        [values, values[:block_length - 1]]
    )

    max_start = n

    n_blocks = int(
        np.ceil(n / block_length)
    )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for b in range(n_bootstrap):

        sample = []

        for _ in range(n_blocks):

            start = rng.integers(
                0,
                max_start,
            )

            block = extended[
                start:start + block_length
            ]

            sample.extend(block)

        sample = np.asarray(
            sample[:n],
            dtype=float,
        )

        bootstrap_means[b] = np.mean(
            sample
        )

    return bootstrap_means

def block_bootstrap_mean_difference(
    event_returns: pd.Series | np.ndarray,
    baseline_returns: pd.Series | np.ndarray,
    n_bootstrap: int = 10_000,
    block_length: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    event = np.asarray(
        event_returns,
        dtype=float,
    )

    baseline = np.asarray(
        baseline_returns,
        dtype=float,
    )

    event = event[np.isfinite(event)]
    baseline = baseline[np.isfinite(baseline)]

    if len(event) == 0:
        raise ValueError(
            "No valid event observations."
        )

    if len(baseline) == 0:
        raise ValueError(
            "No valid baseline observations."
        )

    rng = np.random.default_rng(
        random_state
    )

    observed_difference = (
        event.mean()
        - baseline.mean()
    )

    baseline_bootstrap = (
        moving_block_bootstrap_mean(
            baseline,
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            random_state=random_state,
        )
    )

    # Event observations are resampled independently
    # because the event sample itself is sparse.
    event_bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):

        event_sample = rng.choice(
            event,
            size=len(event),
            replace=True,
        )

        event_bootstrap_means[i] = (
            event_sample.mean()
        )

    differences = (
        event_bootstrap_means
        - baseline_bootstrap
    )

    ci_lower = np.quantile(
        differences,
        0.025,
    )

    ci_upper = np.quantile(
        differences,
        0.975,
    )

    return {
        "event_n": float(len(event)),
        "baseline_n": float(len(baseline)),
        "observed_difference": float(
            observed_difference
        ),
        "bootstrap_se": float(
            np.std(
                differences,
                ddof=1,
            )
        ),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }