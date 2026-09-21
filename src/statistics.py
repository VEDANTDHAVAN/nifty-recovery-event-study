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