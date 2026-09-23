# Quantitative Event Study — Final Summary

## 1. Research question and hypothesis

Following an unusually large negative daily NIFTY 50 return, do subsequent returns differ from the unconditional forward-return baseline? The hypothesis is that extreme negative-return events are followed by a statistically meaningful abnormal forward return.

## 2. Data and design

The checked-in dataset contains 4,585 daily OHLC observations. The development/OOS split is fixed at 2020-01-01. The primary event is a close-to-close return <= -3%. Events within five trading observations of a selected event are excluded. There are 79 raw qualifying days and 53 independent events after filtering: 38 development and 15 OOS.

Forward returns use the next trading day Open as entry and the Close after 1, 3, 5, or 10 trading observations as exit. The baseline uses the same construction for every eligible date. The full-sample analysis uses all eligible observations and is intentionally separate from the 53-event study.

## 3. Full-sample evidence

The full-sample population contains 4,585 rows. The extreme-event population contains 53 rows and the non-event population contains 4,532 rows. Full-sample return buckets are available in `results/return_bucket_results.csv`; the population comparison is in `results/population_comparison.csv`.

The full-sample analysis is descriptive: it shows how current-return magnitude/sign relates to later returns across the entire dataset. It does not establish causality and must not be interpreted as having 4,585 independent extreme events.

## 4. Event-study results

| Period | Horizon | Events | Baseline n | Event mean | Baseline mean | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Development | 1 | 38 | 3001 | -0.2715% | -0.0484% | -0.2231% |
| Development | 3 | 38 | 3001 | 0.1889% | 0.0370% | 0.1519% |
| Development | 5 | 38 | 3001 | 0.8975% | 0.1171% | 0.7805% |
| Development | 10 | 38 | 3001 | 0.4538% | 0.3103% | 0.1435% |
| OOS | 1 | 15 | 1583 | 0.5853% | -0.0529% | 0.6382% |
| OOS | 3 | 15 | 1581 | 0.1316% | 0.0425% | 0.0891% |
| OOS | 5 | 15 | 1579 | 0.3491% | 0.1432% | 0.2059% |
| OOS | 10 | 15 | 1574 | 0.0207% | 0.3890% | -0.3683% |

All bootstrap 95% confidence intervals for the event-minus-baseline difference cross zero. The OOS sample is especially imprecise. Thus, positive point estimates do not constitute robust evidence of a positive abnormal-return effect.

## 5. Robustness and leakage checks

Threshold sensitivity was retained at -2%, -2.5%, -3%, -3.5%, -4%, and -5%; it was not used to select a new primary threshold. Trend and volatility regimes use information available on or before the event date. Regime cells with fewer than 10 events are labelled `descriptive_only`. Canonical schemas reject missing columns and `_x`/`_y` merge artifacts. Forward-return alignment, overlap exclusion, split assignment, threshold detection, and bootstrap observed-difference consistency were regression-tested.

## 6. Limitations

The event sample is small, with only 15 OOS events and some smaller regime cells. Returns across the full sample can overlap in calendar time, and independent bootstrap resampling may not fully capture all serial dependence. The analysis is observational; it does not establish causality, after-cost profitability, investability, or a trading strategy. Multiple horizons, thresholds, and exploratory regimes were examined, so regime patterns should not be treated as confirmatory.

## 7. Conclusion

The full-sample analysis correctly uses the approximately 4,500 daily observations to describe general conditional forward-return behavior. The separate event study correctly uses only the 53 independent <= -3% events. Development contains some positive point estimates, especially at five days, but uncertainty intervals cross zero. The OOS results do not provide robust replication. The analysis does not provide statistically robust evidence that large negative daily market returns are followed by positive abnormal forward returns.
