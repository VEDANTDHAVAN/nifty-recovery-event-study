# AI Usage Note

## Tools used

AI assistance was used through the Codex coding agent, together with the project’s local Python environment, pandas, NumPy, Matplotlib, Jupyter, and the repository’s existing source modules. No external app connector or additional dataset was used.

## How the tools were used

The agent inspected the repository, traced event detection, forward returns, baseline construction, bootstrap inference, regimes, schemas, and notebook execution. It then added a separate full-sample analysis, canonical schema checks, regression tests, final plots, result CSVs, and documentation. The local pipeline was executed against the checked-in OHLC data, and the notebook was executed from a clean kernel.

## My decisions

The research scope was kept as an event study rather than expanded into a trading strategy. The -3% threshold, five-observation overlap exclusion, 1/3/5/10-day horizons, development/OOS split, and bootstrap approach were retained. Full-sample analysis was added because it answers a different question from the 53-event study. Regime cells with fewer than 10 events were treated as descriptive only.

## Suggestions changed or rejected

Suggestions that would have increased the event count by loosening the threshold or removing overlap exclusion were rejected. Additional indicators, machine-learning models, optimization, and broad parameter sweeps were also rejected because they would change the research question and increase overfitting risk. The analysis was revised to use historical volatility information rather than full-sample volatility cutoffs.

## Incorrect AI suggestions identified

An early generated pipeline treated the existing notebook’s split-specific means as authoritative without first checking the split labels. Recalculation exposed differences between old notebook summaries and the clean pipeline; the recomputed outputs were used instead. An initial notebook execution also relied on a live `kagglehub` download and an environment-specific working directory, which was removed in favor of the local checked-in dataset. The environment did not contain `pytest`, so direct regression checks were run and the limitation was reported rather than claiming a full test-suite run.

## What was learned

The central methodological lesson is that 4,500 observations and 53 independent extreme events are not contradictory: they belong to different populations and answer different questions. A full-sample return-bucket analysis describes general conditional behavior, while the event study tests a narrow extreme-event hypothesis. Independent OOS validation and uncertainty intervals are more important than attractive development point estimates. Clear schemas and explicit data contracts also prevent merge artifacts from silently changing the analysis.
