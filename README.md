# AlgoChowk Quant Research

Reproducible NIFTY 50 event-study research. The project asks:

> After an unusually large negative daily market move, do subsequent returns differ from the unconditional baseline?

The research workflow is deliberately narrow:

**Define precisely → Test carefully → Question the result → Validate independently → Explain clearly.**

## Research design

The data contain 4,585 daily OHLC observations from `data/raw/nifty50.csv`. The fixed development/OOS split is 2020-01-01.

The primary event is a close-to-close return of at most -3%:

```text
event_return[t] = Close[t] / Close[t-1] - 1
```

Candidate events within five trading observations of a selected event are excluded. The final sample is 53 events: 38 development and 15 OOS.

Forward returns use the next trading day Open as entry and the Close after 1, 3, 5, or 10 trading observations as exit. The baseline uses the identical return construction for all eligible observations. The event study and full-sample analysis are separate:

- Full sample: all eligible daily observations, used to describe return buckets and unconditional behavior.
- Event study: only selected returns <= -3%, used to test abnormal forward returns.
- OOS validation: the held-out 15-event period is the primary replication layer.

## Methodology and assumptions

- No threshold or holding-period optimization is performed after the design is frozen.
- Event classification uses only information available on the event date.
- Trend and volatility regimes use information available through the event date; volatility cutoffs use historical observations rather than full-sample future information.
- Events are non-overlapping under the five-observation rule.
- Bootstrap differences resample event and baseline returns independently using 10,000 draws and fixed seeds. Confidence intervals are uncertainty estimates, not extra observations.
- Regime cells with fewer than 10 events are labelled `descriptive_only` and are not treated as inferential evidence.
- Results are observational and do not establish causality or a profitable strategy.

## Results

At the primary threshold, raw qualifying days number 79; overlap filtering leaves 53 events.

The full-sample comparison uses all 4,585 observations. The event population has 53 observations and the non-event population has 4,532. Full-sample return buckets are reported in `results/return_bucket_results.csv`.

Event-study mean differences are positive at some horizons in both development and OOS, especially around five days, but all bootstrap 95% confidence intervals cross zero. The 10-day OOS difference is negative. The held-out evidence therefore does not robustly confirm a positive abnormal-return effect.

The conclusion is deliberately limited:

> The analysis does not provide statistically robust evidence that large negative daily returns are followed by positive abnormal forward returns relative to the baseline.

This does not prove that no effect exists. It means this dataset and predefined design do not establish one reliably.

## Reproduce the analysis

Use the project virtual environment:

```powershell
& .\.venv\Scripts\python.exe final_pipeline.py
```

The pipeline writes canonical datasets, tables, figures, and diagnostics under `results/`, and updates `docs/final_research_report.md`.

Run the compact notebook from a clean kernel:

```powershell
& .\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute notebooks/01_data_and_event_study.ipynb --output-dir $env:TEMP
```

The notebook presents dataset validation, full-sample forward returns, return buckets, event results, threshold diagnostics, bootstrap results, regime robustness, and the final assessment.

## Code map

- `src/data_loader.py`: loads and normalizes OHLC data.
- `src/events.py`: event detection and forward-return calculations.
- `src/baseline.py`: unconditional baseline returns.
- `src/full_sample.py`: all-observation forward-return and return-bucket analysis.
- `src/regimes.py`: historical volatility and trend regimes.
- `src/statistics.py`: bootstrap inference.
- `src/schemas.py`: canonical schema enforcement.
- `final_pipeline.py`: reproducible final outputs.
- `notebooks/01_data_and_event_study.ipynb`: concise presentation notebook.
- `docs/final_research_report.md`: final research summary.
- `docs/ai_usage_note.md`: AI-use disclosure and reasoning record.

## Limitations

The event sample is small, especially OOS and within regime cells. Forward returns overlap naturally across non-event observations, and bootstrap assumptions may not fully capture time dependence. The analysis does not model transaction costs, investability, causality, or strategy performance. Multiple horizons, thresholds, and exploratory regime partitions were examined, so regime patterns should not be treated as confirmatory findings.
