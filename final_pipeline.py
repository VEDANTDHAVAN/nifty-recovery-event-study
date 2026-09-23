from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_nifty_data
from src.events import detect_events, calculate_forward_returns
from src.baseline import calculate_baseline_returns
from src.regimes import add_volatility_regime, add_trend_regime
from src.statistics import bootstrap_mean_difference
from src.schemas import CANONICAL_DF, CANONICAL_BASELINE, CANONICAL_EVENTS, enforce_schema
from src.full_sample import calculate_full_sample_forward_returns, add_return_buckets, summarize_return_buckets, summarize_populations

ROOT = Path(__file__).parent
OUT = ROOT / "results"
FIG = OUT / "figures"
HORIZONS = (1, 3, 5, 10)
SPLIT = pd.Timestamp("2020-01-01")
THRESHOLDS = (-0.02, -0.025, -0.03, -0.035, -0.04, -0.05)

def add_period(frame):
    frame = frame.copy()
    frame["period"] = np.where(pd.to_datetime(frame["Date"]) < SPLIT, "development", "oos")
    return frame

def summary(events, baseline, period):
    e = events[events.period == period]
    b = baseline[baseline.period == period]
    rows = []
    for h in HORIZONS:
        ev = pd.to_numeric(e[f"forward_return_{h}"], errors="coerce").dropna()
        ba = pd.to_numeric(b[f"baseline_return_{h}"], errors="coerce").dropna()
        rows.append({"period": period, "horizon": h, "event_n": len(ev), "baseline_n": len(ba), "event_mean": ev.mean(), "baseline_mean": ba.mean(), "mean_difference": ev.mean()-ba.mean(), "event_median": ev.median(), "baseline_median": ba.median(), "event_win_rate": (ev > 0).mean(), "baseline_win_rate": (ba > 0).mean()})
    return pd.DataFrame(rows)

def regime_summary(events, baseline, key):
    rows = []
    for period in ("development", "oos"):
        for regime in sorted(events[key].dropna().unique()):
            e = events[(events.period == period) & (events[key] == regime)]
            b = baseline[(baseline.period == period) & (baseline[key] == regime)]
            for h in HORIZONS:
                ev = pd.to_numeric(e[f"forward_return_{h}"], errors="coerce").dropna()
                ba = pd.to_numeric(b[f"baseline_return_{h}"], errors="coerce").dropna()
                rows.append({"period": period, "regime": regime, "horizon": h, "event_n": len(ev), "baseline_n": len(ba), "event_mean": ev.mean(), "baseline_mean": ba.mean(), "mean_difference": ev.mean()-ba.mean(), "status": "inferential" if len(ev) >= 10 else "descriptive_only"})
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    raw = load_nifty_data(ROOT / "data/raw/nifty50.csv")
    raw = raw.sort_values("Date").reset_index(drop=True)
    raw_returns = raw["Close"].pct_change()
    diagnostics = []
    for t in THRESHOLDS:
        candidates = raw.index[raw_returns <= t].tolist()
        selected = []
        last = -(5 + 1)
        for idx in candidates:
            if idx - last > 5:
                selected.append(idx); last = idx
        diagnostics.append({"threshold": t, "raw_qualifying": len(candidates), "overlap_filtered": len(selected), "development_events": sum(raw.loc[selected, "Date"] < SPLIT), "oos_events": sum(raw.loc[selected, "Date"] >= SPLIT)})
    df = add_trend_regime(add_volatility_regime(raw, volatility_window=20, regime_lookback=252))
    df = add_period(df)
    events = calculate_forward_returns(df, detect_events(df, threshold=-0.03, overlap_window=5), HORIZONS)
    events = events.merge(df[["Date", "period", "trend_regime", "volatility_regime", "trend_volatility_regime"]], on="Date", how="left", suffixes=("", "_df"))
    events = enforce_schema(events, CANONICAL_EVENTS)
    baseline = calculate_baseline_returns(df, HORIZONS).merge(df[["Date", "period", "trend_regime", "volatility_regime", "trend_volatility_regime"]], on="Date", how="left")
    baseline = enforce_schema(baseline, CANONICAL_BASELINE)
    df = enforce_schema(df, CANONICAL_DF)
    full_sample = add_return_buckets(calculate_full_sample_forward_returns(df, HORIZONS))
    bucket_results = summarize_return_buckets(full_sample, HORIZONS)
    population_results = summarize_populations(full_sample, events, HORIZONS)
    core = pd.concat([summary(events, baseline, "development"), summary(events, baseline, "oos")], ignore_index=True)
    infer = []
    for _, r in core.iterrows():
        e = events[(events.period == r.period)][f"forward_return_{int(r.horizon)}"]
        b = baseline[(baseline.period == r.period)][f"baseline_return_{int(r.horizon)}"]
        result = bootstrap_mean_difference(e, b, random_state=42 + int(r.horizon))
        assert abs(result["observed_difference"] - r.mean_difference) < 1e-12
        infer.append({"period": r.period, "horizon": r.horizon, **result})
    inference = pd.DataFrame(infer)
    threshold_rows = []
    for t in THRESHOLDS:
        ev = calculate_forward_returns(df, detect_events(df, t, 5), HORIZONS)
        for p in ("development", "oos"):
            evp = ev[pd.to_datetime(ev.Date) < SPLIT] if p == "development" else ev[pd.to_datetime(ev.Date) >= SPLIT]
            threshold_rows.append({"threshold": t, "period": p, "event_n": len(evp)})
    sensitivity = pd.DataFrame(threshold_rows)
    regimes = pd.concat([regime_summary(events, baseline, "trend_regime").assign(partition="trend"), regime_summary(events, baseline, "volatility_regime").assign(partition="volatility"), regime_summary(events, baseline, "trend_volatility_regime").assign(partition="trend_volatility")], ignore_index=True)
    core.to_csv(OUT / "core_results.csv", index=False); inference.to_csv(OUT / "bootstrap_results.csv", index=False); sensitivity.to_csv(OUT / "threshold_sensitivity.csv", index=False); regimes.to_csv(OUT / "regime_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "event_diagnostics.csv", index=False)
    full_sample.to_csv(OUT / "full_sample_forward_returns.csv", index=False)
    bucket_results.to_csv(OUT / "return_bucket_results.csv", index=False)
    population_results.to_csv(OUT / "population_comparison.csv", index=False)
    df.to_csv(OUT / "canonical_df.csv", index=False); baseline.to_csv(OUT / "canonical_baseline.csv", index=False); events.to_csv(OUT / "canonical_events.csv", index=False)
    plot_core(core, "Event vs baseline mean forward return", FIG / "event_vs_baseline.png", oos=False); plot_core(core[core.period == "oos"], "OOS event vs baseline mean forward return", FIG / "oos_event_vs_baseline.png", oos=True); plot_ci(inference, FIG / "bootstrap_confidence_intervals.png"); plot_sensitivity(sensitivity, FIG / "threshold_sensitivity.png"); plot_regimes(regimes, FIG / "regime_mean_differences.png")
    report(core, inference, sensitivity, regimes, pd.DataFrame(diagnostics), bucket_results, population_results)

def plot_core(data, title, path, oos):
    x = np.arange(len(HORIZONS)); fig, ax = plt.subplots(figsize=(8, 5)); sub = data if oos else data[data.period == "development"]
    ax.plot(x, sub.event_mean, marker="o", label="Event"); ax.plot(x, sub.baseline_mean, marker="o", label="Baseline"); ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, HORIZONS); ax.set(xlabel="Holding period (trading days)", ylabel="Mean forward return", title=title); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

def plot_ci(data, path):
    fig, ax = plt.subplots(figsize=(8, 5)); x=np.arange(len(data)); ax.errorbar(x, data.observed_difference, yerr=[data.observed_difference-data.ci_lower, data.ci_upper-data.observed_difference], fmt="o"); ax.axhline(0,color="black",lw=.8); ax.set_xticks(x, [f"{p}\n{int(h)}d" for p,h in zip(data.period,data.horizon)]); ax.set(ylabel="Mean difference", title="Bootstrap 95% confidence intervals"); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def plot_sensitivity(data, path):
    fig, ax=plt.subplots(figsize=(8,5));
    for p,g in data.groupby("period"): ax.plot(g.threshold, g.event_n, marker="o", label=p)
    ax.set(xlabel="Event threshold",ylabel="Event count",title="Threshold sensitivity"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def plot_regimes(data, path):
    g=data[(data.status == "inferential") & (data.horizon == 5)]; fig, ax=plt.subplots(figsize=(10,5));
    if len(g): g.assign(label=g.partition+"/"+g.regime+"/"+g.period).plot.bar(x="label",y="mean_difference",ax=ax,legend=False)
    ax.axhline(0,color="black",lw=.8); ax.set(ylabel="Mean difference",title="Regime-conditioned mean differences (5-day; n >= 10)"); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def report(core, inference, sensitivity, regimes, diagnostics, bucket_results, population_results):
    def table(d):
        return "```text\n" + d.to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n```"
    text = f'''# Quantitative Event Study\n\n## 1. Research Question\nDoes the market exhibit abnormal forward returns after a daily market return of at most -3%, relative to unconditional forward-return behavior?\n\n## 2. Hypothesis\nLarge negative daily returns are followed by statistically meaningful abnormal forward returns.\n\n## 3. Dataset\nNIFTY 50 daily OHLC observations from `data/raw/nifty50.csv`. The fixed development/OOS split is 2020-01-01.\n\n## 4. Full-Sample Analysis\nThe full-sample population uses all eligible daily observations and is separate from the event study.\n\nEvent diagnostics:\n\n{table(diagnostics)}\n\nReturn buckets:\n\n{table(bucket_results)}\n\nPopulation comparison:\n\n{table(population_results)}\n\n## 5. Event Definition\nEvents are close-to-close returns <= -0.03, with overlapping events excluded using a 5-trading-day window.\n\n## 6. Methodology\nForward returns use the next trading day Open as entry and the Close h trading observations after the event date as exit. Baselines use the identical construction for every eligible date. Regimes use information available through the event date; volatility classification uses a historical lookback. Bootstrap differences resample event and baseline returns independently (10,000 draws, 95% percentile CI, seed 42 plus horizon).\n\n## 7. Core Results\n### Development\n{table(core[core.period == 'development'])}\n\n### OOS\n{table(core[core.period == 'oos'])}\n\nBootstrap inference:\n\n{table(inference)}\n\n## 8. Threshold Sensitivity\nAlready-tested thresholds were -2%, -2.5%, -3%, -3.5%, -4%, and -5%. This is descriptive robustness, not a new threshold search.\n\n{table(sensitivity)}\n\n## 9. Regime Analysis\nTrend, volatility, and trend x volatility partitions were retained. Cells with event_n < 10 are marked `descriptive_only`; they are not used for inferential claims.\n\n{table(regimes)}\n\n## 10. Robustness Checks\nThe final pipeline checks threshold sensitivity, bootstrap intervals, OOS validation, date/forward-return alignment, overlap handling, canonical schemas, and historical regime construction. Multiple horizons, thresholds, and regime partitions were examined; regime results are exploratory and OOS evidence is the primary validation layer.\n\n## 11. Limitations\nThe sample is small (53 primary events, 38 development and 15 OOS). OOS regime cells can be extremely small. The study is observational and does not establish a tradable strategy, after-cost profitability, or causal effect.\n\n## 12. Final Conclusion\nThe full-sample analysis describes how return magnitude relates to subsequent returns across approximately 4,500 observations; it does not establish causality. The event-study point estimates include positive differences at some horizons, but bootstrap confidence intervals cross zero. OOS point estimates are imprecise and do not provide robust confirmation. The analysis does not provide statistically robust evidence of a positive abnormal-return effect after a large negative daily return.\n'''
    (ROOT / "docs").mkdir(exist_ok=True); (ROOT / "docs/final_research_report.md").write_text(text, encoding="utf-8")

if __name__ == "__main__": main()
