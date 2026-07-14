#!/usr/bin/env python3
"""
Statistical diagnostics for rolling 10-day OLS beta (PAAS on SLV).

Returns:
  r_t = ln(P_t / P_{t-1})   from last RTH print each session

Beta:
  β_t = Cov(r_PAAS, r_SLV) / Var(r_SLV)  over the trailing ``lookback`` sessions
        ending at t (inclusive). Needs ``lookback`` valid return pairs.

Outputs:
  - Time series of β_t
  - QQ-plot vs Normal
  - Histogram + KDE
  - Shapiro–Wilk normality test
  - Nyblom L_C constancy test (on β level)
  - CUSUM of demeaned β
  - ACF of β (and Δβ)

Usage::

    python analyze_beta_stats.py --start 20240101 --end 20251231
    python analyze_beta_stats.py --lookback 10 --ticker-a PAAS --ticker-b SLV
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as sm_acf

from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import load_pair_panel


# Nyblom (1989) asymptotic 5% / 1% critical values for L_C (mean constancy)
NYBLOM_CV_05 = 0.470
NYBLOM_CV_01 = 0.748


def daily_closes(panel: pd.DataFrame, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    """Last RTH print per trading date."""
    closes = (
        panel.groupby("Trading date")[[ticker_a, ticker_b]]
        .last()
        .sort_index()
    )
    closes = closes[(closes[ticker_a] > 0) & (closes[ticker_b] > 0)].copy()
    closes.index = closes.index.astype(int)
    return closes


def daily_closes_yahoo(ticker_a: str, ticker_b: str, start: int, end: int) -> pd.DataFrame:
    """Daily adjusted closes via yfinance (fallback when local minute folder is unavailable)."""
    import yfinance as yf

    start_s = pd.Timestamp(str(start)).strftime("%Y-%m-%d")
    end_s = (pd.Timestamp(str(end)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        [ticker_a, ticker_b],
        start=start_s,
        end=end_s,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker_a}/{ticker_b}")

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw[["Close"]].copy()
        closes.columns = [ticker_a]

    closes = closes[[ticker_a, ticker_b]].dropna()
    closes.index = closes.index.tz_localize(None) if closes.index.tz is not None else closes.index
    out = closes.copy()
    out.index = out.index.strftime("%Y%m%d").astype(int)
    out.index.name = "Trading date"
    return out


def load_closes(
    ticker_a: str,
    ticker_b: str,
    start: int,
    end: int,
    folder: str | None,
    source: str,
) -> tuple[pd.DataFrame, str]:
    """Load daily closes from local minute panel or Yahoo."""
    if source == "yahoo":
        return daily_closes_yahoo(ticker_a, ticker_b, start, end), "yahoo"

    if source == "local":
        panel = load_pair_panel(ticker_a, ticker_b, folder, start, end)
        return daily_closes(panel, ticker_a, ticker_b), "local_rth_last"

    # auto: try local, then yahoo
    try:
        panel = load_pair_panel(ticker_a, ticker_b, folder, start, end)
        return daily_closes(panel, ticker_a, ticker_b), "local_rth_last"
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Local data unavailable ({exc}); falling back to Yahoo Finance daily closes.")
        return daily_closes_yahoo(ticker_a, ticker_b, start, end), "yahoo"


def log_returns(closes: pd.DataFrame, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    """r_t = ln(P_t / P_{t-1}) for each ticker."""
    out = pd.DataFrame(index=closes.index)
    out[f"logret_{ticker_a}"] = np.log(closes[ticker_a] / closes[ticker_a].shift(1))
    out[f"logret_{ticker_b}"] = np.log(closes[ticker_b] / closes[ticker_b].shift(1))
    out[ticker_a] = closes[ticker_a]
    out[ticker_b] = closes[ticker_b]
    return out.dropna()


def rolling_ols_beta(
    rets: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookback: int,
) -> pd.Series:
    """
    Rolling OLS β of A on B using the last ``lookback`` log-return pairs ending at t.
    """
    ya = rets[f"logret_{ticker_a}"].to_numpy(dtype=float)
    xb = rets[f"logret_{ticker_b}"].to_numpy(dtype=float)
    dates = rets.index.to_numpy()
    betas = np.full(len(rets), np.nan)

    for i in range(lookback - 1, len(rets)):
        y = ya[i - lookback + 1 : i + 1]
        x = xb[i - lookback + 1 : i + 1]
        xx = float(np.dot(x, x))
        if xx <= 0.0 or not np.isfinite(xx):
            continue
        betas[i] = float(np.dot(x, y) / xx)

    return pd.Series(betas, index=dates, name=f"beta_{lookback}d").dropna()


def shapiro_wilk(x: np.ndarray) -> dict:
    # SciPy caps at 5000; subsample if needed
    sample = x if len(x) <= 5000 else np.random.default_rng(42).choice(x, 5000, replace=False)
    stat, p = stats.shapiro(sample)
    return {
        "test": "Shapiro–Wilk",
        "statistic": float(stat),
        "pvalue": float(p),
        "n": int(len(sample)),
        "reject_H0_normal_5pct": bool(p < 0.05),
        "note": "H0: sample is Normal",
    }


def nyblom_lc(x: np.ndarray) -> dict:
    """
    Nyblom L_C test for constancy of the mean of x.

    L_C = T^{-2} σ̂^{-2} Σ_t S_t² ,  S_t = Σ_{i≤t} (x_i − x̄)
    Reject constancy if L_C > critical value.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 10:
        return {
            "test": "Nyblom L_C",
            "statistic": float("nan"),
            "pvalue": None,
            "n": t,
            "reject_H0_constancy_5pct": None,
            "note": "Too few observations",
        }

    mean = x.mean()
    s = np.cumsum(x - mean)
    sigma2 = float(np.sum((x - mean) ** 2) / t)
    if sigma2 <= 0:
        lc = float("nan")
    else:
        lc = float(np.sum(s ** 2) / (t**2 * sigma2))

    return {
        "test": "Nyblom L_C (mean constancy)",
        "statistic": lc,
        "cv_5pct": NYBLOM_CV_05,
        "cv_1pct": NYBLOM_CV_01,
        "pvalue": None,
        "n": t,
        "reject_H0_constancy_5pct": bool(lc > NYBLOM_CV_05) if np.isfinite(lc) else None,
        "note": "H0: mean of β is constant over time. Reject if L_C > 0.47 (5%).",
    }


def cusum_path(x: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Standardized CUSUM of demeaned series:
      W_t = (1 / (σ √T)) Σ_{i≤t} (x_i − x̄)
    Returns path and approx 5% Brownian-bridge band (±0.948).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    mean = x.mean()
    sigma = float(np.std(x, ddof=1))
    if sigma <= 0 or t < 2:
        return np.zeros(t), 0.948
    w = np.cumsum(x - mean) / (sigma * np.sqrt(t))
    return w, 0.948


def acf_stats(x: np.ndarray, nlags: int = 20) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < nlags + 5:
        nlags = max(1, len(x) // 4)
    acf_vals = sm_acf(x, nlags=nlags, fft=True)
    lb = acorr_ljungbox(x, lags=[min(10, nlags)], return_df=True)
    return {
        "lags": list(range(nlags + 1)),
        "acf": [float(v) for v in acf_vals],
        "ljung_box_stat": float(lb["lb_stat"].iloc[-1]),
        "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[-1]),
        "reject_H0_white_noise_5pct": bool(lb["lb_pvalue"].iloc[-1] < 0.05),
        "note": "Ljung–Box H0: no autocorrelation up to lag 10",
    }


def summary_stats(beta: pd.Series) -> pd.DataFrame:
    x = beta.to_numpy(dtype=float)
    rows = [
        ("N", len(x)),
        ("Mean", float(np.mean(x))),
        ("Std", float(np.std(x, ddof=1))),
        ("Min", float(np.min(x))),
        ("P25", float(np.percentile(x, 25))),
        ("Median", float(np.median(x))),
        ("P75", float(np.percentile(x, 75))),
        ("Max", float(np.max(x))),
        ("Skew", float(stats.skew(x))),
        ("Excess kurtosis", float(stats.kurtosis(x))),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_report_html(
    beta: pd.Series,
    rets: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookback: int,
    descriptive: pd.DataFrame,
    sw: dict,
    nyb: dict,
    acf_beta: dict,
    acf_dbeta: dict,
    source: str,
) -> str:
    x = beta.to_numpy(dtype=float)
    dates = pd.to_datetime(beta.index.astype(str), format="%Y%m%d")
    cusum, band = cusum_path(x)

    # Theoretical normal QQ
    osm, osr = stats.probplot(x, dist="norm")
    # KDE
    kde_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    kde = stats.gaussian_kde(x)
    kde_y = kde(kde_x)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            f"β_{lookback}d time series ({ticker_a} on {ticker_b})",
            "QQ-plot vs Normal",
            "Histogram + KDE",
            "CUSUM of demeaned β (±0.948 ≈ 5% band)",
            f"ACF of β_{lookback}d",
            "ACF of Δβ",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=dates, y=x, mode="lines", name="β_t", line=dict(width=1.5)),
        row=1,
        col=1,
    )
    fig.add_hline(y=float(np.mean(x)), line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=osm[0], y=osm[1], mode="markers", name="Sample", marker=dict(size=5)),
        row=1,
        col=2,
    )
    # Reference line
    qq_fit = np.polyfit(osm[0], osm[1], 1)
    qq_line = qq_fit[0] * osm[0] + qq_fit[1]
    fig.add_trace(
        go.Scatter(x=osm[0], y=qq_line, mode="lines", name="Normal ref", line=dict(color="red")),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Histogram(x=x, nbinsx=30, histnorm="probability density", name="Hist", opacity=0.55),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=kde_x, y=kde_y, mode="lines", name="KDE", line=dict(color="crimson", width=2)),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=dates, y=cusum, mode="lines", name="CUSUM", line=dict(width=1.5)),
        row=2,
        col=2,
    )
    fig.add_hline(y=band, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-band, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=2, col=2)

    lags_b = acf_beta["lags"]
    fig.add_trace(
        go.Bar(x=lags_b, y=acf_beta["acf"], name="ACF(β)", marker_color="steelblue"),
        row=3,
        col=1,
    )
    # approx 95% band
    conf = 1.96 / np.sqrt(len(x))
    fig.add_hline(y=conf, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=3, col=1)

    lags_d = acf_dbeta["lags"]
    fig.add_trace(
        go.Bar(x=lags_d, y=acf_dbeta["acf"], name="ACF(Δβ)", marker_color="darkorange"),
        row=3,
        col=2,
    )
    conf_d = 1.96 / np.sqrt(max(len(x) - 1, 1))
    fig.add_hline(y=conf_d, line_dash="dash", line_color="red", row=3, col=2)
    fig.add_hline(y=-conf_d, line_dash="dash", line_color="red", row=3, col=2)

    fig.update_layout(
        height=1100,
        width=1200,
        title_text=(
            f"β_{lookback}d diagnostics · {ticker_a}/{ticker_b} · "
            f"r=ln(P_t/P_{{t-1}}) · N={len(x)}"
        ),
        showlegend=False,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="β", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample quantiles", row=1, col=2)
    fig.update_xaxes(title_text="β", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="W_t", row=2, col=2)
    fig.update_xaxes(title_text="Lag", row=3, col=1)
    fig.update_yaxes(title_text="ACF", row=3, col=1)
    fig.update_xaxes(title_text="Lag", row=3, col=2)
    fig.update_yaxes(title_text="ACF", row=3, col=2)

    def fmt_row(d: dict) -> str:
        cells = []
        for k, v in d.items():
            if k in ("lags", "acf"):
                continue
            if isinstance(v, float):
                cells.append(f"<tr><td>{k}</td><td>{v:.6g}</td></tr>")
            else:
                cells.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        return "".join(cells)

    desc_html = descriptive.to_html(index=False, float_format=lambda v: f"{v:.6g}")
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    start_d = dates.min().strftime("%Y-%m-%d")
    end_d = dates.max().strftime("%Y-%m-%d")

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>β_{lookback}d stats — {ticker_a}/{ticker_b}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
           margin: 24px; color: #222; max-width: 1240px; }}
    h1, h2 {{ margin-top: 1.4em; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .note {{ color: #555; font-size: 0.95em; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  </style>
</head>
<body>
  <h1>Rolling β_{lookback}d diagnostics</h1>
  <p class="note">
    Pair: <b>{ticker_a}</b> on <b>{ticker_b}</b> ·
    Source: <b>{source}</b> ·
    Return: <code>ln(P_t / P_{{t-1}})</code> ·
    β_t = OLS over trailing {lookback} sessions ending at t ·
    Sample: {start_d} → {end_d} ({len(x)} β points) ·
    Raw daily log-returns: {len(rets)}
  </p>

  <h2>Descriptive stats (β_t)</h2>
  {desc_html}

  <div class="grid">
    <div>
      <h2>Shapiro–Wilk</h2>
      <table>{fmt_row(sw)}</table>
    </div>
    <div>
      <h2>Nyblom L_C</h2>
      <table>{fmt_row(nyb)}</table>
    </div>
  </div>

  <div class="grid">
    <div>
      <h2>ACF / Ljung–Box (β)</h2>
      <table>{fmt_row(acf_beta)}</table>
    </div>
    <div>
      <h2>ACF / Ljung–Box (Δβ)</h2>
      <table>{fmt_row(acf_dbeta)}</table>
    </div>
  </div>

  <h2>Plots</h2>
  {plot_html}

  <p class="note">
    CUSUM: W_t = Σ(β_i−β̄) / (σ√T). Crossing ±0.948 roughly rejects mean constancy at 5%
    (Brownian-bridge approximation). Nyblom L_C uses the same cumulative-sum idea with
    critical value 0.47 (5%).
  </p>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    today = datetime.now().date()
    p = argparse.ArgumentParser(description="β_10d statistical diagnostics")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=int((today - timedelta(days=800)).strftime("%Y%m%d")))
    p.add_argument("--end", type=int, default=int(today.strftime("%Y%m%d")))
    p.add_argument("--lookback", type=int, default=10, help="Rolling window for OLS beta")
    p.add_argument("--data-folder", default=None)
    p.add_argument(
        "--source",
        choices=("auto", "local", "yahoo"),
        default="auto",
        help="Price source: local RTH last print, Yahoo daily, or auto-fallback",
    )
    p.add_argument("--output", default="beta_10d_stats.html")
    p.add_argument("--csv", default="beta_10d_series.csv", help="Also write β series CSV")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.data_folder or Paths().min_folder

    print(f"Loading {args.ticker_a}/{args.ticker_b} [{args.start}, {args.end}] …")
    closes, src = load_closes(
        args.ticker_a, args.ticker_b, args.start, args.end, folder, args.source
    )
    print(f"Source: {src}  ({len(closes)} daily closes)")
    rets = log_returns(closes, args.ticker_a, args.ticker_b)
    beta = rolling_ols_beta(rets, args.ticker_a, args.ticker_b, args.lookback)

    if beta.empty:
        raise SystemExit("No beta estimates — check date range / data.")

    x = beta.to_numpy(dtype=float)
    dx = np.diff(x)

    descriptive = summary_stats(beta)
    sw = shapiro_wilk(x)
    nyb = nyblom_lc(x)
    acf_beta = acf_stats(x, nlags=20)
    acf_dbeta = acf_stats(dx, nlags=20)

    print("\n=== Descriptive (β) ===")
    print(descriptive.to_string(index=False))
    print("\n=== Shapiro–Wilk ===")
    print(f"  W={sw['statistic']:.6f}  p={sw['pvalue']:.4g}  reject_normal@5%={sw['reject_H0_normal_5pct']}")
    print("\n=== Nyblom L_C ===")
    print(
        f"  L_C={nyb['statistic']:.6f}  cv5%={nyb['cv_5pct']}  "
        f"reject_constancy@5%={nyb['reject_H0_constancy_5pct']}"
    )
    print("\n=== Ljung–Box ===")
    print(
        f"  β:  Q={acf_beta['ljung_box_stat']:.4f}  p={acf_beta['ljung_box_pvalue']:.4g}  "
        f"reject_WN@5%={acf_beta['reject_H0_white_noise_5pct']}"
    )
    print(
        f"  Δβ: Q={acf_dbeta['ljung_box_stat']:.4f}  p={acf_dbeta['ljung_box_pvalue']:.4g}  "
        f"reject_WN@5%={acf_dbeta['reject_H0_white_noise_5pct']}"
    )

    out_csv = args.csv
    beta.to_frame().assign(
        **{
            f"logret_{args.ticker_a}": rets.loc[beta.index, f"logret_{args.ticker_a}"],
            f"logret_{args.ticker_b}": rets.loc[beta.index, f"logret_{args.ticker_b}"],
        }
    ).to_csv(out_csv)
    print(f"\nCSV: {out_csv}")

    html = build_report_html(
        beta,
        rets,
        args.ticker_a,
        args.ticker_b,
        args.lookback,
        descriptive,
        sw,
        nyb,
        acf_beta,
        acf_dbeta,
        src,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report: {args.output}")

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
