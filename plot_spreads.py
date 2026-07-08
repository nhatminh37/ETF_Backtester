#!/usr/bin/env python3
"""
Fast standalone spread viewer — no full backtest pipeline.

Plots minute-level RTH **return spreads** for a pair (A, B):

  1. r_A − r_B
  2. r_A − β₅×r_B   (β from prior 5 session daily returns, known pre-open)
  3. r_A − β₁₀×r_B
  4. r_A − β₂₀×r_B
  5. r_A − β_D×r_B   (β from prior session 15-min returns: 9:45, 10:00, …)

Charts show the **cumulative session return spread** (cumsum from open, resets each day).
Use ``--return-view minute`` for raw 1-min return spreads instead.

Usage::

    python plot_spreads.py --date 20260201
    python plot_spreads.py --start 20260101 --end 20260201
    python plot_spreads.py --ticker-a PAAS --ticker-b SLV --date 20260201 --output spreads.html
"""

from __future__ import annotations

import argparse
import os
import tempfile
import webbrowser
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from silver_backtesting.config import Paths
from silver_backtesting.loaders import market_calendar
from silver_backtesting.spread import (
    lookback_beta_by_trading_date,
    ols_beta,
    pair_15min_returns,
)


SPREAD_SPECS: list[tuple[str, str]] = [
    ("r_A − r_B", "spread_raw"),
    ("r_A − β₅×r_B", "spread_b5"),
    ("r_A − β₁₀×r_B", "spread_b10"),
    ("r_A − β₂₀×r_B", "spread_b20"),
    ("r_A − β_D×r_B (prior day 15m)", "spread_bD"),
]


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=45)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(description="Plot beta-adjusted spreads (fast, 2 tickers only).")
    p.add_argument("--ticker-a", default="PAAS", help="Stock / leg A (default: PAAS)")
    p.add_argument("--ticker-b", default="SLV", help="Benchmark / leg B (default: SLV)")
    p.add_argument("--start", type=int, default=default_start, help="First trading date YYYYMMDD")
    p.add_argument("--end", type=int, default=default_end, help="Last trading date YYYYMMDD")
    p.add_argument(
        "--date",
        type=int,
        default=None,
        help="Focus chart on one session (intraday overlay). Default: last date in range.",
    )
    p.add_argument(
        "--mode",
        choices=("day", "range"),
        default="day",
        help="'day' = one session overlay; 'range' = all RTH minutes in window (5 subplots).",
    )
    p.add_argument(
        "--return-view",
        choices=("cumulative", "minute"),
        default="cumulative",
        help="'cumulative' = cumsum of return spread from each session open (default); "
        "'minute' = raw 1-min return spread.",
    )
    p.add_argument(
        "--data-folder",
        default=None,
        help="Minute CSV folder (default: Paths().min_folder)",
    )
    p.add_argument("--output", default=None, help="Write HTML here instead of a temp file")
    p.add_argument("--no-browser", action="store_true", help="Do not open the chart in a browser")
    return p.parse_args()


def _load_ticker_minutes(
    folder: str,
    ticker: str,
    start: int,
    end: int,
    nyse_days: set[int],
) -> pd.DataFrame:
    """Load 1-min equity CSVs for one ticker; keep NYSE RTH rows only."""
    clean = ticker.replace("-US", "")
    suffix = f"{clean}.CSV"
    chunks: list[pd.DataFrame] = []

    try:
        names = os.listdir(folder)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read data folder: {folder}") from exc

    for name in names:
        upper = name.upper()
        if not upper.endswith(suffix):
            continue
        try:
            td = int(name[:8])
        except ValueError:
            continue
        if td < start or td > end or td not in nyse_days:
            continue

        path = os.path.join(folder, name)
        raw = pd.read_csv(path)
        raw.rename(
            columns={
                "Close": "close",
                "Open": "open",
                "Time ET Rounded": "dt_str",
            },
            inplace=True,
        )
        if "dt_str" not in raw.columns:
            continue
        raw["DT"] = pd.to_datetime(raw["dt_str"])
        raw["Trading date"] = td
        rth = (
            raw["DT"].dt.hour.between(9, 15)
            & ((raw["DT"].dt.hour != 9) | (raw["DT"].dt.minute >= 30))
        )
        raw = raw.loc[rth, ["DT", "Trading date", "close"]].copy()
        if not raw.empty:
            chunks.append(raw)

    if not chunks:
        return pd.DataFrame(columns=["DT", "Trading date", ticker])

    out = pd.concat(chunks, ignore_index=True).sort_values("DT")
    out.rename(columns={"close": ticker}, inplace=True)
    return out.drop_duplicates(subset=["DT"], keep="last")


def load_pair_panel(
    ticker_a: str,
    ticker_b: str,
    folder: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Merge two tickers on DT; add ``ind_nyse_rth`` for beta helpers."""
    cal = market_calendar()
    nyse_days = set(cal["Date_cal"].astype(int))

    df_a = _load_ticker_minutes(folder, ticker_a, start, end, nyse_days)
    df_b = _load_ticker_minutes(folder, ticker_b, start, end, nyse_days)
    if df_a.empty or df_b.empty:
        raise RuntimeError(
            f"No minute data for {ticker_a}/{ticker_b} between {start} and {end} in {folder}"
        )

    df = df_a.merge(df_b, on=["DT", "Trading date"], how="inner")
    df["ind_nyse_rth"] = True
    return df.sort_values("DT").reset_index(drop=True)


def prior_session_15m_beta_by_date(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    bar_minutes: int = 15,
) -> pd.Series:
    """
    For session *t*, OLS β from all prior-session 15-min returns.

    15-min marks are resampled RTH closes (≈ 9:45, 10:00, … through session end).
    """
    fifteen = pair_15min_returns(df, ticker, cmdty, bar_minutes)
    if fifteen is None or fifteen.empty:
        return pd.Series(dtype=float)

    by_session: dict[int, float] = {}
    for td, day in fifteen.groupby("Trading date"):
        by_session[int(td)] = ols_beta(day[ticker], day[cmdty])

    dates = sorted(by_session)
    lagged: dict[int, float] = {}
    for i in range(1, len(dates)):
        lagged[dates[i]] = by_session[dates[i - 1]]
    return pd.Series(lagged, name="beta_D")


def _attach_minute_returns(df: pd.DataFrame, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    """1-min pct returns within each session (valid consecutive RTH bars only)."""
    out = df.copy()
    valid = out.groupby("Trading date")["DT"].diff().eq(pd.Timedelta(minutes=1))
    out["ret_a"] = out.groupby("Trading date")[ticker_a].pct_change().where(valid)
    out["ret_b"] = out.groupby("Trading date")[ticker_b].pct_change().where(valid)
    return out


def attach_spread_columns(
    df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    """Add beta columns and five minute return spreads (r_A − β×r_B)."""
    out = _attach_minute_returns(df, ticker_a, ticker_b)
    beta5 = lookback_beta_by_trading_date(
        out, ticker_a, ticker_b, lookback_days=5, lag_sessions=1
    )
    beta10 = lookback_beta_by_trading_date(
        out, ticker_a, ticker_b, lookback_days=10, lag_sessions=1
    )
    beta20 = lookback_beta_by_trading_date(
        out, ticker_a, ticker_b, lookback_days=20, lag_sessions=1
    )
    beta_d = prior_session_15m_beta_by_date(out, ticker_a, ticker_b)

    out["beta5"] = out["Trading date"].map(beta5)
    out["beta10"] = out["Trading date"].map(beta10)
    out["beta20"] = out["Trading date"].map(beta20)
    out["beta_D"] = out["Trading date"].map(beta_d)

    ra, rb = out["ret_a"], out["ret_b"]
    out["spread_raw"] = ra - rb
    out["spread_b5"] = ra - out["beta5"] * rb
    out["spread_b10"] = ra - out["beta10"] * rb
    out["spread_b20"] = ra - out["beta20"] * rb
    out["spread_bD"] = ra - out["beta_D"] * rb
    return out


def plot_spread_columns(df: pd.DataFrame, return_view: str) -> pd.DataFrame:
    """Return columns to plot (cumulative session path or raw minute spreads)."""
    cols = [col for _, col in SPREAD_SPECS]
    if return_view == "minute":
        return df
    out = df.copy()
    for col in cols:
        out[col] = out.groupby("Trading date", group_keys=False)[col].cumsum()
    return out


def _beta_summary_row(df: pd.DataFrame, td: int) -> pd.Series:
    day = df.loc[df["Trading date"] == td].iloc[0]
    return pd.Series(
        {
            "beta5": day["beta5"],
            "beta10": day["beta10"],
            "beta20": day["beta20"],
            "beta_D (prior 15m day)": day["beta_D"],
        }
    )


def _yaxis_title(return_view: str) -> str:
    if return_view == "minute":
        return "1-min return spread"
    return "Cumulative return spread (from open)"


def figure_intraday_day(
    df: pd.DataFrame,
    td: int,
    ticker_a: str,
    ticker_b: str,
    return_view: str = "cumulative",
) -> go.Figure:
    """All five spreads overlaid for one session."""
    sub = df.loc[df["Trading date"] == td].sort_values("DT")
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (label, col), color in zip(SPREAD_SPECS, colors):
        fig.add_trace(
            go.Scatter(
                x=sub["DT"],
                y=sub[col],
                mode="lines",
                name=label,
                line=dict(width=1.5, color=color),
            )
        )

    betas = _beta_summary_row(df, td)
    beta_txt = " | ".join(f"{k}={v:.3f}" if pd.notna(v) else f"{k}=n/a" for k, v in betas.items())

    view_note = "cumulative from open" if return_view == "cumulative" else "1-min returns"
    fig.update_layout(
        title=f"{ticker_a} vs {ticker_b} return spreads — {td} ({view_note})<br><sup>{beta_txt}</sup>",
        xaxis_title="Time (ET)",
        yaxis_title=_yaxis_title(return_view),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=560,
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#999")
    return fig


def figure_range(
    df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    return_view: str = "cumulative",
) -> go.Figure:
    """Five stacked subplots for the full date window."""
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[label for label, _ in SPREAD_SPECS],
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, ((label, col), color) in enumerate(zip(SPREAD_SPECS, colors), start=1):
        fig.add_trace(
            go.Scatter(
                x=df["DT"],
                y=df[col],
                mode="lines",
                name=label,
                line=dict(width=1, color=color),
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#ccc", row=i, col=1)

    view_note = "cumulative from open" if return_view == "cumulative" else "1-min returns"
    fig.update_layout(
        title=(
            f"{ticker_a} vs {ticker_b} — RTH return spreads "
            f"({df['Trading date'].min()}–{df['Trading date'].max()}, {view_note})"
        ),
        template="plotly_white",
        height=900,
    )
    fig.update_xaxes(title_text="Time (ET)", row=5, col=1)
    fig.update_yaxes(title_text=_yaxis_title(return_view))
    return fig


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    ticker_a, ticker_b = args.ticker_a, args.ticker_b

    print(f"Loading {ticker_a} + {ticker_b} from {folder} …")
    df = load_pair_panel(ticker_a, ticker_b, folder, args.start, args.end)
    print(f"  {len(df):,} RTH minutes ({df['Trading date'].nunique()} sessions)")

    df = attach_spread_columns(df, ticker_a, ticker_b)
    plot_df = plot_spread_columns(df, args.return_view)

    focus_td = args.date or int(df["Trading date"].max())
    if focus_td not in set(df["Trading date"].astype(int)):
        raise SystemExit(f"No data for --date {focus_td} in loaded window.")

    print(f"\nBetas used on {focus_td} (fixed for whole session):")
    print(_beta_summary_row(df, focus_td).to_string())

    if args.mode == "day":
        fig = figure_intraday_day(
            plot_df, focus_td, ticker_a, ticker_b, args.return_view
        )
    else:
        fig = figure_range(plot_df, ticker_a, ticker_b, args.return_view)

    if args.output:
        out_path = args.output
        fig.write_html(out_path, include_plotlyjs="cdn")
    else:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", prefix="spreads_", delete=False
        )
        out_path = tmp.name
        tmp.close()
        fig.write_html(out_path, include_plotlyjs="cdn")

    print(f"\nChart written: {out_path}")
    if not args.no_browser:
        webbrowser.open(f"file://{out_path}")


if __name__ == "__main__":
    main()
