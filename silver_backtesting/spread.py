"""Spread feature engineering and statistics."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def get_dynamic_value(row: pd.Series, ticker_col: str, template: str, ret_col: str | None = None) -> float:
    ticker = row[ticker_col]
    col = template.format(ticker=ticker, ret_col=ret_col)
    return row[col] if col in row.index else float("nan")


def select_minutes(starting_min: int, standard_minutes: list[int] | None = None) -> list[int]:
    if standard_minutes is None:
        standard_minutes = [1, 2, 3, 5, 10, 15, 30, 60]
    below = [m for m in standard_minutes if m < starting_min]
    cur = [starting_min]
    above: list[int] = []
    for m in standard_minutes:
        if m > starting_min:
            above = [m]
            break
    return below + cur + above


def forward_spread_pct_return(
    df: pd.DataFrame,
    spread_col: str,
    minutes_list: list[int],
) -> pd.DataFrame:
    """Legacy: % change in dollar spread level. Prefer ``forward_pairs_return``."""
    for m in minutes_list:
        col_name = f"{spread_col}_fwd{m}m"
        fwd = df[spread_col].shift(-m)
        dt_diff = df["DT"].shift(-m) - df["DT"]
        valid = dt_diff.eq(pd.Timedelta(minutes=m))
        df[col_name] = ((fwd - df[spread_col]) / df[spread_col].abs()).where(valid)
    return df


def _last_rth_bar_by_date(
    df: pd.DataFrame,
    price_cols: list[str],
) -> pd.DataFrame:
    """Last NYSE RTH minute per ``Trading date`` (market close proxy)."""
    rth = df.loc[df["ind_nyse_rth"], ["Trading date", "DT", *price_cols]]
    return (
        rth.sort_values("DT")
        .groupby("Trading date", as_index=False)
        .last()
        .rename(columns={"DT": "last_rth_dt"})
    )


def forward_pairs_return(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    minutes_list: list[int],
) -> pd.DataFrame:
    """
    Forward pairs return: r_ticker − r_cmdty over ``m`` minutes.

    Exit time is ``entry + m`` on the clock. If that falls after the last RTH bar
    of the session, the last RTH price is used as the exit (market close).

    Matches a dollar-neutral long-ticker / short-cmdty book. Long-spread PnL uses
    this directly; short-spread PnL flips the sign (handled in the backtester).
    """
    spread_col = f"{ticker}-{cmdty}"
    rth_prices = df.loc[
        df["ind_nyse_rth"], ["Trading date", "DT", ticker, cmdty]
    ].copy()
    last_rth = _last_rth_bar_by_date(df, [ticker, cmdty])
    last_dt_by_td = last_rth.set_index("Trading date")["last_rth_dt"]
    exit_lookup = rth_prices.rename(
        columns={"DT": "exit_dt", ticker: "_exit_t", cmdty: "_exit_c"}
    )

    entry_t = df[ticker]
    entry_c = df[cmdty]

    for m in minutes_list:
        col_name = f"{spread_col}_fwd{m}m"
        exit_target = df["DT"] + pd.to_timedelta(m, unit="m")
        last_dt = df["Trading date"].map(last_dt_by_td)
        exit_dt = exit_target.where(exit_target <= last_dt, last_dt)

        lookup = pd.DataFrame(
            {"Trading date": df["Trading date"].values, "exit_dt": exit_dt.values},
            index=df.index,
        ).merge(exit_lookup, on=["Trading date", "exit_dt"], how="left")

        exit_t = lookup["_exit_t"]
        exit_c = lookup["_exit_c"]
        pairs = (exit_t / entry_t - 1) - (exit_c / entry_c - 1)
        valid = (
            df["ind_nyse_rth"]
            & entry_t.notna()
            & entry_c.notna()
            & exit_t.notna()
            & exit_c.notna()
            & (entry_t != 0)
            & (entry_c != 0)
        )
        df[col_name] = pairs.where(valid)
    return df


def ols_beta(stock_ret: pd.Series, bench_ret: pd.Series) -> float:
    aligned = pd.concat([stock_ret, bench_ret], axis=1, keys=["s", "b"]).dropna()
    if len(aligned) < 2:
        return float("nan")
    var_b = aligned["b"].var()
    if var_b == 0 or pd.isna(var_b):
        return float("nan")
    return float(aligned["s"].cov(aligned["b"]) / var_b)


def pair_15min_returns(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    bar_minutes: int = 15,
) -> pd.DataFrame | None:
    """15-min close-to-close returns within each RTH session."""
    if ticker not in df.columns or cmdty not in df.columns:
        return None
    mask = df["ind_nyse_rth"] == True
    if not mask.any():
        return None

    freq = f"{bar_minutes}min"
    chunks: list[pd.DataFrame] = []
    cols = [ticker, cmdty]
    for td, group in df.loc[mask, ["Trading date", "DT", *cols]].groupby("Trading date"):
        bars = (
            group.set_index("DT")[cols]
            .sort_index()
            .resample(freq, label="right", closed="right")
            .last()
            .dropna(how="all")
        )
        rets = bars.pct_change().dropna(how="any")
        if rets.empty:
            continue
        rets["Trading date"] = td
        chunks.append(rets.reset_index())

    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True)


def lookback_beta_by_trading_date(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    lookback_days: int = 20,
    lag_sessions: int = 1,
    bar_minutes: int = 15,
) -> pd.Series:
    """
    OLS beta (ticker on cmdty) from pooled 15-min returns.

    For session ``t``, uses the prior ``lookback_days`` sessions only when
    ``lag_sessions=1`` (no look-ahead within the trading day).
    """
    fifteen_min = pair_15min_returns(df, ticker, cmdty, bar_minutes)
    if fifteen_min is None or fifteen_min.empty:
        return pd.Series(dtype=float)

    dates = sorted(fifteen_min["Trading date"].unique())
    betas: dict[int, float] = {}
    for i, td in enumerate(dates):
        if i < lookback_days + lag_sessions - 1:
            continue
        window_dates = dates[i - lookback_days - lag_sessions + 1 : i - lag_sessions + 1]
        window = fifteen_min[fifteen_min["Trading date"].isin(window_dates)]
        betas[int(td)] = ols_beta(window[ticker], window[cmdty])
    return pd.Series(betas, name="beta20")


def attach_lookback_beta(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    lookback_days: int = 20,
    lag_sessions: int = 1,
) -> pd.DataFrame:
    spread_col = f"{ticker}-{cmdty}"
    beta_by_date = lookback_beta_by_trading_date(
        df, ticker, cmdty, lookback_days=lookback_days, lag_sessions=lag_sessions
    )
    df[f"{spread_col}_beta20"] = df["Trading date"].map(beta_by_date)
    return df


def forward_beta_hedged_return(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    minutes_list: list[int],
    beta_col: str,
) -> pd.DataFrame:
    """
    Forward beta-hedged return: r_ticker − β·r_cmdty over ``m`` minutes.

    β is the row's hedge ratio (e.g. 20-day lookback). Matches a book sized
    ``N`` on ticker and ``N×β`` on cmdty.
    """
    spread_col = f"{ticker}-{cmdty}"
    rth_prices = df.loc[
        df["ind_nyse_rth"], ["Trading date", "DT", ticker, cmdty]
    ].copy()
    last_rth = _last_rth_bar_by_date(df, [ticker, cmdty])
    last_dt_by_td = last_rth.set_index("Trading date")["last_rth_dt"]
    exit_lookup = rth_prices.rename(
        columns={"DT": "exit_dt", ticker: "_exit_t", cmdty: "_exit_c"}
    )

    entry_t = df[ticker]
    entry_c = df[cmdty]
    beta = df[beta_col]

    for m in minutes_list:
        col_name = f"{spread_col}_fwd{m}m"
        exit_target = df["DT"] + pd.to_timedelta(m, unit="m")
        last_dt = df["Trading date"].map(last_dt_by_td)
        exit_dt = exit_target.where(exit_target <= last_dt, last_dt)

        lookup = pd.DataFrame(
            {"Trading date": df["Trading date"].values, "exit_dt": exit_dt.values},
            index=df.index,
        ).merge(exit_lookup, on=["Trading date", "exit_dt"], how="left")

        exit_t = lookup["_exit_t"]
        exit_c = lookup["_exit_c"]
        r_t = exit_t / entry_t - 1
        r_c = exit_c / entry_c - 1
        hedged = r_t - beta * r_c
        valid = (
            df["ind_nyse_rth"]
            & entry_t.notna()
            & entry_c.notna()
            & exit_t.notna()
            & exit_c.notna()
            & beta.notna()
            & (entry_t != 0)
            & (entry_c != 0)
        )
        df[col_name] = hedged.where(valid)
    return df


def compute_ind_trend(
    df: pd.DataFrame,
    spread_col: str,
    rvol_col: str,
    trend_window: int = 90,
    trend_mult: float = 1.2,
    freq_thresh: float = 0.6,
) -> pd.DataFrame:
    df_trend = df[df["ind_nyse_rth"] == True].copy()

    spread_pct = df_trend[spread_col].pct_change()

    net_direction = (
        spread_pct.groupby(df_trend["DT"].dt.date)
        .rolling(window=trend_window, min_periods=trend_window)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    is_positive = (spread_pct > 0).astype(int)
    is_negative = (spread_pct < 0).astype(int)

    freq_up = (
        is_positive.groupby(df_trend["DT"].dt.date)
        .rolling(window=trend_window, min_periods=trend_window)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    freq_down = (
        is_negative.groupby(df_trend["DT"].dt.date)
        .rolling(window=trend_window, min_periods=trend_window)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    trend_flag = (
        (net_direction.abs() > trend_mult * df_trend[rvol_col])
        & ((freq_up >= freq_thresh) | (freq_down >= freq_thresh))
    ).astype(int)

    df["ind_trend"] = 0
    df.loc[df_trend.index, "ind_trend"] = trend_flag
    return df


def convert_datetime_cols_to_string(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df


def run_stats_spreads(
    df_data: pd.DataFrame,
    zs_list: list[float],
    rp_list: list[str],
    date_range_list: list[str],
    cmdty_list: list[str],
    ticker_list: list[str],
) -> dict[str, pd.DataFrame]:
    df_dict: dict[str, pd.DataFrame] = {}
    dr_count = 0
    for date_range in date_range_list:
        if not re.fullmatch(r"\d{8}-\d{8}", date_range):
            raise ValueError("date_range must be in 'yyyymmdd-yyyymmdd' format")
    for date_range in date_range_list:
        dr_count += 1
        sd = int(date_range.split("-")[0])
        ed = int(date_range.split("-")[1])

        for return_period in rp_list:
            for zs in zs_list:
                zscore_thres = zs
                rows = []

                for ticker in ticker_list:
                    for cmdty in cmdty_list:
                        if cmdty == ticker:
                            continue
                        dates = df_data[
                            (df_data["ind_nyse_rth"] == True)
                            & (df_data["Trading date"].between(sd, ed, inclusive="both"))
                        ]["Trading date"].unique()
                        mask_short = (df_data[f"{ticker}-{cmdty}_zscore"] > zscore_thres) & (
                            df_data["Trading date"].between(sd, ed, inclusive="both")
                        )
                        mask_long = (df_data[f"{ticker}-{cmdty}_zscore"] < -zscore_thres) & (
                            df_data["Trading date"].between(sd, ed, inclusive="both")
                        )

                        short_returns = df_data.loc[mask_short, f"{ticker}-{cmdty}_{return_period}"]
                        short_sum = round(short_returns.sum() * 100, 1)
                        short_avg = round(short_returns.mean() * 100, 2)
                        short_count = short_returns.shape[0]
                        short_sharpe = round(
                            short_returns.mean() / short_returns.std()
                            if short_returns.std() != 0
                            else 0,
                            3,
                        )
                        short_wins = (short_returns < 0).sum()
                        short_losses = (short_returns > 0).sum()
                        short_wl = round(short_wins / short_losses, 2)

                        long_returns = df_data.loc[mask_long, f"{ticker}-{cmdty}_{return_period}"]
                        long_sum = round(long_returns.sum() * 100, 1)
                        long_avg = round(long_returns.mean() * 100, 2)
                        long_count = long_returns.shape[0]
                        long_sharpe = round(
                            long_returns.mean() / long_returns.std()
                            if long_returns.std() != 0
                            else 0,
                            3,
                        )
                        long_wins = (long_returns > 0).sum()
                        long_losses = (long_returns < 0).sum()
                        long_wl = round(long_wins / long_losses, 2)

                        rows.append(
                            {
                                "Ticker-Cmdty": f"{ticker}-{cmdty}",
                                "Zscore threshold": zscore_thres,
                                "Return period": return_period.replace("5m", "05m").replace(
                                    "1m", "01m"
                                ),
                                "Date range": f"{min(dates)}-{max(dates)} ({len(dates)})",
                                "||": "||",
                                "Long Count": long_count,
                                "Long Avg (%)": long_avg,
                                "Long Sum (%)": long_sum,
                                "Long Sharpe": long_sharpe,
                                "Long WL": long_wl,
                                " || ": "||",
                                "Short Count": short_count,
                                "Short Avg (%)": short_avg,
                                "Short Sum (%)": short_sum,
                                "Short Sharpe": short_sharpe,
                                "Short WL": short_wl,
                            }
                        )

                df_dict[f"df_{zscore_thres}_{return_period}_{dr_count}"] = pd.DataFrame(rows)
    return df_dict
