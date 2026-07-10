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


def _rolling_beta_on_window(ra: pd.Series, rb: pd.Series) -> float:
    return ols_beta(ra, rb)


def add_rolling_adapted_spread_columns(
    df: pd.DataFrame,
    window_minutes: int = 30,
    warmup_minutes: int = 30,
) -> pd.DataFrame:
    """Add ``beta_adapted`` and ``spread_bAdapted``; requires ``ret_a`` and ``ret_b``."""
    if "ret_a" not in df.columns or "ret_b" not in df.columns:
        raise ValueError("add_rolling_adapted_spread_columns needs ret_a and ret_b")

    chunks: list[pd.DataFrame] = []
    for _, grp in df.groupby("Trading date", sort=False):
        g = grp.sort_values("DT").copy()
        ra = g["ret_a"].to_numpy()
        rb = g["ret_b"].to_numpy()
        session_start = g["DT"].iloc[0]
        elapsed = (g["DT"] - session_start).dt.total_seconds().to_numpy() / 60.0
        betas = np.full(len(g), np.nan)

        for i in range(len(g)):
            if elapsed[i] < warmup_minutes:
                continue
            start = max(0, i - window_minutes + 1)
            betas[i] = _rolling_beta_on_window(
                pd.Series(ra[start : i + 1]),
                pd.Series(rb[start : i + 1]),
            )

        g["beta_adapted"] = betas
        g["spread_bAdapted"] = g["ret_a"] - g["beta_adapted"] * g["ret_b"]
        chunks.append(g)

    return pd.concat(chunks).sort_values("DT").reset_index(drop=True)


def attach_adapted_beta_spread(
    df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window_minutes: int = 30,
    warmup_minutes: int = 30,
) -> pd.DataFrame:
    """
    Rolling OLS β on 1-min returns within each session.

    ``beta_adapted`` and ``spread_bAdapted`` (= r_A − β_adapted×r_B) are NaN until
    ``warmup_minutes`` have elapsed since the session open (default: first value at 10:00 ET).
    """
    out = df.copy()
    valid = out.groupby("Trading date")["DT"].diff().eq(pd.Timedelta(minutes=1))
    out["ret_a"] = out.groupby("Trading date")[ticker_a].pct_change().where(valid)
    out["ret_b"] = out.groupby("Trading date")[ticker_b].pct_change().where(valid)
    return add_rolling_adapted_spread_columns(out, window_minutes, warmup_minutes)


def pair_daily_returns(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
) -> pd.DataFrame | None:
    """Session-to-session daily returns from last RTH print each day."""
    if ticker not in df.columns or cmdty not in df.columns:
        return None
    mask = df["ind_nyse_rth"] == True
    if not mask.any():
        return None
    daily_close = (
        df.loc[mask, ["Trading date", ticker, cmdty]]
        .groupby("Trading date")[[ticker, cmdty]]
        .last()
        .sort_index()
    )
    return daily_close.pct_change()


def pair_15min_returns(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    bar_minutes: int = 15,
) -> pd.DataFrame | None:
    """Bar close-to-close returns within each RTH session (``bar_minutes`` e.g. 15 or 60)."""
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
) -> pd.Series:
    """
    OLS beta (ticker on cmdty) from daily returns over a rolling session window.

    For session ``t``, uses the prior ``lookback_days`` sessions only when
    ``lag_sessions=1`` (no look-ahead within the trading day).
    """
    daily_ret = pair_daily_returns(df, ticker, cmdty)
    if daily_ret is None or daily_ret.empty:
        return pd.Series(dtype=float)

    dates = list(daily_ret.index)
    betas: dict[int, float] = {}
    for i, td in enumerate(dates):
        if i < lookback_days + lag_sessions - 1:
            continue
        window_dates = dates[i - lookback_days - lag_sessions + 1 : i - lag_sessions + 1]
        window = daily_ret.loc[window_dates]
        betas[int(td)] = ols_beta(window[ticker], window[cmdty])
    return pd.Series(betas, name="beta20")


def attach_lookback_beta(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    lookback_days: int = 20,
    lag_sessions: int = 1,
) -> pd.DataFrame:
    """
    Map OLS beta (ticker on cmdty) onto each row by ``Trading date``.

    With ``lag_sessions=1`` (default), day *t* uses daily returns from the
    prior ``lookback_days`` sessions only — known before the open.
    """
    spread_col = f"{ticker}-{cmdty}"
    beta_by_date = lookback_beta_by_trading_date(
        df, ticker, cmdty, lookback_days=lookback_days, lag_sessions=lag_sessions
    )
    df[f"{spread_col}_beta20"] = df["Trading date"].map(beta_by_date)
    return df


def _spread_vs_session_anchor(
    group: pd.DataFrame,
    level_col: str,
    anchor_hour: int,
    anchor_minute: int,
) -> pd.Series:
    """Spread minus spread at ``anchor_hour:anchor_minute`` ET; NaN before anchor."""
    g = group.sort_values("DT")
    mask = (g["DT"].dt.hour == anchor_hour) & (g["DT"].dt.minute == anchor_minute)
    anchor_rows = g.loc[mask, level_col]
    if anchor_rows.empty:
        t_anchor = g["DT"].iloc[0].normalize() + pd.Timedelta(
            hours=anchor_hour, minutes=anchor_minute
        )
        fallback = g.loc[g["DT"] >= t_anchor]
        if fallback.empty:
            return pd.Series(np.nan, index=g.index)
        anchor_val = float(fallback[level_col].iloc[0])
        anchor_dt = fallback["DT"].iloc[0]
    else:
        anchor_val = float(anchor_rows.iloc[0])
        anchor_dt = g.loc[mask, "DT"].iloc[0]

    out = g[level_col] - anchor_val
    out = out.where(g["DT"] >= anchor_dt)
    return out.reindex(group.index)


def prior_session_bar_beta_by_date(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    bar_minutes: int = 15,
) -> pd.Series:
    """For session *t*, OLS β from all ``bar_minutes`` returns on prior session."""
    bars = pair_15min_returns(df, ticker, cmdty, bar_minutes)
    if bars is None or bars.empty:
        return pd.Series(dtype=float)

    by_session: dict[int, float] = {}
    for td, day in bars.groupby("Trading date"):
        by_session[int(td)] = ols_beta(day[ticker], day[cmdty])

    dates = sorted(by_session)
    lagged: dict[int, float] = {}
    for i in range(1, len(dates)):
        lagged[dates[i]] = by_session[dates[i - 1]]
    return pd.Series(lagged, name=f"beta_{bar_minutes}m")


def prior_session_15m_beta_by_date(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    bar_minutes: int = 15,
) -> pd.Series:
    """For session *t*, OLS β from all 15-min returns on prior session."""
    out = prior_session_bar_beta_by_date(df, ticker, cmdty, bar_minutes)
    out.name = "beta_D"
    return out


def prior_session_1h_beta_by_date(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
) -> pd.Series:
    """For session *t*, OLS β from all 1-hour returns on prior session (β_E)."""
    out = prior_session_bar_beta_by_date(df, ticker, cmdty, bar_minutes=60)
    out.name = "beta_E"
    return out


def _attach_beta_price_spread(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    beta_by_date: pd.Series,
    *,
    beta_col: str,
    level_col: str,
    vs_col: str,
    anchor_hour: int = 9,
    anchor_minute: int = 45,
) -> pd.DataFrame:
    out = df.copy()
    out[beta_col] = out["Trading date"].map(beta_by_date)
    out[level_col] = out[ticker] - out[beta_col] * out[cmdty]
    parts: list[pd.DataFrame] = []
    for _, g in out.groupby("Trading date", sort=False):
        chunk = g.copy()
        chunk[vs_col] = _spread_vs_session_anchor(
            chunk, level_col, anchor_hour, anchor_minute
        )
        parts.append(chunk)
    return pd.concat(parts).sort_values("DT").reset_index(drop=True)


def attach_beta20_price_spread(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    lookback_days: int = 20,
    lag_sessions: int = 1,
    anchor_hour: int = 9,
    anchor_minute: int = 45,
) -> pd.DataFrame:
    """
    Dollar β₂₀ price spread (not returns).

    Adds ``{spread}_beta20``, ``{spread}_price_b20``, ``{spread}_price_b20_vs_945``.
    """
    spread_col = f"{ticker}-{cmdty}"
    beta_by_date = lookback_beta_by_trading_date(
        df, ticker, cmdty, lookback_days=lookback_days, lag_sessions=lag_sessions
    )
    return _attach_beta_price_spread(
        df,
        ticker,
        cmdty,
        beta_by_date,
        beta_col=f"{spread_col}_beta20",
        level_col=f"{spread_col}_price_b20",
        vs_col=f"{spread_col}_price_b20_vs_945",
        anchor_hour=anchor_hour,
        anchor_minute=anchor_minute,
    )


def attach_beta_d_price_spread(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    anchor_hour: int = 9,
    anchor_minute: int = 45,
    bar_minutes: int = 15,
) -> pd.DataFrame:
    """
    Dollar β_D price spread (not returns).

    β_D = OLS on **prior session** 15-min returns. Adds ``{spread}_beta_D``,
    ``{spread}_price_bD``, ``{spread}_price_bD_vs_945``.
    """
    spread_col = f"{ticker}-{cmdty}"
    beta_by_date = prior_session_15m_beta_by_date(df, ticker, cmdty, bar_minutes)
    return _attach_beta_price_spread(
        df,
        ticker,
        cmdty,
        beta_by_date,
        beta_col=f"{spread_col}_beta_D",
        level_col=f"{spread_col}_price_bD",
        vs_col=f"{spread_col}_price_bD_vs_945",
        anchor_hour=anchor_hour,
        anchor_minute=anchor_minute,
    )


def attach_beta_e_price_spread(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    anchor_hour: int = 9,
    anchor_minute: int = 45,
) -> pd.DataFrame:
    """
    Dollar β_E price spread (not returns).

    β_E = OLS on **prior session** 1-hour returns. Adds ``{spread}_beta_E``,
    ``{spread}_price_bE``, ``{spread}_price_bE_vs_945``.
    """
    spread_col = f"{ticker}-{cmdty}"
    beta_by_date = prior_session_1h_beta_by_date(df, ticker, cmdty)
    return _attach_beta_price_spread(
        df,
        ticker,
        cmdty,
        beta_by_date,
        beta_col=f"{spread_col}_beta_E",
        level_col=f"{spread_col}_price_bE",
        vs_col=f"{spread_col}_price_bE_vs_945",
        anchor_hour=anchor_hour,
        anchor_minute=anchor_minute,
    )


def _map_session_beta_ma(
    sub: pd.DataFrame,
    beta_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """Rolling mean of per-session β over prior sessions only (excludes current day)."""
    if not windows:
        return sub

    daily = (
        sub.groupby("Trading date", as_index=False)[beta_col]
        .first()
        .sort_values("Trading date")
    )
    for w in windows:
        ma_col = f"_beta_ma{w}"
        daily[ma_col] = (
            daily[beta_col].rolling(window=w, min_periods=1).mean().shift(1)
        )

    lut = daily.set_index("Trading date")
    out = sub.copy()
    for w in windows:
        out[f"_beta_ma{w}"] = out["Trading date"].map(lut[f"_beta_ma{w}"])
    return out


_BETA_MA_COLORS = ("#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf")


def beta_price_spread_figure(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    *,
    beta: str = "20",
    trading_date: int | None = None,
    show_vs_anchor: bool = True,
    show_level: bool = True,
    show_beta: bool = True,
    beta_ma_windows: list[int] | None = None,
    anchor_label: str = "9:45",
    height: int = 920,
    width: int = 1500,
    line_width: float | None = None,
) -> "go.Figure":
    """
    Large Plotly chart for β dollar price spread (1-min RTH).

    ``beta``: ``"20"`` (20-day daily), ``"D"`` (prior session 15m), or ``"E"`` (prior session 1h).
    ``beta_ma_windows``: session rolling means on β panel (e.g. [5, 10, 20]).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    spread_col = f"{ticker}-{cmdty}"
    specs = {
        "20": {
            "beta_col": f"{spread_col}_beta20",
            "level_col": f"{spread_col}_price_b20",
            "vs_col": f"{spread_col}_price_b20_vs_945",
            "label": "β₂₀",
        },
        "D": {
            "beta_col": f"{spread_col}_beta_D",
            "level_col": f"{spread_col}_price_bD",
            "vs_col": f"{spread_col}_price_bD_vs_945",
            "label": "β_D",
        },
        "E": {
            "beta_col": f"{spread_col}_beta_E",
            "level_col": f"{spread_col}_price_bE",
            "vs_col": f"{spread_col}_price_bE_vs_945",
            "label": "β_E",
        },
    }
    if beta not in specs:
        raise ValueError(f"beta must be '20', 'D', or 'E', got {beta!r}")
    cfg = specs[beta]
    level_col = cfg["level_col"]
    vs_col = cfg["vs_col"]
    beta_col = cfg["beta_col"]
    beta_label = cfg["label"]

    if level_col not in df.columns:
        attach_fns = {
            "20": "attach_beta20_price_spread",
            "D": "attach_beta_d_price_spread",
            "E": "attach_beta_e_price_spread",
        }
        fn = attach_fns[beta]
        raise ValueError(f"Missing {level_col}; call {fn} first.")

    sub = df.loc[df["ind_nyse_rth"] == True].copy()
    if trading_date is not None:
        sub = sub.loc[sub["Trading date"] == trading_date]
    sub = sub.dropna(subset=[level_col]).sort_values("DT")
    if sub.empty:
        raise ValueError("No RTH rows to plot for the selected window.")

    td_label = (
        str(int(trading_date))
        if trading_date is not None
        else f"{int(sub['Trading date'].min())}–{int(sub['Trading date'].max())}"
    )
    n_sessions = sub["Trading date"].nunique()
    if trading_date is not None:
        beta_val = sub[beta_col].dropna().iloc[0] if sub[beta_col].notna().any() else float("nan")
        beta_txt = f"{beta_val:.3f}" if pd.notna(beta_val) else "n/a"
    else:
        beta_txt = f"{n_sessions} sessions ({beta_label} fixed per day)"

    if line_width is None:
        line_width = 2.8 if n_sessions <= 2 else (1.2 if n_sessions <= 30 else 0.6)

    rows = int(show_vs_anchor) + int(show_level) + int(show_beta)
    if rows == 0:
        show_vs_anchor = True
        rows = 1

    titles: list[str] = []
    if show_vs_anchor:
        titles.append(f"Δ spread from {anchor_label} ($) — {ticker} − {beta_label}×{cmdty}")
    if show_level:
        titles.append(f"Spread level ($) — {ticker} − {beta_label}×{cmdty}")
    if show_beta:
        ma_windows = beta_ma_windows if beta_ma_windows is not None else [5, 10, 20]
        ma_tag = f" + MA{','.join(str(w) for w in ma_windows)}" if ma_windows else ""
        titles.append(f"{beta_label} hedge ratio{ma_tag} — {ticker} on {cmdty}")

    row_heights = []
    if show_vs_anchor:
        row_heights.append(0.38)
    if show_level:
        row_heights.append(0.38)
    if show_beta:
        row_heights.append(0.24)
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=titles,
        row_heights=row_heights if len(row_heights) > 1 else None,
    )

    row = 1
    if show_vs_anchor:
        fig.add_trace(
            go.Scatter(
                x=sub["DT"],
                y=sub[vs_col],
                mode="lines",
                name=f"vs {anchor_label} ($)",
                line=dict(width=line_width, color="#1f77b4"),
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=1.5, row=row, col=1)
        row += 1

    if show_level:
        fig.add_trace(
            go.Scatter(
                x=sub["DT"],
                y=sub[level_col],
                mode="lines",
                name="level ($)",
                line=dict(width=line_width, color="#d62728"),
            ),
            row=row,
            col=1,
        )
        row += 1

    if show_beta:
        ma_windows = beta_ma_windows if beta_ma_windows is not None else [5, 10, 20]
        beta_sub = _map_session_beta_ma(
            sub.dropna(subset=[beta_col]), beta_col, ma_windows
        )
        fig.add_trace(
            go.Scatter(
                x=beta_sub["DT"],
                y=beta_sub[beta_col],
                mode="lines",
                name=f"{beta_label} (daily)",
                line=dict(width=line_width, color="#2ca02c"),
            ),
            row=row,
            col=1,
        )
        ma_width = max(1.0, line_width * 0.85)
        for i, w in enumerate(ma_windows):
            ma_col = f"_beta_ma{w}"
            if ma_col not in beta_sub.columns:
                continue
            color = _BETA_MA_COLORS[i % len(_BETA_MA_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=beta_sub["DT"],
                    y=beta_sub[ma_col],
                    mode="lines",
                    name=f"{beta_label} MA{w} (prior sessions)",
                    line=dict(width=ma_width, color=color, dash="dash"),
                ),
                row=row,
                col=1,
            )

    plot_height = height + (180 if show_beta and (show_vs_anchor or show_level) else 0)

    fig.update_layout(
        title=dict(
            text=f"{beta_label} price spread — {ticker} vs {cmdty} ({td_label}, {beta_label}={beta_txt})",
            font=dict(size=22),
        ),
        template="plotly_white",
        height=plot_height,
        width=width,
        font=dict(size=16),
        legend=dict(font=dict(size=14)),
        margin=dict(l=70, r=40, t=90, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (ET)", title_font=dict(size=16), tickfont=dict(size=13), row=rows, col=1)
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=13))

    y_row = 1
    if show_vs_anchor:
        fig.update_yaxes(title_text=f"Δ spread from {anchor_label} ($)", row=y_row, col=1)
        y_row += 1
    if show_level:
        fig.update_yaxes(title_text="Spread level ($)", row=y_row, col=1)
        y_row += 1
    if show_beta:
        fig.update_yaxes(title_text=f"{beta_label} (OLS)", row=y_row, col=1)
    return fig


def beta20_price_spread_figure(
    df: pd.DataFrame,
    ticker: str,
    cmdty: str,
    **kwargs,
) -> "go.Figure":
    """Backward-compatible wrapper for ``beta_price_spread_figure(..., beta='20')``."""
    return beta_price_spread_figure(df, ticker, cmdty, beta="20", **kwargs)


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
