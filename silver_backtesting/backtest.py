"""Spread backtest engine and PnL charting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import StrategyConfig


def pnl_summary_chart(
    df_dict: dict[str, pd.DataFrame],
    trading_date_col: str = "Trading date",
    netpnl_col: str = "Netpnl",
    cum_col: str = "Cumulative_netpnl",
    count_col: str = "count",
) -> tuple[go.Figure, pd.DataFrame]:
    summary_dict: dict[str, dict[str, Any]] = {}
    fig = go.Figure()

    for df_name, df in df_dict.items():
        df_copy = df.copy()
        df_copy[trading_date_col] = pd.to_datetime(
            df_copy[trading_date_col].astype(str), format="%Y%m%d"
        )

        daily_pnl = df_copy[netpnl_col]
        daily_count = df_copy[count_col].replace(0, np.nan)
        cum_pnl = df_copy[cum_col]

        total_pnl = daily_pnl.sum()
        total_trades = daily_count.sum()
        avg_trades_per_day = daily_count.mean()
        num_trading_days = daily_count.count()
        avg_daily_pnl = daily_pnl.mean()
        avg_trade_pnl = daily_pnl.sum() / daily_count.sum()
        sharpe_daily = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
        cummax = cum_pnl.cummax()
        drawdown = cum_pnl - cummax
        max_drawdown = drawdown.min()
        avg_notional = int(df_copy["Notional_sum"].sum() / df_copy["count"].sum())
        max_notional = int(df_copy["Notional_max"].max())

        if "mean" in df_copy.columns:
            avg_daily_pnl_pct = f"{round(df_copy['mean'].mean() * 100, 2)}%"
        else:
            avg_daily_pnl_pct = "-"
        if "sum" in df_copy.columns:
            avg_trade_pnl_pct = (
                f"{round(df_copy['sum'].sum() / df_copy['count'].sum() * 100, 2)}%"
            )
        else:
            avg_trade_pnl_pct = "-"

        summary_dict[df_name] = {
            "Total PnL": f"${round(total_pnl)}",
            "Total trades": int(total_trades),
            "Average trades per day": round(avg_trades_per_day, 1),
            "Number of trading dates": int(num_trading_days),
            "Average daily PnL": f"${round(avg_daily_pnl)} ({avg_daily_pnl_pct})",
            "Average PnL per trade": f"${round(avg_trade_pnl)} ({avg_trade_pnl_pct})",
            "Daily Sharpe": round(sharpe_daily, 1),
            "Max Drawdown": f"{round(max_drawdown)}".replace("$-", "-$"),
            "Notional": f"{avg_notional:,.0f} / {max_notional:,.0f}",
        }

        fig.add_trace(
            go.Scatter(
                x=df_copy[trading_date_col],
                y=cum_pnl,
                mode="lines",
                name=df_name,
            )
        )

    summary_df_all = pd.DataFrame(summary_dict)
    fig.update_layout(
        title="Cumulative PnL Comparison",
        xaxis_title="Trading Date",
        yaxis_title="PnL",
        template="plotly_white",
        legend_title_text="Strategy",
    )
    return fig, summary_df_all


def run_spread_backtest(
    df_data: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    go.Figure,
    pd.DataFrame,
]:
    """Run the spread z-score backtest and return results plus PnL chart."""
    thres = config.backtest_thres
    base_notional = config.backtest_base_notional
    total_slippage = config.backtest_slippage
    spreads = config.backtest_spreads
    rt_col = config.backtest_rt_col
    sd = config.backtest_start
    ed = df_data["Trading date"].max()
    thres_max = config.backtest_thres_max
    side = config.backtest_side

    df_dict: dict[str, pd.DataFrame] = {}
    backtest_pregroup: dict[str, pd.DataFrame] = {}
    backtest_group: dict[str, pd.DataFrame] = {}
    backtest_group2: dict[str, pd.DataFrame] = {}

    for spread in spreads:
        td_mask = df_data["Trading date"].between(sd, ed, inclusive="both")
        beta_col = f"{spread}_beta20"
        if side.lower() == "all":
            side_mask = df_data[f"{spread}_zscore"].abs().between(thres, thres_max)
        elif side.lower() == "long":
            side_mask = df_data[f"{spread}_zscore"].between(-thres_max, -thres)
        elif side.lower() == "short":
            side_mask = df_data[f"{spread}_zscore"].between(thres, thres_max)
        else:
            side_mask = df_data[f"{spread}_zscore"].abs().between(thres, thres_max)

        mask = (
            (df_data["ind_nyse_rth"] == True)
            & (df_data["ind_start_spread"] == 1)
            & df_data[beta_col].notna()
        )
        df_group = df_data[mask & td_mask & side_mask].copy()
        z_abs = df_group[f"{spread}_zscore"].abs()
        df_group["Notional"] = np.where(z_abs > thres, base_notional, np.nan)
        df_group["Notional_hedge"] = df_group["Notional"] * df_group[beta_col]

        z = df_group[f"{spread}_zscore"]
        df_group["ind_zscore2"] = np.where(
            z > thres, 1, np.where(z < -thres, -1, 0)
        )
        df_group[f"{spread}_{rt_col}_adj"] = np.where(
            z > thres,
            -df_group[f"{spread}_{rt_col}"],
            np.where(z < -thres, df_group[f"{spread}_{rt_col}"], np.nan),
        )
        df_pregroup = df_group.copy()
        df_group2 = df_group.groupby("Trading date").agg(
            {f"{spread}_{rt_col}_adj": ["count", "sum", "mean"], "Notional": ["mean", "max"]}
        ).reset_index()
        df_group = df_group.groupby(["Trading date", "ind_zscore2"]).agg(
            {f"{spread}_{rt_col}_adj": ["count", "sum", "mean"], "Notional": ["mean", "max"]}
        ).reset_index()
        df_group.columns = [
            str(c1)
            if c0 == f"{spread}_{rt_col}_adj"
            else f"{c0}_{c1}"
            if c1 not in ("", None)
            else str(c0)
            for c0, c1 in df_group.columns
        ]
        df_group2.columns = [
            str(c1)
            if c0 == f"{spread}_{rt_col}_adj"
            else f"{c0}_{c1}"
            if c1 not in ("", None)
            else str(c0)
            for c0, c1 in df_group2.columns
        ]
        ticker_key = spread.split("-")[0]
        backtest_pregroup[ticker_key] = df_pregroup.copy()
        backtest_group[ticker_key] = df_group.copy()
        df_group2["Notional_sum"] = df_group2["count"] * df_group2["Notional_mean"]
        df_group2["Netpnl"] = (
            (df_group2["mean"] - total_slippage)
            * df_group2["count"]
            * df_group2["Notional_mean"]
        )
        df_group2["Cumulative_netpnl"] = df_group2["Netpnl"].cumsum()
        backtest_group2[ticker_key] = df_group2.copy()
        df_dict[spread] = df_group2

    fig, summary = pnl_summary_chart(
        df_dict,
        trading_date_col="Trading date",
        netpnl_col="Netpnl",
        cum_col="Cumulative_netpnl",
        count_col="count",
    )
    return df_dict, backtest_pregroup, backtest_group, backtest_group2, fig, summary
