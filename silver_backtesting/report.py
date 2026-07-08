"""HTML backtest report generation and browser display."""

from __future__ import annotations

import json
import tempfile
import webbrowser
from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .config import SilverBacktestState, StrategyConfig
from .spread import lookback_beta_by_trading_date, ols_beta, pair_15min_returns


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.6f}"
        return f"{value:.4f}"
    return str(value)


def _dataframe_to_html(df: pd.DataFrame, table_id: str = "") -> str:
    """Render a DataFrame as a styled HTML table."""
    id_attr = f' id="{table_id}"' if table_id else ""
    header = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{escape(_format_cell(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(rows)
    return f"""
    <table{id_attr} class="data-table">
        <thead><tr>{header}</tr></thead>
        <tbody>{body}</tbody>
    </table>
    """


def _config_params_html(config: StrategyConfig) -> str:
    """Build an HTML section listing strategy parameters."""
    params = [
        ("Start date", config.startdate),
        ("Tickers", ", ".join(config.ticker_list)),
        ("Include CN futures", config.ind_include_cn),
        ("Starting min (spread window)", config.starting_min),
        ("Backtest threshold", config.backtest_thres),
        ("Backtest threshold max", config.backtest_thres_max),
        ("Base notional", config.backtest_base_notional),
        ("Slippage", config.backtest_slippage),
        ("Spreads", ", ".join(config.backtest_spreads)),
        ("Return column", config.backtest_rt_col),
        ("Beta lookback (sessions)", config.backtest_beta_lookback),
        ("Signal", "z-score on Δ(A − β×B); β from prior 20 daily returns (pre-open)"),
        ("Spread level", "A − β×B"),
        ("Hedge", "N ticker / N×beta20 cmdty"),
        ("Backtest start", config.backtest_start),
        ("Side", config.backtest_side),
    ]
    rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
        for k, v in params
    )
    return f"""
    <section>
        <h2>Strategy Parameters</h2>
        <table class="params-table">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </section>
    """


def _trade_display_columns(spread: str, rt_col: str, df: pd.DataFrame) -> list[str]:
    """Columns to show in the per-minute trades drill-down."""
    ticker, cmdty = spread.split("-", 1)
    preferred = [
        "DT",
        ticker,
        cmdty,
        spread,
        f"{spread}_beta20",
        f"{spread}_zscore",
        "Side",
        "Notional",
        "Notional_hedge",
        f"{spread}_{rt_col}",
        f"{spread}_{rt_col}_adj",
    ]
    return [c for c in preferred if c in df.columns]


def _trades_by_date(
    pregroup: pd.DataFrame | None,
    spread: str,
    rt_col: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Group minute-level trades by ``Trading date`` for the HTML drill-down."""
    if pregroup is None or pregroup.empty:
        return {}, []

    df = pregroup.copy()
    if "DT" in df.columns:
        df["DT"] = df["DT"].astype(str)
    if "ind_zscore2" in df.columns:
        side_map = {1: "Short beta spread", -1: "Long beta spread", 0: "Flat"}
        df["Side"] = df["ind_zscore2"].map(side_map)

    columns = _trade_display_columns(spread, rt_col, df)
    if not columns:
        return {}, []

    df = df[columns + (["Trading date"] if "Trading date" not in columns else [])]
    by_date: dict[str, list[dict[str, Any]]] = {}
    for td, grp in df.groupby("Trading date"):
        records = []
        for row in grp[columns].to_dict(orient="records"):
            clean = {k: _format_cell(v) for k, v in row.items()}
            records.append(clean)
        by_date[str(int(td))] = records
    return by_date, columns


def _spread_minute_by_date(
    df_data: pd.DataFrame,
    config: StrategyConfig,
    spread: str,
) -> dict[str, list[dict[str, Any]]]:
    """Minute-level beta-adjusted spread grouped by trading date (for drill-down chart)."""
    if spread not in df_data.columns:
        return {}
    ed = df_data["Trading date"].max()
    mask = df_data["Trading date"].between(
        config.backtest_start, ed, inclusive="both"
    ) & (df_data["ind_nyse_rth"] == True)
    sub = (
        df_data.loc[mask, ["Trading date", "DT", spread]]
        .dropna(subset=[spread])
        .sort_values("DT")
    )
    by_date: dict[str, list[dict[str, Any]]] = {}
    for td, grp in sub.groupby("Trading date"):
        by_date[str(int(td))] = [
            {"dt": str(row["DT"]), "spread": float(row[spread])}
            for _, row in grp.iterrows()
        ]
    return by_date


def _interactive_daily_section(
    spread: str,
    daily: pd.DataFrame,
    trade_columns: list[str],
) -> str:
    """Daily PnL table with clickable rows + empty trades panel filled by JS."""
    spread_id = spread.replace("-", "_")
    headers = "".join(f"<th>{escape(str(c))}</th>" for c in daily.columns)
    rows = []
    for _, row in daily.iterrows():
        td = int(row["Trading date"])
        cells = "".join(f"<td>{escape(_format_cell(v))}</td>" for v in row)
        rows.append(
            f'<tr class="daily-row" data-date="{td}" '
            f'onclick="showTrades(\'{spread_id}\', \'{td}\')" title="Click to view trades">'
            f"{cells}</tr>"
        )
    trade_headers = "".join(f"<th>{escape(c)}</th>" for c in trade_columns)
    return f"""
    <section class="daily-section" id="daily-{spread_id}">
        <h2>Daily PnL — {escape(spread)}</h2>
        <p class="hint">Click any date row to see the minute-by-minute spread chart and trades for that day.</p>
        <table class="data-table daily-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
        <div id="trades-panel-{spread_id}" class="trades-panel hidden">
            <h3>Trades on <span id="trades-date-{spread_id}"></span>
                <span id="trades-count-{spread_id}" class="trade-count"></span>
            </h3>
            <div id="spread-chart-{spread_id}" class="intraday-spread-chart"></div>
            <div class="trades-scroll">
                <table class="data-table trades-table">
                    <thead><tr>{trade_headers}</tr></thead>
                    <tbody id="trades-body-{spread_id}"></tbody>
                </table>
            </div>
        </div>
    </section>
    """


_REPORT_STYLES = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 2rem;
            color: #222;
            line-height: 1.5;
        }
        h1 { border-bottom: 2px solid #0066cc; padding-bottom: 0.5rem; }
        h2 { color: #0066cc; margin-top: 2rem; }
        h3 { color: #333; margin-top: 1rem; }
        section { margin-bottom: 2rem; }
        .hint { color: #666; font-size: 14px; margin: 0.25rem 0 0.75rem; }
        .data-table, .params-table {
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
            margin-top: 0.5rem;
        }
        .data-table th, .params-table th {
            background: #0066cc;
            color: white;
            padding: 8px 12px;
            text-align: left;
        }
        .data-table td, .params-table td {
            padding: 6px 12px;
            border: 1px solid #ddd;
        }
        .data-table tr:nth-child(even) { background: #f8f9fa; }
        .daily-row { cursor: pointer; }
        .daily-row:hover { background: #e3f2fd !important; }
        .daily-row.selected { background: #bbdefb !important; outline: 2px solid #0066cc; }
        .chart-container { margin: 1rem 0; }
        .trades-panel {
            margin-top: 1rem;
            padding: 1rem;
            border: 1px solid #0066cc;
            border-radius: 6px;
            background: #fafcff;
        }
        .trades-panel.hidden { display: none; }
        .trades-scroll { max-height: 420px; overflow: auto; }
        .trade-count { color: #666; font-weight: normal; font-size: 14px; }
        .intraday-spread-chart { height: 340px; margin: 1rem 0; }
"""


def _backtest_rth_mask(df_data: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    ed = df_data["Trading date"].max()
    return df_data["Trading date"].between(
        config.backtest_start, ed, inclusive="both"
    ) & (df_data["ind_nyse_rth"] == True)


def _pair_minute_returns(
    df_data: pd.DataFrame,
    config: StrategyConfig,
    ticker: str,
    cmdty: str,
) -> pd.DataFrame | None:
    """1-min pct returns for a pair during the backtest RTH window."""
    if ticker not in df_data.columns or cmdty not in df_data.columns:
        return None
    mask = _backtest_rth_mask(df_data, config)
    if not mask.any():
        return None
    sub = df_data.loc[mask, ["Trading date", ticker, cmdty]].copy()
    sub["_rt"] = sub.groupby("Trading date")[ticker].pct_change()
    sub["_rc"] = sub.groupby("Trading date")[cmdty].pct_change()
    return sub


def _session_corr(group: pd.DataFrame) -> float:
    aligned = group.dropna(subset=["_rt", "_rc"])
    if len(aligned) < 2:
        return float("nan")
    return float(aligned["_rt"].corr(aligned["_rc"]))


def _intraday_beta_series(
    fifteen_min: pd.DataFrame,
    ticker: str,
    cmdty: str,
) -> pd.DataFrame:
    """Beta for each session from 15-min returns that day only (9:45…close)."""
    rows: list[dict[str, float | int]] = []
    for td, day in fifteen_min.groupby("Trading date"):
        rows.append(
            {
                "Trading date": td,
                "intraday_beta": ols_beta(day[ticker], day[cmdty]),
            }
        )
    return pd.DataFrame(rows)


def _lookback_beta_chart_series(
    df_data: pd.DataFrame,
    ticker: str,
    cmdty: str,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Beta by date for the chart (includes day t in the 20-day window)."""
    beta_by_date = lookback_beta_by_trading_date(
        df_data, ticker, cmdty, lookback_days=lookback_days, lag_sessions=0
    )
    out = beta_by_date.reset_index()
    out.columns = ["Trading date", "lookback_beta"]
    return out


def _daily_rolling_correlation_figure(
    df_data: pd.DataFrame | None,
    config: StrategyConfig,
    rolling_days: int = 5,
) -> go.Figure | None:
    """Build a rolling correlation chart for each backtest spread pair."""
    if df_data is None or df_data.empty:
        return None

    fig = go.Figure()
    for spread in config.backtest_spreads:
        ticker, cmdty = spread.split("-", 1)
        sub = _pair_minute_returns(df_data, config, ticker, cmdty)
        if sub is None:
            continue

        daily_corr = (
            sub.groupby("Trading date")[["_rt", "_rc"]]
            .apply(_session_corr, include_groups=False)
            .reset_index(name="daily_corr")
        )
        daily_corr["rolling_corr"] = (
            daily_corr["daily_corr"].rolling(rolling_days, min_periods=1).mean()
        )
        daily_corr["Trading date"] = pd.to_datetime(
            daily_corr["Trading date"].astype(str), format="%Y%m%d"
        )

        fig.add_trace(
            go.Scatter(
                x=daily_corr["Trading date"],
                y=daily_corr["daily_corr"],
                mode="lines",
                name=f"{spread} daily",
                line=dict(width=1, dash="dot"),
                opacity=0.45,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily_corr["Trading date"],
                y=daily_corr["rolling_corr"],
                mode="lines",
                name=f"{spread} {rolling_days}d roll",
                line=dict(width=2),
            )
        )

    if not fig.data:
        return None

    fig.update_layout(
        title=(
            f"{rolling_days}-Day Rolling Correlation "
            "(daily corr of 1-min RTH returns)"
        ),
        xaxis_title="Trading Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
        template="plotly_white",
        legend_title_text="Pair",
        height=420,
    )
    return fig


def _beta_spread_price_figure(
    df_data: pd.DataFrame | None,
    config: StrategyConfig,
) -> go.Figure | None:
    """Every RTH minute of beta-adjusted spread: ticker − β×cmdty."""
    if df_data is None or df_data.empty:
        return None

    ed = df_data["Trading date"].max()
    fig = go.Figure()
    for spread in config.backtest_spreads:
        if spread not in df_data.columns:
            continue
        ticker, cmdty = spread.split("-", 1)
        mask = df_data["Trading date"].between(
            config.backtest_start, ed, inclusive="both"
        ) & (df_data["ind_nyse_rth"] == True)
        minute = (
            df_data.loc[mask, ["DT", spread]]
            .dropna(subset=[spread])
            .sort_values("DT")
        )
        if minute.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=minute["DT"],
                y=minute[spread],
                mode="lines",
                name=f"{ticker} − β×{cmdty}",
            )
        )

    if not fig.data:
        return None

    fig.add_hline(y=0, line_dash="dot", line_color="#999")
    fig.update_layout(
        title="Beta-adjusted spread (1-min RTH): A − β×B",
        xaxis_title="Time",
        yaxis_title="Spread level ($)",
        template="plotly_white",
        legend_title_text="Pair",
        height=480,
    )
    return fig


def _daily_beta_figure(
    df_data: pd.DataFrame | None,
    config: StrategyConfig,
    lookback_days: int = 20,
    bar_minutes: int = 15,
) -> go.Figure | None:
    """Intraday beta (single day) and lookback beta (last N days of 15-min returns)."""
    if df_data is None or df_data.empty:
        return None

    fig = go.Figure()
    for spread in config.backtest_spreads:
        ticker, cmdty = spread.split("-", 1)
        fifteen_min = pair_15min_returns(df_data, ticker, cmdty, bar_minutes)
        if fifteen_min is None or fifteen_min.empty:
            continue

        intraday = _intraday_beta_series(fifteen_min, ticker, cmdty)
        lookback = _lookback_beta_chart_series(
            df_data, ticker, cmdty, lookback_days
        )
        plot_df = intraday.merge(lookback, on="Trading date", how="outer")
        plot_df = plot_df[
            plot_df["Trading date"].between(
                config.backtest_start,
                df_data["Trading date"].max(),
                inclusive="both",
            )
        ]
        plot_df["Trading date"] = pd.to_datetime(
            plot_df["Trading date"].astype(str), format="%Y%m%d"
        )

        fig.add_trace(
            go.Scatter(
                x=plot_df["Trading date"],
                y=plot_df["intraday_beta"],
                mode="lines",
                name=f"{ticker} on {cmdty} intraday",
                line=dict(width=1, dash="dot"),
                opacity=0.55,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["Trading date"],
                y=plot_df["lookback_beta"],
                mode="lines",
                name=f"{ticker} on {cmdty} {lookback_days}d",
                line=dict(width=2),
            )
        )

    if not fig.data:
        return None

    fig.update_layout(
        title=(
            f"Beta — intraday ({bar_minutes}m, day t) vs "
            f"{lookback_days}-day lookback (daily returns, stock on SLV)"
        ),
        xaxis_title="Trading Date",
        yaxis_title="Beta",
        template="plotly_white",
        legend_title_text="Pair",
        height=420,
    )
    return fig


_REPORT_SCRIPT = """
        const TRADES_DATA = __TRADES_JSON__;
        const TRADE_COLUMNS = __COLUMNS_JSON__;
        const SPREAD_DATA = __SPREAD_JSON__;

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function showIntradaySpread(spreadId, date) {
            const spreadKey = spreadId.replace(/_/g, '-');
            const series = (SPREAD_DATA[spreadKey] || {})[String(date)] || [];
            const el = document.getElementById(`spread-chart-${spreadId}`);
            if (!series.length) {
                el.innerHTML = '<p class="hint">No minute spread data for this date.</p>';
                return;
            }
            Plotly.newPlot(
                el,
                [{
                    x: series.map((p) => p.dt),
                    y: series.map((p) => p.spread),
                    mode: 'lines',
                    name: spreadKey,
                }],
                {
                    title: `Beta-adjusted spread — ${date}`,
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'A − β×B ($)' },
                    template: 'plotly_white',
                    shapes: [{
                        type: 'line',
                        x0: 0, x1: 1, xref: 'paper',
                        y0: 0, y1: 0,
                        line: { dash: 'dot', color: '#999' },
                    }],
                    margin: { t: 40, r: 20, b: 40, l: 60 },
                },
                { responsive: true, displayModeBar: true }
            );
        }

        function showTrades(spreadId, date) {
            document.querySelectorAll(`#daily-${spreadId} .daily-row`).forEach(
                (row) => row.classList.remove('selected')
            );
            const selected = document.querySelector(
                `#daily-${spreadId} .daily-row[data-date="${date}"]`
            );
            if (selected) selected.classList.add('selected');

            const spreadKey = spreadId.replace(/_/g, '-');
            const trades = (TRADES_DATA[spreadKey] || {})[String(date)] || [];
            const cols = TRADE_COLUMNS[spreadKey] || (trades[0] ? Object.keys(trades[0]) : []);

            document.getElementById(`trades-date-${spreadId}`).textContent = date;
            document.getElementById(`trades-count-${spreadId}`).textContent =
                `(${trades.length} trade${trades.length === 1 ? '' : 's'})`;

            showIntradaySpread(spreadId, date);

            const tbody = document.getElementById(`trades-body-${spreadId}`);
            tbody.innerHTML = '';
            if (trades.length === 0) {
                tbody.innerHTML = `<tr><td colspan="${cols.length || 1}">No trades</td></tr>`;
            } else {
                trades.forEach((trade) => {
                    const cells = cols.map((c) => `<td>${escapeHtml(trade[c] ?? '')}</td>`).join('');
                    tbody.innerHTML += `<tr>${cells}</tr>`;
                });
            }

            const panel = document.getElementById(`trades-panel-${spreadId}`);
            panel.classList.remove('hidden');
            panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
"""


def build_backtest_html(config: StrategyConfig, state: SilverBacktestState) -> str:
    """Build a self-contained HTML report with clickable daily rows → minute trades."""
    chart_html = ""
    if state.backtest_fig is not None:
        chart_html = state.backtest_fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            config={"displayModeBar": True},
        )

    corr_chart_html = ""
    corr_fig = _daily_rolling_correlation_figure(state.df_data, config)
    if corr_fig is not None:
        corr_chart_html = corr_fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": True},
        )

    beta_chart_html = ""
    beta_fig = _daily_beta_figure(state.df_data, config)
    if beta_fig is not None:
        beta_chart_html = beta_fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": True},
        )

    spread_chart_html = ""
    spread_fig = _beta_spread_price_figure(state.df_data, config)
    if spread_fig is not None:
        spread_chart_html = spread_fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": True},
        )

    summary_html = ""
    if state.backtest_summary is not None and not state.backtest_summary.empty:
        summary_display = state.backtest_summary.copy()
        summary_display.index.name = "Metric"
        summary_display = summary_display.reset_index()
        summary_html = f"""
        <section>
            <h2>Backtest Summary</h2>
            {_dataframe_to_html(summary_display, "summary-table")}
        </section>
        """

    trades_data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    trade_columns: dict[str, list[str]] = {}
    spread_data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    daily_sections: list[str] = []

    for spread in config.backtest_spreads:
        ticker_key = spread.split("-")[0]
        daily = state.backtest_group2.get(ticker_key)
        pregroup = state.backtest_pregroup.get(ticker_key)
        if daily is None or daily.empty:
            continue

        by_date, cols = _trades_by_date(pregroup, spread, config.backtest_rt_col)
        trades_data[spread] = by_date
        trade_columns[spread] = cols
        if state.df_data is not None:
            spread_data[spread] = _spread_minute_by_date(
                state.df_data, config, spread
            )
        daily_sections.append(_interactive_daily_section(spread, daily, cols))

    daily_html = "\n".join(daily_sections)
    trades_json = json.dumps(trades_data)
    columns_json = json.dumps(trade_columns)
    spread_json = json.dumps(spread_data)
    script = (
        _REPORT_SCRIPT.replace("__TRADES_JSON__", trades_json)
        .replace("__COLUMNS_JSON__", columns_json)
        .replace("__SPREAD_JSON__", spread_json)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Silver Backtest Report</title>
    <style>{_REPORT_STYLES}</style>
</head>
<body>
    <h1>Silver Spread Backtest Report</h1>
    {_config_params_html(config)}
    {summary_html}
    <section>
        <h2>Cumulative PnL Chart</h2>
        <div class="chart-container">{chart_html}</div>
    </section>
    <section>
        <h2>Beta-adjusted spread</h2>
        <p class="hint">Every RTH minute: PAAS − β×SLV (β fixed each morning from prior 20 daily returns). Zoom with Plotly controls.</p>
        <div class="chart-container">{spread_chart_html}</div>
    </section>
    <section>
        <h2>Rolling Correlation</h2>
        <p class="hint">Daily correlation of 1-min returns during RTH, with a 5-day rolling average.</p>
        <div class="chart-container">{corr_chart_html}</div>
    </section>
    <section>
        <h2>Beta</h2>
        <p class="hint">
            <strong>Intraday (dotted):</strong> OLS beta from 15-min returns within day t only
            (9:45, 10:00, … through session close).<br>
            <strong>20-day (solid):</strong> OLS beta from daily returns over the last 20 sessions (chart includes day t).<br>
            Backtest hedge β uses prior 20 sessions of daily returns only (no same-day look-ahead).
        </p>
        <div class="chart-container">{beta_chart_html}</div>
    </section>
    {daily_html}
    <script>{script}</script>
</body>
</html>
"""


def show_backtest_report(config: StrategyConfig, state: SilverBacktestState) -> str:
    """Open the backtest HTML report in the default browser via a temp file."""
    html_content = build_backtest_html(config, state)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        prefix="silver_backtest_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(html_content)
        tmp_path = tmp.name
    webbrowser.open(f"file://{tmp_path}")
    return tmp_path
