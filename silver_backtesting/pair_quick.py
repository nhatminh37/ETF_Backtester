"""Fast two-ticker minute panel loader (PAAS/SLV style) for standalone scripts."""

from __future__ import annotations

import os

import pandas as pd

from .config import Paths
from .loaders import market_calendar
from .spread import ols_beta, pair_15min_returns


def load_ticker_minutes(
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
    folder: str | None,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Merge two tickers on DT; add ``ind_nyse_rth`` for beta helpers."""
    folder = folder or Paths().min_folder
    cal = market_calendar()
    nyse_days = set(cal["Date_cal"].astype(int))

    df_a = load_ticker_minutes(folder, ticker_a, start, end, nyse_days)
    df_b = load_ticker_minutes(folder, ticker_b, start, end, nyse_days)
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
    """For session *t*, OLS β from all prior-session 15-min returns."""
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


def attach_diff_ret(
    df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    """Minute diff_ret = r_A − β_D×r_B and session cumulative diff_ret."""
    out = df.copy()
    valid = out.groupby("Trading date")["DT"].diff().eq(pd.Timedelta(minutes=1))
    out["ret_a"] = out.groupby("Trading date")[ticker_a].pct_change().where(valid)
    out["ret_b"] = out.groupby("Trading date")[ticker_b].pct_change().where(valid)

    beta_d = prior_session_15m_beta_by_date(out, ticker_a, ticker_b)
    out["beta_D"] = out["Trading date"].map(beta_d)
    out["diff_ret"] = out["ret_a"] - out["beta_D"] * out["ret_b"]
    out["cum_diff_ret"] = out.groupby("Trading date", group_keys=False)["diff_ret"].cumsum()
    return out
