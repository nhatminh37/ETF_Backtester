"""Data loading, calendar, and ETL helpers."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def modify_contract(contract: str) -> str:
    if len(contract) >= 2 and contract[-2:].isdigit():
        if int(contract[-2:]) < 30:
            return contract[:-2] + contract[-1]
        return contract
    return contract


def convert_ticker(df_all: pd.DataFrame, ticker: str, month_mapping: dict[str, str]) -> pd.DataFrame:
    s = df_all[f"{ticker.upper()}_main"].astype(str)

    ticker_part = s.str[:-2]
    month_code = s.str[-2].map(month_mapping)
    year_part = (20 + s.str[-1].astype(int)).astype(str)

    df_all[f"{ticker.upper()}_main"] = ticker_part + year_part + month_code
    return df_all


def market_calendar() -> pd.DataFrame:
    date_today = datetime.today().strftime("%Y%m%d")
    mktcal = mcal.get_calendar("NYSE")
    mkt_startdate = "20170101"
    mkt_enddate = date_today[:4] + "-" + date_today[4:6] + "-" + date_today[6:]

    df_calendar = pd.DataFrame(
        mcal.date_range(
            mktcal.schedule(start_date=mkt_startdate, end_date=mkt_enddate),
            frequency="1D",
        )
    ).rename(columns={0: "Datetime"})
    df_calendar["Date_cal"] = (
        df_calendar["Datetime"].astype(str).str.slice(start=0, stop=10).str.replace("-", "")
    )
    df_calendar["Date_cal"] = df_calendar["Date_cal"].astype(int)
    df_calendar["Prevdate"] = df_calendar["Date_cal"].shift(1).fillna(0).astype(int)
    df_calendar["min_2days"] = df_calendar["Date_cal"].shift(2)
    df_calendar["min_3days"] = df_calendar["Date_cal"].shift(3)
    df_calendar["min_4days"] = df_calendar["Date_cal"].shift(4)
    df_calendar["min_5days"] = df_calendar["Date_cal"].shift(5)
    df_calendar["Next_date"] = df_calendar["Date_cal"].shift(-1).fillna(0).astype(int)
    df_calendar["5days"] = df_calendar["Date_cal"].shift(5)
    df_calendar["21days"] = df_calendar["Date_cal"].shift(21)
    df_calendar["63days"] = df_calendar["Date_cal"].shift(63)

    df_calendar.drop(columns="Datetime", inplace=True)
    df_calendar["Year"] = (
        df_calendar["Date_cal"].astype(str).str.slice(start=0, stop=4).astype(int)
    )
    df_calendar["Datetime"] = pd.to_datetime(df_calendar["Date_cal"].astype(str), format="%Y%m%d")
    df_calendar["Weeknumber"] = df_calendar["Datetime"].dt.isocalendar().week
    return df_calendar


def compute_30m_bucket(h: int, m: int) -> int:
    if h == 9:
        return 1
    return 2 + (h - 10) * 2 + (m // 30)


def onshore_ETL(
    df_concat: pd.DataFrame,
    folder: str,
    key: str,
    error_dict: dict[str, Any],
) -> pd.DataFrame:
    try:
        df_temp = pd.read_csv(rf"{folder}{key.split('-')[1]}\\{key.split('-')[0]}.CSV")
        df_temp["DT_ET"] = pd.to_datetime(df_temp["Unnamed: 0"])
        df_temp["DT_ET"] = df_temp["DT_ET"].dt.tz_localize("Asia/Shanghai")
        df_temp["DT_ET"] = (
            df_temp["DT_ET"]
            .dt.tz_convert("America/New_York")
            .astype(str)
            .str.replace("-05:00", "")
            .str.replace("-04:00", "")
        )
        df_temp = df_temp[["DT_ET", "open", "close"]].set_index("DT_ET")
        if not df_temp.empty:
            df_concat = pd.concat([df_concat, df_temp])
    except Exception as e:
        error_dict[key] = e
    return df_concat


def offshore_ETL(
    df_concat: pd.DataFrame,
    folder: str,
    mc: str,
    td: list[int],
    error_dict: dict[str, Any],
) -> pd.DataFrame:
    try:
        df_temp = pd.read_csv(rf"{folder}{mc}\\{mc}.CSV")
        df_temp["DT_ET"] = pd.to_datetime(df_temp["time_utc"])
        df_temp["DT_ET"] = (
            df_temp["DT_ET"]
            .dt.tz_convert("America/New_York")
            .astype(str)
            .str.replace("-05:00", "")
            .str.replace("-04:00", "")
        )
        df_temp["Date_ET"] = (
            df_temp["DT_ET"].astype(str).str.slice(stop=10).str.replace("-", "").astype(int)
        )
        df_temp = df_temp[df_temp["Date_ET"].isin(td)]
        df_temp = df_temp[["DT_ET", "open", "close"]].set_index("DT_ET")
        if not df_temp.empty:
            df_concat = pd.concat([df_concat, df_temp])
    except Exception as e:
        error_dict[key] = e
    return df_concat


def active_table(df_mapping: pd.DataFrame, active_table_folder: str) -> pd.DataFrame:
    df_active = pd.read_csv(
        active_table_folder + [x for x in os.listdir(active_table_folder) if ".csv" in x.lower()][-1]
    )
    df_active["Startdate"] = (
        pd.to_datetime(df_active["StDate"], format="%Y-%m-%d").dt.strftime("%Y%m%d").astype(int)
    )
    df_active["Enddate"] = (
        pd.to_datetime(df_active["EdDate"], format="%Y-%m-%d").dt.strftime("%Y%m%d").astype(int)
    )
    df_active.rename(columns={"ActiveContract": "Contract"}, inplace=True)
    df_active.drop(columns=["StDate", "EdDate"], inplace=True)
    df_active = (
        df_active[df_active["Startdate"] >= 20210101]
        .sort_values(by=["Product", "Startdate"])
        .reset_index(drop=True)
    )
    df_active["Product"] = df_active["Product"].str.lower().str.rstrip()
    df_active = df_active.merge(
        df_mapping[["ChainRIC"]].set_index(df_mapping["BloombergCode"].str.lower()),
        left_on="Product",
        right_index=True,
        how="left",
    )
    df_active["RIC Root"] = df_active["ChainRIC"].str.replace("0#", "").str.replace(":", "")
    df_active["RIC"] = df_active["Contract"].loc[df_active["RIC Root"].notna()].apply(modify_contract)
    df_active["RIC"] = df_active.apply(
        lambda row: (
            str(row["RIC"]).replace(row["Product"].upper(), row["RIC Root"])
            if isinstance(row["RIC"], str)
            else row["RIC"]
        ),
        axis=1,
    )
    df_active["Key"] = df_active["Startdate"].astype(str) + df_active["RIC"]
    df_rollover_sd = df_active[["Key"]].set_index("Key")
    df_rollover_sd["ind_rollover"] = 1
    return df_active


def load_XAG(xag_file: str) -> pd.DataFrame:
    df_XAG = pd.read_csv(xag_file)
    df_XAG["DT ET_str"] = (
        pd.to_datetime(df_XAG["datetime"], utc=True).dt.tz_convert("US/Eastern")
    ).astype(str).str.slice(stop=-6)
    df_XAG = df_XAG.sort_values(by="datetime")
    df_XAG["XAG_log_old"] = np.log(df_XAG["close"]).diff()
    df_XAG["XAG_pct_old"] = df_XAG["close"].pct_change()

    df_XAG["DT ET"] = pd.to_datetime(df_XAG["datetime"], utc=True).dt.tz_convert("US/Eastern")
    df_XAG = df_XAG.sort_values("datetime").reset_index(drop=True)

    s = df_XAG["DT ET"].diff().dt.total_seconds()
    valid = s.between(50, 70)
    safe = valid & valid.shift()

    df_XAG.loc[safe, "XAG_log"] = np.log(df_XAG["close"]) - np.log(df_XAG["close"].shift())
    df_XAG.loc[safe, "XAG_pct"] = df_XAG["close"] / df_XAG["close"].shift() - 1
    df_XAG.rename(columns={"close": "XAG_close"}, inplace=True)
    return df_XAG
