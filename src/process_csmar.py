"""
SRTP: CSMAR Auxiliary Data Processing
=====================================
Process industry classification, stock status, and index data
into formats usable for factor neutralization and stock filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_csmar_data():
    """Load all CSMAR auxiliary datasets."""
    data = {}
    files = {
        "index": "csmar_index_daily.csv",
        "industry": "csmar_industry.csv",
        "listing_status": "csmar_listing_status.csv",
        "special_treatment": "csmar_special_treatment.csv",
        "suspend_status": "csmar_suspend_status.csv",
    }
    for key, fname in files.items():
        path = DATA_DIR / fname
        if path.exists():
            try:
                data[key] = pd.read_csv(path, encoding="utf-8-sig")
            except pd.errors.ParserError:
                # Fallback for malformed CSMAR CSV files
                data[key] = pd.read_csv(path, encoding="utf-8-sig",
                                        engine="python", on_bad_lines="skip")
            print(f"[load] {key}: {len(data[key]):,} rows, {list(data[key].columns)}")
        else:
            print(f"[load] {key}: MISSING")
    return data


def process_industry(data):
    """Extract latest industry classification per stock, prioritizing Shenwan 2014."""
    df = data["industry"].copy()

    # Shenwan 2021 revised is preferred
    SW_NAME = "申银万国行业分类2021版"
    SW_2014 = "申银万国行业分类2014修订版"

    # Filter to preferred classifications
    preferred = df[df["IndustryClassificationName"].isin([SW_NAME, SW_2014])]

    if len(preferred) == 0:
        # Try other Shenwan
        sw_mask = df["IndustryClassificationName"].str.contains("申银万国|申万", na=False)
        preferred = df[sw_mask]

    if len(preferred) == 0:
        preferred = df  # Use all

    # Get latest per stock
    preferred["ImplementDate"] = pd.to_datetime(preferred["ImplementDate"])
    industry = preferred.sort_values(["Symbol", "ImplementDate"]).drop_duplicates(
        "Symbol", keep="last"
    )

    # Map stock code to industry code and name
    industry_map = industry[["Symbol", "IndustryCode", "IndustryName"]].copy()
    industry_map.columns = ["stock_code", "industry_code", "industry_name"]
    industry_map["stock_code"] = industry_map["stock_code"].astype(str).str.zfill(6)

    print(f"[industry] {len(industry_map)} stocks mapped to industries")
    print(f"[industry] {industry_map['industry_name'].nunique()} unique industries")

    return industry_map


def process_index_data(data):
    """Build benchmark return series from index data."""
    df = data["index"].copy()

    # Map column names
    col_map = {
        "Indexcd": "index_code",
        "Idxtrd01": "date",
        "Idxtrd02": "open",
        "Idxtrd03": "high",
        "Idxtrd04": "low",
        "Idxtrd05": "close",
        "Idxtrd06": "volume",
        "Idxtrd07": "amount",
        "Idxtrd08": "return_pct",
        "Idxtrd09": "index_name",
    }
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["index_code", "date"])

    # Pivot to get close prices by index
    closes = df.pivot_table(
        values="close", index="date", columns="index_code", aggfunc="first"
    )

    print(f"[index] {len(df)} rows, dates {df['date'].min()} ~ {df['date'].max()}")
    print(f"[index] indices: {df[['index_code', 'index_name']].drop_duplicates().to_dict('records')}")

    return df, closes


def build_stock_filter(data):
    """
    Build stock filtering rules:
    - Mark stocks as ST/special treatment on given dates
    - Mark stocks as suspended on given dates
    """
    # ST filter
    st = data["special_treatment"].copy()
    if "Execudt" in st.columns:
        st["date"] = pd.to_datetime(st["Execudt"])
    elif "Annoudt" in st.columns:
        st["date"] = pd.to_datetime(st["Annoudt"])
    else:
        st["date"] = pd.NaT

    st["stock_code"] = st["Stkcd"].astype(str).str.zfill(6)

    # Build date range when each stock is under ST
    st_periods = []
    for _, row in st.iterrows():
        if pd.notna(row["date"]):
            # ST typically lasts 1 year or until next announcement
            st_periods.append({
                "stock_code": row["stock_code"],
                "start_date": row["date"],
                "end_date": row["date"] + pd.DateOffset(years=1),
            })
    st_df = pd.DataFrame(st_periods) if st_periods else pd.DataFrame()

    # Suspension filter
    suspend = data["suspend_status"].copy()
    suspend["stock_code"] = suspend["Stkcd"].astype(str).str.zfill(6)
    suspend["susp_start"] = pd.to_datetime(suspend["Suspdate"], errors="coerce")
    suspend["susp_end"] = pd.to_datetime(suspend["Resmdate"], errors="coerce")
    suspend = suspend.dropna(subset=["susp_start"])

    print(f"[filter] ST periods: {len(st_df) if not st_df.empty else 0}")
    print(f"[filter] Suspension records: {len(suspend)}")
    print(f"[filter] Stocks with ST history: {st['stock_code'].nunique() if len(st)>0 else 0}")
    print(f"[filter] Stocks with suspension history: {suspend['stock_code'].nunique()}")

    return st_df, suspend


def is_stock_restricted(stock_code, trade_date, st_df, suspend_df):
    """Check if a stock is restricted (ST or suspended) on a given date."""
    trade_date = pd.Timestamp(trade_date)

    # Check ST
    if not st_df.empty:
        st_check = st_df[
            (st_df["stock_code"] == stock_code) &
            (st_df["start_date"] <= trade_date) &
            (st_df["end_date"] >= trade_date)
        ]
        if len(st_check) > 0:
            return True

    # Check suspension
    if not suspend_df.empty:
        susp_check = suspend_df[
            (suspend_df["stock_code"] == stock_code) &
            (suspend_df["susp_start"] <= trade_date)
        ]
        if len(susp_check) > 0:
            return True

    return False


def get_trading_calendar(index_df):
    """Extract trading calendar from index data."""
    trading_days = sorted(index_df["date"].unique())
    return pd.DatetimeIndex(trading_days)


def next_trading_day(date, calendar):
    """Get next trading day >= date + 1 day."""
    date = pd.Timestamp(date)
    target = date + pd.Timedelta(days=1)
    future = calendar[calendar >= target]
    return future[0] if len(future) > 0 else None


def main():
    print("=" * 60)
    print("SRTP CSMAR Auxiliary Data Processing")
    print("=" * 60)

    data = load_csmar_data()

    # Process each dataset
    industry_map = process_industry(data)
    index_df, index_closes = process_index_data(data)
    st_df, suspend_df = build_stock_filter(data)

    # Trading calendar
    calendar = get_trading_calendar(index_df)
    print(f"\n[calendar] Trading days: {len(calendar)}")
    print(f"[calendar] Range: {calendar.min()} ~ {calendar.max()}")

    # Save processed data
    industry_map.to_csv(DATA_DIR / "industry_mapping.csv", index=False)
    index_closes.to_csv(DATA_DIR / "index_closes.csv")
    if not st_df.empty:
        st_df.to_csv(DATA_DIR / "st_periods.csv", index=False)
    suspend_df[["stock_code", "susp_start", "susp_end"]].to_csv(
        DATA_DIR / "suspend_periods.csv", index=False
    )

    print("\n[save] Processed data saved to data/")
    print("  - industry_mapping.csv")
    print("  - index_closes.csv")
    print("  - st_periods.csv")
    print("  - suspend_periods.csv")

    return data, industry_map, index_df


if __name__ == "__main__":
    main()
