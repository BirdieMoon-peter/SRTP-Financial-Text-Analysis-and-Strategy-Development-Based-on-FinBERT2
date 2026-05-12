"""
SRTP: Label Construction from Daily Stock Data
===============================================
Generates supervised labels from baostock daily stock data:
- Forward excess returns (h=1,2,5,10,20)
- Benchmark-adjusted returns
- Volatility, max drawdown, volume change labels
Matches reports to next tradable date with strict look-ahead prevention.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_daily_stock(path=None):
    """Load daily stock data from baostock output."""
    if path is None:
        path = DATA_DIR / "csmar_daily_stock.csv"
    df = pd.read_csv(path, dtype={"stock_code": str, "date": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # Convert numeric columns
    for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[stock] {len(df):,} rows, {df['stock_code'].nunique()} stocks")
    print(f"[stock] Date range: {df['date'].min()} ~ {df['date'].max()}")
    return df


def load_index_data(path=None):
    """Load index data for benchmark adjustment."""
    if path is None:
        path = DATA_DIR / "csmar_index_daily.csv"

    if not path.exists():
        print("[index] No index data, computing without benchmark adjustment")
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    col_map = {
        "Indexcd": "index_code", "Idxtrd01": "date",
        "Idxtrd05": "close", "Idxtrd08": "return_pct",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])

    # Use CSI 300 (code 300) as primary benchmark, 中证全指 (985) as alternative
    for idx_code in [300, 985, 1]:
        idx_df = df[df["index_code"] == idx_code].copy()
        if len(idx_df) > 0:
            idx_df = idx_df.sort_values("date")[["date", "close"]]
            idx_df.columns = ["date", "benchmark_close"]
            print(f"[index] Using index {idx_code}: {len(idx_df)} days")
            return idx_df

    return None


def get_trading_calendar(stock_df):
    """Extract trading calendar from stock data."""
    return sorted(stock_df["date"].unique())


def compute_labels(stock_df, benchmark_df=None, horizons=(1, 2, 5, 10, 20)):
    """
    Compute forward-looking labels for each stock-date.

    Returns DataFrame with columns:
    - stock_code, date
    - fwd_ret_{h}d: forward return over h days
    - fwd_excess_{h}d: benchmark-adjusted forward return
    - fwd_vol_10d: future realized volatility
    - fwd_maxdd_10d: future max drawdown
    - fwd_vol_chg_5d: future volume change
    """
    df = stock_df.copy()
    gb = df.groupby("stock_code")

    labels_list = []
    total_groups = df["stock_code"].nunique()

    for i, (stock, group) in enumerate(gb):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)

        # Forward returns
        for h in horizons:
            if n > h:
                group[f"fwd_ret_{h}d"] = (
                    group["close"].shift(-h) / group["close"] - 1
                ).astype(float)

        # Future volatility (10-day realized)
        if n > 10:
            log_rets = np.log(group["close"] / group["close"].shift(1))
            group["fwd_vol_10d"] = (
                log_rets.shift(-1).rolling(10, min_periods=5).std().values
            )

        # Forward max drawdown (10-day)
        if n > 10:
            maxdds = []
            for j in range(n):
                if j + 10 < n:
                    window = group["close"].iloc[j + 1 : j + 11]
                    peak = window.expanding().max()
                    dd = (window - peak) / peak
                    maxdds.append(dd.min())
                else:
                    maxdds.append(np.nan)
            group["fwd_maxdd_10d"] = maxdds

        # Forward volume change (5-day vs 20-day history)
        if "volume" in group.columns and n > 25:
            hist_vol = group["volume"].rolling(20).mean()
            fwd_vol = group["volume"].shift(-1).rolling(5).mean()
            group["fwd_vol_chg_5d"] = (
                (fwd_vol - hist_vol) / (hist_vol + 1e-8)
            ).astype(float)

        labels_list.append(group)

        if (i + 1) % 500 == 0:
            print(f"  [labels] {i+1}/{total_groups} stocks processed")

    result = pd.concat(labels_list, ignore_index=True)

    # Benchmark-adjusted returns
    if benchmark_df is not None:
        result = result.merge(benchmark_df, on="date", how="left")
        # Compute benchmark forward returns
        benchmark_df_sorted = benchmark_df.sort_values("date")
        for h in horizons:
            bench_ret = benchmark_df_sorted["benchmark_close"].shift(-h) / \
                        benchmark_df_sorted["benchmark_close"] - 1
            bench_ret_map = dict(zip(benchmark_df_sorted["date"], bench_ret))
            result[f"bench_ret_{h}d"] = result["date"].map(bench_ret_map)
            result[f"fwd_excess_{h}d"] = result[f"fwd_ret_{h}d"] - result[f"bench_ret_{h}d"]

    print(f"[labels] Complete: {len(result):,} rows, {len(result.columns)} columns")
    return result


def match_reports_to_labels(reports_df, labels_df, calendar):
    """
    Match reports to their first tradable date's labels.

    Principle: report published on date D -> first tradable date >= D+1.
    Labels at that date become the prediction target.
    """
    reports = reports_df.copy()
    reports["report_date"] = pd.to_datetime(reports["report_date"])

    # Build next-trading-day mapping
    calendar_sorted = sorted(calendar)
    date_to_next = {}
    for d in reports["report_date"].unique():
        d = pd.Timestamp(d)
        candidates = [td for td in calendar_sorted if td >= d + pd.Timedelta(days=1)]
        date_to_next[d] = candidates[0] if candidates else None

    reports["tradable_date"] = reports["report_date"].map(date_to_next)
    matched_before = len(reports)
    reports = reports.dropna(subset=["tradable_date"])
    print(f"[match] Reports matched: {len(reports)}/{matched_before}")

    # Merge with labels on (stock_code, tradable_date = date)
    labels_df = labels_df.rename(columns={"date": "tradable_date"})
    # Ensure matching dtypes
    labels_df["stock_code"] = labels_df["stock_code"].astype(str).str.zfill(6)
    reports["stock_code"] = reports["stock_code"].astype(str).str.zfill(6)
    merged = reports.merge(
        labels_df,
        on=["stock_code", "tradable_date"],
        how="inner"
    )
    print(f"[match] Final merged: {len(merged)} rows")
    print(f"[match] {merged['stock_code'].nunique()} stocks, "
          f"dates {merged['tradable_date'].min()} ~ {merged['tradable_date'].max()}")

    return merged


def discretize_labels(df, label_col, train_start, train_end, n_classes=3):
    """
    Discretize continuous labels into classes using training-set quantiles.
    Prevents look-ahead bias.
    """
    train_mask = (df["tradable_date"] >= train_start) & \
                 (df["tradable_date"] <= train_end)
    train_vals = df.loc[train_mask, label_col].dropna()

    if n_classes == 3:
        lo = train_vals.quantile(0.3)
        hi = train_vals.quantile(0.7)

        def classify(x):
            if pd.isna(x):
                return np.nan
            return 0 if x <= lo else (2 if x > hi else 1)

        return df[label_col].apply(classify)
    else:
        median = train_vals.median()
        return (df[label_col] > median).astype(float)


def main():
    print("=" * 60)
    print("SRTP Label Construction")
    print("=" * 60)

    LOGS_DIR.mkdir(exist_ok=True)

    # Load data
    stock_df = load_daily_stock()
    benchmark_df = load_index_data()
    reports = pd.read_csv(DATA_DIR / "reports_cleaned.csv")

    # Compute labels
    print("\nComputing forward-looking labels...")
    labels_df = compute_labels(stock_df, benchmark_df)

    # Get trading calendar
    calendar = get_trading_calendar(stock_df)
    print(f"Trading days: {len(calendar)}")

    # Match reports to labels
    print("\nMatching reports to labels...")
    merged = match_reports_to_labels(reports, labels_df, calendar)

    # Save
    labels_df.to_csv(DATA_DIR / "stock_labels.csv", index=False)
    merged.to_csv(DATA_DIR / "reports_with_labels.csv", index=False)

    print(f"\n{'='*40}")
    print("Label summary:")
    for h in [1, 2, 5, 10, 20]:
        col = f"fwd_excess_{h}d"
        if col in merged.columns:
            vals = merged[col].dropna()
            print(f"  {col}: mean={vals.mean():.6f}, std={vals.std():.4f}, "
                  f"N={len(vals):,}")

    # Save label statistics to log
    stats = {}
    for col in merged.columns:
        if col.startswith("fwd_"):
            vals = merged[col].dropna()
            stats[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "n": int(len(vals)),
                "skew": float(vals.skew()),
            }

    import json
    with open(LOGS_DIR / "label_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nLabel stats saved to {LOGS_DIR / 'label_stats.json'}")


if __name__ == "__main__":
    main()
