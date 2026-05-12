"""
SRTP: FinBERT Hidden Layer Text Factor Research
Label Construction Module
==============================================
Constructs supervised labels: future excess returns (h=1,2,5,10,20 days),
volatility, max drawdown, volume change, abnormal turnover.
Follows strict time-alignment to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def compute_forward_returns(prices, horizons=(1, 2, 5, 10, 20)):
    """
    Compute forward returns for multiple horizons.
    prices: Series indexed by date, sorted.
    Returns DataFrame with columns: ret_{h}d for each horizon.
    """
    rets = pd.DataFrame(index=prices.index)
    for h in horizons:
        fwd_price = prices.shift(-h)
        rets[f"ret_{h}d"] = (fwd_price / prices - 1.0)
    return rets


def compute_benchmark_adjusted_returns(stock_returns, benchmark_returns):
    """
    Compute benchmark-adjusted excess returns.
    Both are DataFrames indexed by date.
    """
    excess = stock_returns.sub(benchmark_returns, axis=0)
    return excess


def compute_future_volatility(returns, horizon=10, window=20):
    """
    Compute realized volatility over future period.
    """
    future_rets = returns.shift(-1).rolling(horizon).std()
    hist_vol = returns.rolling(window).std()
    return future_rets / (hist_vol + 1e-8)  # normalized vol change


def compute_max_drawdown(prices, horizon=10):
    """Compute max drawdown over next h days."""
    max_dd = pd.Series(np.nan, index=prices.index)
    for i in range(len(prices) - horizon):
        window = prices.iloc[i + 1 : i + 1 + horizon]
        peak = window.expanding().max()
        dd = (window - peak) / peak
        max_dd.iloc[i] = dd.min()
    return max_dd


def compute_volume_change(volume, horizon=5, hist_window=20):
    """Compute future volume change relative to historical average."""
    hist_avg = volume.rolling(hist_window).mean()
    future_avg = volume.shift(-1).rolling(horizon).mean()
    return (future_avg - hist_avg) / (hist_avg + 1e-8)


def compute_abnormal_turnover(turnover, horizon=5, hist_window=60):
    """Flag if turnover enters high percentile."""
    hist_80pct = turnover.rolling(hist_window).quantile(0.8)
    future_max = turnover.shift(-1).rolling(horizon).max()
    return (future_max > hist_80pct).astype(float)


def construct_labels_from_market_data(df_merged, horizons=(1, 2, 5, 10, 20)):
    """
    Main label construction from merged report-market data.

    df_merged must have:
      - report_date, stock_code
      - close_price (daily close)
      - benchmark_close (index close, e.g., CSI 300 or 万得全A)
      - volume (daily volume)
      - turnover (daily turnover rate)

    Returns DataFrame with labels for each report.
    """
    df = df_merged.copy()
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    labels_list = []

    for stock, group in df.groupby("stock_code"):
        group = group.sort_values("date").reset_index(drop=True)

        # Forward returns
        for h in horizons:
            group[f"fwd_ret_{h}d"] = group["close_price"].shift(-h) / group["close_price"] - 1

        # Benchmark-adjusted returns
        for h in horizons:
            group[f"fwd_bench_ret_{h}d"] = group["close_price"].shift(-h) / group["close_price"] - \
                group["benchmark_close"].shift(-h) / group["benchmark_close"]

        # Future volatility
        group["fwd_vol_10d"] = compute_future_volatility(
            group["close_price"].pct_change(), horizon=10
        )

        # Max drawdown
        group["fwd_maxdd_10d"] = compute_max_drawdown(group["close_price"], horizon=10)

        # Volume change
        group["fwd_vol_change_5d"] = compute_volume_change(group["volume"], horizon=5)

        # Abnormal turnover
        group["fwd_abn_turnover_5d"] = compute_abnormal_turnover(group["turnover"], horizon=5)

        labels_list.append(group)

    result = pd.concat(labels_list, ignore_index=True)
    return result


def discretize_labels(df, label_col, train_mask, n_classes=3):
    """
    Discretize continuous labels into classes using training-set quantiles.
    train_mask: boolean Series, True for training samples.
    """
    train_vals = df.loc[train_mask, label_col].dropna()
    if n_classes == 3:
        lo = train_vals.quantile(0.3)
        hi = train_vals.quantile(0.7)

        def classify(x):
            if pd.isna(x):
                return np.nan
            if x <= lo:
                return 0  # down
            elif x <= hi:
                return 1  # neutral
            else:
                return 2  # up

        return df[label_col].apply(classify)

    # For binary
    median = train_vals.median()
    return (df[label_col] > median).astype(float)


def match_reports_to_market(reports_df, market_df, benchmark_df=None):
    """
    Match reports to next tradable date's market data.

    Principle: if report date is non-trading day or published after close,
    first tradable day is next trading day (t+1).
    If intraday timestamp available, can use same day.

    Reports are matched to the NEXT trading day's data for label computation.
    """
    # Get trading calendar from market data
    trading_days = sorted(market_df["date"].unique())

    df = reports_df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])

    # Map report_date to next available trading day
    next_trading = {}
    for d in df["report_date"].unique():
        d = pd.Timestamp(d)
        # Find first trading day >= d+1 (next day)
        candidates = [td for td in trading_days if td >= d + pd.Timedelta(days=1)]
        if candidates:
            next_trading[d] = candidates[0]
        else:
            next_trading[d] = None

    df["tradable_date"] = df["report_date"].map(next_trading)
    df = df.dropna(subset=["tradable_date"])

    # Merge with market data on stock_code and tradable_date
    market_df["date"] = pd.to_datetime(market_df["date"])
    merged = df.merge(
        market_df, left_on=["stock_code", "tradable_date"],
        right_on=["stock_code", "date"], how="inner"
    )

    # Merge benchmark if provided
    if benchmark_df is not None:
        benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
        merged = merged.merge(
            benchmark_df[["date", "close_price"]].rename(
                columns={"close_price": "benchmark_close"}
            ),
            on="date", how="left"
        )

    print(f"[match] {len(df)} reports -> {len(merged)} matched to market data")
    print(f"[match] {df['stock_code'].nunique()} stocks with reports -> "
          f"{merged['stock_code'].nunique()} with market data")

    return merged


def main():
    print("=" * 60)
    print("SRTP Label Construction")
    print("=" * 60)

    # Load reports
    reports = pd.read_csv(DATA_DIR / "reports_cleaned.csv")
    print(f"Reports: {len(reports):,}")

    # This module requires market data from CSMAR
    # For now, print the expected data format

    print("""
    Label construction requires the following CSMAR data:
    1. Daily stock quotes: date, stock_code, open, close, high, low, volume, amount, turnover_rate
    2. Benchmark index: date, close (e.g., 000300 for CSI 300, or 881001 for Wind A)
    3. Stock status: date, stock_code, is_st, is_suspended, is_limit_up, is_limit_down

    When market data is available, use match_reports_to_market() then
    construct_labels_from_market_data() to generate labels.
    """)


if __name__ == "__main__":
    main()
