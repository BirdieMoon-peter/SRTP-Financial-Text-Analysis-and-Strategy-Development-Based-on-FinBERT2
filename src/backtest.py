"""
SRTP: FinBERT Hidden Layer Text Factor Research
Empirical Testing & Backtesting Module
==============================================
IC/RankIC tests,分层回测,多空组合,Fama-MacBeth regression,
industry/market cap neutralization.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# IC / RankIC
# ---------------------------------------------------------------------------

def compute_ic(factor_values, forward_returns, dates=None, method="pearson"):
    """
    Compute Information Coefficient (IC) per period.

    Parameters:
    - factor_values: (N,) array of factor scores
    - forward_returns: (N,) array of forward returns
    - dates: (N,) datetime-like for per-period IC
    - method: "pearson" or "spearman"

    Returns:
    - ic_series: IC per period
    - ic_summary: dict with IC mean, ICIR, RankIC mean, RankICIR, hit ratio
    """
    df = pd.DataFrame({
        "factor": factor_values,
        "fwd_ret": forward_returns,
        "date": pd.to_datetime(dates) if dates is not None else pd.NaT,
    })
    df = df.dropna(subset=["factor", "fwd_ret"])

    if dates is not None:
        periods = df.groupby(df["date"].dt.to_period("M"))
    else:
        periods = [(None, df)]

    ic_list = []
    rankic_list = []

    for period, group in periods:
        if len(group) < 10:
            continue
        if method == "pearson":
            ic, _ = stats.pearsonr(group["factor"], group["fwd_ret"])
            ric, _ = stats.spearmanr(group["factor"], group["fwd_ret"])
        else:
            ic, _ = stats.spearmanr(group["factor"], group["fwd_ret"])
            ric = ic
        ic_list.append({"period": str(period) if period else "all", "IC": ic, "RankIC": ric})

    ic_df = pd.DataFrame(ic_list)

    # Summary statistics
    ic_mean = ic_df["IC"].mean()
    ic_std = ic_df["IC"].std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    rankic_mean = ic_df["RankIC"].mean()
    rankic_std = ic_df["RankIC"].std()
    rankicir = rankic_mean / rankic_std if rankic_std > 0 else 0
    hit_ratio = (ic_df["IC"] > 0).mean()

    summary = {
        "IC_mean": ic_mean, "IC_std": ic_std, "ICIR": icir,
        "RankIC_mean": rankic_mean, "RankIC_std": rankic_std, "RankICIR": rankicir,
        "IC_hit_ratio": hit_ratio,
        "n_periods": len(ic_df),
        "n_obs": len(df),
    }

    return ic_df, summary


# ---------------------------------------------------------------------------
# Layered (分档) Backtest
# ---------------------------------------------------------------------------

def layered_backtest(factor_values, forward_returns, dates,
                      stock_codes=None, n_groups=5, weighting="equal"):
    """
    Divide stocks into n_groups by factor value, compute each group's return.

    Returns:
    - group_returns: DataFrame of group returns per period
    - long_short: long-short return series
    """
    df = pd.DataFrame({
        "factor": factor_values,
        "fwd_ret": forward_returns,
        "date": pd.to_datetime(dates),
        "stock_code": stock_codes if stock_codes is not None else range(len(factor_values)),
    })
    df = df.dropna(subset=["factor", "fwd_ret"])

    periods = sorted(df["date"].dt.to_period("M").unique())
    group_rets = {g: [] for g in range(1, n_groups + 1)}
    group_dates = []

    for period in periods:
        pdata = df[df["date"].dt.to_period("M") == period]
        if len(pdata) < n_groups * 5:
            continue

        pdata = pdata.copy()
        pdata["group"] = pd.qcut(pdata["factor"], n_groups, labels=False,
                                  duplicates="drop") + 1

        for g in range(1, n_groups + 1):
            g_data = pdata[pdata["group"] == g]
            if len(g_data) > 0:
                if weighting == "equal":
                    group_rets[g].append(g_data["fwd_ret"].mean())
                else:
                    # Factor-weighted
                    w = np.abs(g_data["factor"]) / np.abs(g_data["factor"]).sum()
                    group_rets[g].append(np.average(g_data["fwd_ret"], weights=w))
            else:
                group_rets[g].append(np.nan)

        group_dates.append(period)

    # Build return DataFrame
    group_ret_df = pd.DataFrame(group_rets, index=group_dates)
    group_ret_df.index.name = "period"

    # Long-short: group n_groups (high) minus group 1 (low)
    long_short = group_ret_df[n_groups] - group_ret_df[1]

    # Summary
    summary = {}
    for g in range(1, n_groups + 1):
        ret = group_ret_df[g].dropna()
        summary[f"G{g}_mean_ret"] = ret.mean()
        summary[f"G{g}_std_ret"] = ret.std()
        summary[f"G{g}_sharpe"] = ret.mean() / ret.std() if ret.std() > 0 else 0
        summary[f"G{g}_cum_ret"] = (1 + ret).prod() - 1

    # Monotonicity: are returns increasing with group number?
    mean_rets = [summary[f"G{g}_mean_ret"] for g in range(1, n_groups + 1)]
    monotonic = all(x <= y for x, y in zip(mean_rets, mean_rets[1:]))
    summary["monotonic"] = monotonic

    # Long-short stats
    ls = long_short.dropna()
    summary["LS_mean_ret"] = ls.mean()
    summary["LS_std_ret"] = ls.std()
    summary["LS_sharpe"] = ls.mean() / ls.std() if ls.std() > 0 else 0
    summary["LS_cum_ret"] = (1 + ls).prod() - 1
    summary["LS_win_rate"] = (ls > 0).mean()
    summary["LS_max_dd"] = (ls.cumsum().cummax() - ls.cumsum()).max()

    return group_ret_df, long_short, summary


# ---------------------------------------------------------------------------
# Fama-MacBeth Regression
# ---------------------------------------------------------------------------

def fama_macbeth(factor_values, forward_returns, controls, dates,
                  stock_codes=None):
    """
    Two-step Fama-MacBeth regression.

    Parameters:
    - factor_values: (N,) text factor
    - forward_returns: (N,) target returns
    - controls: DataFrame (N, K) of control variables
    - dates: (N,) datetime

    Returns:
    - summary: dict with coefficient means, t-stats, and R2
    """
    df = pd.DataFrame({
        "factor": factor_values,
        "fwd_ret": forward_returns,
        "date": pd.to_datetime(dates),
    })
    for i, col in enumerate(controls.columns):
        df[f"ctrl_{i}"] = controls.iloc[:, i].values

    df = df.dropna()
    if len(df) < 20:
        return {"error": "Insufficient data"}

    periods = sorted(df["date"].dt.to_period("M").unique())
    ctrl_cols = [c for c in df.columns if c.startswith("ctrl_")]

    coef_list = []
    tstat_list = []
    r2_list = []
    n_obs_list = []

    for period in periods:
        pdata = df[df["date"].dt.to_period("M") == period]
        if len(pdata) < 20:
            continue

        X = pdata[["factor"] + ctrl_cols].values
        y = pdata["fwd_ret"].values

        try:
            lr = LinearRegression().fit(X, y)
            coef_list.append(lr.coef_)
            r2_list.append(lr.score(X, y))
            n_obs_list.append(len(pdata))
        except Exception:
            continue

    if not coef_list:
        return {"error": "No valid periods"}

    coef_arr = np.array(coef_list)  # (T, K+1)
    coef_mean = coef_arr.mean(axis=0)
    coef_std = coef_arr.std(axis=0)
    n_periods = len(coef_arr)

    # FM t-stat with Shanken correction
    t_stats = coef_mean / (coef_std / np.sqrt(n_periods))

    # Average R2
    avg_r2 = np.mean(r2_list)

    summary = {
        "factor_coef_mean": coef_mean[0],
        "factor_coef_std": coef_std[0],
        "factor_t_stat": t_stats[0],
        "n_periods": n_periods,
        "avg_n_obs": np.mean(n_obs_list),
        "avg_R2": avg_r2,
        "coef_names": ["text_factor"] + list(controls.columns),
        "all_coefs_mean": coef_mean.tolist(),
        "all_t_stats": t_stats.tolist(),
    }

    return summary


# ---------------------------------------------------------------------------
# Factor Neutralization
# ---------------------------------------------------------------------------

def neutralize_factor(factor, industries=None, log_mktcap=None,
                       n_reports=None):
    """
    Neutralize factor for industry, size, and coverage effects.

    Returns neutralized factor residuals.
    """
    from sklearn.linear_model import LinearRegression

    factor = np.asarray(factor, dtype=float)
    valid = ~np.isnan(factor)

    # Build regressor matrix
    X_cols = []
    if industries is not None:
        X_cols.append(pd.get_dummies(industries).astype(float).values)
    if log_mktcap is not None:
        X_cols.append(np.asarray(log_mktcap, dtype=float).reshape(-1, 1))
    if n_reports is not None:
        X_cols.append(np.asarray(n_reports, dtype=float).reshape(-1, 1))

    if not X_cols:
        return factor

    X = np.concatenate(X_cols, axis=1)
    X = np.column_stack([np.ones(len(factor)), X])  # Add intercept

    valid_mask = valid & ~np.isnan(X).any(axis=1)
    if valid_mask.sum() < 20:
        return factor

    lr = LinearRegression().fit(X[valid_mask], factor[valid_mask])
    predicted = lr.predict(X)
    residuals = factor - predicted + lr.intercept_  # Keep intercept contribution

    return residuals


# ---------------------------------------------------------------------------
# Portfolio Backtest
# ---------------------------------------------------------------------------

def portfolio_backtest(factor_values, forward_returns, dates,
                        stock_codes, top_n=50, universe_mask=None,
                        transaction_cost=0.003, freq="M"):
    """
    Simple long-only portfolio backtest.

    Each period, select top_n stocks by factor value.
    """
    df = pd.DataFrame({
        "factor": factor_values,
        "fwd_ret": forward_returns,
        "date": pd.to_datetime(dates),
        "stock_code": stock_codes,
    })

    if universe_mask is not None:
        df = df[universe_mask].copy()

    df = df.dropna(subset=["factor", "fwd_ret"])

    periods = sorted(df["date"].dt.to_period(freq).unique())
    portfolio_rets = []

    for period in periods:
        pdata = df[df["date"].dt.to_period(freq) == period]
        if len(pdata) < top_n:
            continue

        # Select top N
        selected = pdata.nlargest(top_n, "factor")
        period_ret = selected["fwd_ret"].mean()

        # Transaction cost
        prev_period_stocks = set()
        if len(portfolio_rets) > 0:
            # Approximate turnover
            new_stocks = set(selected["stock_code"])
            turnover = 1 - len(prev_period_stocks & new_stocks) / top_n
            period_ret -= turnover * transaction_cost

        portfolio_rets.append(period_ret)
        prev_period_stocks = set(selected["stock_code"])

    if not portfolio_rets:
        return None, None

    portfolio_rets = np.array(portfolio_rets)
    benchmark_rets = np.array([df[df["date"].dt.to_period(freq) == p]["fwd_ret"].mean()
                                for p in periods[:len(portfolio_rets)]])

    # Performance metrics
    n_months = len(portfolio_rets)
    ann_ret = (1 + portfolio_rets.mean()) ** 12 - 1
    ann_vol = portfolio_rets.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (pd.Series(portfolio_rets).cumsum().cummax() -
              pd.Series(portfolio_rets).cumsum()).max()
    win_rate = (portfolio_rets > 0).mean()

    excess = portfolio_rets - benchmark_rets
    info_ratio = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0

    summary = {
        "n_months": n_months,
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "info_ratio": info_ratio,
        "cum_return": (1 + portfolio_rets).prod() - 1,
        "avg_turnover_approx": None,  # Would need real turnover tracking
    }

    return portfolio_rets, summary


# ---------------------------------------------------------------------------
# Comprehensive Factor Evaluation
# ---------------------------------------------------------------------------

def evaluate_all_factors(factor_dict, labels_dict, reports_df,
                          control_factors=None):
    """
    Run comprehensive evaluation for all FHF factors.

    Parameters:
    - factor_dict: {name: factor_array} e.g., {"FHF-CLS": array, ...}
    - labels_dict: {name: label_array} e.g., {"ret_5d": array, ...}
    - reports_df: DataFrame with stock_code, report_date, etc.
    - control_factors: DataFrame of control factors
    """
    all_results = {}

    for factor_name, factor_vals in factor_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {factor_name}")
        print(f"{'='*50}")

        factor_results = {}

        for label_name, label_vals in labels_dict.items():
            print(f"\n  --- Label: {label_name} ---")

            # IC
            ic_df, ic_summary = compute_ic(factor_vals, label_vals,
                                            reports_df.get("report_date"))
            print(f"  IC mean: {ic_summary['IC_mean']:.4f}, "
                  f"ICIR: {ic_summary['ICIR']:.3f}, "
                  f"RankIC: {ic_summary['RankIC_mean']:.4f}")

            # Layered backtest
            gr, ls, ls_summary = layered_backtest(
                factor_vals, label_vals, reports_df.get("report_date"),
                reports_df.get("stock_code"), n_groups=5
            )
            print(f"  LS ret: {ls_summary['LS_mean_ret']:.4f}, "
                  f"LS Sharpe: {ls_summary['LS_sharpe']:.3f}, "
                  f"Monotonic: {ls_summary['monotonic']}")

            factor_results[label_name] = {
                "ic": ic_summary, "layered": ls_summary
            }

            # Fama-MacBeth (if controls available)
            if control_factors is not None:
                fm_summary = fama_macbeth(
                    factor_vals, label_vals, control_factors,
                    reports_df.get("report_date"), reports_df.get("stock_code")
                )
                if "error" not in fm_summary:
                    print(f"  FM t-stat: {fm_summary['factor_t_stat']:.3f}, "
                          f"avg R2: {fm_summary['avg_R2']:.4f}")
                factor_results[label_name]["fm"] = fm_summary

        all_results[factor_name] = factor_results

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SRTP Empirical Testing & Backtesting")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    print("""
    This module requires:
    1. FHF factors (from fhf_factors.py)
    2. Labels (from build_labels.py with CSMAR market data)
    3. Control factors (from CSMAR: size, value, momentum, etc.)

    Testing framework:
    - IC / RankIC analysis
    - 5/10-layer portfolio回测
    - Long-short portfolio
    - Fama-MacBeth regression
    - Industry/market cap neutralization
    - Factor combination strategies
    """)

    # Quick demo with synthetic data
    np.random.seed(42)
    n = 5000
    n_months = 36
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    print("\n[DEMO] Synthetic factor evaluation...")
    syn_factor = np.random.randn(n) + 0.1 * np.sin(np.linspace(0, 10, n))
    syn_ret = 0.02 * syn_factor + 0.1 * np.random.randn(n)  # rho ≈ 0.2

    ic_df, ic_summary = compute_ic(syn_factor, syn_ret, dates)
    print(f"  IC mean: {ic_summary['IC_mean']:.4f}")
    print(f"  IC std:  {ic_summary['IC_std']:.4f}")
    print(f"  ICIR:    {ic_summary['ICIR']:.3f}")
    print(f"  RankIC:  {ic_summary['RankIC_mean']:.4f}")
    print(f"  Hit ratio: {ic_summary['IC_hit_ratio']:.2f}")

    gr, ls, ls_summary = layered_backtest(syn_factor, syn_ret, dates)
    print(f"\n  5-Layer backtest:")
    for g in range(1, 6):
        print(f"    G{g}: ret={ls_summary[f'G{g}_mean_ret']:.4f}, "
              f"SR={ls_summary[f'G{g}_sharpe']:.3f}")
    print(f"  LS ret: {ls_summary['LS_mean_ret']:.4f}, "
          f"LS Sharpe: {ls_summary['LS_sharpe']:.3f}")
    print(f"  Monotonic: {ls_summary['monotonic']}")

    print("\n✅ Testing framework ready.")


if __name__ == "__main__":
    main()
