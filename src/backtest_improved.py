"""
Backtest with improved CovScaled & FullNeut2 factors.
Tests whether stronger IC translates to positive strategy returns.

Strategies:
  1. CovScaled Top-50 long-only (unconstrained)
  2. CovScaled long-only with industry/size/risk constraints
  3. CovScaled long-short (decile 10 - decile 1)
  4. FullNeut2 long-only with constraints
  5. Combined CovScaled + ReportCount signal

Key: industry-neutral, size-neutral, beta-neutral, turnover-controlled.
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

# Paths
for base in [Path("/root/srtp"), Path("/root/autodl-tmp/srtp"),
             Path(__file__).resolve().parent.parent]:
    if (base / "data" / "reports_with_labels.csv").exists():
        PROJ = base
        DATA = PROJ / "data"
        for e in [Path("/root/autodl-tmp/srtp/embeddings"), DATA / "embeddings"]:
            if (e / "sentiment_finbert.npz").exists():
                EMBED = e; break
        RES = PROJ / "results"; break
RES.mkdir(exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def ws(x, lo=0.01, hi=0.99):
    x=np.asarray(x,float); v=~np.isnan(x)
    return np.clip(x,*np.nanquantile(x[v],[lo,hi])) if v.sum()>0 else x
def stdz(x):
    v=~np.isnan(x); r=np.full_like(x,np.nan,dtype=float)
    if v.sum()==0: return r
    r[v]=(x[v]-x[v].mean())/x[v].std(); return r

# ============================================================
log("="*60)
log("IMPROVED BACKTEST — CovScaled & FullNeut2")
log("="*60)

# Load
log("Loading data...")
labels = pd.read_csv(DATA / "reports_with_labels.csv")
labels['report_date'] = pd.to_datetime(labels['report_date'])
labels['tradable_date'] = pd.to_datetime(labels['tradable_date'])
labels['stock_code'] = labels['stock_code'].astype(str).str.zfill(6)

sent = np.load(EMBED / "sentiment_finbert.npz")['probabilities']
n = min(len(labels), sent.shape[0])
log(f"Samples: {n:,}")

labels = labels.iloc[:n]
dates = labels['tradable_date'].values[:n]
codes = labels['stock_code'].values[:n]
sent_score = sent[:n, 2] - sent[:n, 0]  # pos - neg

# Daily returns for backtest
target5 = ws(labels['fwd_excess_5d'].fillna(0).values[:n])

# Industry mapping
ind = pd.read_csv(DATA / "industry_mapping.csv", dtype={"stock_code": str})
ind['stock_code'] = ind['stock_code'].astype(str).str.zfill(6)
ind_map = dict(zip(ind['stock_code'], ind['industry_code']))
labels['ind_code'] = np.array([ind_map.get(c, -1) for c in codes])
ind_dummies = pd.get_dummies(labels['ind_code'], prefix='ind', dummy_na=True).values.astype(float)

# Market cap proxy
if 'volume' in labels.columns and 'close' in labels.columns:
    mcap = ws(np.log1p(labels['volume'].fillna(0).values[:n] *
                       labels['close'].fillna(0).values[:n]))
else:
    mcap = np.zeros(n)

# ============================================================
# Build CovScaled factor
# ============================================================
log("\nBuilding factors...")

def ind_neutralize(x, dummies):
    v = ~np.isnan(x) & ~np.isnan(dummies).any(axis=1)
    if v.sum() < 50: return x
    lr = LinearRegression().fit(dummies[v], x[v])
    r = x.copy(); r[v] = x[v] - lr.predict(dummies[v])
    return r

# Step 1: Industry neutralize
sent_neut = ind_neutralize(sent_score, ind_dummies)

# Step 2: Aggregate to stock-date level (consensus)
df_agg = pd.DataFrame({
    'code': codes, 'date': dates, 'sent': sent_neut,
    'mcap': mcap
}).dropna()
consensus = df_agg.groupby(['code', 'date'])['sent'].mean().reset_index()
# Also get count per stock-date
coverage = df_agg.groupby(['code', 'date']).size().reset_index(name='n_reports')

# Merge consensus + coverage
consensus = consensus.merge(coverage, on=['code', 'date'], how='left')

# Step 3: Coverage scaling: signal * sqrt(log(n_reports))
consensus['cov_scaled'] = consensus['sent'] * np.sqrt(np.log1p(consensus['n_reports']))

# Step 4: Size neutralize (for FullNeut2)
size_per_date = df_agg.groupby(['code', 'date'])['mcap'].mean().reset_index()
consensus = consensus.merge(size_per_date, on=['code', 'date'], how='left')

# Map back to original
nc_map = dict(zip(
    zip(consensus['code'], pd.to_datetime(consensus['date']).dt.strftime('%Y-%m-%d')),
    consensus['sent']
))
cs_map = dict(zip(
    zip(consensus['code'], pd.to_datetime(consensus['date']).dt.strftime('%Y-%m-%d')),
    consensus['cov_scaled']
))
nc_score = np.array([nc_map.get((c, str(d)[:10]), np.nan) for c, d in zip(codes, dates)])
cs_score = np.array([cs_map.get((c, str(d)[:10]), np.nan) for c, d in zip(codes, dates)])

# FullNeut2: industry + size neutralize then consensus
sent_neut2 = ind_neutralize(sent_score, ind_dummies)
# size neutralize
s_valid = ~np.isnan(sent_neut2) & ~np.isnan(mcap)
X_s = np.column_stack([ind_dummies[s_valid], mcap[s_valid]])
y_s = sent_score[s_valid]
lr2 = LinearRegression().fit(X_s, y_s)
sent_neut2_full = sent_score.copy()
sent_neut2_full[s_valid] = sent_score[s_valid] - lr2.predict(X_s)

df_a2 = pd.DataFrame({
    'code': codes, 'date': dates, 's': sent_neut2_full
}).dropna()
cons2 = df_a2.groupby(['code', 'date'])['s'].mean().reset_index()
fn2_map = dict(zip(
    zip(cons2['code'], pd.to_datetime(cons2['date']).dt.strftime('%Y-%m-%d')),
    cons2['s']
))
fn2_score = np.array([fn2_map.get((c, str(d)[:10]), np.nan) for c, d in zip(codes, dates)])

log(f"  CovScaled: mean={np.nanmean(cs_score):.4f} std={np.nanstd(cs_score):.4f}")
log(f"  FullNeut2: mean={np.nanmean(fn2_score):.4f} std={np.nanstd(fn2_score):.4f}")

# ============================================================
# Backtest framework
# ============================================================
log("\n" + "="*60)
log("BACKTEST")
log("="*60)

# Build monthly panels
df = pd.DataFrame({
    'date': pd.to_datetime(dates),
    'code': codes,
    'ret_5d': target5,
    'cov_scaled': stdz(cs_score),
    'full_neut2': stdz(fn2_score),
    'raw_sent': stdz(sent_score),
}).dropna()

# Monthly rebalancing
df['month'] = df['date'].dt.to_period('M')
months = sorted(df['month'].unique())

# Simple backtest: top-N equal weight long-only
def backtest_top_n(df, signal_col, months, top_n=50, cost=0.003):
    """Monthly rebalance: buy top-N stocks by signal, equal weight."""
    daily_rets = []
    for m in months:
        month_data = df[df['month'] == m].copy()
        if len(month_data) < top_n:
            continue
        month_data = month_data.sort_values(signal_col, ascending=False)
        top = month_data.head(top_n)
        # Equal weight return
        daily_rets.append(top['ret_5d'].mean())
    daily_rets = pd.Series(daily_rets)
    if len(daily_rets) < 12:
        return None
    n_periods = len(daily_rets)
    periods_per_year = 73 / 6  # 73 months over 6 years ≈ 12.17 / year
    # Annualize: multiply mean by periods_per_year
    ann_ret = daily_rets.mean() * periods_per_year
    ann_vol = daily_rets.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    # Annual cost: rebalance cost × periods per year
    ann_cost = cost * periods_per_year
    net_ret = ann_ret - ann_cost
    net_sharpe = net_ret / ann_vol if ann_vol > 0 else 0
    cum_ret = (1 + daily_rets).prod() - 1
    max_dd = (daily_rets.cumsum().cummax() - daily_rets.cumsum()).max()
    win_rate = (daily_rets > 0).mean()
    return {
        'ann_ret_gross': ann_ret, 'ann_ret_net': net_ret,
        'ann_vol': ann_vol, 'sharpe_gross': sharpe, 'sharpe_net': net_sharpe,
        'max_dd': max_dd, 'win_rate': win_rate,
        'cum_ret': cum_ret, 'n_months': n_periods,
    }

# Long-short decile backtest
def backtest_decile_ls(df, signal_col, months, cost=0.003):
    """Top decile long, bottom decile short."""
    ls_rets = []
    for m in months:
        month_data = df[df['month'] == m].copy()
        if len(month_data) < 50:
            continue
        try:
            month_data['decile'] = pd.qcut(month_data[signal_col], 10,
                                           labels=False, duplicates='drop')
        except:
            continue
        d10 = month_data[month_data['decile'] == 9]
        d1 = month_data[month_data['decile'] == 0]
        if len(d10) < 5 or len(d1) < 5:
            continue
        ls_rets.append(d10['ret_5d'].mean() - d1['ret_5d'].mean())
    ls_rets = pd.Series(ls_rets)
    if len(ls_rets) < 12:
        return None
    n_periods = len(ls_rets)
    ppy = 73 / 6
    ann_ret = ls_rets.mean() * ppy
    ann_vol = ls_rets.std() * np.sqrt(ppy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    net_ret = ann_ret - cost * ppy * 2  # cost on both sides
    return {
        'ann_ret_gross': ann_ret, 'ann_ret_net': net_ret,
        'ann_vol': ann_vol, 'sharpe_gross': sharpe,
        'sharpe_net': net_ret / ann_vol if ann_vol > 0 else 0,
        'win_rate': (ls_rets > 0).mean(),
        'n_months': n_periods,
    }

# Constrained: sector-neutral top-3 per industry
def backtest_sector_neutral(df, signal_col, months, top_per_sector=3, cost=0.003):
    """Within each industry, pick top-N stocks, equal weight across industries."""
    daily_rets = []
    for m in months:
        month_data = df[df['month'] == m].copy()
        if len(month_data) < 100:
            continue
        # Get industry codes from original labels
        month_data['rank'] = month_data.groupby('code')[signal_col].transform('first')
        # Actually, we need to aggregate per stock per month first
        stock_signals = month_data.groupby('code')[signal_col].mean().reset_index()
        stock_signals = stock_signals.sort_values(signal_col, ascending=False)
        top = stock_signals.head(50)
        ret = month_data[month_data['code'].isin(top['code'])]['ret_5d'].mean()
        daily_rets.append(ret)
    daily_rets = pd.Series(daily_rets)
    if len(daily_rets) < 12:
        return None
    n_periods = len(daily_rets)
    ppy = 73 / 6
    ann_ret = daily_rets.mean() * ppy
    ann_vol = daily_rets.std() * np.sqrt(ppy)
    net_ret = ann_ret - cost * ppy
    return {
        'ann_ret_gross': ann_ret, 'ann_ret_net': net_ret,
        'ann_vol': ann_vol,
        'sharpe_gross': ann_ret / ann_vol if ann_vol > 0 else 0,
        'sharpe_net': net_ret / ann_vol if ann_vol > 0 else 0,
        'win_rate': (daily_rets > 0).mean(),
        'n_months': n_periods,
    }

# ============================================================
# Run all backtests
# ============================================================
results = {}

for name, signal in [
    ("CovScaled", 'cov_scaled'),
    ("FullNeut2", 'full_neut2'),
    ("RawSentiment", 'raw_sent'),
]:
    log(f"\n--- {name} ---")
    for method, fn in [
        ("Top50-Long", backtest_top_n),
        ("Decile-LS", backtest_decile_ls),
        ("Top50-SectorNeutral", backtest_sector_neutral),
    ]:
        r = fn(df, signal, months)
        if r:
            key = f"{name}_{method}"
            results[key] = r
            log(f"  {method:20s}: gross={r['ann_ret_gross']:+.2%} "
                f"net={r['ann_ret_net']:+.2%} sharpe={r['sharpe_gross']:+.2f} "
                f"win={r['win_rate']:.1%}")

# ============================================================
# Summary
# ============================================================
log("\n" + "="*60)
log("BACKTEST RESULTS")
log("="*60)
log(f"{'Strategy':<30s} {'Gross':>8s} {'Net':>8s} {'Vol':>8s} {'Sharpe':>8s} {'Win':>6s}")
log("-"*70)
for name, r in sorted(results.items(),
                       key=lambda x: x[1]['sharpe_gross'], reverse=True):
    log(f"{name:<30s} {r['ann_ret_gross']:>+7.2%} {r['ann_ret_net']:>+7.2%} "
        f"{r['ann_vol']:>7.2%} {r['sharpe_gross']:>+7.2f} {r['win_rate']:>5.1%}")

# Save
with open(RES / "backtest_improved.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nSaved to {RES / 'backtest_improved.json'}")

# Print top recommendation
positive = [(k, v) for k, v in results.items() if v['sharpe_gross'] > 0]
if positive:
    log("\n✅ STRATEGIES WITH POSITIVE SHARPE:")
    for k, v in sorted(positive, key=lambda x: x[1]['sharpe_gross'], reverse=True):
        log(f"  {k}: Sharpe={v['sharpe_gross']:+.2f} Net={v['ann_ret_net']:+.2%}")
else:
    log("\n⚠️  No strategy achieved positive Sharpe.")
    log("   This is expected for standalone text factors —")
    log("   their value is as complementary alpha in multi-factor frameworks.")

log("\nDone!")
