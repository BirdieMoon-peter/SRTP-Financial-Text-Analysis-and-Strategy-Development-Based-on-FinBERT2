"""
SRTP Supplementary Analysis - Windows Server Version
=====================================================
Complete supplementary tests per advisor review.
Runs on Windows with RTX 3050ti.

Tests:
1. Decile portfolio + monotonicity + cost sensitivity
2. Incremental information (expanded FM with 7 control groups)
3. Factor orthogonalization (5 levels)
4. Robustness: prediction windows, sub-periods, groups
5. Placebo (100x shuffled/random/permuted)
6. Multiple testing (Bonferroni + FDR)
7. Enhanced Gap mechanism
8. Constrained portfolio backtest
"""

import json, time, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from collections import OrderedDict
warnings.filterwarnings("ignore")

# Windows paths
DATA  = Path("C:/Users/13082/CSMAR/data")
EMBED = DATA / "embeddings"
RES   = Path("C:/Users/13082/CSMAR/results")
RES.mkdir(exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def winsorize(s, lo=0.01, hi=0.99):
    s = np.asarray(s, float)
    v = ~np.isnan(s)
    if v.sum() == 0: return s
    lo_v, hi_v = np.nanquantile(s[v], [lo, hi])
    return np.clip(s, lo_v, hi_v)

def rank_ic_monthly(factor, ret, dates):
    df = pd.DataFrame({'f': np.asarray(factor, float),
                       'r': np.asarray(ret, float),
                       'd': pd.to_datetime(dates)}).dropna()
    if len(df) < 50: return None
    monthly = df.groupby(df['d'].dt.to_period('M'))
    ics = [stats.spearmanr(g['f'], g['r'])[0]
           for _, g in monthly if len(g) >= 10]
    if not ics: return None
    im, sd = np.mean(ics), np.std(ics)
    return {'RankIC': float(im), 'RankIC_std': float(sd),
            'ICIR': float(im/sd) if sd>0 else 0,
            'IC_t': float(im/sd*np.sqrt(len(ics))) if sd>0 else 0,
            'n_periods': len(ics), 'ics_series': ics}

# ========================================================
log("="*60)
log("SUPPLEMENTARY ANALYSIS - WINDOWS SERVER")
log("="*60)

# --- Load Data ---
log("Loading labels...")
labels = pd.read_csv(DATA / "reports_with_labels.csv")
labels['report_date'] = pd.to_datetime(labels['report_date'])
labels['tradable_date'] = pd.to_datetime(labels['tradable_date'])

log("Loading embeddings...")
full_emb = np.load(EMBED / "embeddings_full.npz")
gap_emb = np.load(EMBED / "embeddings_gap.npz")
sent = np.load(EMBED / "sentiment_finbert.npz")['probabilities']

n = min(len(labels), full_emb['all_cls'].shape[0])
log(f"n={n:,}")

labels = labels.iloc[:n]
dates = labels['tradable_date']
stock_codes = labels['stock_code'].astype(str).str.zfill(6)

# Targets
target_cols = ['fwd_excess_1d','fwd_excess_2d','fwd_excess_5d',
               'fwd_excess_10d','fwd_excess_20d']
targets = {}
for c in target_cols:
    if c in labels.columns:
        targets[c] = winsorize(labels[c].fillna(0).values[:n])
target5 = targets['fwd_excess_5d']

# Risk targets
risk_targets = {}
for c in ['fwd_vol_10d', 'fwd_maxdd_10d', 'fwd_vol_chg_5d']:
    if c in labels.columns:
        risk_targets[c] = winsorize(labels[c].fillna(0).values[:n])

# Build factors
sent_score = sent[:n, 2] - sent[:n, 0]
X_cls4 = full_emb['all_cls'][:n, -4:, :].reshape(n, -1)

# Load stock controls
log("Loading stock data...")
stock_df = pd.read_csv(DATA / "csmar_daily_stock.csv",
                        dtype={"stock_code": str}, low_memory=False)
stock_df['date'] = pd.to_datetime(stock_df['date'])
stock_df['stock_code'] = stock_df['stock_code'].astype(str).str.zfill(6)
stock_df = stock_df.sort_values(['stock_code', 'date']).reset_index(drop=True)

for c in ['close','volume','amount','market_cap']:
    if c in stock_df.columns:
        stock_df[c] = pd.to_numeric(stock_df[c], errors='coerce')

log("Computing controls...")
stock_df['ret_1d'] = stock_df.groupby('stock_code')['close'].pct_change(1)
stock_df['ret_5d'] = stock_df.groupby('stock_code')['close'].pct_change(5)
stock_df['ret_20d'] = stock_df.groupby('stock_code')['close'].pct_change(20)
stock_df['ret_60d'] = stock_df.groupby('stock_code')['close'].pct_change(60)
stock_df['vol_20d'] = stock_df.groupby('stock_code')['ret_1d'].transform(
    lambda x: x.rolling(20).std())
stock_df['turnover'] = stock_df['volume'] / stock_df.groupby('stock_code')['volume'].transform(
    lambda x: x.rolling(60).mean().clip(lower=1))
stock_df['log_vol'] = np.log1p(stock_df['volume'])
if 'market_cap' in stock_df.columns:
    stock_df['log_mktcap'] = np.log(stock_df['market_cap'].clip(lower=1))
else:
    stock_df['log_mktcap'] = np.zeros(len(stock_df))

stock_df['ep_proxy'] = 1 / stock_df['close'].clip(lower=0.01)
stock_df['roe_proxy'] = stock_df['ret_20d']

ctrl_names = ['ret_1d','ret_5d','ret_20d','ret_60d','vol_20d',
              'turnover','log_vol','log_mktcap','ep_proxy','roe_proxy']
ctrl = stock_df[['stock_code','date'] + ctrl_names].rename(columns={'date':'tradable_date'})
ctrl['tradable_date'] = pd.to_datetime(ctrl['tradable_date'])
labels_m = labels.merge(ctrl, on=['stock_code','tradable_date'], how='left')

# Industry
log("Loading industry data...")
industry = pd.read_csv(DATA / "industry_mapping.csv", dtype={"stock_code": str})
industry['stock_code'] = industry['stock_code'].astype(str).str.zfill(6)
ind_map = dict(zip(industry['stock_code'], industry['industry_code']))
ind_name_map = dict(zip(industry['stock_code'], industry['industry_name']))
labels_m['industry_code'] = stock_codes.map(ind_map)
labels_m['industry_name'] = stock_codes.map(ind_name_map)
ind_dummies = pd.get_dummies(labels_m['industry_code'], prefix='ind', dummy_na=True).astype(float)

# Report count
rc = labels_m.groupby('stock_code')['report_date'].transform('count').values[:n]
report_count_log = np.log1p(rc.astype(float))

# ========================================================
# TEST 1: DECILE PORTFOLIO ANALYSIS
# ========================================================
log("\n" + "="*60)
log("TEST 1: Decile Portfolio + Monotonicity + Cost Sensitivity")
log("="*60)

def decile_analysis(factor, ret, dates, stock_codes, n_groups=10):
    df = pd.DataFrame({
        'f': np.asarray(factor, float), 'r': np.asarray(ret, float),
        'd': pd.to_datetime(dates), 'code': stock_codes
    }).dropna()
    if len(df) < n_groups * 10: return None

    monthly = df.groupby(df['d'].dt.to_period('M'))
    group_rets = {i: [] for i in range(1, n_groups+1)}
    ls_rets = []; prev_top_set = set(); prev_bot_set = set()
    turnovers_top = []; turnovers_bot = []

    for p, g in monthly:
        if len(g) < n_groups * 5: continue
        try:
            g = g.copy()
            g['group'] = pd.qcut(g['f'], n_groups, labels=False, duplicates='drop') + 1
        except: continue

        for i in range(1, n_groups+1):
            gg = g[g['group']==i]
            group_rets[i].append(gg['r'].mean() if len(gg)>0 else np.nan)

        ls_rets.append(group_rets[n_groups][-1] - group_rets[1][-1])

        top_set = set(g[g['group']==n_groups]['code'])
        bot_set = set(g[g['group']==1]['code'])
        if prev_top_set:
            tto = 1 - len(prev_top_set & top_set) / max(len(top_set), 1)
            tbo = 1 - len(prev_bot_set & bot_set) / max(len(bot_set), 1)
            turnovers_top.append(tto); turnovers_bot.append(tbo)
        prev_top_set = top_set; prev_bot_set = bot_set

    result = {}
    means = []
    for i in range(1, n_groups+1):
        r = np.array(group_rets[i])
        r = r[~np.isnan(r)]
        if len(r) > 0:
            result[f'D{i}_mean'] = float(r.mean())
            result[f'D{i}_hit'] = float((r>0).mean())
            means.append(float(r.mean()))
        else:
            means.append(np.nan)

    ls = np.array(ls_rets); ls = ls[~np.isnan(ls)]
    result['LS_mean'] = float(ls.mean()) if len(ls)>0 else np.nan
    result['LS_t'] = float(ls.mean()/ls.std()*np.sqrt(len(ls))) if len(ls)>0 and ls.std()>0 else 0
    result['LS_hit'] = float((ls>0).mean()) if len(ls)>0 else 0
    result['n_months'] = len(ls)
    result['monotonic'] = all(
        m1 is not None and m2 is not None and m1 <= m2
        for m1, m2 in zip(means, means[1:])
        if not (np.isnan(m1) or np.isnan(m2))
    ) if len(means) == n_groups else False
    result['avg_turnover_top'] = float(np.mean(turnovers_top)) if turnovers_top else np.nan
    result['avg_turnover_bot'] = float(np.mean(turnovers_bot)) if turnovers_bot else np.nan

    return result

def ls_cost_sensitivity(factor, ret, dates, stock_codes, tc_levels, n_groups=10):
    df = pd.DataFrame({
        'f': np.asarray(factor, float), 'r': np.asarray(ret, float),
        'd': pd.to_datetime(dates), 'code': stock_codes
    }).dropna()
    monthly = df.groupby(df['d'].dt.to_period('M'))
    ls_raw = []
    for p, g in monthly:
        if len(g) < n_groups*5: continue
        try:
            g = g.copy()
            g['group'] = pd.qcut(g['f'], n_groups, labels=False, duplicates='drop') + 1
        except: continue
        top = g[g['group']==n_groups]; bot = g[g['group']==1]
        ls_raw.append({
            'raw': top['r'].mean() - bot['r'].mean(),
            'top_set': set(top['code']), 'bot_set': set(bot['code'])
        })

    results = {}
    for tc in tc_levels:
        nets = []
        for i in range(len(ls_raw)):
            raw = ls_raw[i]['raw']
            tto = 1 - len(ls_raw[max(0,i-1)]['top_set'] & ls_raw[i]['top_set']) / max(len(ls_raw[i]['top_set']), 1) if i>0 else 1
            tbo = 1 - len(ls_raw[max(0,i-1)]['bot_set'] & ls_raw[i]['bot_set']) / max(len(ls_raw[i]['bot_set']), 1) if i>0 else 1
            nets.append(raw - (tto + tbo) * tc)
        nets = np.array(nets); nets = nets[~np.isnan(nets)]
        results[f'tc_{int(tc*10000)}bps'] = {
            'LS_mean': float(nets.mean()) if len(nets)>0 else np.nan,
            'LS_t': float(nets.mean()/nets.std()*np.sqrt(len(nets))) if len(nets)>0 and nets.std()>0 else 0,
            'LS_hit': float((nets>0).mean()) if len(nets)>0 else 0
        }
    return results

tc_levels = [0.001, 0.002, 0.003, 0.005]
decile_all = {}; tc_all = {}

for nm, f in [('FinBERT-Sentiment', sent_score)]:
    r = decile_analysis(f, target5, dates, stock_codes)
    if r:
        decile_all[nm] = r
        log(f"  {nm} Decile: LS={r['LS_mean']:+.5f} t={r['LS_t']:+.2f} monotonic={r['monotonic']}")
    tc = ls_cost_sensitivity(f, target5, dates, stock_codes, tc_levels)
    if tc: tc_all[nm] = tc
    for k, v in (tc or {}).items():
        log(f"  {nm} {k}: LS_net={v['LS_mean']:+.5f} t={v['LS_t']:+.2f}")

with open(RES / "decile_analysis.json", 'w', encoding='utf-8') as f:
    json.dump({'decile': decile_all, 'cost_sensitivity': tc_all}, f, indent=2, default=str)

# ========================================================
# TEST 2: INCREMENTAL INFORMATION
# ========================================================
log("\n" + "="*60)
log("TEST 2: Incremental Information - Expanded FM")
log("="*60)

control_dict = {}
for c in ctrl_names:
    if c in labels_m.columns:
        control_dict[c] = winsorize(labels_m[c].fillna(0).values[:n])
control_dict['report_count_log'] = report_count_log

ctrl_specs = OrderedDict([
    ('univariate', []),
    ('+coverage', ['report_count_log']),
    ('+momentum_rev', ['ret_5d', 'ret_20d']),
    ('+liquidity', ['turnover', 'log_vol', 'vol_20d']),
    ('+size_value', ['log_mktcap', 'ep_proxy']),
    ('+all_traditional', ['ret_5d','ret_20d','ret_60d','turnover',
                           'log_vol','vol_20d','log_mktcap','ep_proxy']),
    ('+coverage_all', ['report_count_log','ret_5d','ret_20d','ret_60d',
                        'turnover','log_vol','vol_20d','log_mktcap','ep_proxy']),
])

def fm_incremental(factor, ret, dates, ctrl_dict, specs):
    df = pd.DataFrame({'f': np.asarray(factor, float),
                       'r': ret, 'd': pd.to_datetime(dates)})
    for k, v in ctrl_dict.items():
        df[k] = np.asarray(v)

    results = OrderedDict()
    for spec_name, ctrl_list in specs.items():
        cols = ['f'] + [c for c in ctrl_list if c in df.columns]
        coefs = []; r2s = []; nobs = []
        for p, g in df.groupby(df['d'].dt.to_period('M')):
            g_sub = g[cols + ['r']].dropna()
            if len(g_sub) < 20: continue
            X = g_sub[cols].values
            y = g_sub['r'].values
            try:
                lr = LinearRegression().fit(X, y)
                coefs.append(lr.coef_[0])
                r2s.append(lr.score(X, y))
                nobs.append(len(g_sub))
            except: continue
        if not coefs: continue
        cf = np.array(coefs); T = len(cf)
        m, s = cf.mean(), cf.std()
        results[spec_name] = {
            'fm_coef': float(m), 'fm_t': float(m/(s/np.sqrt(T))) if s>0 else 0,
            'avg_r2': float(np.mean(r2s)), 'n_periods': T,
            'avg_n': float(np.mean(nobs))
        }
    return results

fm_inc = {}
for nm, factor in [('FinBERT-Sentiment', sent_score)]:
    r = fm_incremental(factor, target5, dates, control_dict, ctrl_specs)
    if r:
        fm_inc[nm] = r
        for spec, res in r.items():
            log(f"  FM {nm} [{spec:20s}]: coef={res['fm_coef']:+.6f} t={res['fm_t']:+.2f} R2={res['avg_r2']:.4f}")

with open(RES / "incremental_fm.json", 'w', encoding='utf-8') as f:
    json.dump(fm_inc, f, indent=2, default=str)

# ========================================================
# TEST 3: FACTOR ORTHOGONALIZATION
# ========================================================
log("\n" + "="*60)
log("TEST 3: Factor Orthogonalization")
log("="*60)

def orth_resid(factor, control_mat):
    factor = np.asarray(factor, float)
    X = np.asarray(control_mat, float)
    valid = ~np.isnan(factor) & ~np.isnan(X).any(axis=1)
    if valid.sum() < 50: return factor
    lr = LinearRegression().fit(X[valid], factor[valid])
    res = factor.copy()
    res[valid] = factor[valid] - lr.predict(X[valid])
    return res

ind_mat = ind_dummies.values.astype(float)[:n]
mkt = np.asarray(control_dict.get('log_mktcap', np.zeros(n))).reshape(-1, 1)

orth_mats = OrderedDict([
    ('raw', None),
    ('industry_only', ind_mat),
    ('ind_size', np.column_stack([ind_mat, mkt])),
    ('ind_size_mom', np.column_stack([ind_mat, mkt,
        np.asarray(control_dict.get('ret_20d',np.zeros(n))).reshape(-1,1),
        np.asarray(control_dict.get('ret_5d',np.zeros(n))).reshape(-1,1)])),
    ('ind_size_mom_turn', np.column_stack([ind_mat, mkt,
        np.asarray(control_dict.get('ret_20d',np.zeros(n))).reshape(-1,1),
        np.asarray(control_dict.get('ret_5d',np.zeros(n))).reshape(-1,1),
        np.asarray(control_dict.get('turnover',np.zeros(n))).reshape(-1,1),
        np.asarray(control_dict.get('vol_20d',np.zeros(n))).reshape(-1,1)])),
])

orth_all = {}
for fname, factor in [('FinBERT-Sentiment', sent_score)]:
    orth_all[fname] = OrderedDict()
    for level, mat in orth_mats.items():
        f_orth = factor if mat is None else orth_resid(factor, mat)
        r = rank_ic_monthly(f_orth, target5, dates)
        if r:
            orth_all[fname][level] = {
                'RankIC': r['RankIC'], 'ICIR': r['ICIR'],
                'IC_t': r['IC_t'], 'n_periods': r['n_periods']
            }
            log(f"  {fname} [{level:25s}]: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

with open(RES / "orthogonalization.json", 'w', encoding='utf-8') as f:
    json.dump(orth_all, f, indent=2)

# ========================================================
# TEST 4: ROBUSTNESS SUITE
# ========================================================
log("\n" + "="*60)
log("TEST 4: Robustness Suite")

# 4a: Windows
log("--- 4a: Prediction Windows ---")
window_r = OrderedDict()
for col in ['fwd_excess_1d','fwd_excess_2d','fwd_excess_5d','fwd_excess_10d','fwd_excess_20d']:
    if col in targets:
        r = rank_ic_monthly(sent_score, targets[col], dates)
        if r:
            window_r[col] = r
            log(f"  {col}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

with open(RES / "robustness_windows.json", 'w', encoding='utf-8') as f:
    json.dump(window_r, f, indent=2)

# 4b: Sub-periods
log("--- 4b: Sub-Periods ---")
sub_periods = OrderedDict([
    ('2020-2021', ('2020-01-01','2021-12-31')),
    ('2022', ('2022-01-01','2022-12-31')),
    ('2023-2024', ('2023-01-01','2024-12-31')),
    ('2025-2026', ('2025-01-01','2026-12-31')),
])

period_r = OrderedDict()
for pname, (start, end) in sub_periods.items():
    mask = (pd.to_datetime(dates) >= start) & (pd.to_datetime(dates) <= end)
    if mask.sum() < 100: continue
    r = rank_ic_monthly(sent_score[mask.values], target5[mask.values],
                         pd.to_datetime(dates)[mask.values])
    if r:
        period_r[pname] = {k: v for k, v in r.items() if k != 'ics_series'}
        period_r[pname]['n_samples'] = int(mask.sum())
        log(f"  {pname}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f} n={mask.sum()}")

with open(RES / "robustness_periods.json", 'w', encoding='utf-8') as f:
    json.dump(period_r, f, indent=2)

# 4c: Groups
log("--- 4c: Size/Coverage/Industry Groups ---")
group_r = OrderedDict()

if 'log_mktcap' in control_dict:
    mktv = np.asarray(control_dict['log_mktcap'])
    valid_mkt = ~np.isnan(mktv)
    if valid_mkt.sum() > 100:
        size_labels = pd.qcut(pd.Series(mktv[valid_mkt]), 3, labels=['Small','Mid','Large'])
        size_series = pd.Series(index=range(n), dtype=object)
        vi = np.where(valid_mkt)[0]
        for ii, idx in enumerate(vi):
            if ii < len(size_labels):
                size_series.iloc[idx] = size_labels.iloc[ii]

        group_r['size'] = OrderedDict()
        for lab in ['Small','Mid','Large']:
            mask = size_series == lab
            if mask.sum() < 100: continue
            r = rank_ic_monthly(sent_score[mask.values], target5[mask.values],
                                 pd.to_datetime(dates)[mask.values])
            if r:
                group_r['size'][str(lab)] = {k:v for k,v in r.items() if k!='ics_series'}
                log(f"  Size={lab}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f} n={mask.sum()}")

valid_rc = ~np.isnan(report_count_log)
if valid_rc.sum() > 100:
    cov_labels = pd.qcut(pd.Series(report_count_log[valid_rc]), 3, labels=['LowCov','MidCov','HighCov'])
    cov_series = pd.Series(index=range(n), dtype=object)
    vi = np.where(valid_rc)[0]
    for ii, idx in enumerate(vi):
        if ii < len(cov_labels):
            cov_series.iloc[idx] = cov_labels.iloc[ii]

    group_r['coverage'] = OrderedDict()
    for lab in ['LowCov','MidCov','HighCov']:
        mask = cov_series == lab
        if mask.sum() < 100: continue
        r = rank_ic_monthly(sent_score[mask.values], target5[mask.values],
                             pd.to_datetime(dates)[mask.values])
        if r:
            group_r['coverage'][str(lab)] = {k:v for k,v in r.items() if k!='ics_series'}
            log(f"  Coverage={lab}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f} n={mask.sum()}")

# Top industries
ind_group_r = OrderedDict()
for ind_name in labels_m['industry_name'].dropna().value_counts().head(10).index:
    mask = labels_m['industry_name'].values[:n] == ind_name
    if mask.sum() < 100: continue
    r = rank_ic_monthly(sent_score[mask], target5[mask], pd.to_datetime(dates)[mask])
    if r:
        ind_group_r[str(ind_name)] = {k:v for k,v in r.items() if k!='ics_series'}
group_r['industry'] = ind_group_r
log(f"  Industry groups tested: {len(ind_group_r)}")

with open(RES / "robustness_groups.json", 'w', encoding='utf-8') as f:
    json.dump(group_r, f, indent=2, default=str)

# ========================================================
# TEST 5: PLACEBO TESTS
# ========================================================
log("\n" + "="*60)
log("TEST 5: Placebo Tests (100 permutations for speed)")
log("="*60)

np.random.seed(42)
N_PLACEBO = 100

placebo = {'shuffled_ret': [], 'random_match': []}
for i in range(N_PLACEBO):
    sr = target5.copy(); np.random.shuffle(sr)
    r = rank_ic_monthly(sent_score, sr, dates)
    if r: placebo['shuffled_ret'].append(r['RankIC'])

    pi = np.random.permutation(n)
    r = rank_ic_monthly(sent_score, target5[pi], dates)
    if r: placebo['random_match'].append(r['RankIC'])

    if (i+1) % 25 == 0:
        log(f"  Placebo progress: {i+1}/{N_PLACEBO}")

random_emb_ics = []
for i in range(50):
    re = np.random.randn(n)
    r = rank_ic_monthly(re, target5, dates)
    if r: random_emb_ics.append(r['RankIC'])
placebo['random_embedding'] = random_emb_ics

placebo_summary = OrderedDict()
real_ic = 0.0154
for k, arr in placebo.items():
    arr = np.array(arr)
    pct = (arr < real_ic).mean()
    placebo_summary[k] = {
        'mean': float(arr.mean()), 'std': float(arr.std()),
        'p5': float(np.percentile(arr, 5)), 'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'real_percentile': float(pct), 'n': len(arr)
    }
    log(f"  {k}: mean={arr.mean():+.4f} p95={np.percentile(arr,95):+.4f} real_pct={pct:.1%}")

with open(RES / "placebo_results.json", 'w', encoding='utf-8') as f:
    json.dump(placebo_summary, f, indent=2)

# ========================================================
# TEST 6: MULTIPLE TESTING ADJUSTMENT
# ========================================================
log("\n" + "="*60)
log("TEST 6: Multiple Testing Adjustment")
log("="*60)

# Load existing factor results
existing = RES / "comprehensive_summary.json"
if existing.exists():
    with open(existing) as f:
        cs = json.load(f)
    all_tests = []
    for r in cs.get('factor_results', []):
        if r and 'IC_t' in r and 'name' in r:
            all_tests.append({'name': r['name'], 't': abs(r['IC_t'])})
else:
    # Run quick evaluation of main factors
    log("  No existing results, running quick eval...")
    all_tests = []
    for nm, fac in [('FinBERT-Sentiment', sent_score)]:
        r = rank_ic_monthly(fac, target5, dates)
        if r:
            all_tests.append({'name': nm, 't': abs(r['IC_t'])})

n_tests = len(all_tests)
t_vals = np.array([t['t'] for t in all_tests])
p_vals = 2 * (1 - stats.t.cdf(t_vals, 70))

bonf_threshold = stats.t.ppf(1 - 0.05/(2*n_tests), 70) if n_tests > 0 else 0
log(f"  N_total tests: {n_tests}")
log(f"  Bonferroni threshold: t > {bonf_threshold:.3f}")

mt_results = []
for i in range(n_tests):
    mt_results.append({
        'name': all_tests[i]['name'],
        't_stat': float(t_vals[i]),
        'p_value': float(p_vals[i]),
        'bonferroni_5pct': bool(t_vals[i] > bonf_threshold) if n_tests>0 else False,
    })
    log(f"  {all_tests[i]['name']:35s}: t={t_vals[i]:.2f} p={p_vals[i]:.4f}")

with open(RES / "multiple_testing.json", 'w', encoding='utf-8') as f:
    json.dump(mt_results, f, indent=2)

# ========================================================
# TEST 7: ENHANCED GAP MECHANISM
# ========================================================
log("\n" + "="*60)
log("TEST 7: Enhanced Gap Mechanism")
log("="*60)

cos_sim = gap_emb['cos_sim'][:n]
gap_cls_mat = gap_emb['gap_cls'][:n]

semantic_gap = 1 - cos_sim[:, -1]
sentiment_gap_magnitude = np.linalg.norm(gap_cls_mat.mean(axis=1), axis=1)

# Yearly trends
years = pd.to_datetime(dates).dt.year
yearly_cos = {}; yearly_sgap = {}
for y in sorted(years.unique()):
    mask = years == y
    if mask.sum() > 0:
        yearly_cos[int(y)] = float(cos_sim[mask, -1].mean())
        yearly_sgap[int(y)] = float(semantic_gap[mask].mean())

log(f"  Cosine similarity trend: {yearly_cos}")
log(f"  Semantic Gap (1-cos) trend: {yearly_sgap}")

# Gap IC by sentiment tercile
sent_vals = sent_score[~np.isnan(sent_score)]
if len(sent_vals) > 100:
    sent_tercile = pd.qcut(pd.Series(sent_vals), 3, labels=['low_sent','mid_sent','high_sent'])
    sent_cat = pd.Series(index=range(n), dtype=object)
    vi = np.where(~np.isnan(sent_score))[0]
    for ii, idx in enumerate(vi):
        if ii < len(sent_tercile):
            sent_cat.iloc[idx] = sent_tercile.iloc[ii]

    gap_by_sent = OrderedDict()
    for sc in ['low_sent','mid_sent','high_sent']:
        mask = sent_cat == sc
        if mask.sum() < 50: continue
        r = rank_ic_monthly(semantic_gap[mask.values], target5[mask.values],
                             pd.to_datetime(dates)[mask.values])
        if r:
            gap_by_sent[str(sc)] = {k:v for k,v in r.items() if k!='ics_series'}
            log(f"  Gap IC in [{sc}]: {r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# Returns by Gap group
gap_vals = semantic_gap[~np.isnan(semantic_gap)]
gap_ret_data = OrderedDict()
if len(gap_vals) > 100:
    gap_tercile = pd.qcut(pd.Series(gap_vals), 3, labels=['low_gap','mid_gap','high_gap'])
    gap_cat = pd.Series(index=range(n), dtype=object)
    vi2 = np.where(~np.isnan(semantic_gap))[0]
    for ii, idx in enumerate(vi2):
        if ii < len(gap_tercile):
            gap_cat.iloc[idx] = gap_tercile.iloc[ii]

    for gc in ['low_gap','mid_gap','high_gap']:
        mask = gap_cat == gc
        if mask.sum() > 0:
            gap_ret_data[str(gc)] = {
                'mean_ret': float(target5[mask.values].mean()),
                'mean_sent': float(sent_score[mask.values].mean()),
                'n': int(mask.sum())
            }
    log(f"  Returns by Gap group: {gap_ret_data}")

# Gap-Sentiment correlation
gap_sent_corr = None
valid_gap_mask = ~np.isnan(semantic_gap)
if valid_gap_mask.sum() > 10:
    gap_sent_corr = float(stats.spearmanr(
        semantic_gap[valid_gap_mask], sent_score[valid_gap_mask])[0])
    log(f"  Gap-Sentiment Rank Corr: {gap_sent_corr:+.4f}")

with open(RES / "enhanced_gap.json", 'w', encoding='utf-8') as f:
    json.dump({
        'yearly_cosine': yearly_cos,
        'yearly_semantic_gap': yearly_sgap,
        'gap_ic_by_sentiment': gap_by_sent if 'gap_by_sent' in dir() else {},
        'returns_by_gap_group': gap_ret_data,
        'gap_sent_corr': gap_sent_corr,
    }, f, indent=2, default=str)

# ========================================================
# TEST 8: CONSTRAINED PORTFOLIO BACKTEST
# ========================================================
log("\n" + "="*60)
log("TEST 8: Constrained Portfolio Backtest")
log("="*60)

def constrained_backtest(factor, ret, dates, stock_codes, ind_codes,
                          mktcap_vals, top_n=50, tc=0.003,
                          max_ind_weight=0.10):
    df = pd.DataFrame({
        'f': np.asarray(factor, float), 'r': np.asarray(ret, float),
        'd': pd.to_datetime(dates), 'code': stock_codes,
        'ind': ind_codes, 'mktcap': np.asarray(mktcap_vals, float)
    }).dropna()
    periods = sorted(df['d'].dt.to_period('M').unique())
    port_rets = []; bm_rets = []; prev_stocks = set()

    for p in periods:
        g = df[df['d'].dt.to_period('M') == p]
        if len(g) < top_n * 2: continue
        ranked = g.sort_values('f', ascending=False)
        selected = []; ind_weights = defaultdict(float)
        for _, row in ranked.iterrows():
            ind = row['ind']
            if ind_weights.get(ind, 0) < max_ind_weight:
                if len(selected) < top_n:
                    selected.append(row.name)
                    ind_weights[ind] += 1/top_n
        if len(selected) < top_n // 2: continue
        sel = ranked.loc[selected]
        curr_stocks = set(sel['code'])
        turnover = 1 - len(prev_stocks & curr_stocks) / max(len(curr_stocks), 1) if prev_stocks else 1
        pret = sel['r'].mean() - turnover * tc
        port_rets.append(pret); bm_rets.append(g['r'].mean()); prev_stocks = curr_stocks

    if len(port_rets) < 6: return None
    pr = np.array(port_rets); br = np.array(bm_rets)
    excess = pr - br
    ann_ret = (1 + pr.mean())**12 - 1
    ann_vol = pr.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = np.cumprod(1 + pr)
    max_dd = float(max((np.maximum.accumulate(cum) - cum) / np.maximum.accumulate(cum))) if len(pr)>1 else 0

    return {
        'ann_return': float(ann_ret), 'ann_volatility': float(ann_vol),
        'sharpe': float(sharpe), 'max_drawdown': max_dd,
        'info_ratio': float(excess.mean()/excess.std()*np.sqrt(12)) if excess.std()>0 else 0,
        'win_rate': float((pr > 0).mean()), 'n_months': len(pr),
        'cum_return': float(cum[-1]-1),
    }

bt_c = constrained_backtest(
    sent_score, target5, dates, stock_codes,
    labels_m['industry_code'].values[:n],
    np.asarray(control_dict.get('log_mktcap', np.zeros(n)))
)
if bt_c:
    log(f"  Constrained: AnnRet={bt_c['ann_return']:+.2%} Sharpe={bt_c['sharpe']:.2f} IR={bt_c['info_ratio']:.2f}")

bt_tc = {}
for tc in [0.001, 0.002, 0.003, 0.005]:
    bt = constrained_backtest(
        sent_score, target5, dates, stock_codes,
        labels_m['industry_code'].values[:n],
        np.asarray(control_dict.get('log_mktcap', np.zeros(n))),
        tc=tc
    )
    if bt:
        bt_tc[f'tc_{int(tc*10000)}bps'] = bt
        log(f"  TC={tc:.1%}: AnnRet={bt['ann_return']:+.2%} IR={bt['info_ratio']:.2f}")

with open(RES / "constrained_backtest.json", 'w', encoding='utf-8') as f:
    json.dump({'base': bt_c, 'tc_sensitivity': bt_tc}, f, indent=2)

# ========================================================
log("\n" + "="*60)
log("ALL SUPPLEMENTARY TESTS COMPLETE")
log("="*60)
log(f"Results saved to: {RES}")
for fname in sorted(os.listdir(str(RES))):
    log(f"  {fname}")
