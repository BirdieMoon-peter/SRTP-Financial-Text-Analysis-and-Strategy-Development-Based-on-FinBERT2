import json
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

for base in [Path(__file__).resolve().parent.parent, Path('/root/srtp'), Path('C:/Users/13082/CSMAR')]:
    if (base / 'data' / 'reports_with_labels.csv').exists():
        PROJ = base
        break
else:
    raise FileNotFoundError('Cannot locate project data directory')

DATA = PROJ / 'data'
EMBED = DATA / 'embeddings'
RES = PROJ / 'results'
RES.mkdir(exist_ok=True)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def winsorize(values, lo=0.01, hi=0.99):
    x = np.asarray(values, dtype=float)
    valid = ~np.isnan(x)
    if valid.sum() == 0:
        return x
    qlo, qhi = np.nanquantile(x[valid], [lo, hi])
    y = x.copy()
    y[valid] = np.clip(y[valid], qlo, qhi)
    return y


def monthly_rank_ic(factor, ret, dates, min_obs=20):
    df = pd.DataFrame({
        'factor': np.asarray(factor, dtype=float),
        'ret': np.asarray(ret, dtype=float),
        'date': pd.to_datetime(dates),
    }).dropna()
    ics = []
    months = []
    for month, group in df.groupby(df['date'].dt.to_period('M')):
        if len(group) < min_obs:
            continue
        ic = stats.spearmanr(group['factor'], group['ret'])[0]
        if not np.isnan(ic):
            ics.append(float(ic))
            months.append(str(month))
    if not ics:
        return None
    arr = np.asarray(ics, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    t = float(mean / (std / np.sqrt(len(arr)))) if std > 0 else 0.0
    return {
        'rank_ic': mean,
        'ic_std': std,
        'ic_t': t,
        'icir': float(mean / std) if std > 0 else 0.0,
        'hit_ratio': float((arr > 0).mean()),
        'n_periods': int(len(arr)),
        'series': [{'month': m, 'ic': float(v)} for m, v in zip(months, arr)],
    }


def newey_west_t(values, lag=4):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return 0.0
    centered = x - x.mean()
    gamma0 = np.dot(centered, centered) / n
    var = gamma0
    for ell in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - ell / (lag + 1.0)
        gamma = np.dot(centered[ell:], centered[:-ell]) / n
        var += 2.0 * weight * gamma
    se = np.sqrt(max(var, 0.0) / n)
    return float(x.mean() / se) if se > 0 else 0.0


def neutralize(factor, controls):
    f = np.asarray(factor, dtype=float)
    X = np.asarray(controls, dtype=float)
    valid = ~np.isnan(f) & ~np.isnan(X).any(axis=1)
    out = f.copy()
    if valid.sum() < 50:
        return out
    model = LinearRegression().fit(X[valid], f[valid])
    out[valid] = f[valid] - model.predict(X[valid])
    return out


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    prev = 1.0
    n = len(p)
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        val = min(prev, p[idx] * n / original_rank)
        adjusted[idx] = val
        prev = val
    return adjusted


def load_inputs():
    labels = pd.read_csv(DATA / 'reports_with_labels.csv')
    labels['stock_code'] = labels['stock_code'].astype(str).str.zfill(6)
    labels['report_date'] = pd.to_datetime(labels['report_date'])
    labels['tradable_date'] = pd.to_datetime(labels['tradable_date'])

    sent_path = EMBED / 'sentiment_finbert.npz'
    if not sent_path.exists():
        raise FileNotFoundError(f'Missing {sent_path}')
    sent = np.load(sent_path)['probabilities']
    n = min(len(labels), len(sent))
    labels = labels.iloc[:n].copy()
    sent_score = sent[:n, 2] - sent[:n, 0]

    target_cols = [c for c in ['fwd_excess_1d', 'fwd_excess_2d', 'fwd_excess_5d',
                                'fwd_excess_10d', 'fwd_excess_20d'] if c in labels.columns]
    targets = {c: winsorize(labels[c].values[:n]) for c in target_cols}

    stock = pd.read_csv(DATA / 'csmar_daily_stock.csv',
                        dtype={'stock_code': str}, low_memory=False)
    stock['stock_code'] = stock['stock_code'].astype(str).str.zfill(6)
    stock['date'] = pd.to_datetime(stock['date'])
    stock = stock.sort_values(['stock_code', 'date']).reset_index(drop=True)
    stock['close'] = pd.to_numeric(stock['close'], errors='coerce')
    stock['volume'] = pd.to_numeric(stock.get('volume', 0), errors='coerce')
    stock['amount'] = pd.to_numeric(stock.get('amount', 0), errors='coerce')
    stock['turn'] = pd.to_numeric(stock.get('turn', 0), errors='coerce')

    stock['ret_5d'] = stock.groupby('stock_code')['close'].pct_change(5)
    stock['ret_20d'] = stock.groupby('stock_code')['close'].pct_change(20)
    stock['ret_60d'] = stock.groupby('stock_code')['close'].pct_change(60)
    stock['vol_20d'] = stock.groupby('stock_code')['close'].pct_change().transform(
        lambda s: s.rolling(20).std())
    stock['log_amount'] = np.log(stock['amount'].clip(lower=1))
    stock['log_volume'] = np.log1p(stock['volume'])
    stock['turnover_proxy'] = stock['turn']

    controls = stock[['stock_code', 'date', 'ret_5d', 'ret_20d', 'ret_60d',
                       'vol_20d', 'turnover_proxy', 'log_amount', 'log_volume']]
    controls = controls.rename(columns={'date': 'tradable_date'})
    merged = labels.merge(controls, on=['stock_code', 'tradable_date'], how='left')

    ind_path = DATA / 'industry_mapping.csv'
    if ind_path.exists():
        industry = pd.read_csv(ind_path, dtype={'stock_code': str})
        industry['stock_code'] = industry['stock_code'].astype(str).str.zfill(6)
        ind_map = dict(zip(industry['stock_code'], industry['industry_code']))
        merged['industry_code'] = merged['stock_code'].map(ind_map)
    else:
        merged['industry_code'] = 'unknown'

    industry_dummies = pd.get_dummies(merged['industry_code'], dummy_na=True).astype(float).values
    report_count = merged.groupby('stock_code')['report_date'].transform('count').astype(float).values
    merged['report_count_log'] = np.log1p(report_count)

    return labels, merged, sent_score, targets, industry_dummies


def factor_summary(name, factor, target, dates):
    result = monthly_rank_ic(factor, target, dates)
    if result is None:
        return None
    p = 2.0 * (1.0 - stats.t.cdf(abs(result['ic_t']), df=max(result['n_periods'] - 1, 1)))
    nw_t = newey_west_t([row['ic'] for row in result['series']], lag=4)
    return {
        'name': name,
        'rank_ic': result['rank_ic'],
        'ic_t': result['ic_t'],
        'newey_west_t': nw_t,
        'icir': result['icir'],
        'hit_ratio': result['hit_ratio'],
        'n_periods': result['n_periods'],
        'p_value': float(p),
    }


def run_factor_tests(merged, sent_score, target5, dates, industry_dummies):
    control_cols = ['log_amount', 'ret_20d', 'ret_5d', 'turnover_proxy', 'vol_20d', 'log_volume']
    control_matrix = merged[control_cols].fillna(0).astype(float).values
    full_matrix = np.column_stack([industry_dummies, control_matrix])

    factors = OrderedDict()
    factors['FinBERT-Sentiment-Raw'] = sent_score
    factors['FinBERT-Sentiment-IndustryNeutral'] = neutralize(sent_score, industry_dummies)
    factors['FinBERT-Sentiment-FullNeutral'] = neutralize(sent_score, full_matrix)

    consensus = pd.DataFrame({
        'stock_code': merged['stock_code'],
        'tradable_date': merged['tradable_date'],
        'factor': sent_score,
    })
    consensus_value = consensus.groupby(['stock_code', 'tradable_date'])['factor'].transform('mean').values
    factors['FinBERT-Sentiment-Consensus'] = consensus_value
    factors['FinBERT-Sentiment-NeutConsensus'] = neutralize(consensus_value, industry_dummies)

    out = []
    for name, factor in factors.items():
        item = factor_summary(name, factor, target5, dates)
        if item:
            out.append(item)

    pvals = [x['p_value'] for x in out]
    if pvals:
        fdr = benjamini_hochberg(pvals)
        bonf = np.minimum(np.asarray(pvals) * len(pvals), 1.0)
        for item, fdr_p, bonf_p in zip(out, fdr, bonf):
            item['fdr_p'] = float(fdr_p)
            item['bonferroni_p'] = float(bonf_p)
    return out, factors


def run_incremental_fm(merged, factor, target, dates):
    specs = OrderedDict([
        ('TextOnly', []),
        ('Coverage', ['report_count_log']),
        ('Style', ['log_amount', 'ret_20d', 'ret_5d']),
        ('Liquidity', ['turnover_proxy', 'vol_20d', 'log_volume']),
        ('AllControls', ['report_count_log', 'log_amount', 'ret_20d', 'ret_5d',
                         'ret_60d', 'turnover_proxy', 'vol_20d', 'log_volume']),
    ])
    df = pd.DataFrame({'factor': factor, 'ret': target, 'date': pd.to_datetime(dates)})
    for col in sorted({c for cols in specs.values() for c in cols}):
        df[col] = merged[col].fillna(0).astype(float).values

    results = []
    for name, controls in specs.items():
        cols = ['factor'] + controls
        coefs = []
        r2s = []
        for _, group in df.groupby(df['date'].dt.to_period('M')):
            g = group[cols + ['ret']].dropna()
            if len(g) < 30:
                continue
            model = LinearRegression().fit(g[cols].values, g['ret'].values)
            coefs.append(float(model.coef_[0]))
            r2s.append(float(model.score(g[cols].values, g['ret'].values)))
        if coefs:
            arr = np.asarray(coefs, dtype=float)
            se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.nan
            results.append({
                'spec': name,
                'coef': float(arr.mean()),
                't': float(arr.mean() / se) if se and se > 0 else 0.0,
                'newey_west_t': newey_west_t(arr, lag=4),
                'avg_r2': float(np.mean(r2s)),
                'n_periods': int(len(arr)),
            })
    return results


def run_windows(sent_score, targets, dates):
    out = []
    for col, target in targets.items():
        item = factor_summary(f'Sentiment_on_{col}', sent_score, target, dates)
        if item:
            out.append(item)
    return out


def run_placebo(sent_score, target, dates, real_ic, n_iter=500):
    rng = np.random.default_rng(42)
    shuffled = []
    random_factor = []
    for _ in range(n_iter):
        shuffled_target = target.copy()
        rng.shuffle(shuffled_target)
        s = monthly_rank_ic(sent_score, shuffled_target, dates)
        if s:
            shuffled.append(s['rank_ic'])
        rf = rng.normal(size=len(sent_score))
        r = monthly_rank_ic(rf, target, dates)
        if r:
            random_factor.append(r['rank_ic'])

    def summarize(values):
        arr = np.asarray(values, dtype=float)
        return {
            'mean': float(arr.mean()),
            'std': float(arr.std(ddof=1)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'real_percentile': float((arr < real_ic).mean()),
            'n': int(len(arr)),
        }

    return {
        'shuffled_returns': summarize(shuffled),
        'random_factor': summarize(random_factor),
    }


def main():
    log('Loading inputs')
    labels, merged, sent_score, targets, industry_dummies = load_inputs()
    if 'fwd_excess_5d' not in targets:
        raise KeyError('fwd_excess_5d is required')
    target5 = targets['fwd_excess_5d']
    dates = labels['tradable_date']

    log('Running factor tests')
    factor_tests, factors = run_factor_tests(merged, sent_score, target5, dates, industry_dummies)
    raw_ic = next((x['rank_ic'] for x in factor_tests if x['name'] == 'FinBERT-Sentiment-Raw'), 0.0)

    log('Running incremental Fama-MacBeth')
    fm = run_incremental_fm(merged, factors['FinBERT-Sentiment-NeutConsensus'], target5, dates)

    log('Running prediction-window robustness')
    windows = run_windows(sent_score, targets, dates)

    log('Running placebo tests')
    placebo = run_placebo(sent_score, target5, dates, raw_ic, n_iter=500)

    summary = OrderedDict([
        ('sample', {
            'n_reports': int(len(labels)),
            'n_stocks': int(labels['stock_code'].nunique()),
            'start_date': str(pd.to_datetime(labels['tradable_date']).min().date()),
            'end_date': str(pd.to_datetime(labels['tradable_date']).max().date()),
        }),
        ('factor_tests', factor_tests),
        ('incremental_fm', fm),
        ('window_robustness', windows),
        ('placebo', placebo),
        ('interpretation_rules', {
            'delete_strong_hidden_superiority_claim': True,
            'state_hidden_layers_as_complementary_only': True,
            'state_strategy_as_alpha_input_not_standalone': True,
            'downgrade_gap_mechanism_if_not_supported_by_current_outputs': True,
        }),
    ])

    out_json = RES / 'literature_enhancement_summary.json'
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    rows = []
    for section, records in [('factor_tests', factor_tests), ('incremental_fm', fm),
                              ('window_robustness', windows)]:
        for record in records:
            row = {'section': section}
            row.update(record)
            rows.append(row)
    pd.DataFrame(rows).to_csv(RES / 'literature_enhancement_tables.csv', index=False)
    log(f'Saved {out_json}')


if __name__ == '__main__':
    main()
