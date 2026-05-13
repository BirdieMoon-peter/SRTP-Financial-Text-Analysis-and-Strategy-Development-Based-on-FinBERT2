"""
======================================================================
SRTP: Signal Enhancement Experiments — Beyond NeutConsensus
======================================================================
Focus: Push the post-processing approach further.

Methods tested:
  1. Analyst track-record weighting (better analysts get more weight)
  2. Sentiment change vs. absolute sentiment (delta-sentiment)
  3. Multi-period consensus (current + previous month)
  4. Coverage-aware scaling (adjust signal strength by coverage count)
  5. Tail-sentiment (extreme sentiment more informative?)
  6. Sector-relative sentiment (within-sector ranking)
  7. Liquidity-filtered signal (only trade liquid stocks)

Run on server:  python src/experiments_postprocess.py
======================================================================
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# Paths — Embeddings may be on mounted data disk
EMBED = None
DATA  = None
RES   = None

for base in [Path("/root/srtp"), Path("/root/autodl-tmp/srtp"),
             Path(__file__).resolve().parent.parent]:
    if (base / "data" / "reports_with_labels.csv").exists():
        PROJ = base
        DATA = PROJ / "data"
        for emb_base in [Path("/root/autodl-tmp/srtp/embeddings"),
                         DATA / "embeddings",
                         PROJ / "embeddings"]:
            if (emb_base / "embeddings_full.npz").exists():
                EMBED = emb_base
                break
        RES = PROJ / "results"
        break

if DATA is None or EMBED is None:
    raise FileNotFoundError(
        "Cannot find project root or embeddings. "
        "Check /root/srtp/data/ and /root/autodl-tmp/srtp/embeddings/"
    )

RES.mkdir(exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def ws(x, lo=0.01, hi=0.99):
    x=np.asarray(x,float); v=~np.isnan(x)
    return np.clip(x,*np.nanquantile(x[v],[lo,hi])) if v.sum()>0 else x
def stdz(x):
    v=~np.isnan(x); r=np.full_like(x,np.nan,dtype=float)
    if v.sum()==0: return r
    r[v]=(x[v]-x[v].mean())/x[v].std(); return r

def eval_factor(name, factor, ret, dates):
    df=pd.DataFrame({'f':np.asarray(factor,float),'r':np.asarray(ret,float),
                     'd':pd.to_datetime(dates)}).dropna()
    if len(df)<50: return None
    ics=[stats.spearmanr(g['f'],g['r'])[0]
         for _,g in df.groupby(df['d'].dt.to_period('M')) if len(g)>=10]
    if len(ics)<6: return None
    im,sd=np.mean(ics),np.std(ics); T=len(ics)
    try:
        df['g']=pd.qcut(df['f'],5,labels=False,duplicates='drop')+1
    except: df['g']=1
    gr={g:df[df['g']==g]['r'].mean() for g in range(1,6)}
    return {'name':name,'RankIC':float(im),
            'ICIR':float(im/sd) if sd>0 else 0,
            'IC_t':float(im/sd*np.sqrt(T)) if sd>0 else 0,
            'LS':float(gr.get(5,np.nan)-gr.get(1,np.nan)),
            'G1':float(gr.get(1,0)),'G5':float(gr.get(5,0)),
            'hit_ratio':float(np.mean([ic>0 for ic in ics])),
            'n_periods':T,'n':len(df)}

# ============================================================
def main():
    log("="*60)
    log("SRTP Signal Enhancement — Post-Processing")
    log("="*60)

    # Load data
    labels = pd.read_csv(DATA / "reports_with_labels.csv")
    labels['report_date'] = pd.to_datetime(labels['report_date'])
    labels['tradable_date'] = pd.to_datetime(labels['tradable_date'])
    labels['stock_code'] = labels['stock_code'].astype(str).str.zfill(6)

    sent = np.load(EMBED / "sentiment_finbert.npz")['probabilities']
    n = min(len(labels), sent.shape[0])
    log(f"Samples: {n:,}")

    labels = labels.iloc[:n]
    target5 = ws(labels['fwd_excess_5d'].fillna(0).values[:n])
    dates = labels['tradable_date'].values[:n]
    codes = labels['stock_code'].astype(str).str.zfill(6).values[:n]
    report_dates = labels['report_date'].values[:n]

    # Baseline sentiment
    sent_score = sent[:n, 2] - sent[:n, 0]  # pos - neg
    sent_pos = sent[:n, 2]
    sent_neg = sent[:n, 0]

    # Load industry mapping
    ind = pd.read_csv(DATA / "industry_mapping.csv", dtype={"stock_code": str})
    ind['stock_code'] = ind['stock_code'].astype(str).str.zfill(6)
    ind_map = dict(zip(ind['stock_code'], ind['industry_code']))
    labels['ind_code'] = [ind_map.get(c, -1) for c in codes]
    ind_dummies = pd.get_dummies(labels['ind_code'], prefix='ind', dummy_na=True)
    ind_dummies = ind_dummies.values.astype(float)

    def ind_neutralize(x):
        v = ~np.isnan(x) & ~np.isnan(ind_dummies).any(axis=1)
        if v.sum() < 50: return x
        lr = LinearRegression().fit(ind_dummies[v], x[v])
        r = x.copy(); r[v] = x[v] - lr.predict(ind_dummies[v])
        return r

    # Market cap proxy: use volume if available, else constant
    if 'volume' in labels.columns:
        mcap_proxy = ws(np.log1p(labels['volume'].fillna(0).values[:n]))
    else:
        mcap_proxy = np.zeros(n)

    results = []

    # ---- Baseline ----
    log("\n--- Baselines ---")
    r = eval_factor("Baseline-Raw", sent_score, target5, dates)
    if r: results.append(r); log(f"  Raw: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # NeutConsensus (current best)
    sent_neut = ind_neutralize(sent_score)
    df_agg = pd.DataFrame({
        'code': codes, 'date': dates, 'sent_neut': sent_neut
    }).dropna()
    consensus = df_agg.groupby(['code', 'date'])['sent_neut'].mean().reset_index()
    # Map back to original index
    nc_map = {}
    for _, row in consensus.iterrows():
        nc_map[(row['code'], str(row['date'])[:10])] = row['sent_neut']
    nc_score = np.array([nc_map.get((c, str(d)[:10]), np.nan)
                         for c, d in zip(codes, dates)])
    nc_score = stdz(nc_score)

    r = eval_factor("NeutConsensus", nc_score, target5, dates)
    if r: results.append(r); log(f"  NeutConsensus: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # ---- EXP 1: Sentiment Change (Delta-Sentiment) ----
    log("\n--- EXP 1: Sentiment Change ---")
    # For each stock, compute the change in sentiment from previous report
    df_s = pd.DataFrame({
        'code': codes, 'rdate': pd.to_datetime(report_dates),
        'sent': sent_score, 'sent_pos': sent_pos, 'sent_neg': sent_neg
    }).sort_values(['code', 'rdate'])

    # Previous sentiment per stock
    df_s['sent_prev'] = df_s.groupby('code')['sent'].shift(1)
    df_s['sent_pos_prev'] = df_s.groupby('code')['sent_pos'].shift(1)

    delta_sent = df_s['sent'].values - df_s['sent_prev'].values
    delta_sent = ws(delta_sent)

    r = eval_factor("DeltaSentiment", delta_sent, target5, dates)
    if r: results.append(r); log(f"  Delta-Sent: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Delta + Neut + Consensus
    delta_nc = ind_neutralize(delta_sent)
    r = eval_factor("DeltaSent-Neut", delta_nc, target5, dates)
    if r: results.append(r); log(f"  Delta-Neut: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # ---- EXP 2: Analyst Track-Record Weighting ----
    log("\n--- EXP 2: Analyst Track-Record ---")
    if 'analyst_code' in labels.columns or 'analyst_name' in labels.columns:
        analyst_col = 'analyst_code' if 'analyst_code' in labels.columns else 'analyst_name'
        analysts = labels[analyst_col].values[:n]

        # Rolling: past 12-month average IC per analyst
        dates_pd = pd.to_datetime(dates)
        months = sorted(dates_pd.dt.to_period('M').unique())
        analyst_weights = np.ones(n)

        for month in months:
            train_start = month - pd.offsets.MonthEnd(12)
            tm = (dates_pd.dt.to_period('M') >= train_start) & (dates_pd.dt.to_period('M') < month)
            xm = dates_pd.dt.to_period('M') == month
            if tm.sum() < 100 or xm.sum() < 10:
                continue
            # Per-analyst hit rate in training period
            df_train = pd.DataFrame({
                'analyst': analysts[tm.values],
                'sent': sent_score[tm.values],
                'ret': target5[tm.values],
            }).dropna()
            df_train['correct'] = np.sign(df_train['sent']) == np.sign(df_train['ret'])
            analyst_hit = df_train.groupby('analyst')['correct'].mean()
            # Map to test period
            test_analysts = analysts[xm.values]
            weights = np.array([analyst_hit.get(a, 0.5) for a in test_analysts])
            analyst_weights[xm.values] = np.clip(weights, 0.3, 0.7)

        weighted_sent = stdz(sent_score * analyst_weights)
        r = eval_factor("AnalystWeighted", weighted_sent, target5, dates)
        if r: results.append(r); log(f"  AnalystW: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")
    else:
        log("  No analyst identifier column — skipping")

    # ---- EXP 3: Coverage-Aware Signal Scaling ----
    log("\n--- EXP 3: Coverage-Aware Scaling ---")
    # Count reports per stock per month
    df_cnt = pd.DataFrame({
        'code': codes, 'month': pd.to_datetime(dates).strftime('%Y-%m')
    })
    coverage = df_cnt.groupby(['code', 'month']).size().reset_index(name='n_reports')
    cov_map = {}
    for _, row in coverage.iterrows():
        cov_map[(row['code'], row['month'])] = row['n_reports']
    n_reports = np.array([cov_map.get((c, str(d)[:7]), 1)
                          for c, d in zip(codes, dates)])
    log_n_reports = np.log1p(n_reports)

    # Signal * sqrt(log coverage) — amplify when there's more consensus
    cov_scaled = stdz(nc_score * np.sqrt(log_n_reports))
    r = eval_factor("NeutCons-CovScaled", cov_scaled, target5, dates)
    if r: results.append(r); log(f"  CovScaled: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Only use signal when coverage >= 3
    high_cov_mask = n_reports >= 3
    nc_highcov = nc_score.copy()
    nc_highcov[~high_cov_mask] = 0  # neutral when low coverage
    nc_highcov = stdz(nc_highcov)
    r = eval_factor("NeutCons-HighCovOnly", nc_highcov, target5, dates)
    if r: results.append(r); log(f"  HighCov: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # ---- EXP 4: Tail Sentiment ----
    log("\n--- EXP 4: Tail Sentiment ---")
    # Extreme sentiment (top/bottom 20%) may be more informative
    sent_pct = pd.Series(sent_score).rank(pct=True).values
    tail_mask = (sent_pct < 0.2) | (sent_pct > 0.8)
    sent_tail = sent_score.copy()
    sent_tail[~tail_mask] = 0  # neutral for middle 60%
    sent_tail = stdz(ind_neutralize(sent_tail))

    r = eval_factor("Sentiment-Tail", sent_tail, target5, dates)
    if r: results.append(r); log(f"  Tail: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Tail + Consensus
    df_tail = pd.DataFrame({
        'code': codes, 'date': dates, 'sent_tail': sent_tail
    }).dropna()
    tail_cons = df_tail.groupby(['code', 'date'])['sent_tail'].mean().reset_index()
    tc_map = {}
    for _, row in tail_cons.iterrows():
        tc_map[(row['code'], str(row['date'])[:10])] = row['sent_tail']
    tc_score = np.array([tc_map.get((c, str(d)[:10]), np.nan)
                         for c, d in zip(codes, dates)])
    tc_score = stdz(tc_score)
    r = eval_factor("TailSent-NeutCons", tc_score, target5, dates)
    if r: results.append(r); log(f"  Tail+NC: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # ---- EXP 5: Sector-Relative Sentiment ----
    log("\n--- EXP 5: Sector-Relative Sentiment ---")
    # Within each industry, rank stocks by sentiment
    df_sec = pd.DataFrame({
        'code': codes, 'date': dates, 'ind': labels['ind_code'].values[:n],
        'sent': sent_score
    }).dropna()
    # Rank within each (date, industry) group
    df_sec['sent_rank'] = df_sec.groupby(['date', 'ind'])['sent'].rank(pct=True)
    sector_rank = df_sec['sent_rank'].values
    sector_rank_neut = ind_neutralize(sector_rank)

    r = eval_factor("SectorRank", sector_rank, target5, dates)
    if r: results.append(r); log(f"  SectorRank: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    r = eval_factor("SectorRank-Neut", sector_rank_neut, target5, dates)
    if r: results.append(r); log(f"  SectorR-Neut: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # ---- EXP 6: Size-Neutral + Industry-Neutral + Momentum-Neutral ----
    log("\n--- EXP 6: Full Neutralization ---")
    # Construct a more complete neutralization
    # Industry
    sent_n1 = ind_neutralize(sent_score)
    # Industry + Size
    size_valid = ~np.isnan(mcap_proxy)
    if size_valid.sum() > 100:
        X_size = np.column_stack([ind_dummies, mcap_proxy])
        v = ~np.isnan(sent_score) & ~np.isnan(X_size).any(axis=1)
        lr = LinearRegression().fit(X_size[v], sent_score[v])
        sent_n2 = sent_score.copy()
        sent_n2[v] = sent_score[v] - lr.predict(X_size[v])
    else:
        sent_n2 = sent_n1

    # Now aggregate to consensus
    for name, s in [("FullNeut1-Ind", sent_n1), ("FullNeut2-IndSize", sent_n2)]:
        df_a = pd.DataFrame({
            'code': codes, 'date': dates, 's': stdz(s)
        }).dropna()
        cons = df_a.groupby(['code', 'date'])['s'].mean().reset_index()
        cmap = {}
        for _, row in cons.iterrows():
            cmap[(row['code'], str(row['date'])[:10])] = row['s']
        cs = np.array([cmap.get((c, str(d)[:10]), np.nan)
                       for c, d in zip(codes, dates)])
        cs = stdz(cs)
        r = eval_factor(name, cs, target5, dates)
        if r: results.append(r); log(f"  {name}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # ---- EXP 7: Liquidity Filter ----
    log("\n--- EXP 7: Liquidity Filter ---")
    if 'volume' in labels.columns and 'close' in labels.columns:
        # Dollar volume = volume * close
        dvol = ws(labels['volume'].fillna(0).values[:n] *
                  labels['close'].fillna(0).values[:n])
        liquid_mask = dvol > np.nanmedian(dvol)
        nc_liquid = nc_score.copy()
        nc_liquid[~liquid_mask] = 0  # neutral for illiquid stocks
        nc_liquid = stdz(nc_liquid)
        r = eval_factor("NeutCons-Liquid", nc_liquid, target5, dates)
        if r: results.append(r); log(f"  Liquid: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # ---- EXP 8: Combined Best Methods ----
    log("\n--- EXP 8: Best Combination ---")
    # Combine the most promising enhancements
    combined = stdz(
        0.5 * nc_score +           # NeutConsensus (best baseline)
        0.2 * stdz(delta_nc) +     # Delta sentiment
        0.2 * stdz(sector_rank) +  # Sector-relative
        0.1 * stdz(sent_tail)      # Tail sentiment
    )
    r = eval_factor("Enhanced-Combo", combined, target5, dates)
    if r: results.append(r); log(f"  Combo: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Rolling-optimized combo: learn weights via rolling regression
    dates_idx = pd.to_datetime(dates)
    months_combo = sorted(dates_idx.to_period('M').unique())
    combo_rolling = np.full(n, np.nan)

    component_factors = np.column_stack([
        stdz(nc_score),
        stdz(delta_nc),
        stdz(sector_rank),
        stdz(sent_tail),
    ])
    # Fill NaN with 0 (neutral) for Ridge regression
    component_factors = np.nan_to_num(component_factors, nan=0.0)

    for month in months_combo:
        ts = month - pd.offsets.MonthEnd(24)
        tm = (dates_idx.to_period('M') >= ts) & (dates_idx.to_period('M') < month)
        xm = dates_idx.to_period('M') == month
        if tm.sum() < 100 or xm.sum() < 10:
            continue
        F_train = component_factors[tm]
        y_train = target5[tm]
        valid = ~np.isnan(y_train) & ~np.isnan(F_train).any(axis=1)
        if valid.sum() < 100:
            continue
        F_train, y_train = F_train[valid], y_train[valid]
        from sklearn.linear_model import RidgeCV
        ridge = RidgeCV(alphas=np.logspace(-1, 2, 10))
        ridge.fit(F_train, y_train)
        combo_rolling[xm] = ridge.predict(component_factors[xm])

    combo_rolling = stdz(combo_rolling)
    r = eval_factor("Enhanced-Combo-Ridge", combo_rolling, target5, dates)
    if r: results.append(r); log(f"  ComboRidge: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # ---- SAVE (before combo to ensure results are saved) ----
    log("\n" + "="*60)
    log("RESULTS SUMMARY")
    log("="*60)
    results.sort(key=lambda x: abs(x['RankIC']), reverse=True)
    for r in results[:25]:
        sig = "***" if abs(r['IC_t']) > 2.58 else ("**" if abs(r['IC_t']) > 1.96 else "")
        log(f"  {r['name']:<30s} IC={r['RankIC']:>+8.4f}  "
            f"ICIR={r['ICIR']:>7.3f}  t={r['IC_t']:>+7.2f}{sig}  "
            f"hit={r['hit_ratio']:.1%}")

    out = RES / "postprocess_enhanced.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved to {out}")

    pd.DataFrame(results).to_csv(RES / "postprocess_enhanced.csv", index=False)
    log("Done!")
    return


if __name__ == "__main__":
    main()
