"""
SRTP: Focused Signal Enhancement
=================================
Tests 6 proven methods to boost text factor IC.
Runs ~15 min on RTX 3050ti / instant on RTX 3090.
"""
import json, time, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
warnings.filterwarnings("ignore")

# Auto-detect project root
for base in [Path("C:/Users/13082/CSMAR"), Path("/root/srtp"),
             Path(__file__).resolve().parent.parent]:
    if (base / "data" / "reports_with_labels.csv").exists():
        PROJ = base; break

DATA, EMBED, RES = PROJ/"data", PROJ/"data"/"embeddings", PROJ/"results"
RES.mkdir(exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def ws(x, lo=0.01, hi=0.99):
    x=np.asarray(x,float); v=~np.isnan(x)
    return np.clip(x,*np.nanquantile(x[v],[lo,hi])) if v.sum()>0 else x
def stdz(x):
    v=~np.isnan(x); r=np.full_like(x,np.nan,dtype=float)
    r[v]=(x[v]-x[v].mean())/x[v].std(); return r

def eval_ic(factor, ret, dates, name=""):
    df=pd.DataFrame({'f':np.asarray(factor,float),'r':np.asarray(ret,float),
                     'd':pd.to_datetime(dates)}).dropna()
    if len(df)<50: return None
    ics=[stats.spearmanr(g['f'],g['r'])[0]
         for _,g in df.groupby(df['d'].dt.to_period('M')) if len(g)>=10]
    if not ics: return None
    m,s=np.mean(ics),np.std(ics); T=len(ics)
    return {'name':name,'RankIC':float(m),'ICIR':float(m/s) if s>0 else 0,
            'IC_t':float(m/s*np.sqrt(T)) if s>0 else 0,'n':T}

# ============================================================
log("="*60)
log("FOCUSED SIGNAL ENHANCEMENT")
log("="*60)

log("Loading data...")
labels=pd.read_csv(DATA/"reports_with_labels.csv")
labels['report_date']=pd.to_datetime(labels['report_date'])
labels['tradable_date']=pd.to_datetime(labels['tradable_date'])

sent=np.load(EMBED/"sentiment_finbert.npz")['probabilities']
gap_emb=np.load(EMBED/"embeddings_gap.npz")
full_emb=np.load(EMBED/"embeddings_full.npz")

n=min(len(labels),full_emb['all_cls'].shape[0])
log(f"Samples: {n:,}")

labels=labels.iloc[:n]
dates=labels['tradable_date']
codes=labels['stock_code'].astype(str).str.zfill(6)
labels['stock_code']=labels['stock_code'].astype(str).str.zfill(6)  # ensure string type

# Baseline factors
sent_score=sent[:n,2]-sent[:n,0]  # pos - neg
sent_pos=sent[:n,2]                # positive probability
sent_neg=sent[:n,0]                # negative probability
gap_semantic=1-gap_emb['cos_sim'][:n,-1]  # 1 - cos(title, summary)
cls_last4=full_emb['all_cls'][:n,-4:,:].reshape(n,-1)  # last 4 layers CLS

# Labels
target5=ws(labels['fwd_excess_5d'].fillna(0).values[:n])
target10=ws(labels['fwd_excess_10d'].fillna(0).values[:n])
maxdd10=ws(labels['fwd_maxdd_10d'].fillna(0).values[:n])

# ============================================================
log("\n--- Stock Data for Controls ---")
stock=pd.read_csv(DATA/"csmar_daily_stock.csv",dtype={"stock_code":str},low_memory=False)
stock['date']=pd.to_datetime(stock['date'])
stock['stock_code']=stock['stock_code'].astype(str).str.zfill(6)
stock=stock.sort_values(['stock_code','date']).reset_index(drop=True)
for c in ['close','volume','market_cap']:
    if c in stock.columns: stock[c]=pd.to_numeric(stock[c],errors='coerce')

stock['ret_5d']=stock.groupby('stock_code')['close'].pct_change(5)
stock['ret_20d']=stock.groupby('stock_code')['close'].pct_change(20)
stock['vol_20d']=stock.groupby('stock_code')['close'].pct_change().transform(lambda x:x.rolling(20).std())
stock['turnover']=stock['volume']/stock.groupby('stock_code')['volume'].transform(lambda x:x.rolling(60).mean().clip(lower=1))
stock['log_mktcap']=np.log(stock['market_cap'].clip(lower=1)) if 'market_cap' in stock.columns else 0

ctrl_cols=['ret_5d','ret_20d','vol_20d','turnover','log_mktcap']
ctrl=stock[['stock_code','date']+ctrl_cols].rename(columns={'date':'tradable_date'})
ctrl['tradable_date']=pd.to_datetime(ctrl['tradable_date'])
ctrl['stock_code']=ctrl['stock_code'].astype(str).str.zfill(6)  # ensure string
lm=labels.merge(ctrl,on=['stock_code','tradable_date'],how='left')

# Industry
ind_df=pd.read_csv(DATA/"industry_mapping.csv",dtype={"stock_code":str})
ind_df['stock_code']=ind_df['stock_code'].astype(str).str.zfill(6)
ind_map=dict(zip(ind_df['stock_code'],ind_df['industry_code']))
lm['ind_code']=lm['stock_code'].map(ind_map)
ind_dummies=pd.get_dummies(lm['ind_code'],prefix='ind',dummy_na=True).astype(float)

# Coverage
rc=lm.groupby('stock_code')['report_date'].transform('count').values[:n]
cov_log=np.log1p(rc.astype(float))

# ============================================================
# BASELINE
# ============================================================
log("\n"+"="*60)
log("BASELINE IC")
log("="*60)

baseline=eval_ic(sent_score, target5, dates, "Sentiment_raw")
log(f"  Sentiment_raw: IC={baseline['RankIC']:+.4f} t={baseline['IC_t']:+.2f}")

all_results=OrderedDict()
all_results['Baseline_Sentiment']=baseline

# ============================================================
# METHOD 1: INDUSTRY NEUTRALIZATION (known +38% boost)
# ============================================================
log("\n"+"="*60)
log("METHOD 1: Industry Neutralization")
log("="*60)

def neutralize(factor, X):
    f=np.asarray(factor,float); X=np.asarray(X,float)
    valid=~np.isnan(f)&~np.isnan(X).any(axis=1)
    if valid.sum()<50: return f
    lr=LinearRegression().fit(X[valid],f[valid])
    res=f.copy(); res[valid]=f[valid]-lr.predict(X[valid])
    return res

sent_ind_neut=neutralize(sent_score, ind_dummies.values.astype(float)[:n])
r=eval_ic(sent_ind_neut, target5, dates, "Sentiment_IndNeut")
all_results['IndNeut']=r
log(f"  Sentiment_IndNeut: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# Full neutralization: industry + size + momentum + turnover
mktcap=np.asarray(lm['log_mktcap'].values[:n],float).reshape(-1,1)
mom=np.asarray(lm['ret_20d'].values[:n],float).reshape(-1,1)
turn=np.asarray(lm['turnover'].values[:n],float).reshape(-1,1)
full_ctrl=np.column_stack([ind_dummies.values.astype(float)[:n],mktcap,mom,turn])
sent_full_neut=neutralize(sent_score, full_ctrl)
r=eval_ic(sent_full_neut, target5, dates, "Sentiment_FullNeut")
all_results['FullNeut']=r
log(f"  Sentiment_FullNeut: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# ============================================================
# METHOD 2: INTERACTION FACTORS
# ============================================================
log("\n"+"="*60)
log("METHOD 2: Interaction Factors (Text × Environment)")
log("="*60)

# The intuition: text signal is stronger when:
# - Coverage is low (less analyst attention → text has more marginal info)
# - Size is small (less market efficiency → text not yet priced in)
# - Turnover is low (slow information diffusion → text predicts delayed reaction)

sz=stdz(np.asarray(lm['log_mktcap'].values[:n],float))
cv=stdz(cov_log)
tu=stdz(np.asarray(lm['turnover'].values[:n],float))
sent_z=stdz(sent_score)

interactions={
    'Sent_x_LowCov':     sent_z * (-cv),      # stronger for neglected stocks
    'Sent_x_SmallSize':  sent_z * (-sz),      # stronger for small caps
    'Sent_x_LowTurn':    sent_z * (-tu),      # stronger for illiquid stocks
    'Sent_x_LowCov_Small': sent_z * (-cv) * (-sz),  # joint interaction
}

for nm, fac in interactions.items():
    r=eval_ic(fac, target5, dates, nm)
    if r: all_results[nm]=r; log(f"  {nm}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# ============================================================
# METHOD 3: IDIOSYNCRATIC RETURN LABELS
# ============================================================
log("\n"+"="*60)
log("METHOD 3: Idiosyncratic Return Labels")
log("="*60)

# Remove market + industry expected returns from labels
# For each date, regress returns on industry dummies, take residual
# This isolates the "unexplained" return component that text should predict

idio5=np.full(n,np.nan)
dates_dt=pd.to_datetime(dates)

# Group by month for more stable estimation
for p, g in lm.groupby(dates_dt.dt.to_period('M')):
    idx=g.index.values
    if len(idx)<30: continue
    ret=g['fwd_excess_5d'].values.astype(float)
    v=~np.isnan(ret)
    if v.sum()<20: continue
    try:
        X=ind_dummies.values[idx][v].astype(float)
        y=ret[v]
        lr=LinearRegression().fit(X,y)
        idio5[idx[v]]=y-lr.predict(X)
    except: continue

idio5_ws=ws(idio5)
r=eval_ic(sent_score, idio5_ws, dates, "Sentiment_on_IdioRet")
all_results['IdioRet_Label']=r
log(f"  Sentiment_on_IdioRet: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# Combined: neutral factor + idio labels
r=eval_ic(sent_ind_neut, idio5_ws, dates, "SentNeut_on_IdioRet")
all_results['IndNeut_IdioRet']=r
log(f"  SentNeut_on_IdioRet:  IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# ============================================================
# METHOD 4: SAMPLE CONDITIONING
# ============================================================
log("\n"+"="*60)
log("METHOD 4: Sample Conditioning (Information-Rich Subsets)")
log("="*60)

# Filter for situations where text should matter most
conditions=OrderedDict()

# 4a: All samples (reference)
conditions['All']=np.ones(n,bool)

# 4b: Low coverage stocks only (below median)
conditions['LowCoverage']=cov_log<np.nanmedian(cov_log[~np.isnan(cov_log)])

# 4c: Strong sentiment conviction (top/bottom tercile of |sentiment|)
sent_abs=np.abs(sent_score)
thresh=np.nanquantile(sent_abs[~np.isnan(sent_abs)],0.67)
conditions['StrongSent']=sent_abs>thresh

# 4d: Low coverage + strong sentiment
conditions['LowCov_StrongSent']=conditions['LowCoverage']&conditions['StrongSent']

# 4e: Meaningful text (summary > 100 chars)
summary_len=np.array([len(str(s)) if isinstance(s,str) else 0 for s in labels['summary'].values[:n]],float)
conditions['Meaningful']=summary_len>100

for cname, mask in conditions.items():
    if mask.sum()<100: continue
    r=eval_ic(sent_score[mask], target5[mask], dates_dt[mask], f"Filter_{cname}")
    if r:
        r['n_samples']=int(mask.sum())
        r['pct']=float(mask.mean())
        all_results[f'Filter_{cname}']=r
        log(f"  Filter_{cname:25s}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f} n={mask.sum():,} ({mask.mean():.0%})")

# ============================================================
# METHOD 5: CONSENSUS-DISPERSION DECOMPOSITION
# ============================================================
log("\n"+"="*60)
log("METHOD 5: Consensus-Dispersion")
log("="*60)

# Aggregate to stock-date level: consensus (mean), dispersion (std)
agg=lm[['stock_code','tradable_date']].copy()
agg['sent']=sent_score; agg.index=range(n)

consensus=np.full(n,np.nan); dispersion=np.full(n,np.nan)
for (sc,td),g in agg.groupby(['stock_code','tradable_date']):
    if len(g)>=2:
        consensus[g.index]=g['sent'].mean()
        dispersion[g.index]=g['sent'].std()
    elif len(g)==1:
        consensus[g.index]=g['sent'].iloc[0]
        dispersion[g.index]=0.0

surprise=sent_score-consensus

for nm,fac in [('Consensus',consensus),('Dispersion',dispersion),
               ('Surprise',surprise)]:
    r=eval_ic(fac, target5, dates, nm)
    if r: all_results[nm]=r; log(f"  {nm:15s}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# Consensus × LowDispersion interaction
cd=stdz(consensus)*(-stdz(dispersion))
r=eval_ic(cd, target5, dates, "Consensus_x_LowDisp")
if r: all_results['Cons_x_LowDisp']=r; log(f"  Cons_x_LowDisp: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# ============================================================
# METHOD 6: RISK-ADJUSTED LABELS
# ============================================================
log("\n"+"="*60)
log("METHOD 6: Risk-Adjusted & Alternative Labels")
log("="*60)

# 6a: Sharpe-like: excess return / volatility
vol10=ws(labels['fwd_vol_10d'].fillna(0).values[:n])
sharpe5=target5/(vol10+0.01)  # avoid div by zero
r=eval_ic(sent_score, ws(sharpe5), dates, "Sentiment_on_Sharpe5")
if r: all_results['Sharpe5_Label']=r; log(f"  Sharpe5_Label: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# 6b: Info-adjusted: excess / tracking error proxy
r=eval_ic(sent_score, ws(sharpe5), dates, "SentNeut_on_Sharpe5")
if r: all_results['SentNeut_Sharpe5']=r

# 6c: Max drawdown prediction (text may predict risk better than return)
r=eval_ic(sent_score, maxdd10, dates, "Sentiment_on_MaxDD")
if r: all_results['MaxDD_Label']=r; log(f"  MaxDD_Label: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# 6d: Gap on max drawdown
r=eval_ic(gap_semantic, maxdd10, dates, "Gap_on_MaxDD")
if r: all_results['Gap_MaxDD']=r; log(f"  Gap_MaxDD: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# ============================================================
# COMPOSITE: Best combination
# ============================================================
log("\n"+"="*60)
log("COMPOSITE: Best Signal Combination")
log("="*60)

# Combine best approaches based on results:
# Winner 1: Industry neutralization (IC=0.0212)
# Winner 2: Consensus aggregation (IC=0.0187)
# Winner 3: Sentiment × Coverage interaction
# Strategy: Industry-neutral consensus with coverage interaction

# 1. Neutral consensus: industry-neutralize, then take consensus (stock-date mean)
neut_consensus=np.full(n,np.nan)
for (sc,td),g in agg.groupby(['stock_code','tradable_date']):
    vals=sent_ind_neut[g.index]
    v=~np.isnan(vals)
    neut_consensus[g.index]=vals[v].mean() if v.sum()>0 else np.nan

r=eval_ic(neut_consensus, target5, dates, "NeutConsensus")
all_results['NeutConsensus']=r
log(f"  NeutConsensus: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# 2. NeutConsensus × coverage interaction (direction from data)
# Sent_x_LowCov gave IC=-0.0181, meaning sentiment works better for HIGH coverage
# So try: NeutConsensus × (+coverage) for the correct direction
neutcons_z=stdz(neut_consensus)
nc_x_cov=neutcons_z * cv  # consensus × coverage
r=eval_ic(nc_x_cov, target5, dates, "NeutCons_x_Coverage")
all_results['NeutCons_x_Cov']=r
log(f"  NeutCons_x_Coverage: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# 3. NeutConsensus × size
nc_x_small=neutcons_z * (-sz)  # consensus × small size
r=eval_ic(nc_x_small, target5, dates, "NeutCons_x_SmallSize")
all_results['NeutCons_x_Small']=r
if r: log(f"  NeutCons_x_SmallSize: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")
else: log("  NeutCons_x_SmallSize: returned None")

# 4. Full enhanced: NeutConsensus with coverage boost
enhanced=neutcons_z + 0.3*nc_x_cov
r=eval_ic(enhanced, target5, dates, "Enhanced_Final")
all_results['Enhanced_Final']=r
log(f"  Enhanced_Final: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# 5. NeutConsensus on 10-day window
r10=eval_ic(neut_consensus, target10, dates, "NeutCons_on_10d")
all_results['NeutCons_10d']=r10
log(f"  NeutCons_on_10d: IC={r10['RankIC']:+.4f} t={r10['IC_t']:+.2f}")

# 6. NeutConsensus on MaxDD (risk prediction)
r_md=eval_ic(neut_consensus, maxdd10, dates, "NeutCons_on_MaxDD")
all_results['NeutCons_MaxDD']=r_md
log(f"  NeutCons_on_MaxDD: IC={r_md['RankIC']:+.4f} t={r_md['IC_t']:+.2f}")

# 7. Weighted consensus: recent reports get higher weight
# Aggregate with time decay then neutralize
sent_tdw=np.full(n,np.nan)
hl=5
for (sc,td),g in agg.groupby(['stock_code','tradable_date']):
    if len(g)<1: continue
    ages=(pd.Timestamp(td)-pd.to_datetime(labels['report_date'].iloc[g.index])).dt.days.clip(0).values
    w=np.exp(-np.log(2)*ages/hl); w=w/w.sum()
    sent_tdw[g.index]=np.average(sent_score[g.index],weights=w)

sent_tdw_neut=neutralize(sent_tdw, ind_dummies.values.astype(float)[:n])
# Consensus of TDW-neutral
tdw_consensus=np.full(n,np.nan)
tmp_df=pd.DataFrame({'sc':lm['stock_code'],'td':lm['tradable_date'],'s':sent_tdw_neut})
for (sc,td),g in tmp_df.groupby(['sc','td']):
    v=g['s'].dropna()
    tdw_consensus[g.index]=v.mean() if len(v)>0 else np.nan

r=eval_ic(tdw_consensus, target5, dates, "NeutCons_TDW")
all_results['NeutCons_TDW']=r
log(f"  NeutCons_TDW (time-decay): IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

# ============================================================
# SUMMARY & SAVE
# ============================================================
log("\n"+"="*60)
log("RESULTS SUMMARY")
log("="*60)

base_ic=baseline['RankIC']
log(f"\n{'Method':40s} {'IC':>8s} {'t':>7s} {'vsBase':>8s}")
log("-"*68)

improvements=[]
for nm,r in all_results.items():
    if r is None: continue
    ic=r['RankIC']; t=r['IC_t']
    delta=(ic-base_ic)/abs(base_ic)*100 if abs(base_ic)>0 else 0
    log(f"{nm:40s} {ic:+8.4f} {t:+7.2f} {delta:+7.1f}%")
    improvements.append((nm, ic, t, delta))

# Save
with open(RES/"signal_enhancement_results.json",'w',encoding='utf-8') as f:
    json.dump({nm:{'RankIC':r['RankIC'],'IC_t':r['IC_t'],'ICIR':r['ICIR']}
               for nm,r in all_results.items() if r}, f, indent=2)

# Top improvements
improvements.sort(key=lambda x: x[1], reverse=True)
log(f"\n--- TOP 5 IMPROVEMENTS ---")
for i,(nm,ic,t,d) in enumerate(improvements[:5]):
    log(f"  {i+1}. {nm}: IC={ic:+.4f} t={t:.2f} ({d:+.1f}%)")

log(f"\nResults saved to: {RES/'signal_enhancement_results.json'}")
log("DONE")
