"""
RTX 3090 Optimized Embedding Extraction
24GB VRAM - batch_size=64, max GPU utilization
"""
import os, sys, gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path("/root/srtp")
MODEL_DIR = BASE / "models" / "finbert-tone-chinese"
DATA_DIR = BASE / "data"
EMBED_DIR = DATA_DIR / "embeddings"
EMBED_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
MAX_LENGTH = 256

class ReportDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts; self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        t = self.texts[idx]
        if not isinstance(t, str) or len(t.strip()) < 2: t = "[PAD]"
        e = self.tokenizer(t, truncation=True, padding="max_length",
                          max_length=MAX_LENGTH, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0)}

def extract(model, tokenizer, texts, device):
    ds = ReportDataset(texts, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    cls_list, all_cls_list, mean_list = [], [], []
    model.eval(); n = len(dl); t0 = time.time()
    with torch.no_grad():
        for i, b in enumerate(dl):
            ids = b["input_ids"].to(device); am = b["attention_mask"].to(device)
            hs = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
            stacked = torch.stack(hs[1:], dim=0)  # (L, B, S, H)
            all_cls_list.append(stacked[:,:,0,:].permute(1,0,2).cpu().numpy())
            me = (stacked * am.unsqueeze(0).unsqueeze(-1).float()).sum(2) / am.unsqueeze(0).unsqueeze(-1).float().sum(2).clamp(min=1)
            mean_list.append(me.permute(1,0,2).cpu().numpy())
            cls_list.append(hs[-1][:,0,:].cpu().numpy())
            if (i+1)%50==0:
                e = time.time()-t0; s = (i+1)*BATCH_SIZE/e
                print(f"  [{min((i+1)*BATCH_SIZE,len(texts))}/{len(texts)}] {s:.0f}/s, ETA:{max(0,(n-i-1)*BATCH_SIZE/s):.0f}s, GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB")
    e = time.time()-t0
    print(f"  Done: {e:.0f}s ({len(texts)/e:.0f}/s)")
    return {"last_cls": np.concatenate(cls_list), "all_cls": np.concatenate(all_cls_list),
            "all_mean": np.concatenate(mean_list), "n_layers": len(hs)-1, "hidden_dim": hs[-1].shape[-1]}

def extract_sentiment(model, tokenizer, texts, device):
    ds = ReportDataset(texts, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    logits, probs = [], []
    model.eval()
    with torch.no_grad():
        for b in dl:
            o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            logits.append(o.logits.cpu().numpy())
            probs.append(torch.softmax(o.logits, -1).cpu().numpy())
    return {"logits": np.concatenate(logits), "probabilities": np.concatenate(probs)}

def main():
    print("="*60); print("RTX 3090 Embedding Extraction"); print("="*60)
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    df = pd.read_csv(DATA_DIR/"reports_cleaned.csv")
    print(f"Reports: {len(df):,}")

    tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    m = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), output_hidden_states=True).to(dev)
    print(f"Model: {m.config.num_hidden_layers}L, {m.config.hidden_size}D")

    titles = df["title"].fillna("").astype(str).tolist()
    summaries = df["summary"].fillna("").astype(str).tolist()
    configs = {"title": titles, "summary": summaries,
               "full": [f"{t} [SEP] {s}" for t,s in zip(titles,summaries)]}

    embs = {}
    for name, texts in configs.items():
        print(f"\n{'='*40}\n[{name}] {len(texts)} texts")
        torch.cuda.empty_cache(); import gc; gc.collect()
        emb = extract(m, tk, texts, dev)
        embs[name] = emb
        np.savez_compressed(EMBED_DIR/f"embeddings_{name}.npz",
                           last_cls=emb["last_cls"], all_cls=emb["all_cls"], all_mean=emb["all_mean"])
        with open(EMBED_DIR/f"embeddings_{name}_meta.json","w") as f:
            json.dump({"n_samples":len(texts),"n_layers":emb["n_layers"],"hidden_dim":emb["hidden_dim"]},f)
        # Free memory: delete embeddings for this config after saving
        size_mb = os.path.getsize(EMBED_DIR/f"embeddings_{name}.npz") / 1e6
        print(f"  Saved: {size_mb:.0f}MB. Clear memory.")
        del embs[name]; gc.collect()

    # Gap features
    if "title" in embs and "summary" in embs:
        print("\n=== Gap features ===")
        ta = embs["title"]["all_cls"]; sa = embs["summary"]["all_cls"]
        tm = embs["title"]["all_mean"]; sm = embs["summary"]["all_mean"]
        gap_c = ta - sa; gap_m = tm - sm
        cos_sim = np.zeros((ta.shape[0], ta.shape[1]))
        for l in range(ta.shape[1]):
            tn = np.linalg.norm(ta[:,l,:], axis=1) + 1e-8
            sn = np.linalg.norm(sa[:,l,:], axis=1) + 1e-8
            cos_sim[:,l] = (ta[:,l,:] * sa[:,l,:]).sum(1) / (tn * sn)
        np.savez_compressed(EMBED_DIR/"embeddings_gap.npz", gap_cls=gap_c, gap_mean=gap_m, cos_sim=cos_sim)
        print("  Gap saved")

    # Sentiment
    print("\n=== Sentiment ===")
    torch.cuda.empty_cache()
    sent = extract_sentiment(m, tk, configs["full"], dev)
    np.savez_compressed(EMBED_DIR/"sentiment_finbert.npz", **sent)
    print("  Sentiment saved")

    print(f"\n{'='*60}\nCOMPLETE!")
    for f in sorted(os.listdir(EMBED_DIR)):
        print(f"  {f:40s} {os.path.getsize(EMBED_DIR/f)/1e6:.0f}MB")

if __name__ == "__main__":
    main()
