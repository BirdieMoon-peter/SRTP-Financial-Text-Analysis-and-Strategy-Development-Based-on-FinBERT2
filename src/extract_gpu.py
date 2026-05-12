"""
GPU-accelerated FinBERT Embedding Extraction
=============================================
Run on Windows with CUDA GPU (RTX 3050 Ti).
Extracts hidden states from all FinBERT layers.
"""

import os
import sys
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths
BASE_DIR = Path(r"C:\Users\13082\CSMAR")
MODEL_DIR = BASE_DIR / "models" / "finbert-tone-chinese"
DATA_DIR = BASE_DIR / "data"
EMBED_DIR = DATA_DIR / "embeddings"
EMBED_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8  # RTX 3050 Ti 4GB - conservative for hidden states
MAX_LENGTH = 192  # Most summaries <250 chars, Chinese chars ~1-2 tokens


class ReportDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if not isinstance(text, str) or len(text.strip()) < 2:
            text = "[PAD]"
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def extract_embeddings(model, tokenizer, texts, device, batch_size=BATCH_SIZE):
    dataset = ReportDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    all_last_cls = []
    all_all_cls = []
    all_all_mean = []

    model.eval()
    n_batches = len(dataloader)
    start_time = time.time()
    n_layers = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (L+1) tensors

            if n_layers is None:
                n_layers = len(hidden_states) - 1

            # Stack layers 1..L (skip embedding)
            stacked = torch.stack(hidden_states[1:], dim=0)  # (L, B, S, H)
            L = stacked.shape[0]

            # CLS token
            cls_per_layer = stacked[:, :, 0, :]  # (L, B, H)
            all_all_cls.append(cls_per_layer.permute(1, 0, 2).cpu().numpy())

            # Mean pooling
            mask_expanded = attention_mask.unsqueeze(0).unsqueeze(-1).float()  # (1, B, S, 1)
            sum_per_layer = (stacked * mask_expanded).sum(dim=2)  # (L, B, H)
            count = mask_expanded.sum(dim=2).clamp(min=1)
            mean_per_layer = sum_per_layer / count
            all_all_mean.append(mean_per_layer.permute(1, 0, 2).cpu().numpy())

            # Last layer CLS
            all_last_cls.append(hidden_states[-1][:, 0, :].cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                done = (batch_idx + 1) * batch_size
                speed = done / elapsed
                eta = (len(texts) - done) / speed
                mem = torch.cuda.memory_allocated() / 1e9
                print(f"  [{done}/{len(texts)}] speed={speed:.0f}/s, ETA={eta:.0f}s, GPU={mem:.1f}GB")

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.0f}s ({len(texts)/elapsed:.1f} texts/s)")

    return {
        "last_cls": np.concatenate(all_last_cls, axis=0),
        "all_cls": np.concatenate(all_all_cls, axis=0),
        "all_mean": np.concatenate(all_all_mean, axis=0),
        "n_layers": n_layers,
        "hidden_dim": all_last_cls[0].shape[-1],
    }


def build_input_texts(df):
    titles = df["title"].fillna("").astype(str).tolist()
    summaries = df["summary"].fillna("").astype(str).tolist()
    return {
        "title": titles,
        "summary": summaries,
        "full": [f"{t} [SEP] {s}" for t, s in zip(titles, summaries)],
    }


def extract_sentiment(model, tokenizer, texts, device, batch_size=BATCH_SIZE):
    """Extract FinBERT final sentiment outputs."""
    dataset = ReportDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    all_logits = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_logits.append(logits)
            all_probs.append(probs)
    return {
        "logits": np.concatenate(all_logits, axis=0),
        "probabilities": np.concatenate(all_probs, axis=0),
    }


def main():
    print("=" * 60)
    print("GPU Embedding Extraction")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print(f"Batch size: {BATCH_SIZE}")

    # Load data
    df = pd.read_csv(DATA_DIR / "reports_cleaned.csv")
    print(f"Loaded {len(df):,} reports")

    # Load model
    print(f"Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), output_hidden_states=True
    )
    model.to(device)
    print(f"Layers: {model.config.num_hidden_layers}, Hidden: {model.config.hidden_size}")

    # Build inputs
    inputs = build_input_texts(df)
    for name, texts in inputs.items():
        print(f"  [{name}] {len(texts)} texts")

    # Extract embeddings for each config
    all_embs = {}
    for name, texts in inputs.items():
        print(f"\n{'='*40}")
        print(f"Extracting [{name}]...")
        torch.cuda.empty_cache()
        gc.collect()

        emb = extract_embeddings(model, tokenizer, texts, device)
        all_embs[name] = emb

        npz_path = EMBED_DIR / f"embeddings_{name}.npz"
        np.savez_compressed(npz_path, last_cls=emb["last_cls"],
                            all_cls=emb["all_cls"], all_mean=emb["all_mean"])
        with open(EMBED_DIR / f"embeddings_{name}_meta.json", "w") as f:
            json.dump({"n_samples": len(texts), "n_layers": emb["n_layers"],
                       "hidden_dim": emb["hidden_dim"]}, f)
        print(f"  Saved: {npz_path}")

    # Gap features
    if "title" in all_embs and "summary" in all_embs:
        print(f"\n{'='*40}")
        print("Computing gap features...")
        t = all_embs["title"]; s = all_embs["summary"]
        gap_cls = t["all_cls"] - s["all_cls"]
        gap_mean = t["all_mean"] - s["all_mean"]
        cos_sim = np.array([
            (t["all_cls"][:, l, :] * s["all_cls"][:, l, :]).sum(axis=1) /
            (np.linalg.norm(t["all_cls"][:, l, :], axis=1) *
             np.linalg.norm(s["all_cls"][:, l, :], axis=1) + 1e-8)
            for l in range(all_embs["title"]["n_layers"])
        ]).T
        np.savez_compressed(EMBED_DIR / "embeddings_gap.npz",
                            gap_cls=gap_cls, gap_mean=gap_mean, cos_sim=cos_sim)
        print("  Gap features saved")

    # Sentiment baseline
    torch.cuda.empty_cache()
    print(f"\n{'='*40}")
    print("Extracting sentiment baseline...")
    sent = extract_sentiment(model, tokenizer, inputs["full"], device)
    np.savez_compressed(EMBED_DIR / "sentiment_finbert.npz", **sent)
    print("  Sentiment saved")

    print(f"\n{'='*60}")
    print("COMPLETE. Files:")
    for f in sorted(os.listdir(EMBED_DIR)):
        size_mb = os.path.getsize(EMBED_DIR / f) / 1024 / 1024
        print(f"  {f:45s} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
