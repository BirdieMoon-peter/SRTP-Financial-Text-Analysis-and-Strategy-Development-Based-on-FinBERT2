"""
SRTP: FinBERT Hidden Layer Text Factor Research
Embedding Extraction Module
==============================================
Extracts hidden states from all layers of FinBERT for
three input configurations: title-only, summary-only, full.
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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "finbert-tone-chinese"
DATA_DIR = PROJECT_ROOT / "data"
EMBED_DIR = PROJECT_ROOT / "data" / "embeddings"


class ReportDataset(Dataset):
    """Dataset for batch encoding of report texts."""

    def __init__(self, texts, tokenizer, max_length=512):
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
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def build_input_texts(df):
    """Build three input configurations from reports data."""
    titles = df["title"].fillna("").astype(str).tolist()
    summaries = df["summary"].fillna("").astype(str).tolist()
    input_configs = {
        "title": titles,
        "summary": summaries,
        "full": [f"{t} [SEP] {s}" for t, s in zip(titles, summaries)],
    }
    return input_configs


def extract_embeddings_batch(model, tokenizer, texts, config_name,
                              device, batch_size=32, max_length=512):
    """
    Extract hidden states from all layers for a batch of texts.

    Returns:
        dict with keys:
        - last_cls: (N, 768)  final layer CLS
        - all_cls:  (N, L, 768)  CLS from each layer
        - all_mean: (N, L, 768)  mean pooling from each layer
        - hidden_states: list of (N, L, hidden) for layer aggregation
    """
    dataset = ReportDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    all_last_cls = []
    all_all_cls = []
    all_all_mean = []
    n_layers = None

    model.eval()
    n_batches = len(dataloader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # hidden_states: tuple of (batch, seq_len, hidden), length L+1 (embeddings + layers)
            hidden_states = outputs.hidden_states
            if n_layers is None:
                n_layers = len(hidden_states) - 1  # exclude embedding layer

            # Stack layer outputs: (L+1, B, S, H) -> use layers 1..L
            stacked = torch.stack(hidden_states[1:], dim=0)  # (L, B, S, H)
            L = stacked.shape[0]

            # CLS token (position 0) from each layer
            cls_per_layer = stacked[:, :, 0, :]  # (L, B, H)
            all_all_cls.append(cls_per_layer.permute(1, 0, 2).cpu().numpy())  # (B, L, H)

            # Last layer CLS
            all_last_cls.append(hidden_states[-1][:, 0, :].cpu().numpy())  # (B, H)

            # Mean pooling per layer (exclude CLS and SEP positions for richer semantics)
            # Use attention_mask for proper mean
            mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(0)  # (1, B, S, 1)
            mask_expanded = mask_expanded.float()
            # Sum over non-padding tokens, divide by count
            sum_per_layer = (stacked * mask_expanded).sum(dim=2)  # (L, B, H)
            count_per_layer = mask_expanded.sum(dim=2).clamp(min=1)  # (L, B, 1)
            mean_per_layer = sum_per_layer / count_per_layer  # (L, B, H)
            all_all_mean.append(mean_per_layer.permute(1, 0, 2).cpu().numpy())  # (B, L, H)

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                speed = (batch_idx + 1) * batch_size / elapsed
                eta = (n_batches - batch_idx - 1) * batch_size / speed
                print(f"  [{config_name}] Batch {batch_idx+1}/{n_batches} | "
                      f"Speed: {speed:.0f} texts/s | ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"  [{config_name}] Done in {elapsed:.0f}s ({len(texts)/elapsed:.1f} texts/s)")

    result = {
        "last_cls": np.concatenate(all_last_cls, axis=0),     # (N, 768)
        "all_cls": np.concatenate(all_all_cls, axis=0),       # (N, L, 768)
        "all_mean": np.concatenate(all_all_mean, axis=0),     # (N, L, 768)
        "n_layers": n_layers,
        "hidden_dim": all_last_cls[0].shape[-1],
    }
    return result


def compute_title_summary_gap(emb_title, emb_summary):
    """Compute title-summary semantic gap features."""
    # Concatenate along layer dim for gap computation
    # gap = title - summary per layer
    gap_cls = emb_title["all_cls"] - emb_summary["all_cls"]       # (N, L, H)
    gap_mean = emb_title["all_mean"] - emb_summary["all_mean"]    # (N, L, H)

    # Cosine similarity between title and summary per layer
    cos_sim = []
    for l in range(emb_title["n_layers"]):
        t = emb_title["all_cls"][:, l, :]
        s = emb_summary["all_cls"][:, l, :]
        sim = (t * s).sum(axis=1) / (
            np.linalg.norm(t, axis=1) * np.linalg.norm(s, axis=1) + 1e-8
        )
        cos_sim.append(sim)
    cos_sim = np.stack(cos_sim, axis=1)  # (N, L)

    # Euclidean distance
    euclidean_dist = np.linalg.norm(
        emb_title["all_cls"] - emb_summary["all_cls"], axis=2  # (N, L)
    )

    return {
        "gap_cls": gap_cls,
        "gap_mean": gap_mean,
        "cos_sim": cos_sim,
        "euclidean_dist": euclidean_dist,
    }


def extract_finbert_sentiment(model, tokenizer, texts, device, batch_size=32, max_length=512):
    """Extract FinBERT final sentiment outputs (for baseline comparison)."""
    dataset = ReportDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()  # (B, 3): neg, neutral, pos
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            all_logits.append(logits)
            all_probs.append(probs)

    logits = np.concatenate(all_logits, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    return {"logits": logits, "probabilities": probs}


def main():
    print("=" * 60)
    print("SRTP Embedding Extraction")
    print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(DATA_DIR / "reports_cleaned.csv")
    print(f"Loaded {len(df):,} reports")

    # Load model
    print(f"\nLoading FinBERT from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), output_hidden_states=True
    )
    model.to(device)
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")

    # Build inputs
    input_configs = build_input_texts(df)
    for name, texts in input_configs.items():
        # Count non-null texts
        valid = [t for t in texts if isinstance(t, str) and len(t) >= 2]
        n_invalid = len(texts) - len(valid)
        print(f"  [{name}] {len(texts)} texts, "
              f"avg len: {sum(len(t) for t in valid)/max(len(valid),1):.0f} chars"
              + (f", invalid: {n_invalid}" if n_invalid else ""))

    # Create embeddings directory
    EMBED_DIR.mkdir(exist_ok=True)

    # Optional: limit samples for quick testing
    quick_test = os.environ.get("QUICK_TEST", "").lower() in ("1", "true", "yes")
    if quick_test:
        n_test = min(1000, len(df))
        print(f"\n[QUICK TEST MODE] Using {n_test} samples")
        df = df.head(n_test)
        for name in input_configs:
            input_configs[name] = input_configs[name][:n_test]

    batch_size = 16 if device.type == "cpu" else 32
    if device.type == "mps":
        batch_size = 8  # MPS can be memory-limited

    # Extract embeddings for each config
    all_embeddings = {}
    for config_name, texts in input_configs.items():
        print(f"\n{'='*40}")
        print(f"Extracting [{config_name}] embeddings...")
        print(f"  Texts: {len(texts)}, Batch size: {batch_size}")

        emb = extract_embeddings_batch(
            model, tokenizer, texts, config_name, device, batch_size
        )
        all_embeddings[config_name] = emb

        # Save per-config
        save_path = EMBED_DIR / f"embeddings_{config_name}.npz"
        np.savez_compressed(
            save_path,
            last_cls=emb["last_cls"],
            all_cls=emb["all_cls"],
            all_mean=emb["all_mean"],
        )
        print(f"  Saved to {save_path} (shape: last_cls {emb['last_cls'].shape})")

        # Save metadata
        meta = {
            "config_name": config_name,
            "n_samples": len(texts),
            "n_layers": emb["n_layers"],
            "hidden_dim": emb["hidden_dim"],
            "last_cls_shape": list(emb["last_cls"].shape),
            "all_cls_shape": list(emb["all_cls"].shape),
        }
        with open(EMBED_DIR / f"embeddings_{config_name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # Compute title-summary gap features
    if "title" in all_embeddings and "summary" in all_embeddings:
        print(f"\n{'='*40}")
        print("Computing title-summary gap features...")
        gap_features = compute_title_summary_gap(
            all_embeddings["title"], all_embeddings["summary"]
        )
        gap_path = EMBED_DIR / "embeddings_gap.npz"
        np.savez_compressed(gap_path, **gap_features)
        print(f"  Gap features saved to {gap_path}")

        # Save gap meta
        gap_meta = {k: list(v.shape) for k, v in gap_features.items()}
        with open(EMBED_DIR / "embeddings_gap_meta.json", "w") as f:
            json.dump(gap_meta, f, indent=2)

    # Also extract FinBERT final sentiment (for baseline)
    print(f"\n{'='*40}")
    print("Extracting FinBERT final sentiment outputs...")
    sentiment = extract_finbert_sentiment(
        model, tokenizer, input_configs["full"], device, batch_size
    )
    sentiment_path = EMBED_DIR / "sentiment_finbert.npz"
    np.savez_compressed(sentiment_path, **sentiment)
    print(f"  Sentiment saved to {sentiment_path}")

    print(f"\n{'='*60}")
    print("Embedding extraction complete.")
    print(f"Files saved in {EMBED_DIR}")
    for f in sorted(os.listdir(EMBED_DIR)):
        size_mb = os.path.getsize(EMBED_DIR / f) / 1024 / 1024
        print(f"  {f:45s} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
