"""
SRTP: FinBERT Hidden Layer Text Factor Research
Data Preprocessing Module
==============================================
Cleans analyst report data: dedup, standardize codes/dates,
validate entities, align time points.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_reports(path=None):
    """Load raw analyst reports CSV."""
    if path is None:
        path = PROJECT_ROOT / "data" / "reports.csv"
    df = pd.read_csv(path)
    print(f"[load] Raw records: {len(df):,}")
    print(f"[load] Columns: {list(df.columns)}")
    print(f"[load] Date range: {df['report_date'].min()} ~ {df['report_date'].max()}")
    return df


def clean_reports(df):
    """Clean and deduplicate analyst reports."""
    initial = len(df)

    # Drop fully duplicate rows
    df = df.drop_duplicates().copy()
    print(f"[clean] After drop_duplicates: {len(df):,} (removed {initial - len(df)})")

    # Standardize report_date
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"])

    # Standardize stock_code: ensure 6-digit string
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.zfill(6)

    # Clean title and summary
    df["title"] = df["title"].astype(str).str.strip()
    df["summary"] = df["summary"].astype(str).str.strip()

    # Drop empty title AND empty summary
    df = df[~(df["title"].str.len() < 2)]
    df = df[~(df["summary"].str.len() < 5)]

    # Drop rows where summary is just placeholder
    placeholder_patterns = [
        "支撑评级的要点", "投资要点", "事件概述", "核心观点",
        "主要内容", "报告摘要", "报告亮点"
    ]
    for pat in placeholder_patterns:
        df = df[df["summary"] != pat]

    # Remove stock_code prefix from summary (e.g., "宁波银行(002142)\n...")
    df["summary"] = df["summary"].str.replace(
        r"^[^(\n]*\(?\d{6}\)?\s*\n?", "", regex=True
    ).str.strip()

    # Keep only valid stock codes
    df = df[df["stock_code"].str.match(r"^\d{6}$")]

    # Sort by date
    df = df.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

    print(f"[clean] Final records: {len(df):,} (removed {initial - len(df)})")
    return df


def compute_text_stats(df):
    """Compute text length statistics."""
    df["title_len"] = df["title"].str.len()
    df["summary_len"] = df["summary"].str.len()

    print(f"[stats] Title length  - mean: {df['title_len'].mean():.0f}, "
          f"median: {df['title_len'].median():.0f}, max: {df['title_len'].max()}")
    print(f"[stats] Summary length - mean: {df['summary_len'].mean():.0f}, "
          f"median: {df['summary_len'].median():.0f}, max: {df['summary_len'].max()}")

    return df


def add_date_features(df):
    """Add year, month, quarter features for stratified analysis."""
    df["year"] = df["report_date"].dt.year
    df["month"] = df["report_date"].dt.month
    df["quarter"] = df["report_date"].dt.quarter
    return df


def main():
    print("=" * 60)
    print("SRTP Data Preprocessing")
    print("=" * 60)

    df = load_reports()
    df = clean_reports(df)
    df = compute_text_stats(df)
    df = add_date_features(df)

    # Save cleaned data
    out_path = PROJECT_ROOT / "data" / "reports_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[save] Cleaned data saved to {out_path}")
    print(f"[save] Records: {len(df):,}")
    print(f"[save] Unique stocks: {df['stock_code'].nunique():,}")
    print(f"[save] Date range: {df['report_date'].min()} ~ {df['report_date'].max()}")

    # Summary stats
    print(f"\n{'='*40}")
    print("Yearly distribution:")
    print(df["year"].value_counts().sort_index())
    print(f"\nQuarterly distribution:")
    print(df["quarter"].value_counts().sort_index())

    return df


if __name__ == "__main__":
    df = main()
