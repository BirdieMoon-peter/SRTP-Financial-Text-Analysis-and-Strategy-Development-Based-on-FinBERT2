#!/usr/bin/env python3
"""
SRTP: FinBERT Hidden Layer Text Factor Research
Main Pipeline Runner
==============================================
Orchestrates the complete research pipeline:
  1. Data Preprocessing
  2. CSMAR Market Data Collection
  3. Label Construction
  4. Embedding Extraction
  5. FHF Factor Construction
  6. Empirical Testing & Backtesting
  7. Interpretability Analysis
  8. Results Export for Thesis

Usage:
  python src/run_pipeline.py --step preprocess    # Run specific step
  python src/run_pipeline.py --all                 # Run all steps
  python src/run_pipeline.py --from embeddings     # Run from embeddings onward
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
EMBED_DIR = DATA_DIR / "embeddings"


def setup_dirs():
    for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR, EMBED_DIR]:
        d.mkdir(exist_ok=True)


def log_step(step_name, start_time, success=True, extra=None):
    """Log step execution to LOGS_DIR."""
    elapsed = time.time() - start_time
    log_entry = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "success": success,
        "extra": extra or {},
    }
    log_file = LOGS_DIR / "pipeline_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    status = "✅" if success else "❌"
    print(f"\n{status} [{step_name}] completed in {elapsed:.0f}s")
    return log_entry


def step_preprocess():
    """Step 1: Data preprocessing."""
    from preprocess import main as preprocess_main
    df = preprocess_main()
    return {"records": len(df), "stocks": df["stock_code"].nunique()}


def step_download_models():
    """Step 2: Download models locally."""
    from download_models import main as download_main
    download_main()
    return {"models": os.listdir(PROJECT_ROOT / "models")}


def step_extract_embeddings(quick_test=False):
    """Step 3: Extract FinBERT embeddings."""
    from extract_embeddings import main as extract_main
    if quick_test:
        os.environ["QUICK_TEST"] = "1"
    extract_main()
    return {"embeddings_dir": str(EMBED_DIR)}


def step_build_labels():
    """Step 4: Build labels from market data."""
    from build_labels import main as labels_main
    labels_main()
    return {"status": "requires CSMAR data"}


def step_build_factors():
    """Step 5: Build FHF factors."""
    from fhf_factors import main as factors_main
    factors_main()
    return {"status": "tested on synthetic data"}


def step_backtest():
    """Step 6: Run empirical tests and backtesting."""
    from backtest import main as backtest_main
    backtest_main()
    return {"status": "tested on synthetic data"}


def step_interpretability():
    """Step 7: Interpretability analysis."""
    # Placeholder - to be implemented
    return {"status": "pending"}


STEPS = {
    "preprocess": step_preprocess,
    "download-models": step_download_models,
    "embeddings": step_extract_embeddings,
    "labels": step_build_labels,
    "factors": step_build_factors,
    "backtest": step_backtest,
    "interpretability": step_interpretability,
}


def main():
    parser = argparse.ArgumentParser(description="SRTP Research Pipeline")
    parser.add_argument("--step", choices=list(STEPS.keys()), help="Run a specific step")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--from", dest="from_step", choices=list(STEPS.keys()),
                        help="Run from a specific step onward")
    parser.add_argument("--quick-test", action="store_true",
                        help="Use subset of data for quick testing")
    args = parser.parse_args()

    setup_dirs()

    # Determine which steps to run
    if args.all:
        steps_to_run = list(STEPS.keys())
    elif args.from_step:
        step_names = list(STEPS.keys())
        idx = step_names.index(args.from_step)
        steps_to_run = step_names[idx:]
    elif args.step:
        steps_to_run = [args.step]
    else:
        parser.print_help()
        return

    print("=" * 60)
    print("SRTP Research Pipeline")
    print(f"Steps to run: {steps_to_run}")
    print(f"Quick test mode: {args.quick_test}")
    print("=" * 60)

    pipeline_results = {}
    for step_name in steps_to_run:
        print(f"\n{'='*50}")
        print(f"Running: {step_name}")
        print(f"{'='*50}")

        start_time = time.time()
        try:
            fn = STEPS[step_name]
            if step_name == "embeddings":
                result = fn(quick_test=args.quick_test)
            else:
                result = fn()
            extra = result if result else {}
            log_step(step_name, start_time, success=True, extra=extra)
            pipeline_results[step_name] = "success"
        except Exception as e:
            import traceback
            traceback.print_exc()
            log_step(step_name, start_time, success=False, extra={"error": str(e)})
            pipeline_results[step_name] = f"failed: {e}"
            if not args.all:
                break  # Stop on error unless --all

    # Summary
    print(f"\n{'='*60}")
    print("Pipeline Summary")
    print(f"{'='*60}")
    for step, result in pipeline_results.items():
        status = "✅" if result == "success" else "❌"
        print(f"  {status} {step}: {result}")


if __name__ == "__main__":
    main()
