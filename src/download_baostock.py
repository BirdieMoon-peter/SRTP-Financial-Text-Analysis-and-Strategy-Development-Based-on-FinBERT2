"""
Robust baostock downloader with timeout, skip on error, and resume support.
"""

import os, sys, time, socket
from pathlib import Path
import pandas as pd
import numpy as np

# Set global socket timeout to prevent hanging on network calls
socket.setdefaulttimeout(30)

import baostock as bs

DATA_DIR = Path(r"C:\Users\13082\CSMAR\data")
OUTPUT_FILE = DATA_DIR / "csmar_daily_stock.csv"
LOG_FILE = DATA_DIR / "baostock_progress.txt"


def baostock_code(code):
    code = str(code).zfill(6)
    first = code[0]
    if first == '6': return f"sh.{code}"
    elif first in ('0', '3'): return f"sz.{code}"
    elif first in ('9', '8'): return f"bj.{code}"
    return f"sh.{code}"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Load stocks
    reports = pd.read_csv(DATA_DIR / "reports_cleaned.csv", dtype={"stock_code": str})
    all_stocks = sorted(reports["stock_code"].unique())
    print(f"Total stocks: {len(all_stocks)}")

    # Resume: read already downloaded stock codes
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, dtype={"stock_code": str})
        existing_stocks = set(existing["stock_code"].unique())
        stocks_to_download = [s for s in all_stocks if s not in existing_stocks]
        print(f"Already downloaded: {len(existing_stocks)}, Remaining: {len(stocks_to_download)}")
    else:
        existing_stocks = set()
        stocks_to_download = all_stocks
        existing = None

    if not stocks_to_download:
        print("All stocks downloaded!")
        return

    bs.login()
    print("Baostock login OK")

    buffer = []  # Accumulate data for batch save
    n_done = len(existing_stocks)
    n_total = len(all_stocks)
    start_time = time.time()
    errors = 0
    consecutive_errors = 0
    save_interval = 100

    for i, code in enumerate(stocks_to_download):
        bs_code = baostock_code(code)

        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus",
                start_date='2020-01-01', end_date='2026-05-11',
                frequency="d", adjustflag="2"
            )

            if rs.error_code != '0':
                errors += 1
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"  [{n_done + i + 1}/{n_total}] {code} ERROR: {rs.error_msg}")
                if consecutive_errors >= 20:
                    print(f"  Too many errors, backing off for 30s...")
                    time.sleep(30)
                    consecutive_errors = 0
                continue

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if rows:
                df_code = pd.DataFrame(rows, columns=rs.fields)
                df_code["stock_code"] = code
                buffer.append(df_code)
                consecutive_errors = 0

        except Exception as e:
            errors += 1
            consecutive_errors += 1
            if consecutive_errors <= 3:
                print(f"  [{n_done + i + 1}] {code} EXCEPTION: {e}")
            time.sleep(3)
            continue

        # Save batch every save_interval stocks
        if len(buffer) >= save_interval:
            batch = pd.concat(buffer, ignore_index=True)
            if existing is not None:
                batch = pd.concat([existing, batch], ignore_index=True)
                batch = batch.drop_duplicates(subset=["stock_code", "date"])
            batch.to_csv(OUTPUT_FILE, index=False)
            existing = batch
            n_done += len(buffer)
            elapsed = time.time() - start_time
            eta = elapsed / max(n_done - len(existing_stocks), 1) * (n_total - n_done)
            print(f"  [{n_done}/{n_total}] {n_done*100/n_total:.0f}% | "
                  f"{len(batch):,} rows | ETA: {eta/60:.0f}min | {errors} err")
            buffer = []

        # Rate limiting
        if (i + 1) % 20 == 0:
            time.sleep(1)

    # Final save
    if buffer:
        batch = pd.concat(buffer, ignore_index=True)
        if existing is not None:
            batch = pd.concat([existing, batch], ignore_index=True)
            batch = batch.drop_duplicates(subset=["stock_code", "date"])
        batch.to_csv(OUTPUT_FILE, index=False)
        existing = batch

    bs.logout()
    elapsed = time.time() - start_time
    n_done_final = len(existing["stock_code"].unique()) if existing is not None else 0
    print(f"\nDone! {n_done_final}/{n_total} stocks, {errors} errors, {elapsed/60:.1f} min")
    print(f"Output: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
