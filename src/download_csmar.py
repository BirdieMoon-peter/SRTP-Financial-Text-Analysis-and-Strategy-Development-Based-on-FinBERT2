#!/usr/bin/env python3
"""
CSMAR data acquisition utility for the SRTP project.

The official CSMAR Python interface is Windows-oriented and exposes metadata
and query endpoints under cn.gtadata.com. This script follows that interface
style while keeping credentials out of source files and disk:

    set CSMAR_USERNAME=...
    set CSMAR_PASSWORD=...
    python download_csmar.py list-dbs
    python download_csmar.py discover --keywords 股票 交易 行情 指数 行业
    python download_csmar.py download --dataset daily_stock

Outputs are CSV files in the configured output directory. The login token is
kept only in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


DEFAULT_BASE_URL = os.environ.get("CSMAR_BASE_URL", "https://data.csmar.com")
DEFAULT_OUTPUT_DIR = Path(os.environ.get("CSMAR_OUTPUT_DIR", "C:/Users/ssh_agent/CSMAR/data"))
DEFAULT_START_DATE = os.environ.get("CSMAR_START_DATE", "2020-01-01")
DEFAULT_END_DATE = os.environ.get("CSMAR_END_DATE", "2026-05-08")
DEFAULT_PAGE_SIZE = int(os.environ.get("CSMAR_PAGE_SIZE", "100000"))


DATASET_SPECS = {
    "daily_stock": {
        "table_candidates": ["TRD_Dalyr", "STK_MKT_Dalyr", "TRD_Daily", "TRD_Dalyr1"],
        "field_candidates": [
            "Stkcd", "Trddt", "Opnprc", "Hiprc", "Loprc", "Clsprc",
            "Dnshrtrd", "Dnvaltrd", "Dretwd", "Dretnd", "Dsmvosd", "Dsmvtll",
            "Adjprcwd", "Adjprcnd", "Markettype", "Capchgdt", "Trdsta",
            "PreClosePrice", "ChangeRatio", "LimitDown", "LimitUp", "LimitStatus",
        ],
        "date_field": "Trddt",
        "condition_template": "{date_field}>='{start}' and {date_field}<='{end}'",
        "output": "csmar_daily_stock.csv",
    },
    "index_daily": {
        "table_candidates": ["IDX_Idxtrd", "TRD_Index", "IDX_Daily", "IDX_Idxtrd1"],
        "field_candidates": [
            "Indexcd", "Idxtrd01", "Idxtrd02", "Idxtrd03", "Idxtrd04",
            "Idxtrd05", "Idxtrd06", "Idxtrd07", "Idxtrd08", "Idxtrd09",
        ],
        "date_field": "Idxtrd01",
        "condition_template": (
            "Indexcd in ('000001','000300','000905','000852','000985') "
            "and {date_field}>='{start}' and {date_field}<='{end}'"
        ),
        "output": "csmar_index_daily.csv",
    },
    "daily_derived": {
        "table_candidates": ["STK_MKT_DALYR"],
        "field_candidates": [
            "SecurityID", "TradingDate", "Symbol", "ShortName", "Ret", "PE", "PB",
            "PCF", "PS", "Turnover", "CirculatedMarketValue", "ChangeRatio",
            "Amount", "Liquidility",
        ],
        "date_field": "TradingDate",
        "condition_template": "{date_field}>='{start}' and {date_field}<='{end}'",
        "output": "csmar_daily_derived.csv",
    },
    "suspend_status": {
        "table_candidates": ["TSR_Stkstat"],
        "field_candidates": [
            "Stkcd", "Stknme", "Annctime", "Type", "Suspdate", "Susptime",
            "Resmdate", "Resmtime", "Timeperd", "Reason",
        ],
        "date_field": None,
        "condition_template": "1=1",
        "output": "csmar_suspend_status.csv",
    },
    "special_treatment": {
        "table_candidates": ["SPT_Trdchg"],
        "field_candidates": [
            "Stkcd", "Stknmebc", "Stknmeac", "Chgtype", "Annoudt",
            "Chgreas", "Chgrsdis", "Content", "Execudt",
        ],
        "date_field": None,
        "condition_template": "1=1",
        "output": "csmar_special_treatment.csv",
    },
    "listing_status": {
        "table_candidates": ["STK_ITEMCHANGE"],
        "field_candidates": [
            "InstitutionID", "SecurityID", "Symbol", "DeclareDate", "ChangeDate",
            "ChangedItem", "ValueBefore", "ValueAfter", "VALUE", "ReasonID",
            "Comments",
        ],
        "date_field": None,
        "condition_template": "1=1",
        "output": "csmar_listing_status.csv",
    },
    "stock_status": {
        "table_candidates": ["TRD_Dalyr", "STK_Status", "STK_STPT", "STK_Suspend"],
        "field_candidates": [
            "Stkcd", "Trddt", "Trdsta", "Markettype", "Stknme", "Stkcd",
            "ST", "IsST", "Suspended", "Lsttrddt",
        ],
        "date_field": "Trddt",
        "condition_template": "{date_field}>='{start}' and {date_field}<='{end}'",
        "output": "csmar_stock_status.csv",
    },
    "industry": {
        "table_candidates": ["STK_INDUSTRYCLASS", "STK_Industry", "TRD_Co", "STK_LISTEDCOINFO", "LC_Industry"],
        "field_candidates": [
            "InstitutionID", "Symbol", "IndustryClassificationID",
            "IndustryClassificationName", "ImplementDate", "IndustryCode", "IndustryName",
        ],
        "date_field": None,
        "condition_template": "1=1",
        "output": "csmar_industry.csv",
    },
}


class CsmarError(RuntimeError):
    """Raised when CSMAR returns an error response."""


@dataclass
class QueryResult:
    count: int
    rows: int
    output_path: Path


class CsmarClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, verify_ssl: bool = False):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.token: str | None = None
        self.lang = os.environ.get("CSMAR_LANG", "0")
        self.belong = os.environ.get("CSMAR_BELONG", "0")

    @property
    def endpoints(self) -> dict[str, str]:
        return {
            "login": f"{self.base_url}/api/csmar-main/login",
            "list_dbs": f"{self.base_url}/api/csmar-main/python/listDbs",
            "list_tables": f"{self.base_url}/api/csmar-main/python/listTables",
            "list_fields": f"{self.base_url}/api/csmar-main/python/listFields",
            "pack": f"{self.base_url}/api/csmar-main/python/pack",
            "pack_result": f"{self.base_url}/api/csmar-main/python/getPackResult",
            "query": f"{self.base_url}/api/csmar-single/pythonQuery/query",
            "query_count": f"{self.base_url}/api/csmar-single/pythonQuery/getDataCount",
        }

    def headers(self, *, json_content: bool = False) -> dict[str, str]:
        if not self.token:
            raise CsmarError("Not logged in")
        headers = {"Lang": self.lang, "Token": self.token, "belong": self.belong}
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def login(self, username: str, password: str) -> dict:
        payload = {
            "account": username,
            "pwd": password,
            "force": "1",
            "clientType": os.environ.get("CSMAR_CLIENT_TYPE", "5"),
            "version": os.environ.get("CSMAR_API_VERSION", "1.0.2"),
        }
        resp = self.session.post(
            self.endpoints["login"],
            json=payload,
            headers={"Content-Type": "application/json", "lang": self.lang, "belong": self.belong},
            timeout=30,
        )
        data = self._decode(resp)
        if data.get("code") != 0:
            raise CsmarError(f"CSMAR login failed: {data.get('msg') or data}")
        self.token = data["data"]["token"]
        return data

    def list_databases(self) -> list:
        resp = self.session.get(self.endpoints["list_dbs"], headers=self.headers(), timeout=60)
        data = self._decode(resp)
        return self._require_success(data)

    def list_tables(self, database_name: str) -> list:
        resp = self.session.get(
            self.endpoints["list_tables"],
            params={"dbName": database_name},
            headers=self.headers(),
            timeout=60,
        )
        data = self._decode(resp)
        return self._require_success(data)

    def list_fields(self, table_name: str) -> list:
        resp = self.session.get(
            self.endpoints["list_fields"],
            params={"table": table_name},
            headers=self.headers(),
            timeout=60,
        )
        data = self._decode(resp)
        return self._require_success(data)

    def query_count(
        self,
        table_name: str,
        columns: list[str],
        condition: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> int:
        payload = {"columns": columns, "condition": condition, "table": table_name}
        if start_time:
            payload["startTime"] = start_time
        if end_time:
            payload["endTime"] = end_time
        resp = self.session.post(
            self.endpoints["query_count"],
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers(json_content=True),
            timeout=120,
        )
        data = self._decode(resp)
        result = self._require_success(data)
        return int(result or 0)

    def query_page(
        self,
        table_name: str,
        columns: list[str],
        condition: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[dict]:
        payload = {"columns": columns, "condition": condition, "table": table_name}
        if start_time:
            payload["startTime"] = start_time
        if end_time:
            payload["endTime"] = end_time
        resp = self.session.post(
            self.endpoints["query"],
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers(json_content=True),
            timeout=300,
        )
        data = self._decode(resp)
        result = self._require_success(data)
        if isinstance(result, dict) and "previewDatas" in result:
            return result["previewDatas"] or []
        if isinstance(result, list):
            return result
        return []

    def pack_request(
        self,
        table_name: str,
        columns: list[str],
        condition: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> str:
        payload = {"columns": columns, "condition": condition, "table": table_name}
        if start_time:
            payload["startTime"] = start_time
        if end_time:
            payload["endTime"] = end_time
        resp = self.session.post(
            self.endpoints["pack"],
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers(json_content=True),
            timeout=120,
        )
        data = self._decode(resp)
        sign_code = self._require_success(data)
        if not sign_code:
            raise CsmarError(f"CSMAR pack request returned empty sign code: {data}")
        return str(sign_code)

    def get_pack_result(self, sign_code: str) -> dict:
        resp = self.session.get(
            f"{self.endpoints['pack_result']}/{sign_code}",
            headers=self.headers(),
            timeout=120,
        )
        data = self._decode(resp)
        result = self._require_success(data)
        if not isinstance(result, dict):
            raise CsmarError(f"Unexpected pack result for {sign_code}: {result}")
        return result

    def download_pack(
        self,
        table_name: str,
        columns: list[str],
        condition: str,
        output_path: Path,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        poll_seconds: float = 20.0,
        timeout_seconds: int = 1800,
        append: bool = False,
    ) -> QueryResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not append and output_path.exists():
            output_path.unlink()

        sign_code = self.pack_request(table_name, columns, condition, start_time, end_time)
        print(f"[pack] {table_name}: signCode={sign_code}")
        deadline = time.time() + timeout_seconds
        last_percentage = None

        while time.time() < deadline:
            result = self.get_pack_result(sign_code)
            status = str(result.get("status"))
            percentage = result.get("percentage")
            if percentage != last_percentage:
                print(f"[pack] {output_path.name}: status={status}, percentage={percentage}")
                last_percentage = percentage
            if status == "1":
                file_url = result.get("filePath")
                if not file_url:
                    raise CsmarError(f"Pack result has no filePath: {result}")
                rows = self._download_pack_csv(file_url, output_path, append=append)
                return QueryResult(count=rows, rows=rows, output_path=output_path)
            if status == "0":
                raise CsmarError(f"CSMAR pack failed: {result}")
            time.sleep(poll_seconds)

        raise CsmarError(f"Timed out waiting for CSMAR pack result: {sign_code}")

    def _download_pack_csv(self, file_url: str, output_path: Path, *, append: bool) -> int:
        with tempfile.TemporaryDirectory(prefix="csmar_pack_") as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "pack.zip"
            with self.session.get(file_url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with zip_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            unpack_dir = tmp_path / "unpacked"
            unpack_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(unpack_dir)
            csv_files = sorted(unpack_dir.rglob("*.csv"))
            if not csv_files:
                raise CsmarError("Pack ZIP did not contain CSV files")
            rows = copy_csv_files(csv_files, output_path, append=append)
            print(f"[pack] {output_path.name}: copied {rows:,} rows")
            return rows

    def download_query(
        self,
        table_name: str,
        columns: list[str],
        condition: str,
        output_path: Path,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        sleep_seconds: float = 1.0,
        append: bool = False,
    ) -> QueryResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = self.query_count(table_name, columns, condition, start_time, end_time)
        print(f"[count] {table_name}: {count:,} rows")
        rows_written = 0

        mode = "a" if append else "w"
        with output_path.open(mode, newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if not append:
                writer.writeheader()

            offset = 0
            while offset < count:
                page_condition = f"{condition} limit {offset},{page_size}"
                rows = self.query_page(table_name, columns, page_condition, start_time, end_time)
                if not rows:
                    print(f"[warn] empty page at offset {offset:,}; stopping")
                    break
                writer.writerows(rows)
                rows_written += len(rows)
                offset += len(rows)
                print(f"[page] {output_path.name}: {rows_written:,}/{count:,}")
                if len(rows) < page_size:
                    break
                time.sleep(sleep_seconds)

        return QueryResult(count=count, rows=rows_written, output_path=output_path)

    def _decode(self, resp: requests.Response) -> dict:
        try:
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            text = resp.text[:500] if hasattr(resp, "text") else ""
            raise CsmarError(f"HTTP/JSON error: {exc}; body={text}") from exc

    @staticmethod
    def _require_success(data: dict):
        if data.get("code") != 0:
            raise CsmarError(data.get("msg") or str(data))
        return data.get("data")


def login_from_env(args: argparse.Namespace) -> CsmarClient:
    username = os.environ.get("CSMAR_USERNAME")
    password = os.environ.get("CSMAR_PASSWORD")
    if not username or not password:
        raise SystemExit("Set CSMAR_USERNAME and CSMAR_PASSWORD in the environment.")
    client = CsmarClient(base_url=args.base_url, verify_ssl=args.verify_ssl)
    client.login(username, password)
    print("[login] success")
    return client


def normalize_name(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("tableName", "name", "dbName", "title", "label", "id"):
            if key in item and item[key]:
                return str(item[key])
        return json.dumps(item, ensure_ascii=False)
    return str(item)


def field_names(fields: Iterable) -> list[str]:
    names: list[str] = []
    for item in fields:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            for key in ("field", "fieldName", "name", "columnName", "id"):
                if item.get(key):
                    names.append(str(item[key]))
                    break
    return names


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def iter_date_chunks(start: str, end: str, chunk_years: int) -> list[tuple[str, str]]:
    if chunk_years <= 0:
        return [(start, end)]
    start_date = parse_iso_date(start)
    end_date = parse_iso_date(end)
    chunks: list[tuple[str, str]] = []
    current = start_date
    while current <= end_date:
        chunk_end = min(add_years(current, chunk_years) - timedelta(days=1), end_date)
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)
    return chunks


def render_condition(template: str, date_field: str | None, start: str, end: str) -> str:
    return template.format(date_field=date_field or "Trddt", start=start, end=end)


def detect_csv_encoding(path: Path) -> str:
    sample = path.read_bytes()[:8192]
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8"


def copy_csv_files(csv_files: list[Path], output_path: Path, *, append: bool) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and output_path.exists() and output_path.stat().st_size > 0 else "w"
    has_header = mode == "a"
    rows_written = 0

    with output_path.open(mode, newline="", encoding="utf-8-sig") as out_f:
        writer = csv.writer(out_f)
        for csv_path in csv_files:
            encoding = detect_csv_encoding(csv_path)
            with csv_path.open("r", newline="", encoding=encoding) as in_f:
                reader = csv.reader(in_f)
                header = next(reader, None)
                if header and not has_header:
                    writer.writerow(header)
                    has_header = True
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
    return rows_written


def choose_existing_table(client: CsmarClient, candidates: list[str]) -> str | None:
    for table in candidates:
        try:
            client.list_fields(table)
            return table
        except Exception as exc:
            print(f"[discover] table not usable: {table} ({exc})")
    return None


def choose_columns(client: CsmarClient, table_name: str, candidates: list[str]) -> list[str]:
    fields = client.list_fields(table_name)
    names = field_names(fields)
    if not names:
        raise CsmarError(f"No fields returned for table {table_name}")
    name_map = {name.lower(): name for name in names}
    selected = []
    for cand in candidates:
        actual = name_map.get(cand.lower())
        if actual and actual not in selected:
            selected.append(actual)
    if not selected:
        selected = names[: min(20, len(names))]
    return selected


def print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_list_dbs(args: argparse.Namespace) -> None:
    client = login_from_env(args)
    dbs = [normalize_name(x) for x in client.list_databases()]
    print_json(dbs)


def cmd_debug_login(args: argparse.Namespace) -> None:
    username = os.environ.get("CSMAR_USERNAME")
    password = os.environ.get("CSMAR_PASSWORD")
    if not username or not password:
        raise SystemExit("Set CSMAR_USERNAME and CSMAR_PASSWORD in the environment.")
    client = CsmarClient(base_url=args.base_url, verify_ssl=args.verify_ssl)
    data = client.login(username, password)
    safe = json.loads(json.dumps(data, ensure_ascii=False))
    if isinstance(safe.get("data"), dict) and safe["data"].get("token"):
        safe["data"]["token"] = "[REDACTED]"
    print_json(safe)


def cmd_discover(args: argparse.Namespace) -> None:
    client = login_from_env(args)
    keywords = args.keywords or []
    dbs = client.list_databases()
    rows = []
    for db in dbs:
        db_name = normalize_name(db)
        if keywords and not any(k.lower() in db_name.lower() for k in keywords):
            scan_tables = True
        else:
            scan_tables = True
        if not scan_tables:
            continue
        try:
            tables = client.list_tables(db_name)
        except Exception as exc:
            rows.append({"database": db_name, "error": str(exc)})
            continue
        for table in tables:
            table_name = normalize_name(table)
            haystack = f"{db_name} {table_name}".lower()
            if keywords and not any(k.lower() in haystack for k in keywords):
                continue
            rows.append({"database": db_name, "table": table_name})
    print_json(rows)


def cmd_fields(args: argparse.Namespace) -> None:
    client = login_from_env(args)
    fields = client.list_fields(args.table)
    print_json(fields)


def cmd_preview(args: argparse.Namespace) -> None:
    client = login_from_env(args)
    columns = args.columns.split(",") if args.columns else choose_columns(client, args.table, [])
    condition = args.condition or "1=1 limit 0,20"
    rows = client.query_page(args.table, columns, condition, args.start_date, args.end_date)
    print_json(rows[: args.limit])


def cmd_download(args: argparse.Namespace) -> None:
    client = login_from_env(args)
    output_dir = Path(args.output_dir)
    specs = DATASET_SPECS if args.dataset == "all" else {args.dataset: DATASET_SPECS[args.dataset]}
    summary = []

    for name, spec in specs.items():
        print(f"\n=== dataset: {name} ===")
        table_name = args.table or choose_existing_table(client, spec["table_candidates"])
        if not table_name:
            print(f"[skip] no usable table for {name}")
            continue
        columns = args.columns.split(",") if args.columns else choose_columns(
            client, table_name, spec["field_candidates"]
        )
        date_field = args.date_field or spec.get("date_field")
        output_path = output_dir / spec["output"]
        print(f"[table] {table_name}")
        print(f"[columns] {columns}")
        chunks = iter_date_chunks(args.start_date, args.end_date, args.chunk_years) if date_field else [(args.start_date, args.end_date)]
        total_count = 0
        total_rows = 0
        chunk_summaries = []
        for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
            condition_template = args.condition or spec["condition_template"]
            condition = render_condition(condition_template, date_field, chunk_start, chunk_end)
            print(f"[chunk] {chunk_start} to {chunk_end}")
            print(f"[condition] {condition}")
            if args.method == "pack":
                result = client.download_pack(
                    table_name,
                    columns,
                    condition,
                    output_path,
                    start_time=chunk_start if date_field else None,
                    end_time=chunk_end if date_field else None,
                    poll_seconds=args.poll_seconds,
                    timeout_seconds=args.pack_timeout,
                    append=chunk_index > 0,
                )
            else:
                result = client.download_query(
                    table_name,
                    columns,
                    condition,
                    output_path,
                    start_time=chunk_start if date_field else None,
                    end_time=chunk_end if date_field else None,
                    page_size=args.page_size,
                    sleep_seconds=args.sleep_seconds,
                    append=chunk_index > 0,
                )
            total_count += result.count
            total_rows += result.rows
            chunk_summaries.append({
                "start_date": chunk_start,
                "end_date": chunk_end,
                "count": result.count,
                "rows": result.rows,
            })
        summary.append({
            "dataset": name,
            "table": table_name,
            "columns": columns,
            "count": total_count,
            "rows": total_rows,
            "output": str(output_path),
            "chunks": chunk_summaries,
        })

    summary_name = "csmar_download_summary.json" if args.dataset == "all" else f"csmar_download_summary_{args.dataset}.json"
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "datasets": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[summary] {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSMAR data acquisition utility")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--verify-ssl", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-dbs")
    p.set_defaults(func=cmd_list_dbs)

    p = sub.add_parser("debug-login")
    p.set_defaults(func=cmd_debug_login)

    p = sub.add_parser("discover")
    p.add_argument("--keywords", nargs="*", default=[])
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("fields")
    p.add_argument("--table", required=True)
    p.set_defaults(func=cmd_fields)

    p = sub.add_parser("preview")
    p.add_argument("--table", required=True)
    p.add_argument("--columns")
    p.add_argument("--condition")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("download")
    p.add_argument("--dataset", choices=["all"] + sorted(DATASET_SPECS), default="all")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--method", choices=["pack", "query"], default="pack")
    p.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--chunk-years", type=int, default=3)
    p.add_argument("--poll-seconds", type=float, default=20.0)
    p.add_argument("--pack-timeout", type=int, default=1800)
    p.add_argument("--table")
    p.add_argument("--columns")
    p.add_argument("--date-field")
    p.add_argument("--condition")
    p.set_defaults(func=cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except CsmarError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
