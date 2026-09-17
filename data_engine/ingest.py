"""
data_engine/ingest.py
=====================
Async bulk download + extraction + parquet conversion for monthly Binance
USDT-M futures aggTrades.

Per (symbol, year-month):
  1. download the monthly .zip (async, bounded concurrency, retries)
  2. extract the single CSV and stream-convert it to a zstd parquet
  3. verify agg_trade_id is strictly monotonic (the DIB ordering key)
  4. retain/delete the raw zip per config.KEEP_RAW_ZIPS
  5. record row count / size in data/manifest.json

Resume is safe: downloads and parquets are written to a ".part" file and
atomically renamed, so a file present under its final name is complete.

Run:
    python -m data_engine.ingest --dry-run                          # list, no download
    python -m data_engine.ingest --symbols SUIUSDT --start 2023-05 --end 2023-06
    python -m data_engine.ingest                                   # full pipeline
"""

import argparse
import asyncio
import json
import logging
import os
import tempfile
import zipfile
from collections import Counter
from datetime import datetime

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

from data_engine import config

logger = logging.getLogger("data_engine.ingest")

_PA_TYPES = {
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
}


# --- Month / work-item enumeration -------------------------------------------------

def _iter_months(start_ym, end_ym):
    y, m = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _last_complete_month():
    now = datetime.now()
    y, m = now.year, now.month - 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}-{m:02d}"


def build_work_items(symbols=None, start=None, end=None):
    symbols = symbols or config.SYMBOLS
    end = end or _last_complete_month()
    items = []
    for sym in symbols:
        start_ym = start or config.INCEPTION[sym]
        items.extend((sym, ym) for ym in _iter_months(start_ym, end))
    return items


# --- Manifest -----------------------------------------------------------------------

def load_manifest():
    if os.path.exists(config.MANIFEST_PATH):
        with open(config.MANIFEST_PATH) as fh:
            return json.load(fh)
    return {}


def save_manifest(manifest):
    os.makedirs(os.path.dirname(config.MANIFEST_PATH), exist_ok=True)
    tmp = config.MANIFEST_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, config.MANIFEST_PATH)


# --- Download -----------------------------------------------------------------------

async def _download_zip(session, symbol, ym, semaphore):
    """Download one monthly zip; return local path, or None if the month is absent (404)."""
    url = config.monthly_url(symbol, ym)
    dest = config.zip_path(symbol, ym)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    timeout = aiohttp.ClientTimeout(sock_read=config.REQUEST_TIMEOUT_SECONDS, total=None)

    async with semaphore:
        last_err = None
        for attempt in range(config.MAX_RETRIES):
            try:
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status == 404:
                        return None  # predates listing -> skip silently
                    resp.raise_for_status()

                    expected = int(resp.headers.get("Content-Length", 0))
                    tmp = dest + ".part"
                    written = 0
                    with open(tmp, "wb") as fh:
                        async for chunk in resp.content.iter_chunked(1 << 20):
                            fh.write(chunk)
                            written += len(chunk)
                    if expected and written != expected:
                        raise aiohttp.ClientError(f"incomplete download: {written} != {expected} bytes")
                    os.replace(tmp, dest)
                    return dest
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_err = exc
                if attempt < config.MAX_RETRIES - 1:
                    await asyncio.sleep(config.RETRY_BACKOFF_SECONDS[attempt])
        raise last_err


# --- Extraction + conversion ---------------------------------------------------------

def _csv_to_parquet(csv_path, parquet_path):
    """Stream a CSV into a parquet file, verifying strict agg_trade_id monotonicity."""
    read_opts = pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True)
    parse_opts = pc.ParseOptions(delimiter=",")
    convert_opts = pc.ConvertOptions(column_types={
        name: _PA_TYPES[config.AGGTRADES_DTYPES[name]]
        for name in config.AGGTRADES_COLUMNS
    })

    reader = pc.open_csv(csv_path, read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts)

    actual = set(reader.schema.names)
    expected = set(config.AGGTRADES_COLUMNS)
    if actual != expected:
        raise ValueError(f"unexpected CSV columns {sorted(actual)} (expected {sorted(expected)})")

    writer = pq.ParquetWriter(parquet_path, reader.schema, compression=config.PARQUET_COMPRESSION)

    row_count = 0
    violations = 0
    last_id = None
    for batch in reader:
        n = batch.num_rows
        if n:
            ids = batch.column("agg_trade_id").to_numpy(zero_copy_only=False)
            if np.any(ids[1:] <= ids[:-1]):
                violations += int(np.sum(ids[1:] <= ids[:-1]))
            if last_id is not None and int(ids[0]) <= last_id:
                violations += 1
            last_id = int(ids[-1])
        writer.write_batch(batch)
        row_count += n
    writer.close()

    if violations:
        raise ValueError(f"{violations} non-monotonic agg_trade_id rows")

    return row_count


def _convert_zip_to_parquet(zip_path, parquet_path):
    """Extract the single CSV from a zip and stream-convert it to parquet."""
    tmp_parquet = parquet_path + ".part"
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path) as zf:
                members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if len(members) != 1:
                    raise ValueError(f"expected exactly 1 CSV in {os.path.basename(zip_path)}, got {len(members)}")
                csv_path = zf.extract(members[0], tmpdir)
            row_count = _csv_to_parquet(csv_path, tmp_parquet)
        os.replace(tmp_parquet, parquet_path)  # atomic -> resume-safe
        return row_count
    except Exception:
        if os.path.exists(tmp_parquet):
            os.remove(tmp_parquet)
        raise


# --- Per-item orchestration -----------------------------------------------------------

async def _process(session, symbol, ym, semaphore):
    key = f"{symbol}/{ym}"
    parquet = config.parquet_path(symbol, ym)

    if os.path.exists(parquet):
        return {"key": key, "status": "skip"}

    zip_dest = await _download_zip(session, symbol, ym, semaphore)
    if zip_dest is None:
        return {"key": key, "status": "not_found"}

    rows = await asyncio.to_thread(_convert_zip_to_parquet, zip_dest, parquet)

    if not config.KEEP_RAW_ZIPS:
        os.remove(zip_dest)

    size = os.path.getsize(parquet)
    logger.info("%s: %d rows -> %.1f MB", key, rows, size / 1e6)
    return {"key": key, "status": "ok", "rows": rows, "size": size}


# --- Main -----------------------------------------------------------------------------

async def _run(items):
    manifest = load_manifest()
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_DOWNLOADS)

    logger.info("%d (symbol, month) items across %d symbols", len(items), len({s for s, _ in items}))

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_process(session, sym, ym, semaphore) for sym, ym in items],
            return_exceptions=True,
        )

    counts = Counter()
    total_rows = 0
    for r in results:
        if isinstance(r, Exception):
            counts["failed"] += 1
            logger.error("item failed: %s", r)
        else:
            counts[r["status"]] += 1
            if r["status"] == "ok":
                total_rows += r["rows"]
                manifest[r["key"]] = {"status": "ok", "rows": r["rows"], "size": r["size"]}

    save_manifest(manifest)
    logger.info("DONE: ok=%d skip=%d not_found=%d failed=%d  total_rows=%d",
                counts["ok"], counts["skip"], counts["not_found"], counts["failed"], total_rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Bulk aggTrades ingestion from data.binance.vision")
    parser.add_argument("--symbols", nargs="*", help="subset of symbols (default: all)")
    parser.add_argument("--start", help="override start YYYY-MM")
    parser.add_argument("--end", help="override end YYYY-MM (default: last complete month)")
    parser.add_argument("--dry-run", action="store_true", help="list work items without downloading")
    args = parser.parse_args()

    items = build_work_items(symbols=args.symbols, start=args.start, end=args.end)

    if args.dry_run:
        per = Counter(sym for sym, _ in items)
        for sym in sorted(per):
            months = [ym for s, ym in items if s == sym]
            print(f"{sym}: {len(months)} months ({months[0]} .. {months[-1]})")
        print(f"total: {len(items)} (symbol, month) items")
        return

    asyncio.run(_run(items))


if __name__ == "__main__":
    main()
