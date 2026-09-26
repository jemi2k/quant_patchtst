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
    """Download one monthly zip; return local path, or None if the month is absent (404).

    Skips the download if a complete zip already exists locally (resume).
    """
    url = config.monthly_url(symbol, ym)
    dest = config.zip_path(symbol, ym)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        return dest  # already downloaded -> reuse

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

class _OrderingViolation(Exception):
    """Trades are not in agg_trade_id order; the in-memory sort fallback repairs this."""


def _csv_options(has_header):
    """Shared pyarrow read/parse/convert options for streaming and sorted paths."""
    if has_header:
        read_opts = pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True)
    else:
        # Header-less dump (older futures months): supply names, first row is data.
        read_opts = pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True,
                                   column_names=config.AGGTRADES_COLUMNS)
    parse_opts = pc.ParseOptions(delimiter=",")
    convert_opts = pc.ConvertOptions(column_types={
        name: _PA_TYPES[config.AGGTRADES_DTYPES[name]]
        for name in config.AGGTRADES_COLUMNS
    })
    return read_opts, parse_opts, convert_opts


def _csv_to_parquet_sorted(csv_path, parquet_path, has_header=True):
    """Fallback: load the whole CSV, sort by (transact_time, agg_trade_id),
    drop exact duplicates, and write. Repairs out-of-order months."""
    read_opts, parse_opts, convert_opts = _csv_options(has_header)
    table = pc.read_csv(csv_path, read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts)

    actual = set(table.schema.names)
    expected = set(config.AGGTRADES_COLUMNS)
    if actual != expected:
        raise ValueError(f"unexpected CSV columns {sorted(actual)} (expected {sorted(expected)})")

    n_before = table.num_rows
    table = table.sort_by([("transact_time", "ascending"), ("agg_trade_id", "ascending")])

    # Drop adjacent exact duplicates (same agg_trade_id AND transact_time).
    ids = table.column("agg_trade_id").to_numpy(zero_copy_only=False)
    times = table.column("transact_time").to_numpy(zero_copy_only=False)
    keep = np.ones(table.num_rows, dtype=bool)
    if table.num_rows > 1:
        same = (ids[1:] == ids[:-1]) & (times[1:] == times[:-1])
        keep[1:] = ~same
    if not keep.all():
        table = table.filter(pa.array(keep))

    pq.write_table(table, parquet_path, compression=config.PARQUET_COMPRESSION)

    dropped = n_before - table.num_rows
    if dropped:
        logger.warning("sorted fallback dropped %d exact-duplicate rows", dropped)
    return table.num_rows


def _csv_to_parquet(csv_path, parquet_path, has_header=True):
    """Stream a CSV into a parquet file.

    Deduplicates adjacent duplicate agg_trade_id rows (a known Binance export
    glitch). Raises _OrderingViolation on decreasing IDs or non-exact duplicates,
    which the caller repairs via the in-memory sort fallback.
    """
    read_opts, parse_opts, convert_opts = _csv_options(has_header)
    reader = pc.open_csv(csv_path, read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts)

    actual = set(reader.schema.names)
    expected = set(config.AGGTRADES_COLUMNS)
    if actual != expected:
        raise ValueError(f"unexpected CSV columns {sorted(actual)} (expected {sorted(expected)})")

    writer = pq.ParquetWriter(parquet_path, reader.schema, compression=config.PARQUET_COMPRESSION)

    row_count = 0
    dropped = 0
    last_id = None
    last_vals = None  # (transact_time, price, quantity) of the last written row

    try:
        for batch in reader:
            n = batch.num_rows
            if n:
                ids = batch.column("agg_trade_id").to_numpy(zero_copy_only=False)

                # Strict decreases are true reordering -> sort fallback.
                if n > 1 and np.any(ids[1:] < ids[:-1]):
                    raise _OrderingViolation("strictly-decreasing agg_trade_id rows")
                if last_id is not None and int(ids[0]) < last_id:
                    raise _OrderingViolation("strictly-decreasing agg_trade_id rows at batch boundary")

                # Mark adjacent duplicate agg_trade_id rows for removal (keep first).
                drop = np.zeros(n, dtype=bool)
                if n > 1:
                    drop[1:] = ids[1:] == ids[:-1]
                if last_id is not None and int(ids[0]) == last_id:
                    drop[0] = True

                if drop.any():
                    # A duplicate must be an exact copy (same time/price/qty), never
                    # two distinct trades that happen to share an id.
                    for c in ("transact_time", "price", "quantity"):
                        cur = batch.column(c).to_numpy(zero_copy_only=False)
                        prev = np.empty(n, dtype=cur.dtype)
                        prev[0] = last_vals[c] if last_vals is not None else cur[0]
                        prev[1:] = cur[:-1]
                        if (cur[drop] != prev[drop]).any():
                            raise _OrderingViolation(f"duplicate agg_trade_id rows differ on {c}")
                    batch = batch.filter(pa.array(~drop))
                    dropped += int(drop.sum())

                # Track the last written row for the next batch's boundary check.
                if batch.num_rows > 0:
                    last = batch.num_rows - 1
                    last_id = batch.column("agg_trade_id")[last].as_py()
                    last_vals = {
                        "transact_time": batch.column("transact_time")[last].as_py(),
                        "price": batch.column("price")[last].as_py(),
                        "quantity": batch.column("quantity")[last].as_py(),
                    }

            writer.write_batch(batch)
            row_count += batch.num_rows
    finally:
        writer.close()

    if dropped:
        logger.warning("dropped %d duplicate agg_trade_id rows", dropped)

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
                # Binance futures aggTrades are mixed-format: older months are
                # header-less, newer ones carry a header. Peek the first line.
                with zf.open(members[0]) as fh:
                    first_line = fh.readline().decode("utf-8", errors="replace").strip()
                has_header = first_line.lower().startswith("agg_trade_id")
                csv_path = zf.extract(members[0], tmpdir)
            try:
                row_count = _csv_to_parquet(csv_path, tmp_parquet, has_header=has_header)
            except _OrderingViolation:
                logger.warning("out-of-order trades in %s -> in-memory sort fallback", os.path.basename(zip_path))
                row_count = _csv_to_parquet_sorted(csv_path, tmp_parquet, has_header=has_header)
        os.replace(tmp_parquet, parquet_path)  # atomic -> resume-safe
        return row_count
    except Exception:
        if os.path.exists(tmp_parquet):
            os.remove(tmp_parquet)
        raise


# --- Per-item orchestration -----------------------------------------------------------

async def _process(session, symbol, ym, semaphore):
    key = f"{symbol}/{ym}"
    try:
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
    except Exception as exc:
        logger.error("%s FAILED: %s", key, exc, exc_info=True)
        return {"key": key, "status": "failed"}


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
