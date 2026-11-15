"""
data_engine/ingest.py  (v2)
===========================
Async bulk download + extraction + parquet conversion for monthly Binance
USDT-M futures aggTrades, with strict source verification.

Per (symbol, year-month):
  1. fetch the OFFICIAL sha256 checksum (.zip.CHECKSUM) for the month
  2. reuse the local zip ONLY if its recorded sha256 matches the official one
     (otherwise preserve the stale zip under its original name + suffix and
     re-download), download with retries, verify sha256 before accepting
  3. reuse the local parquet ONLY if it exists AND its manifest entry matches
     the current official zip checksum (no silent skips of stale data)
  4. stream-convert the CSV to zstd parquet with full-column validation:
       - agg_trade_id strictly increasing is the invariant
       - adjacent duplicate ids are compared on ALL columns:
           * exact duplicates (all columns equal) are dropped and counted
           * conflicting duplicates with different underlying trade ids are
             KEPT (distinct executions sharing an id - Binance glitch),
             recorded in the manifest
           * any other conflict is recorded as an anomaly, never silently
             dropped (default: keep first, prefer daily-archive record when a
             daily fill is configured)
       - decreasing ids fall back to a deterministic sorted rebuild
  5. optionally merge DAILY archive files for dates the monthly rollup is
     missing (fills); monthly rows win on id collision, overlap is counted
  6. record everything in data/manifest.json: status, rows, sizes, zip sha256,
     csv rows, dropped/kept counts, repairs, provenance, content hash
  7. exit non-zero if any requested month failed or is absent (404)

Resume is safe: downloads and parquets are written to ".part" and atomically
renamed. Existing parquets are never silently trusted - the manifest +
official checksum gate every skip decision.

Run:
    python -m data_engine.ingest --dry-run
    python -m data_engine.ingest --symbols SUIUSDT --start 2023-05 --end 2023-06
    python -m data_engine.ingest --force            # rebuild everything
    python -m data_engine.ingest --reverify-zips    # re-hash all zips first
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import zipfile
from collections import Counter

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

ALL_COLS = config.AGGTRADES_COLUMNS  # ["agg_trade_id","price","quantity","first_trade_id","last_trade_id","transact_time","is_buyer_maker"]


# --- Month / work-item enumeration -------------------------------------------------

def _iter_months(start_ym, end_ym):
    y, m = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m, y = 1, y + 1


def build_work_items(symbols=None, start=None, end=None):
    """Enumerate (symbol, ym). end defaults to config.DATA_END (frozen boundary,
    NOT the wall clock), start defaults to the verified per-symbol INCEPTION."""
    symbols = symbols or config.SYMBOLS
    end = end or config.DATA_END
    if end > config.DATA_END:
        raise ValueError(f"requested end {end} exceeds frozen boundary {config.DATA_END}")
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


# --- Thread helper (Python 3.7/3.8 compatible) --------------------------------------

async def _run_in_thread(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


# --- Checksums ----------------------------------------------------------------------

async def _fetch_checksum(session, symbol, ym):
    """Return the OFFICIAL sha256 for the month, downloading the .CHECKSUM sidecar
    to config.CHECKSUM_DIR if absent. Returns None if the month does not exist (404)."""
    out = config.checksum_path(symbol, ym)
    if os.path.exists(out):
        with open(out) as fh:
            parts = fh.read().strip().split()
            if parts:
                return parts[0].lower()
    url = config.checksum_url(symbol, ym)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(4):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                text = await resp.text()
                with open(out, "w") as fh:
                    fh.write(text)
                return text.strip().split()[0].lower()
        except aiohttp.ClientError:
            if attempt == 3:
                raise
            await asyncio.sleep(1 + attempt)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_stale_name(zip_path):
    """Where a replaced zip is preserved (never deleted)."""
    return zip_path.replace(".zip", ".zip.stale")


# --- Download -----------------------------------------------------------------------

async def _download_zip(session, symbol, ym, want_sha, semaphore):
    """Ensure the local zip matches the official checksum. Returns the zip path,
    or None if the month does not exist (404)."""
    url = config.monthly_url(symbol, ym)
    dest = config.zip_path(symbol, ym)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        if sha256_file(dest) == want_sha:
            return dest
        # Stale: preserve the old zip (source record), then re-download.
        stale = _archive_stale_name(dest)
        if not os.path.exists(stale):
            os.replace(dest, stale)
            logger.warning("%s: local zip checksum mismatch -> preserved as %s, re-downloading",
                           symbol + "/" + ym, os.path.basename(stale))
        else:
            os.remove(dest)  # a previous stale copy already exists; drop this one

    timeout = aiohttp.ClientTimeout(sock_read=config.REQUEST_TIMEOUT_SECONDS, total=None)
    async with semaphore:
        last_err = None
        for attempt in range(config.MAX_RETRIES):
            try:
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status == 404:
                        return None
                    resp.raise_for_status()
                    expected = int(resp.headers.get("Content-Length", 0))
                    tmp = dest + ".part"
                    h = hashlib.sha256()
                    written = 0
                    with open(tmp, "wb") as fh:
                        async for chunk in resp.content.iter_chunked(1 << 20):
                            fh.write(chunk)
                            h.update(chunk)
                            written += len(chunk)
                    if expected and written != expected:
                        raise aiohttp.ClientError(f"incomplete download: {written} != {expected} bytes")
                    if h.hexdigest() != want_sha:
                        raise aiohttp.ClientError(
                            f"checksum mismatch after download (got {h.hexdigest()[:12]}..., want {want_sha[:12]}...)")
                    os.replace(tmp, dest)
                    return dest
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_err = exc
                if os.path.exists(dest + ".part"):
                    os.remove(dest + ".part")
                if attempt < config.MAX_RETRIES - 1:
                    await asyncio.sleep(config.RETRY_BACKOFF_SECONDS[attempt % len(config.RETRY_BACKOFF_SECONDS)])
        raise last_err


# --- Conversion ---------------------------------------------------------------------

class _OrderingViolation(Exception):
    """agg_trade_ids decrease (true reordering) - repaired by the sorted fallback."""


def _csv_options(has_header):
    if has_header:
        read_opts = pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True)
    else:
        read_opts = pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True,
                                   column_names=ALL_COLS)
    parse_opts = pc.ParseOptions(delimiter=",")
    convert_opts = pc.ConvertOptions(column_types={
        name: _PA_TYPES[config.AGGTRADES_DTYPES[name]] for name in ALL_COLS
    })
    return read_opts, parse_opts, convert_opts


def _check_columns(names):
    actual = set(names)
    expected = set(ALL_COLS)
    if actual != expected:
        raise ValueError(f"unexpected CSV columns {sorted(actual)} (expected {sorted(expected)})")


def _rows_equal(table, i, j):
    """All seven columns equal between rows i and j of the table."""
    for c in ALL_COLS:
        if table.column(c)[i].as_py() != table.column(c)[j].as_py():
            return False
    return True


def _apply_duplicate_rule(table, keep_mask, stats):
    """Mark exact-duplicate rows for removal. Conflicting duplicates (same id,
    different values) are KEPT when their underlying trade ids differ (distinct
    executions - Binance glitch); every kept conflict is recorded. Never silently
    deletes conflicting records."""
    n = table.num_rows
    ids = table.column("agg_trade_id").to_numpy(zero_copy_only=False)
    if n < 2:
        return
    dup_pos = np.where(ids[1:] == ids[:-1])[0]
    for i in dup_pos.tolist():
        i = int(i)
        if _rows_equal(table, i + 1, i):
            keep_mask[i + 1] = False
            stats["dropped_exact_dups"] += 1
        else:
            ft0 = table.column("first_trade_id")[i].as_py()
            lt0 = table.column("last_trade_id")[i].as_py()
            ft1 = table.column("first_trade_id")[i + 1].as_py()
            lt1 = table.column("last_trade_id")[i + 1].as_py()
            if (ft0, lt0) != (ft1, lt1):
                # distinct underlying trades that share an id: keep both
                stats["kept_conflicts"].append({
                    "agg_trade_id": int(ids[i + 1]),
                    "underlying": [[int(ft0), int(lt0)], [int(ft1), int(lt1)]],
                })
            else:
                # same underlying trade ids but conflicting fields: anomaly.
                # Keep both rows (preferring nothing silently); flag loudly.
                stats["kept_conflicts"].append({
                    "agg_trade_id": int(ids[i + 1]),
                    "underlying": [[int(ft0), int(lt0)], [int(ft1), int(lt1)]],
                    "same_underlying_conflict": True,
                })
                logger.warning("agg_trade_id %d: conflicting rows share underlying trade ids - kept both, flagged",
                               int(ids[i + 1]))


def _convert_table_to_parquet(table, parquet_path, stats):
    """Sort deterministically, apply the duplicate rule, write, verify."""
    table = table.sort_by([("agg_trade_id", "ascending"), ("transact_time", "ascending")])
    ids = table.column("agg_trade_id").to_numpy(zero_copy_only=False)
    if len(ids) > 1 and np.any(ids[1:] < ids[:-1]):
        raise RuntimeError("internal: sort did not produce increasing ids")

    keep = np.ones(table.num_rows, dtype=bool)
    _apply_duplicate_rule(table, keep, stats)
    if not keep.all():
        table = table.filter(pa.array(keep))

    # no nulls anywhere
    for c in ALL_COLS:
        if table.column(c).null_count != 0:
            raise ValueError(f"{table.column(c).null_count} nulls in column {c}")

    pq.write_table(table, parquet_path, compression=config.PARQUET_COMPRESSION)
    return table.num_rows


def _convert_csv_to_parquet(csv_path, parquet_path, has_header, stats):
    """Streaming fast path. Falls back to a deterministic sorted rebuild on
    ordering violations."""
    read_opts, parse_opts, convert_opts = _csv_options(has_header)
    reader = pc.open_csv(csv_path, read_options=read_opts, parse_options=parse_opts,
                         convert_options=convert_opts)
    _check_columns(reader.schema.names)

    # parquet_path is already a .part staging path (caller cleans up on failure)
    tmp = parquet_path
    writer = pq.ParquetWriter(tmp, reader.schema, compression=config.PARQUET_COMPRESSION)
    row_count = 0
    csv_rows = 0
    prev = None  # full previous row values (all columns)
    try:
        for batch in reader:
            n = batch.num_rows
            csv_rows += n
            if n == 0:
                continue
            ids = batch.column("agg_trade_id").to_numpy(zero_copy_only=False)
            if n > 1 and np.any(ids[1:] < ids[:-1]):
                raise _OrderingViolation("strictly-decreasing agg_trade_id rows")
            if prev is not None and int(ids[0]) < prev["agg_trade_id"]:
                raise _OrderingViolation("strictly-decreasing agg_trade_id rows at batch boundary")

            drop = np.zeros(n, dtype=bool)
            if n > 1:
                drop[1:] = ids[1:] == ids[:-1]
            if prev is not None and int(ids[0]) == prev["agg_trade_id"]:
                drop[0] = True

            if drop.any():
                idxs = np.where(drop)[0]
                for i in idxs:
                    if i == 0:
                        prev_row = prev
                        cur_row = {c: batch.column(c)[0].as_py() for c in ALL_COLS}
                    else:
                        prev_row = {c: batch.column(c)[i - 1].as_py() for c in ALL_COLS}
                        cur_row = {c: batch.column(c)[i].as_py() for c in ALL_COLS}
                    if prev_row == cur_row:
                        stats["dropped_exact_dups"] += 1
                    else:
                        ft0, lt0 = prev_row["first_trade_id"], prev_row["last_trade_id"]
                        ft1, lt1 = cur_row["first_trade_id"], cur_row["last_trade_id"]
                        stats["kept_conflicts"].append({
                            "agg_trade_id": int(cur_row["agg_trade_id"]),
                            "underlying": [[int(ft0), int(lt0)], [int(ft1), int(lt1)]],
                            **({"same_underlying_conflict": True} if (ft0, lt0) == (ft1, lt1) else {}),
                        })
                        drop[i] = False  # conflicting duplicate: KEEP
                        logger.warning("agg_trade_id %d: conflicting duplicate kept (%s)",
                                       int(cur_row["agg_trade_id"]),
                                       "same underlying ids - anomaly" if (ft0, lt0) == (ft1, lt1)
                                       else "distinct underlying trades")
                batch = batch.filter(pa.array(~drop))

            if batch.num_rows > 0:
                last = batch.num_rows - 1
                prev = {c: batch.column(c)[last].as_py() for c in ALL_COLS}
            writer.write_batch(batch)
            row_count += batch.num_rows
    finally:
        writer.close()

    if stats["dropped_exact_dups"]:
        logger.warning("dropped %d exact-duplicate rows", stats["dropped_exact_dups"])
    stats["csv_rows"] = csv_rows
    return row_count


def _convert_csv_to_parquet_sorted(csv_path, parquet_path, has_header, stats):
    """Deterministic rebuild for months whose archive rows are out of order."""
    read_opts, parse_opts, convert_opts = _csv_options(has_header)
    table = pc.read_csv(csv_path, read_options=read_opts, parse_options=parse_opts,
                        convert_options=convert_opts)
    _check_columns(table.schema.names)
    stats["csv_rows"] = table.num_rows
    stats["sorted_fallback"] = True
    return _convert_table_to_parquet(table, parquet_path, stats)


def _convert_zip_to_parquet(zip_path, parquet_path, stats, symbol, fills=None):
    """Convert a monthly zip CSV to parquet; optionally merge daily-archive fills
    for dates the monthly rollup is missing. Monthly rows win on id collision."""
    tmp_parquet = parquet_path + ".part"
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path) as zf:
                members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if len(members) != 1:
                    raise ValueError(f"expected exactly 1 CSV in {os.path.basename(zip_path)}, got {len(members)}")
                with zf.open(members[0]) as fh:
                    first_line = fh.readline().decode("utf-8", errors="replace").strip()
                has_header = first_line.lower().startswith("agg_trade_id")
                csv_path = zf.extract(members[0], tmpdir)

            if not fills:
                try:
                    row_count = _convert_csv_to_parquet(csv_path, tmp_parquet, has_header, stats)
                except _OrderingViolation:
                    logger.warning("out-of-order trades in %s -> deterministic sorted rebuild",
                                   os.path.basename(zip_path))
                    row_count = _convert_csv_to_parquet_sorted(csv_path, tmp_parquet, has_header, stats)
                os.replace(tmp_parquet, parquet_path)
                return row_count

            # Merge path: monthly + daily fills, dedupe by agg_trade_id (monthly wins)
            read_opts, parse_opts, convert_opts = _csv_options(has_header)
            monthly = pc.read_csv(csv_path, read_options=read_opts, parse_options=parse_opts,
                                  convert_options=convert_opts)
            _check_columns(monthly.schema.names)
            stats["monthly_rows"] = monthly.num_rows
            parts = [monthly]
            stats["fills"] = []
            for day in sorted(fills):
                zpath = config.daily_zip_path(symbol, day)
                with zipfile.ZipFile(zpath) as zf:
                    mems = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if len(mems) != 1:
                        raise ValueError(f"daily {day}: expected 1 CSV, got {len(mems)}")
                    with zf.open(mems[0]) as fh:
                        first_line = fh.readline().decode("utf-8", errors="replace").strip()
                    d_has_header = first_line.lower().startswith("agg_trade_id")
                    d_read, d_parse, d_conv = _csv_options(d_has_header)
                    d_csv = zf.extract(mems[0], tmpdir)
                daily = pc.read_csv(d_csv, read_options=d_read, parse_options=d_parse,
                                    convert_options=d_conv)
                _check_columns(daily.schema.names)
                if daily.num_rows == 0:
                    raise ValueError(f"daily {day}: 0 rows - nothing to fill")
                stats["fills"].append({"date": day, "rows": daily.num_rows})
                parts.append(daily)

            merged = pa.concat_tables(parts)
            monthly_ids = set(monthly.column("agg_trade_id").to_pylist())
            offset = monthly.num_rows
            dids = merged.column("agg_trade_id").to_numpy(zero_copy_only=False)[offset:]
            daily_keep = np.ones(len(dids), dtype=bool)
            dropped_overlap = 0
            for i, d in enumerate(dids):
                if int(d) in monthly_ids:
                    daily_keep[i] = False
                    dropped_overlap += 1
            merged = pa.concat_tables([monthly, merged.slice(offset, merged.num_rows - offset).filter(pa.array(daily_keep))])
            stats["fill_overlap_dropped"] = dropped_overlap

            row_count = _convert_table_to_parquet(merged, tmp_parquet, stats)
            os.replace(tmp_parquet, parquet_path)
            return row_count
    except Exception:
        if os.path.exists(tmp_parquet):
            os.remove(tmp_parquet)
        raise


# --- Content hash (deterministic over decoded records) -------------------------------

def parquet_content_hash(parquet_path):
    """Deterministic digest over the decoded column bytes in schema order -
    independent of compression/encoding, reproducible across rebuilds."""
    pf = pq.ParquetFile(parquet_path)
    h = hashlib.sha256()
    names = pf.schema_arrow.names
    for batch in pf.iter_batches(batch_size=1 << 20):
        h.update(len(batch).to_bytes(8, "little"))
        for c in names:
            arr = batch.column(c)
            bufs = arr.buffers() or []
            for b in bufs:
                if b is not None:
                    h.update(bytes(b))
    return h.hexdigest()


# --- Per-item orchestration -----------------------------------------------------------

async def _process(session, symbol, ym, semaphore, manifest, force, fills_map):
    key = f"{symbol}/{ym}"
    stats = {"dropped_exact_dups": 0, "kept_conflicts": [], "sorted_fallback": False}
    try:
        want_sha = await _fetch_checksum(session, symbol, ym)
        if want_sha is None:
            manifest[key] = {"status": "not_found"}
            return {"key": key, "status": "not_found"}

        parquet = config.staging_parquet_path(symbol, ym)
        entry = manifest.get(key, {})

        # Skip ONLY when the existing parquet is verified against the current source.
        if not force and os.path.exists(parquet) and entry.get("status") == "ok" \
                and entry.get("zip_sha256") == want_sha:
            try:
                if pq.ParquetFile(parquet).metadata.num_rows == entry.get("rows"):
                    return {"key": key, "status": "skip_verified"}
            except Exception:
                pass  # unreadable -> rebuild

        zip_dest = await _download_zip(session, symbol, ym, want_sha, semaphore)
        if zip_dest is None:
            manifest[key] = {"status": "not_found"}
            return {"key": key, "status": "not_found"}

        fills = fills_map.get((symbol, ym))
        rows = await _run_in_thread(_convert_zip_to_parquet, zip_dest, parquet, stats, symbol, fills)

        content_hash = await _run_in_thread(parquet_content_hash, parquet)
        size = os.path.getsize(parquet)

        manifest[key] = {
            "status": "ok",
            "rows": rows,
            "size": size,
            "zip_sha256": want_sha,
            "parquet_content_hash": content_hash,
            **{k: v for k, v in stats.items() if v not in (0, [], False)},
        }
        logger.info("%s: %d rows -> %.1f MB%s", key, rows, size / 1e6,
                    f" (fills: {[f['date'] for f in stats.get('fills', [])]})" if stats.get("fills") else "")
        return {"key": key, "status": "ok", "rows": rows, "size": size}
    except Exception as exc:
        logger.error("%s FAILED: %s", key, exc, exc_info=True)
        manifest[key] = {"status": "failed", "error": str(exc)[:500]}
        return {"key": key, "status": "failed"}


# --- Main -----------------------------------------------------------------------------

async def _run(items, force=False, reverify=False, fills_map=None):
    fills_map = fills_map or {}
    manifest = load_manifest()
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_DOWNLOADS)

    if reverify:
        # re-hash every existing zip and record it (repeatable trust check)
        for sym in config.SYMBOLS:
            zdir = os.path.join(config.RAW_DIR, sym)
            if not os.path.isdir(zdir):
                continue
            for fn in sorted(os.listdir(zdir)):
                if fn.endswith(".zip"):
                    ym = fn.split("-aggTrades-")[1][:7]
                    key = f"{sym}/{ym}"
                    if manifest.get(key, {}).get("status") == "ok":
                        got = sha256_file(os.path.join(zdir, fn))
                        manifest[key]["zip_sha256"] = got
        save_manifest(manifest)
        logger.info("re-verified zip checksums and recorded them in the manifest")

    logger.info("%d (symbol, month) items across %d symbols", len(items), len({s for s, _ in items}))

    async def _periodic_save():
        while True:
            await asyncio.sleep(60)
            save_manifest(manifest)

    saver = asyncio.ensure_future(_periodic_save())

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_process(session, sym, ym, semaphore, manifest, force, fills_map) for sym, ym in items],
            return_exceptions=True,
        )

    saver.cancel()

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

    save_manifest(manifest)
    logger.info("DONE: ok=%d skip_verified=%d not_found=%d failed=%d  total_rows=%d",
                counts["ok"], counts["skip_verified"], counts["not_found"], counts["failed"], total_rows)

    if counts["failed"] or counts["not_found"]:
        logger.error("ingest incomplete: %d failed, %d not_found", counts["failed"], counts["not_found"])
        return 1
    return 0


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Verified aggTrades ingestion from data.binance.vision")
    parser.add_argument("--symbols", nargs="*", help="subset of symbols (default: all)")
    parser.add_argument("--start", help="override start YYYY-MM")
    parser.add_argument("--end", help="override end YYYY-MM (default: config.DATA_END)")
    parser.add_argument("--dry-run", action="store_true", help="list work items without downloading")
    parser.add_argument("--force", action="store_true", help="rebuild parquets even if verified")
    parser.add_argument("--reverify-zips", action="store_true",
                        help="re-hash all local zips and refresh manifest sha256 records")
    args = parser.parse_args()

    items = build_work_items(symbols=args.symbols, start=args.start, end=args.end)

    if args.dry_run:
        per = Counter(sym for sym, _ in items)
        for sym in sorted(per):
            months = [ym for s, ym in items if s == sym]
            print(f"{sym}: {len(months)} months ({months[0]} .. {months[-1]})")
        print(f"total: {len(items)} (symbol, month) items")
        return 0

    sys.exit(asyncio.run(_run(items, force=args.force, reverify=args.reverify_zips)))


if __name__ == "__main__":
    main()
