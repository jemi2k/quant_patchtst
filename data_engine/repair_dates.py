"""
data_engine/repair_dates.py
===========================
Evidence-driven repair of archive gaps using Binance DAILY aggTrades files.

Flow:
  1. stream every parquet in <dir> and compute per-date row counts
  2. zero-row dates after the symbol's first trade are candidate gaps
  3. for each candidate date, probe the daily archive:
       - 404            -> genuinely no records (recorded, no action)
       - zip exists     -> download + checksum-verify, count rows
         - 0 rows       -> genuinely empty daily file (recorded, no action)
         - rows > 0     -> add to the fills map for that (symbol, month)
  4. run ingest --force only for the affected months with the fills map
     (monthly rows win on id collision; overlap is counted)

Read-only apart from: daily zips under data/raw/.../daily/, rebuilt parquets in
the configured staging dir, and manifest updates via ingest.

Run:
    python -m data_engine.repair_dates [--dir data/parquet_staging]
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import zipfile

import aiohttp
import pyarrow.csv as pc
import pyarrow.parquet as pq

from data_engine import config
from data_engine.ingest import _iter_months

logger = logging.getLogger("data_engine.repair_dates")

MS_DAY = 86400000


def _utc_date(ms):
    import datetime
    return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")


def scan_zero_dates(parquet_dir):
    """Stream all parquets; return {symbol: {first_trade_ms, zero_dates: {date: ym}}}."""
    out = {}
    for sym in config.SYMBOLS:
        months = list(_iter_months(config.INCEPTION[sym], config.DATA_END))
        day_counts = {}
        first_trade = None
        for ym in months:
            pp = os.path.join(parquet_dir, sym, f"{sym}-aggTrades-{ym}.parquet")
            if not os.path.exists(pp):
                logger.warning("%s/%s missing parquet", sym, ym)
                continue
            pf = pq.ParquetFile(pp)
            for batch in pf.iter_batches(batch_size=1 << 20, columns=["transact_time"]):
                tt = batch.column("transact_time").to_numpy(zero_copy_only=False)
                if len(tt) == 0:
                    continue
                if first_trade is None or int(tt.min()) < first_trade:
                    first_trade = int(tt.min())
                import numpy as np
                days = (tt // MS_DAY).astype(np.int64)
                uniq, cnts = np.unique(days, return_counts=True)
                for d, c in zip(uniq.tolist(), cnts.tolist()):
                    day_counts[d] = day_counts.get(d, 0) + int(c)
        # every calendar date between first_trade and last month end must have rows
        import datetime
        first_day = first_trade // MS_DAY if first_trade else None
        end_ym = config.DATA_END
        y, m = map(int, end_ym.split("-"))
        if m == 12:
            end = datetime.datetime(y + 1, 1, 1, tzinfo=datetime.timezone.utc)
        else:
            end = datetime.datetime(y, m + 1, 1, tzinfo=datetime.timezone.utc)
        last_day = int(end.timestamp() * 1000) // MS_DAY - 1
        zero_dates = {}
        if first_day:
            d = first_day
            while d <= last_day:
                if day_counts.get(d, 0) == 0:
                    zero_dates[_utc_date(d * MS_DAY)] = _utc_date(d * MS_DAY)[:7]
                d += 1
        out[sym] = {"first_trade": first_trade, "zero_dates": zero_dates,
                    "total_days_checked": (last_day - first_day + 1) if first_day else 0}
    return out


async def _head(session, url):
    for attempt in range(4):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                return resp.status
        except aiohttp.ClientError:
            if attempt == 3:
                raise
            await asyncio.sleep(1 + attempt)


async def _download_daily(session, sym, day, want_sha):
    url = config.daily_url(sym, day)
    dest = config.daily_zip_path(sym, day)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        h = hashlib.sha256()
        with open(dest, "rb") as fh:
            for c in iter(lambda: fh.read(8 << 20), b""):
                h.update(c)
        if h.hexdigest() == want_sha:
            return dest
    for attempt in range(6):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=300, sock_read=120)) as resp:
                resp.raise_for_status()
                tmp = dest + ".part"
                h = hashlib.sha256()
                with open(tmp, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(1 << 20):
                        fh.write(chunk)
                        h.update(chunk)
                if h.hexdigest() != want_sha:
                    os.remove(tmp)
                    raise RuntimeError("checksum mismatch")
                os.replace(tmp, dest)
                return dest
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
            if attempt == 5:
                raise
            await asyncio.sleep([2, 4, 8, 16, 30][attempt])


def _count_daily_rows(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        mems = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(mems) != 1:
            raise ValueError(f"daily zip {os.path.basename(zip_path)}: {len(mems)} csv members")
        with zf.open(mems[0]) as fh:
            first = fh.readline().decode("utf-8", errors="replace").strip()
        csv_path = zf.extract(mems[0], "/tmp")
    has_header = first.lower().startswith("agg_trade_id")
    opts = (pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True) if has_header else
            pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True,
                           column_names=config.AGGTRADES_COLUMNS))
    try:
        reader = pc.open_csv(csv_path, read_options=opts)
        return sum(b.num_rows for b in reader)
    finally:
        os.remove(csv_path)


async def probe_and_fill(zero_dates_by_sym):
    """Download daily zips for candidate dates; return (fills_map, report)."""
    fills_map = {}
    report = {"recoverable": [], "genuine_empty": [], "daily_missing": [], "errors": []}
    async with aiohttp.ClientSession() as session:
        for sym, info in zero_dates_by_sym.items():
            for date, ym in sorted(info["zero_dates"].items()):
                url = config.daily_url(sym, date)
                try:
                    status = await _head(session, url)
                except Exception as exc:
                    report["errors"].append(f"{sym}/{date}: probe failed {exc}")
                    continue
                if status == 404:
                    report["daily_missing"].append(f"{sym}/{date}")
                    continue
                ck_url = url + ".CHECKSUM"
                try:
                    async with session.get(ck_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status == 404:
                            report["errors"].append(f"{sym}/{date}: no daily CHECKSUM sidecar")
                            continue
                        want = (await resp.text()).strip().split()[0].lower()
                except aiohttp.ClientError as exc:
                    report["errors"].append(f"{sym}/{date}: checksum fetch failed {exc}")
                    continue
                try:
                    zp = await _download_daily(session, sym, date, want)
                    n = _count_daily_rows(zp)
                except Exception as exc:
                    report["errors"].append(f"{sym}/{date}: {exc}")
                    continue
                if n == 0:
                    report["genuine_empty"].append(f"{sym}/{date} (daily zip has 0 rows)")
                else:
                    fills_map.setdefault((sym, ym), []).append(date)
                    report["recoverable"].append(f"{sym}/{date}: {n} rows -> fill {ym}")
    return fills_map, report


async def _main(parquet_dir):
    from data_engine.ingest import _run

    logger.info("scanning %s for zero-row dates", parquet_dir)
    scan = scan_zero_dates(parquet_dir)
    n_candidates = sum(len(v["zero_dates"]) for v in scan.values())
    logger.info("candidate zero-row dates: %d", n_candidates)
    for sym, info in scan.items():
        logger.info("  %s: first trade %s, %d days checked, %d zero-row dates",
                    sym, _utc_date(info["first_trade"]) if info["first_trade"] else "-",
                    info["total_days_checked"], len(info["zero_dates"]))

    fills_map, report = await probe_and_fill(scan)
    print(json.dumps(report, indent=1))
    with open("data/repair_report.json", "w") as fh:
        json.dump({"scan": {s: {k: v for k, v in d.items() if k != "zero_dates"} for s, d in scan.items()},
                   "report": report, "fills_map": {f"{s}/{ym}": days for (s, ym), days in sorted(fills_map.items())}},
                  fh, indent=2)

    if not fills_map:
        logger.info("no recoverable gaps found - nothing to repair")
        return 0

    # rebuild only the affected months with fills
    items = [(s, ym) for (s, ym) in sorted(fills_map)]
    logger.info("repairing %d months with daily fills", len(items))
    rc = await _run(items, force=True, fills_map=fills_map)
    return rc


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Repair archive gaps from daily aggTrades")
    ap.add_argument("--dir", default=config.STAGING_PARQUET_DIR)
    args = ap.parse_args()
    sys.exit(asyncio.run(_main(args.dir)))


if __name__ == "__main__":
    main()
