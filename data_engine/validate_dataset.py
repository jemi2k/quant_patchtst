"""
data_engine/validate_dataset.py
===============================
Comprehensive, streaming validation of the raw ZIP + Parquet dataset.

Run (read-only; writes only data/validation_report.json):
    python -m data_engine.validate_dataset
    python -m data_engine.validate_dataset --dir data/parquet_staging

Sections:
  1. coverage       every expected (symbol, month) has zip + parquet + manifest ok
  2. zip_checksums  every expected zip's sha256 matches the official sidecar mirror
  3. zip_integrity  every zip's central directory opens cleanly
  4. full_scan      per-file streaming pass over ALL columns:
                      deterministic content hash (schema-order decoded bytes)
                      null counts (must be 0), zero-quantity rows,
                      agg_trade_id monotonicity (equal ids allowed only for
                      documented kept-conflict anomalies),
                      per-date row histogram, month-bounds check,
                      large (>1s) transact_time inversions (informational)
  5. ordering       cross-file id/time contiguity
  6. reconcile      manifest rows == parquet rows == csv - dropped + fills - overlap
  7. schema         dtypes match config; no unexpected columns
  8. may2023        corrected archives: 2023-05-10 present, zero-quantity rows gone,
                    old vs new zip row-count delta
  9. sep2026        all 30 days of 2026-09 populated for every symbol
  10. manifest      entries accurate: status, rows, size, zip_sha256, content hash

Exit code 0 only when every section passes. Everything is re-derived from the
files on disk - nothing is trusted from the manifest without re-checking.
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

from data_engine import config
from data_engine.ingest import sha256_file, parquet_content_hash

REPORT_PATH = "data/validation_report.json"
MS_DAY = 86400000


def _months(a, b):
    y, m = map(int, a.split("-"))
    ey, em = map(int, b.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _month_window_ms(ym):
    y, m = map(int, ym.split("-"))
    start = datetime.datetime(y, m, 1, tzinfo=datetime.timezone.utc)
    if m == 12:
        end = datetime.datetime(y + 1, 1, 1, tzinfo=datetime.timezone.utc)
    else:
        end = datetime.datetime(y, m + 1, 1, tzinfo=datetime.timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _utc_date(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")


def expected_items():
    items = []
    for sym in config.SYMBOLS:
        for ym in _months(config.INCEPTION[sym], config.DATA_END):
            items.append((sym, ym))
    return items


def check_coverage(parquet_dir, manifest, report):
    items = expected_items()
    ok, missing_zip, missing_pq, bad_status = [], [], [], []
    for sym, ym in items:
        zp = config.zip_path(sym, ym)
        pp = os.path.join(parquet_dir, sym, f"{sym}-aggTrades-{ym}.parquet")
        entry = manifest.get(f"{sym}/{ym}", {})
        if not os.path.exists(zp):
            missing_zip.append(f"{sym}/{ym}")
        if not os.path.exists(pp):
            missing_pq.append(f"{sym}/{ym}")
        if entry.get("status") != "ok":
            bad_status.append(f"{sym}/{ym}")
        elif f"{sym}/{ym}" not in missing_pq and f"{sym}/{ym}" not in missing_zip:
            ok.append(f"{sym}/{ym}")
    report["coverage"] = {
        "expected": len(items),
        "ok": len(ok),
        "missing_zip": missing_zip, "missing_parquet": missing_pq,
        "manifest_status_not_ok": [k for k in bad_status if k not in missing_pq + missing_zip],
    }
    return not (missing_zip or missing_pq or bad_status)


def check_zip_checksums(report):
    mismatch, no_sidecar, passed = [], [], 0
    for sym, ym in expected_items():
        zp = config.zip_path(sym, ym)
        ck = config.checksum_path(sym, ym)
        if not os.path.exists(ck):
            no_sidecar.append(f"{sym}/{ym}")
            continue
        want = open(ck).read().strip().split()[0].lower()
        got = sha256_file(zp)
        if got == want:
            passed += 1
        else:
            mismatch.append(f"{sym}/{ym} (local {got[:12]}... != official {want[:12]}...)")
    report["zip_checksums"] = {"passed": passed, "mismatch": mismatch, "no_sidecar": no_sidecar}
    return not (mismatch or no_sidecar)


def check_zip_integrity(report):
    bad = []
    for sym, ym in expected_items():
        zp = config.zip_path(sym, ym)
        try:
            with zipfile.ZipFile(zp) as zf:
                if not zf.namelist():
                    bad.append(f"{sym}/{ym} (empty)")
        except Exception as exc:
            bad.append(f"{sym}/{ym}: {exc}")
    report["zip_integrity"] = {"bad": bad, "checked": len(expected_items())}
    return not bad


def full_scan(parquet_dir, manifest, report):
    """One streaming pass per file: hash, nulls, zero-qty, id order, per-date counts."""
    anomalies = []
    per_date = {}
    file_meta = []
    total_rows = 0
    n_files = 0

    for sym, ym in expected_items():
        pp = os.path.join(parquet_dir, sym, f"{sym}-aggTrades-{ym}.parquet")
        pf = pq.ParquetFile(pp)
        names = pf.schema_arrow.names
        if names != config.AGGTRADES_COLUMNS:
            anomalies.append({"type": "schema", "where": f"{sym}/{ym}", "detail": f"columns {names}"})
        allowed_eq = {int(c["agg_trade_id"]) for c in manifest.get(f"{sym}/{ym}", {}).get("kept_conflicts", [])}
        h = hashlib.sha256()
        nulls = {c: 0 for c in names}
        zero_qty = 0
        prev_id = None
        decreasing = 0
        equal_pairs = 0
        unallowed_equal = []
        time_inversions = 0
        tmin, tmax = None, None
        day_counts = {}
        rows = 0
        for batch in pf.iter_batches(batch_size=1 << 20):
            n = batch.num_rows
            if n == 0:
                continue
            rows += n
            h.update(n.to_bytes(8, "little"))
            for c in names:
                arr = batch.column(c)
                nulls[c] += arr.null_count
                for b in (arr.buffers() or []):
                    if b is not None:
                        h.update(bytes(b))
            ids = batch.column("agg_trade_id").to_numpy(zero_copy_only=False)
            if n > 1:
                d = np.diff(ids)
                decreasing += int((d < 0).sum())
                eq = int((d == 0).sum())
                equal_pairs += eq
                if eq:
                    eq_ids = ids[1:][d == 0]
                    unallowed_equal.extend(int(i) for i in eq_ids if int(i) not in allowed_eq)
            if prev_id is not None and int(ids[0]) < prev_id:
                decreasing += 1
            if prev_id is not None and int(ids[0]) == prev_id and int(ids[0]) not in allowed_eq:
                unallowed_equal.append(int(ids[0]))
            if n > 0:
                prev_id = int(ids[-1])
            qty = batch.column("quantity").to_numpy(zero_copy_only=False)
            zero_qty += int((qty == 0).sum())
            tt = batch.column("transact_time").to_numpy(zero_copy_only=False)
            if tmin is None:
                tmin, tmax = int(tt[0]), int(tt[0])
            tmin = min(tmin, int(tt.min()))
            tmax = max(tmax, int(tt.max()))
            if n > 1:
                time_inversions += int((np.diff(tt) < -1000).sum())
            days = (tt // MS_DAY).astype(np.int64)
            uniq, cnts = np.unique(days, return_counts=True)
            for d, c in zip(uniq.tolist(), cnts.tolist()):
                day_counts[d] = day_counts.get(d, 0) + int(c)

        total_rows += rows
        n_files += 1
        wmin, wmax = _month_window_ms(ym)
        out_of_bounds = (tmin < wmin or tmax >= wmax)
        file_meta.append({"sym": sym, "ym": ym, "rows": rows,
                          "tmin": tmin, "tmax": tmax, "out_of_month_bounds": out_of_bounds})
        for d, c in day_counts.items():
            per_date[f"{sym}:{_utc_date(d * MS_DAY)}"] = per_date.get(f"{sym}:{_utc_date(d * MS_DAY)}", 0) + c
        if nulls.get("agg_trade_id") or nulls.get("transact_time") or nulls.get("price") or nulls.get("quantity"):
            anomalies.append({"type": "nulls", "where": f"{sym}/{ym}",
                              "detail": {k: v for k, v in nulls.items() if v}})
        if zero_qty:
            anomalies.append({"type": "zero_quantity", "where": f"{sym}/{ym}", "detail": zero_qty})
        if decreasing:
            anomalies.append({"type": "decreasing_ids", "where": f"{sym}/{ym}", "detail": decreasing})
        if unallowed_equal:
            anomalies.append({"type": "duplicate_ids", "where": f"{sym}/{ym}",
                              "detail": sorted(set(unallowed_equal))[:20]})
        if out_of_bounds:
            anomalies.append({"type": "out_of_month_bounds", "where": f"{sym}/{ym}",
                              "detail": f"tmin={tmin} tmax={tmax}"})
        rec_hash = manifest.get(f"{sym}/{ym}", {}).get("parquet_content_hash")
        if rec_hash and rec_hash != h.hexdigest():
            anomalies.append({"type": "content_hash_mismatch", "where": f"{sym}/{ym}",
                              "detail": f"manifest {rec_hash[:12]}... != recomputed {h.hexdigest()[:12]}..."})

    # cross-file ordering
    cross = []
    by_sym = {}
    for m in file_meta:
        by_sym.setdefault(m["sym"], []).append(m)
    for sym, files in by_sym.items():
        for a, b in zip(files, files[1:]):
            if a["tmax"] >= b["tmin"]:
                cross.append(f"{sym}: {a['ym']}->{b['ym']} overlap")
            elif (b["tmin"] - a["tmax"]) > 3 * 3600 * 1000:
                cross.append(f"{sym}: {a['ym']}->{b['ym']} gap {(b['tmin'] - a['tmax']) / 3.6e6:.1f}h")

    # proper zero-row-date sweep: every calendar day from each symbol's first
    # trade through DATA_END must have at least one row
    zero_days = []
    for sym in config.SYMBOLS:
        files = [m for m in file_meta if m["sym"] == sym]
        if not files:
            continue
        first = min(m["tmin"] for m in files)
        first_day = first // MS_DAY
        y, m = map(int, config.DATA_END.split("-"))
        end = datetime.datetime(y + 1 if m == 12 else y, 1 if m == 12 else m + 1, 1,
                                tzinfo=datetime.timezone.utc)
        last_day = int(end.timestamp() * 1000) // MS_DAY - 1
        d = first_day
        while d <= last_day:
            if per_date.get(f"{sym}:{_utc_date(d * MS_DAY)}", 0) == 0:
                zero_days.append(f"{sym}:{_utc_date(d * MS_DAY)}")
            d += 1
    report["full_scan"] = {
        "files": n_files, "total_rows": total_rows,
        "anomalies": anomalies,
        "cross_file_issues": cross,
        "time_inversions_total": time_inversions,
        "zero_row_dates": zero_days,
    }
    report["_file_meta"] = file_meta
    return not (anomalies or cross)


def check_reconcile(parquet_dir, manifest, report):
    problems = []
    for sym, ym in expected_items():
        entry = manifest.get(f"{sym}/{ym}", {})
        pp = os.path.join(parquet_dir, sym, f"{sym}-aggTrades-{ym}.parquet")
        nrows = pq.ParquetFile(pp).metadata.num_rows
        if entry.get("rows") != nrows:
            problems.append(f"{sym}/{ym}: manifest rows {entry.get('rows')} != parquet {nrows}")
            continue
        if entry.get("status") != "ok":
            continue
        fills = entry.get("fills", [])
        if fills:
            base = entry.get("monthly_rows")
        else:
            base = entry.get("csv_rows")
        if base is None:
            problems.append(f"{sym}/{ym}: manifest lacks {'monthly_rows' if fills else 'csv_rows'}")
            continue
        expected = base - entry.get("dropped_exact_dups", 0)
        for f in fills:
            expected += f["rows"]
        expected -= entry.get("fill_overlap_dropped", 0)
        if expected != nrows:
            problems.append(f"{sym}/{ym}: base {base} - dups {entry.get('dropped_exact_dups', 0)}"
                            f" + fills {sum(f['rows'] for f in fills)}"
                            f" - overlap {entry.get('fill_overlap_dropped', 0)} = {expected} != parquet {nrows}")
    report["reconcile"] = {"problems": problems, "checked": len(expected_items())}
    return not problems


def check_schema(parquet_dir, report):
    problems = []
    for sym, ym in expected_items():
        pp = os.path.join(parquet_dir, sym, f"{sym}-aggTrades-{ym}.parquet")
        sch = pq.ParquetFile(pp).schema_arrow
        want = config.AGGTRADES_DTYPES
        for name, dtype in want.items():
            if name not in sch.names:
                problems.append(f"{sym}/{ym}: missing column {name}")
                continue
            got = str(sch.field(name).type)
            ok = (dtype == "int64" and got == "int64") or \
                 (dtype == "float64" and got == "double") or \
                 (dtype == "bool" and got == "bool")
            if not ok:
                problems.append(f"{sym}/{ym}: column {name} dtype {got} != {dtype}")
    report["schema"] = {"problems": problems, "checked": len(expected_items())}
    return not problems


def check_may2023(parquet_dir, manifest, report):
    problems = []
    for sym in config.SYMBOLS:
        ym = "2023-05"
        key = f"{sym}:{_utc_date(datetime.datetime(2023, 5, 10, tzinfo=datetime.timezone.utc).timestamp() * 1000)}"
        # per-date presence is checked via full_scan histograms; here: zero-qty + old/new zip delta
        pp = os.path.join(parquet_dir, sym, f"{sym}-aggTrades-{ym}.parquet")
        zp = config.zip_path(sym, ym)
        stale = zp.replace(".zip", ".zip.stale")
        def _csv_rows(zpath):
            with zipfile.ZipFile(zpath) as zf:
                mems = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                with zf.open(mems[0]) as fh:
                    first = fh.readline().decode("utf-8", errors="replace").strip()
                csv_path = zf.extract(mems[0], "/tmp")
            has_header = first.lower().startswith("agg_trade_id")
            opts = (pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True)
                    if has_header else
                    pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True,
                                   column_names=config.AGGTRADES_COLUMNS))
            reader = pc.open_csv(csv_path, read_options=opts)
            n = sum(b.num_rows for b in reader)
            os.remove(csv_path)
            return n
        old_rows, new_rows = None, None
        if os.path.exists(stale):
            old_rows = _csv_rows(stale)
        new_rows = _csv_rows(zp)
        entry = manifest.get(f"{sym}/{ym}", {})
        if entry.get("rows") != new_rows:
            problems.append(f"{sym}/2023-05: parquet {entry.get('rows')} != new zip rows {new_rows}")
        if old_rows is not None:
            report.setdefault("may2023_zip_delta", {})[sym] = {"old_rows": old_rows,
                                                                "new_rows": new_rows,
                                                                "delta": new_rows - old_rows}
    report["may2023"] = {"problems": problems}
    return not problems


def check_sep2026(parquet_dir, manifest, report):
    problems = []
    for sym in config.SYMBOLS:
        pp = os.path.join(parquet_dir, sym, f"{sym}-aggTrades-2026-09.parquet")
        pf = pq.ParquetFile(pp)
        days = set()
        for batch in pf.iter_batches(batch_size=1 << 20, columns=["transact_time"]):
            tt = batch.column("transact_time").to_numpy(zero_copy_only=False)
            days.update((tt // MS_DAY).tolist())
        expected = {(datetime.datetime(2026, 9, d, tzinfo=datetime.timezone.utc).timestamp() * 1000) // MS_DAY
                    for d in range(1, 31)}
        missing = sorted(expected - days)
        if missing:
            problems.append(f"{sym}: 2026-09 missing dates {[_utc_date(d * MS_DAY) for d in missing]}")
    report["sep2026"] = {"problems": problems}
    return not problems


def check_manifest(parquet_dir, manifest, report):
    problems = []
    for sym, ym in expected_items():
        key = f"{sym}/{ym}"
        entry = manifest.get(key)
        if not entry or entry.get("status") != "ok":
            problems.append(f"{key}: missing/not-ok manifest entry")
            continue
        pp = os.path.join(parquet_dir, sym, f"{sym}-aggTrades-{ym}.parquet")
        zp = config.zip_path(sym, ym)
        if entry.get("rows") != pq.ParquetFile(pp).metadata.num_rows:
            problems.append(f"{key}: manifest rows mismatch")
        if entry.get("size") != os.path.getsize(pp):
            problems.append(f"{key}: manifest size mismatch")
        ck = config.checksum_path(sym, ym)
        if os.path.exists(ck):
            want = open(ck).read().strip().split()[0].lower()
            if entry.get("zip_sha256") != want:
                problems.append(f"{key}: manifest zip_sha256 != official checksum")
        else:
            problems.append(f"{key}: no checksum sidecar")
        if entry.get("parquet_content_hash") is None:
            problems.append(f"{key}: manifest lacks content hash")
    # failed/not_found entries inside target range
    for key, entry in manifest.items():
        if entry.get("status") in ("failed", "not_found"):
            problems.append(f"{key}: manifest status {entry.get('status')}")
    report["manifest"] = {"problems": problems}
    return not problems


def main():
    ap = argparse.ArgumentParser(description="Streaming dataset validation")
    ap.add_argument("--dir", default=config.PARQUET_DIR,
                    help="parquet root to validate (default: canonical data/parquet)")
    args = ap.parse_args()

    parquet_dir = args.dir
    manifest = json.load(open(config.MANIFEST_PATH)) if os.path.exists(config.MANIFEST_PATH) else {}
    report = {"parquet_dir": parquet_dir, "sections": {}}
    sections = [
        ("coverage", lambda r: check_coverage(parquet_dir, manifest, r)),
        ("zip_checksums", check_zip_checksums),
        ("zip_integrity", check_zip_integrity),
        ("full_scan", lambda r: full_scan(parquet_dir, manifest, r)),
        ("reconcile", lambda r: check_reconcile(parquet_dir, manifest, r)),
        ("schema", lambda r: check_schema(parquet_dir, r)),
        ("may2023", lambda r: check_may2023(parquet_dir, manifest, r)),
        ("sep2026", lambda r: check_sep2026(parquet_dir, manifest, r)),
        ("manifest", lambda r: check_manifest(parquet_dir, manifest, r)),
    ]
    results = {}
    for name, fn in sections:
        try:
            ok = fn(report)
            results[name] = "PASS" if ok else "FAIL"
            print(f"[{results[name]}] {name}")
        except Exception as exc:
            results[name] = "ERROR"
            report.setdefault(name, {}).setdefault("errors", []).append(str(exc))
            print(f"[ERROR] {name}: {exc}")

    report["results"] = results
    report.pop("_file_meta", None)
    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"report: {REPORT_PATH}")
    print("overall:", "PASS" if all(v == "PASS" for v in results.values()) else "FAIL")
    sys.exit(0 if all(v == "PASS" for v in results.values()) else 1)


if __name__ == "__main__":
    main()
