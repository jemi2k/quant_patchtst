"""
data_engine/verify_source_parity.py
===================================
INDEPENDENT source-to-parquet parity verification.

Re-derives the expected parquet content directly from the raw ZIP CSVs (and
daily repair ZIPs), using its own transformation code (not ingest's), and
compares a canonical content digest + row count against the actual parquet.

Transform applied (mirrors the documented pipeline rules):
  1. read the monthly zip CSV (header auto-detect)
  2. if manifest records daily fills for the month: read each daily zip,
     drop daily rows whose agg_trade_id already exists in the monthly CSV
     (monthly wins), append the rest
  3. sort by (agg_trade_id, transact_time) - the canonical order
  4. drop only adjacent rows that are byte-identical across ALL 7 columns
     (exact duplicates); conflicting duplicates are KEPT
  5. canonical digest = sha256 over per-batch (row count + schema-order
     column buffers) - identical procedure to the parquet-side digest

The parquet-side digest is recomputed from the parquet on disk here, so a
match proves: raw sources + documented transforms == parquet records,
independently of any hash recorded at build time.

Run: python -m data_engine.verify_source_parity [--symbols ETHUSDT] [--months 2022-09]
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

from data_engine import config

ALL_COLS = config.AGGTRADES_COLUMNS
PA_TYPES = {
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
}


def _read_zip_csv(zip_path):
    """Return a pyarrow Table of the single CSV inside the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        mems = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(mems) != 1:
            raise ValueError(f"{zip_path}: {len(mems)} csv members")
        with zf.open(mems[0]) as fh:
            first = fh.readline().decode("utf-8", errors="replace").strip()
        has_header = first.lower().startswith("agg_trade_id")
        tmpdir = tempfile.mkdtemp()
        csv_path = zf.extract(mems[0], tmpdir)
    try:
        if has_header:
            ro = pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True)
        else:
            ro = pc.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True,
                                column_names=ALL_COLS)
        co = pc.ConvertOptions(column_types={c: PA_TYPES[config.AGGTRADES_DTYPES[c]] for c in ALL_COLS})
        table = pc.read_csv(csv_path, read_options=ro, convert_options=co)
        assert set(table.schema.names) == set(ALL_COLS), table.schema.names
        return table
    finally:
        os.remove(csv_path)
        os.rmdir(tmpdir)


def _canonical_digest(batches, n_rows):
    """Boundary-invariant digest: per batch, per column, hash the contiguous
    numpy bytes of the decoded column. Batch segmentation therefore does not
    affect the digest (concatenation of contiguous slices is partition-free);
    only row content and row order do."""
    h = hashlib.sha256()
    seen = 0
    for batch in batches:
        n = batch.num_rows
        seen += n
        for c in ALL_COLS:
            arr = batch.column(c).to_numpy(zero_copy_only=False)
            h.update(arr.tobytes())
    assert seen == n_rows, f"digest row mismatch {seen} != {n_rows}"
    return h.hexdigest()


def _digest_table(table):
    """Independent transform: sort, drop exact dups, canonical digest."""
    table = table.sort_by([("agg_trade_id", "ascending"), ("transact_time", "ascending")])
    ids = table.column("agg_trade_id").to_numpy(zero_copy_only=False)
    n = table.num_rows
    keep = np.ones(n, dtype=bool)
    dropped = 0
    if n > 1:
        eq = np.where(ids[1:] == ids[:-1])[0]
        for i in eq.tolist():
            i = int(i)
            same = True
            for c in ALL_COLS:
                if table.column(c)[i].as_py() != table.column(c)[i + 1].as_py():
                    same = False
                    break
            if same:
                keep[i + 1] = False
                dropped += 1
    if not keep.all():
        table = table.filter(pa.array(keep))
    digest = _canonical_digest(table.to_batches(max_chunksize=1 << 20), table.num_rows)
    return digest, table.num_rows, dropped


def _digest_parquet(parquet_path):
    pf = pq.ParquetFile(parquet_path)
    return _canonical_digest(pf.iter_batches(batch_size=1 << 20), pf.metadata.num_rows), pf.metadata.num_rows


def verify_month(sym, ym, manifest):
    zip_path = config.zip_path(sym, ym)
    parquet_path = config.parquet_path(sym, ym)
    entry = manifest.get(f"{sym}/{ym}", {})

    monthly = _read_zip_csv(zip_path)
    parts = [monthly]
    for f in entry.get("fills", []):
        day = f["date"]
        daily = _read_zip_csv(config.daily_zip_path(sym, day))
        monthly_ids = set(monthly.column("agg_trade_id").to_pylist())
        dids = daily.column("agg_trade_id").to_numpy(zero_copy_only=False)
        keep = np.array([int(d) not in monthly_ids for d in dids])
        if not keep.all():
            daily = daily.filter(pa.array(keep))
        parts.append(daily)
    table = pa.concat_tables(parts)

    # apply documented source-data corrections (deterministic, provenance-recorded)
    if sym == "SOLUSDT" and ym == "2025-07":
        from data_engine.corrections import apply_sol_20250716
        table, _stats = apply_sol_20250716(table)

    exp_digest, exp_rows, dropped = _digest_table(table)
    got_digest, got_rows = _digest_parquet(parquet_path)

    ok = (exp_digest == got_digest) and (exp_rows == got_rows)
    return {
        "sym": sym, "ym": ym,
        "ok": ok,
        "expected_rows": exp_rows,
        "parquet_rows": got_rows,
        "dropped_exact_dups": dropped,
        "fills": [f["date"] for f in entry.get("fills", [])],
        "expected_digest": exp_digest[:16],
        "parquet_digest": got_digest[:16],
    }


def main():
    ap = argparse.ArgumentParser(description="Independent source-to-parquet parity verification")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--months", nargs="*", help="restrict to specific YYYY-MM")
    ap.add_argument("--out", default="data/source_parity_report.json")
    args = ap.parse_args()

    manifest = json.load(open(config.MANIFEST_PATH))
    symbols = args.symbols or config.SYMBOLS
    results = []
    problems = []
    for sym in symbols:
        for ym in (args.months or [e.split("/")[1] for e in manifest if e.startswith(sym + "/")]):
            if f"{sym}/{ym}" not in manifest or manifest[f"{sym}/{ym}"].get("status") != "ok":
                continue
            r = verify_month(sym, ym, manifest)
            results.append(r)
            status = "OK " if r["ok"] else "MISMATCH"
            print(f"[{status}] {sym}/{ym}: expected={r['expected_rows']:,} parquet={r['parquet_rows']:,} "
                  f"dups_dropped={r['dropped_exact_dups']:,} fills={r['fills']} "
                  f"digest {r['expected_digest']} == {r['parquet_digest']}", flush=True)
            if not r["ok"]:
                problems.append(f"{sym}/{ym}")

    with open(args.out, "w") as fh:
        json.dump({"results": results, "problems": problems}, fh, indent=2)
    print(f"\nchecked {len(results)} months; problems: {problems if problems else 'NONE'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
