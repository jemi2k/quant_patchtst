"""
data_engine/corrections.py
===========================
Documented source-data corrections, derived deterministically from
checksum-verified reference archives.

Correction 1 - SOLUSDT 2025-07-16 quantity defect (upstream Binance export
defect, present identically in the daily and monthly aggTrades archives):
  - 3,689 agg rows carry quantity exactly 100x the true aggregated quantity
    (true = sum of raw trades over [first_trade_id, last_trade_id])
  - raw trade 2468302189 (0.66 SOL @ 171.97, maker) was dropped from the
    aggTrades archive; it is folded into the preceding record 926014272-A
    (same price/side, +34 ms - Binance aggregation semantics). The folded
    record adopts the last constituent trade's transact_time and last_trade_id.
  - agg_trade_id 926014272 remains duplicated (both records are distinct real
    executions; no id is invented, none is dropped)
  - agg row 926014100 is UNRESOLVED: its underlying trade ids are absent from
    every public archive; it is retained verbatim and flagged.

Reference archives (kept under data/raw/aggTrades/SOLUSDT/trades-ref/):
  - SOLUSDT-trades-2025-07.zip       (monthly raw trades, full coverage)
  - SOLUSDT-trades-2025-07-16.zip    (daily raw trades)
  - SOLUSDT-aggTrades-2025-07-16.zip (daily aggTrades, defect demonstrator)

Usage:
    python -m data_engine.corrections   # rebuild SOLUSDT 2025-07 parquet in staging
"""

import datetime
import hashlib
import json
import os
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_engine import config

TRADES_REF = "data/raw/aggTrades/SOLUSDT/trades-ref/SOLUSDT-trades-2025-07.zip"

DAY_LO = int(datetime.datetime(2025, 7, 16, tzinfo=datetime.timezone.utc).timestamp() * 1000)
DAY_HI = int(datetime.datetime(2025, 7, 17, tzinfo=datetime.timezone.utc).timestamp() * 1000)

FOLD = {
    "agg_trade_id": 926014272,
    "first_trade_id": 2468302188,
    "new_last_trade_id": 2468302189,
    # timestamp convention: the FIRST constituent execution's timestamp
    # (raw trade 2468302188 @ 16:43:03.594 UTC - also the original archive value)
    "new_transact_time": 1752684183594,
    "added_quantity": 0.66,              # raw trade 2468302189 quantity
}
UNRESOLVED_ID = 926014100


def _trades_map():
    with zipfile.ZipFile(TRADES_REF) as zf:
        csv = zf.read([n for n in zf.namelist() if n.endswith(".csv")][0]).decode("utf-8", errors="replace")
    lines = csv.strip().split("\n")
    if lines[0].lower().startswith("id"):
        lines = lines[1:]
    pos = {}
    for ln in lines:
        p = ln.split(",")
        t = int(p[4])
        if DAY_LO <= t < DAY_HI:
            pos[int(p[0])] = float(p[2])
    return pos


def apply_sol_20250716(table):
    """Edit only 2025-07-16 rows in place (value edits; row order unchanged).
    Returns (new_table, stats)."""
    pos = _trades_map()
    # writeable copies (to_numpy may return read-only views into chunk buffers)
    ids = np.array(table["agg_trade_id"].to_numpy(zero_copy_only=False), copy=True)
    tt = np.array(table["transact_time"].to_numpy(zero_copy_only=False), copy=True)
    qty = np.array(table["quantity"].to_numpy(zero_copy_only=False), copy=True)
    ft = np.array(table["first_trade_id"].to_numpy(zero_copy_only=False), copy=True)
    lt = np.array(table["last_trade_id"].to_numpy(zero_copy_only=False), copy=True)

    mask = (tt >= DAY_LO) & (tt < DAY_HI)
    idx = np.where(mask)[0].tolist()
    corrected = 0
    exact = 0
    unresolved = []

    for i in idx:
        i = int(i)
        if int(ids[i]) == FOLD["agg_trade_id"] and int(ft[i]) == FOLD["first_trade_id"]:
            # folded dropped trade 2468302189 into this record
            s = sum(pos[j] for j in range(int(ft[i]), int(lt[i]) + 1))
            assert abs(qty[i] / s - 100.0) < 1e-6, "fold record is not 100x inflated"
            qty[i] = s + FOLD["added_quantity"]
            lt[i] = FOLD["new_last_trade_id"]
            tt[i] = FOLD["new_transact_time"]
            corrected += 1
            continue
        try:
            s = sum(pos[j] for j in range(int(ft[i]), int(lt[i]) + 1))
        except KeyError:
            partial = sum(pos[j] for j in range(int(ft[i]), int(lt[i]) + 1) if j in pos)
            if partial > 0 and abs(partial - qty[i]) < max(1e-6, 1e-6 * abs(qty[i])):
                exact += 1  # range has archive gaps; present ids already sum to qty
            else:
                unresolved.append(int(ids[i]))
            continue
        if abs(s - qty[i]) < 1e-6:
            exact += 1
        elif abs(qty[i] / s - 100.0) < 1e-6:
            qty[i] = s
            corrected += 1
        else:
            raise ValueError(f"unexpected mismatch at agg_trade_id {int(ids[i])}: qty={qty[i]} true={s}")

    assert corrected == 3689, f"expected 3689 corrections, got {corrected}"
    assert unresolved == [UNRESOLVED_ID], f"unexpected unresolved set: {unresolved}"

    out = table
    for name, arr in (("quantity", qty), ("last_trade_id", lt), ("transact_time", tt)):
        idxf = out.schema.get_field_index(name)
        out = out.set_column(idxf, name, pa.array(arr))
    stats = {"corrected_rows": corrected, "exact_rows": exact,
             "unresolved": unresolved, "day_rows": len(idx)}
    return out, stats


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    from data_engine.ingest import parquet_content_hash, load_manifest, save_manifest
    from data_engine.verify_source_parity import _read_zip_csv, _canonical_digest

    # Derive the pristine table from the checksum-verified source zip (idempotent)
    zip_path = config.zip_path("SOLUSDT", "2025-07")
    table = _read_zip_csv(zip_path)
    table = table.sort_by([("agg_trade_id", "ascending"), ("transact_time", "ascending")])

    # Guard: the zip-derived table must equal the pristine pre-correction parquet.
    # Note: the parquet side must use iter_batches (row-group-aligned, matching
    # the parity verifier), NOT Table.to_batches (chunk-boundary-aligned) -
    # the interleaved multi-column digest is segmentation-sensitive.
    pristine = "data/parquet_old/aggTrades/SOLUSDT/SOLUSDT-aggTrades-2025-07.parquet"
    if os.path.exists(pristine):
        pf = pq.ParquetFile(pristine)
        d1 = _canonical_digest(table.to_batches(max_chunksize=1 << 20), table.num_rows)
        d2 = _canonical_digest(pf.iter_batches(batch_size=1 << 20), pf.metadata.num_rows)
        assert d1 == d2, "zip-derived table diverges from pristine parquet - aborting"

    dst = config.staging_parquet_path("SOLUSDT", "2025-07")
    new_table, stats = apply_sol_20250716(table)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(new_table, dst, compression=config.PARQUET_COMPRESSION)

    manifest = load_manifest()
    entry = manifest["SOLUSDT/2025-07"]
    entry["repair"] = {
        "type": "quantity_correction_2025_07_16",
        "corrected_rows": stats["corrected_rows"],
        "folded_trade": {"trade_id": FOLD["new_last_trade_id"],
                         "into_agg_trade_id": FOLD["agg_trade_id"],
                         "added_quantity": FOLD["added_quantity"]},
        "unresolved": [{"agg_trade_id": UNRESOLVED_ID,
                        "reason": "underlying trade ids absent from all public archives; "
                                  "retained verbatim; max residual notional ~USD 42.6k"}],
        "reference_archives": {
            "SOLUSDT-trades-2025-07.zip": sha256_file(TRADES_REF),
            "SOLUSDT-trades-2025-07-16.zip":
                sha256_file("data/raw/aggTrades/SOLUSDT/trades-ref/SOLUSDT-trades-2025-07-16.zip"),
            "SOLUSDT-aggTrades-2025-07-16.zip":
                sha256_file("data/raw/aggTrades/SOLUSDT/trades-ref/SOLUSDT-aggTrades-2025-07-16.zip"),
        },
        "method": "quantity = sum of raw trades over [first_trade_id, last_trade_id]; "
                  "folded record adopts the last constituent trade timestamp",
    }
    entry["parquet_content_hash"] = parquet_content_hash(dst)
    entry["size"] = os.path.getsize(dst)
    save_manifest(manifest)

    print("correction stats:", stats)
    print("staging parquet:", dst)
    print("manifest SOLUSDT/2025-07 updated with repair provenance")


if __name__ == "__main__":
    main()
