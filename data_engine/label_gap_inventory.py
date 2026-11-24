"""Inventory canonical ID discontinuities and documented uncertain observations for the frozen run."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pyarrow.parquet as pq

from data_engine import config
from data_engine.label_source import _file_identity
from data_engine.label_run import ROOT, PLAN, MANIFEST, _save


EVENT_LIMIT_PER_KIND = 1000
BATCH_SIZE = 1_000_000


def _point(path, row, ordinal, agg_id, stamp):
    seconds, milliseconds = divmod(int(stamp), 1000)
    utc = datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=1000 * milliseconds)
    return dict(file=path, row_zero_based=int(row), ordinal_zero_based=int(ordinal),
                agg_trade_id=int(agg_id), time_ms=int(stamp), utc=utc.isoformat(timespec="milliseconds"))


def scan_asset(asset, streams, manifest, manifest_digest):
    months = streams[0]["months"]
    if any(stream["months"] != months for stream in streams):
        raise ValueError(f"{asset}: configurations disagree on month coverage")
    source = json.loads((ROOT / "sources" / (asset + ".json")).read_text())
    if (source["asset"] != asset or source["status"] != "PASS" or source["manifest_sha256"] != manifest_digest
            or set(source["files"]) != set(months)):
        raise ValueError(f"{asset}: missing matching verified source certificate")
    kinds = ("id_jump", "allowed_duplicate", "invalid_id_order", "unresolved", "time_inversion")
    counts = dict.fromkeys(kinds, 0)
    events = {kind: [] for kind in kinds}
    closed_breaks = {stream["dib_config"]: 0 for stream in streams}
    manifest_notes = {}
    previous = None
    offset = skipped_ids = total_breaks = 0
    started = monotonic()

    def record(kind, event):
        counts[kind] += 1
        if len(events[kind]) < EVENT_LIMIT_PER_KIND:
            events[kind].append(event)

    for name in months:
        path = Path(config.PARQUET_DIR) / asset / name
        entry = manifest[f"{asset}/{name[-15:-8]}"]
        mark = source["files"][name]
        identity = list(_file_identity(path))
        if (identity != mark["identity"] or mark["digest"] != entry["parquet_content_hash"]
                or mark["rows"] != entry["rows"] or entry["status"] != "ok"):
            raise ValueError(f"source differs from frozen certificate: {path}")
        allowed = {item["agg_trade_id"]: len(item["underlying"]) - 1 for item in entry.get("kept_conflicts", [])}
        uncertain = {item["agg_trade_id"]: item for item in entry.get("repair", {}).get("unresolved", [])}
        seen_duplicates, seen_uncertain = Counter(), Counter()
        notes = {key: entry[key] for key in ("fills", "dropped_exact_dups", "sorted_fallback", "repair", "anomalies")
                 if entry.get(key)}
        if notes:
            manifest_notes[name] = notes
        rows = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["agg_trade_id", "transact_time"], batch_size=BATCH_SIZE,
                                          use_threads=False):
            if any(col.null_count for col in batch.columns):
                raise ValueError(f"null ID/time: {path}")
            ids, times = (col.to_numpy() for col in batch.columns)
            if ids.dtype != np.int64 or times.dtype != np.int64 or not len(ids):
                raise ValueError(f"unexpected ID/time batch: {path}")
            steps = np.empty_like(ids)
            steps[0] = 1 if previous is None else int(ids[0]) - previous["agg_trade_id"]
            np.subtract(ids[1:], ids[:-1], out=steps[1:])
            inversions = np.empty(len(times), dtype=bool)
            inversions[0] = previous is not None and times[0] < previous["time_ms"]
            np.less(times[1:], times[:-1], out=inversions[1:])
            breaks = steps > 1
            unresolved = np.isin(ids, list(uncertain)) if uncertain else None
            observed = (steps != 1) | inversions
            if unresolved is not None:
                breaks |= unresolved
                observed |= unresolved
            for index in np.flatnonzero(observed):
                current = _point(name, rows + index, offset + index, ids[index], times[index])
                prior = (previous if index == 0 else
                         _point(name, rows + index - 1, offset + index - 1, ids[index - 1], times[index - 1]))
                delta = int(steps[index])
                if delta > 1:
                    record("id_jump", dict(previous=prior, current=current, id_step=delta,
                                           skipped_id_values=delta-1))
                    skipped_ids += delta - 1
                elif delta != 1:
                    agg_id = int(ids[index])
                    if delta == 0 and seen_duplicates[agg_id] < allowed.get(agg_id, 0):
                        record("allowed_duplicate", dict(previous=prior, current=current))
                        seen_duplicates[agg_id] += 1
                    else:
                        record("invalid_id_order", dict(previous=prior, current=current, id_step=delta))
                if inversions[index]:
                    record("time_inversion", dict(previous=prior, current=current))
                if unresolved is not None and unresolved[index]:
                    agg_id = int(ids[index])
                    seen_uncertain[agg_id] += 1
                    record("unresolved", dict(current=current, manifest=uncertain[agg_id]))
            break_indices = np.flatnonzero(breaks)
            total_breaks += len(break_indices)
            if len(break_indices):
                for stream in streams:
                    closed_breaks[stream["dib_config"]] += int(
                        np.searchsorted(break_indices, stream["closed_trades"] - offset))
            offset += len(ids)
            rows += len(ids)
            previous = _point(name, rows-1, offset-1, ids[-1], times[-1])
        if rows != mark["rows"] or list(_file_identity(path)) != identity:
            raise ValueError(f"wrong row count or source modified during scan: {path}")
        if dict(seen_duplicates) != allowed or set(seen_uncertain) != set(uncertain):
            raise ValueError(f"manifest-declared duplicate or uncertain ID absent from tape: {path}")
    return dict(asset=asset, rows=offset, months=len(months), counts=counts,
                skipped_id_values=skipped_ids, unique_break_positions=total_breaks,
                unique_break_positions_in_closed_prefix_by_config=closed_breaks, events=events,
                all_positions_recorded={kind: counts[kind] == len(events[kind]) for kind in kinds},
                manifest_provenance_notes=manifest_notes, seconds=round(monotonic()-started, 3))


def main():
    plan_bytes, manifest_bytes = PLAN.read_bytes(), MANIFEST.read_bytes()
    plan, manifest = json.loads(plan_bytes), json.loads(manifest_bytes)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != plan["manifest_sha256"]:
        raise ValueError("run manifest digest mismatch")
    assets = {}
    for asset in config.SYMBOLS:
        streams = [stream for stream in plan["streams"] if stream["asset"] == asset]
        result = scan_asset(asset, streams, manifest, manifest_digest)
        print(asset, result["counts"], result["seconds"], flush=True)
        assets[asset] = result
    if PLAN.read_bytes() != plan_bytes or MANIFEST.read_bytes() != manifest_bytes:
        raise ValueError("frozen run changed during scan")
    valid_order = not any(a["counts"]["invalid_id_order"] or a["counts"]["time_inversion"] for a in assets.values())
    report = dict(source="canonical rows in frozen monthly file order, including unclosed tails",
                  projected_columns=["agg_trade_id", "transact_time"], batch_size=BATCH_SIZE,
                  manifest_sha256=manifest_digest, freeze_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                  status="PASS" if valid_order else "FAIL", event_limit_per_kind_per_asset=EVENT_LIMIT_PER_KIND,
                  interpretation="An ID jump marks unobserved integer ID values. It does not establish "
                                 "that executions or barrier crossings are missing. The approved policy flags "
                                 "the row after each jump and each manifest-declared unresolved record. "
                                 "Allowed duplicates do not create a break. No extra time-gap rule is applied.",
                  assets=assets)
    _save(ROOT / "gaps.json", report)
    return int(not valid_order)


if __name__ == "__main__":
    raise SystemExit(main())
