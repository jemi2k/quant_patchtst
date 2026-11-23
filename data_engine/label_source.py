"""Verified, read-only replay of the price tape underlying one frozen DIB generation."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

from data_engine import config as data_config
from data_engine.ingest import parquet_content_hash
from data_engine.labels import TradeBar


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    with path.open() as stream:
        return json.load(stream)


def _batches(files, batch_size, max_gap_ms, known_gaps, closed_rows):
    previous_id = previous_time = None
    for path, entry, identity in files:
        if not closed_rows:
            return
        if _file_identity(path) != identity:
            raise ValueError(f"canonical input changed after hash verification: {path}")
        allowed = {item["agg_trade_id"] for item in entry.get("kept_conflicts", [])}
        uncertain = {item["agg_trade_id"] for item in
                     entry.get("repair", {}).get("unresolved", [])}
        for batch in pq.ParquetFile(path).iter_batches(
                columns=["price", "transact_time", "agg_trade_id"], batch_size=batch_size):
            if not batch.num_rows:
                continue
            if not closed_rows:
                return
            if batch.num_rows > closed_rows:
                batch = batch.slice(0, closed_rows)
            closed_rows -= batch.num_rows
            if any(col.null_count for col in batch.columns):
                raise ValueError(f"null trade in {path}")
            prices, times, ids = (col.to_numpy() for col in batch.columns)
            prior_times = np.empty_like(times)
            prior_times[1:] = times[:-1]
            prior_times[0] = times[0] if previous_time is None else previous_time
            changes = np.diff(ids, prepend=ids[0] - 1 if previous_id is None else previous_id)
            if (np.any(times < prior_times) or np.any(changes < 0)
                    or not set(ids[changes == 0]).issubset(allowed)):
                raise ValueError(f"transaction time, ID order or undocumented duplicate in {path}")
            breaks = changes > 1
            if uncertain:
                breaks |= np.isin(ids, list(uncertain))
            if max_gap_ms is not None:
                breaks |= times - prior_times > max_gap_ms
            for start, end in known_gaps:
                breaks |= (times >= start) & (prior_times < end)
            previous_id, previous_time = ids[-1], times[-1]
            yield prices, times, breaks


def _file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def iter_generation_bars(asset, dib_config, generation, *, bars_root="data/bars",
                         trades_root=data_config.PARQUET_DIR, manifest_path=data_config.MANIFEST_PATH,
                         batch_size=131072, max_trade_gap_ms=None, known_gaps=(), verified_source=None):
    """Yield completed bars only, pinned by registry ID and verified source hashes.

    known_gaps are externally identified [start_ms, end_ms) UTC intervals.
    Ties retain canonical row order. Time regressions are a hard source failure.
    """
    if (not re.fullmatch(r"[A-Za-z0-9_]+", asset) or not re.fullmatch(r"[A-Za-z0-9_]+", dib_config)
            or not re.fullmatch(r"gen-[A-Za-z0-9_-]+", generation)):
        raise ValueError("expected asset, config and explicit generation identifiers")
    known_gaps = tuple(known_gaps)
    if (batch_size < 1 or (max_trade_gap_ms is not None and max_trade_gap_ms <= 0)
            or any(start >= end for start, end in known_gaps)):
        raise ValueError("invalid batch size or gap policy")
    root = Path(bars_root) / asset / dib_config
    entries = [item for item in _json(root / "generations.json")["generations"]
               if item["id"] == generation]
    if len(entries) != 1:
        raise ValueError(f"unknown or ambiguous generation: {generation}")
    stem = f"{asset}-{dib_config}-dib"
    base = root / generation / (stem + ".parquet")
    meta_path = base.with_name(stem + "-meta.json")
    tail_path = base.with_suffix(".parquet.tail.json")
    for path in (base, meta_path, tail_path):
        if _sha256(path) != entries[0]["artifacts"].get(path.name):
            raise ValueError(f"generation digest mismatch: {path}")
    meta, tail = _json(meta_path), _json(tail_path)
    if (meta["symbol"] != asset or meta["config"]["id"] != dib_config or meta["tail"] != tail
            or tail["num_trades"] < 0):
        raise ValueError("DIB metadata identity/tail mismatch")
    manifest = _json(Path(manifest_path))
    if verified_source is not None and (
            verified_source["asset"] != asset or verified_source["status"] != "PASS"
            or verified_source["manifest_sha256"] != _sha256(Path(manifest_path))):
        raise ValueError("source verification belongs to a different asset or manifest")
    files = []
    last_month = None
    for name in meta["input_months"]:
        match = re.fullmatch(re.escape(asset) + r"-aggTrades-(\d{4}-\d\d)\.parquet", name)
        if not match:
            raise ValueError(f"invalid source name: {name}")
        month = match.group(1)
        date = datetime.strptime(month, "%Y-%m")
        number = date.year * 12 + date.month
        if last_month is not None and number != last_month + 1:
            raise ValueError("unordered or missing source month")
        last_month = number
        path = Path(trades_root) / asset / name
        entry = manifest.get(f"{asset}/{month}", {})
        identity = _file_identity(path)
        parquet = pq.ParquetFile(path)
        if (entry.get("status") != "ok" or entry.get("rows") != parquet.metadata.num_rows
                or not parquet.metadata.num_rows):
            raise ValueError(f"unverified month or wrong count: {path}")
        for col, kind in (("price", "double"), ("transact_time", "int64"), ("agg_trade_id", "int64")):
            if col not in parquet.schema_arrow.names or str(parquet.schema_arrow.field(col).type) != kind:
                raise ValueError(f"unexpected source schema: {path}: {col}")
        if verified_source is None:
            digest = parquet_content_hash(str(path))
        else:
            certificate = verified_source["files"][name]
            if tuple(certificate["identity"]) != identity or certificate["rows"] != entry["rows"]:
                raise ValueError(f"source changed since verification: {path}")
            digest = certificate["digest"]
        if digest != entry.get("parquet_content_hash"):
            raise ValueError(f"canonical input digest mismatch: {path}")
        files.append((path, entry, identity))
    if not files or len(files) != meta["input_files"]:
        raise ValueError("inconsistent source month count")
    fields = ["open_time", "close_time", "open", "high", "low", "close", "num_trades"]
    dib = pq.ParquetFile(base)
    counts = dib.read(columns=["num_trades"])["num_trades"].to_numpy()
    closed_rows = int(counts.sum())
    if (counts.dtype.kind != "i" or len(counts) != meta["total_bars"] or np.any(counts <= 0)
            or closed_rows + tail["num_trades"] != sum(entry["rows"] for _, entry, _ in files)):
        raise ValueError("DIB trade counts and tail do not partition canonical inputs")
    batches = _batches(files, batch_size, max_trade_gap_ms, known_gaps, closed_rows)
    current = None
    cursor = 0
    for batch in dib.iter_batches(columns=fields, batch_size=4096):
        for open_time, close_time, opened, high, low, close, remaining in zip(
                *(col.to_pylist() for col in batch.columns)):
            pieces = []
            while remaining:
                if current is None or cursor == len(current[0]):
                    current = next(batches, None)
                    if current is None:
                        raise ValueError("trade tape ended inside a completed DIB")
                    cursor = 0
                take = min(remaining, len(current[0]) - cursor)
                pieces.append(tuple(arr[cursor:cursor + take] for arr in current))
                cursor += take
                remaining -= take
            arrays = pieces[0] if len(pieces) == 1 else tuple(map(np.concatenate, zip(*pieces)))
            bar = TradeBar(open_time, close_time, opened, high, low, close, *arrays)
            bar.validate()
            yield bar
    for path, _, identity in files:
        if _file_identity(path) != identity:
            raise ValueError(f"canonical input changed during labeling: {path}")
