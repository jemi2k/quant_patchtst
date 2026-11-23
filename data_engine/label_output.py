"""Publish one reviewed label configuration for one explicit DIB generation."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from data_engine import config as source_config
from data_engine.label_source import iter_generation_bars, _sha256
from data_engine.labels import APPROVED_CONFIG, LabelConfig, TASKS, iter_labels


LABEL_VERSION = 2


def label_schema():
    fields = [(name, pa.string()) for name in ("asset", "dib_config", "generation")]
    fields.extend((name, pa.int64()) for name in
                  ("decision_bar_index", "decision_time", "decision_ordinal",
                   "history_start_bar_index", "history_start_time", "entry_time",
                   "entry_ordinal", "vertical_bar_index"))
    fields.extend((name, pa.float64()) for name in ("entry_price", "sigma"))
    for task in TASKS:
        fields.extend((task + "_" + name, dtype) for name, dtype in (
            ("label", pa.int8()), ("valid", pa.bool_()), ("start_time", pa.int64()),
            ("start_ordinal", pa.int64()), ("end_time", pa.int64()), ("end_ordinal", pa.int64()),
            ("end_bar_index", pa.int64()), ("duration_ms", pa.int64()), ("duration_bars", pa.int64()),
            ("reason", pa.string()),
            ("target_price", pa.float64()), ("stop_price", pa.float64())))
    return pa.schema(fields)


def write_generation_labels(destination, *, asset, dib_config, generation, config: LabelConfig,
                            bars_root="data/bars", trades_root=source_config.PARQUET_DIR,
                            manifest_path=source_config.MANIFEST_PATH, batch_size=131072,
                            max_trade_gap_ms=None, known_gaps=(), verified_source=None):
    """Explicit one-stream operation. Requires an existing output directory and new filename."""
    if config != APPROVED_CONFIG:
        raise ValueError("historical labels require the approved LabelConfig")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    known_gaps = tuple(known_gaps)
    manifest_path = Path(manifest_path)
    manifest_digest = _sha256(manifest_path)
    specification = dict(version=LABEL_VERSION, asset=asset, dib_config=dib_config, generation=generation,
                         parameters=asdict(config), manifest_sha256=manifest_digest,
                         max_trade_gap_ms=max_trade_gap_ms, known_gaps=known_gaps,
                         barriers="log_target_2_stop_1", expiry="valid_zero",
                         tape="completed_DIBs_only", time_unit="UTC epoch milliseconds")
    schema = label_schema().with_metadata(
        {b"label_spec": json.dumps(specification, sort_keys=True, allow_nan=False).encode()})
    bars = iter_generation_bars(asset, dib_config, generation, bars_root=bars_root,
                                trades_root=trades_root, manifest_path=manifest_path,
                                batch_size=batch_size, max_trade_gap_ms=max_trade_gap_ms,
                                known_gaps=known_gaps, verified_source=verified_source)
    rows = iter_labels(bars, asset=asset, dib_config=dib_config, generation=generation, config=config)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".labels-", suffix=".tmp") as tmp:
        count, buffer = 0, []
        with pq.ParquetWriter(tmp.name, schema, compression="zstd") as writer:
            for row in rows:
                buffer.append(row)
                count += 1
                if len(buffer) == 4096:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        if _sha256(manifest_path) != manifest_digest:
            raise ValueError("manifest changed during label generation")
        os.fsync(tmp.fileno())
        os.link(tmp.name, destination)
    return count
