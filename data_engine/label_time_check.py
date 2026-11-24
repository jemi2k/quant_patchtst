"""Find the first strict timestamp inversion on each DIB source tape, in stored row order."""

import argparse
import json
import os
from pathlib import Path
import tempfile
from time import monotonic

import numpy as np
import pyarrow.parquet as pq

from data_engine import config, dib_configs


def _source_plan(asset, manifest):
    names = None
    generations = {}
    for condition in dib_configs.CONFIGS:
        current = (Path(dib_configs.OUT_DIR) / asset / condition / "current").resolve(strict=True)
        meta = json.loads((current / f"{asset}-{condition}-dib-meta.json").read_text())
        if (meta["symbol"], meta["config"]["id"]) != (asset, condition):
            raise ValueError(f"metadata identity mismatch for {asset}/{condition}")
        if names is not None and names != meta["input_months"]:
            raise ValueError(f"input months differ across {asset} configurations")
        names = meta["input_months"]
        generations[condition] = {"generation": current.name, "tail_rows": meta["tail"]["num_trades"]}
    if not names or names != sorted(set(names)):
        raise ValueError(f"missing, duplicate, or unordered source names for {asset}")
    inputs = []
    for name in names:
        month = name[-15:-8]
        record = manifest[f"{asset}/{month}"]
        if name != f"{asset}-aggTrades-{month}.parquet" or record["status"] != "ok":
            raise ValueError(f"invalid canonical source {name}")
        inputs.append((Path(config.PARQUET_DIR) / asset / name, record["rows"]))
    total = sum(rows for _, rows in inputs)
    for details in generations.values():
        details["closed_rows"] = total - details["tail_rows"]
    return inputs, generations


def _observation(path, local, global_offset, timestamp):
    return {"file": str(path), "row_zero_based": local, "asset_row_zero_based": global_offset,
            "transact_time_ms": int(timestamp)}


def _scan(inputs, batch_size):
    previous = None
    global_start = 0
    for path, expected in inputs:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected or not expected:
            raise ValueError(f"manifest row count mismatch or empty file: {path}")
        local_start = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["transact_time"], use_threads=False):
            times = batch.column(0)
            if times.null_count:
                raise ValueError(f"null transact_time in {path}")
            values = times.to_numpy()
            if values.dtype != np.int64:
                raise ValueError(f"transact_time must be int64 in {path}")
            if previous is not None and values[0] < previous["transact_time_ms"]:
                return previous, _observation(path, local_start, global_start, values[0])
            falls = values[1:] < values[:-1]
            if falls.any():
                first = int(falls.argmax())
                return (_observation(path, local_start + first, global_start + first, values[first]),
                        _observation(path, local_start + first + 1, global_start + first + 1, values[first + 1]))
            local_start += len(values)
            global_start += len(values)
            previous = _observation(path, local_start - 1, global_start - 1, values[-1])
    return None


def check(asset, manifest, batch_size=1_000_000):
    started = monotonic()
    inputs, generations = _source_plan(asset, manifest)
    pair = _scan(inputs, batch_size)
    total_rows = sum(count for _, count in inputs)
    report = {"asset": asset, "canonical_input_months": len(inputs), "canonical_input_rows": total_rows,
              "batch_size": batch_size, "projected_columns": ["transact_time"], "generations": generations,
              "first_inversion": None, "prefix_rows_checked": total_rows}
    if pair:
        previous, current = pair
        report["first_inversion"] = {"previous": previous, "current": current,
                                     "backstep_ms": previous["transact_time_ms"] - current["transact_time_ms"]}
        report["prefix_rows_checked"] = current["asset_row_zero_based"] + 1
    for details in generations.values():
        details["time_inversion_in_closed_bars"] = bool(pair and
                                                       pair[1]["asset_row_zero_based"] < details["closed_rows"])
    report["elapsed_seconds"] = round(monotonic() - started, 3)
    return report


def _write_atomic(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                     delete=False) as temporary:
        json.dump(report, temporary, indent=2, allow_nan=False)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary.name, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", choices=config.SYMBOLS, default=config.SYMBOLS)
    parser.add_argument("--output", type=Path, default=Path("data/labels/preflight"))
    args = parser.parse_args()
    manifest = json.loads(Path(config.MANIFEST_PATH).read_text())
    blocked = False
    for asset in args.assets:
        report = check(asset, manifest)
        _write_atomic(args.output / f"{asset}.json", report)
        print(asset, report["first_inversion"], report["elapsed_seconds"], flush=True)
        blocked |= any(info["time_inversion_in_closed_bars"] for info in report["generations"].values())
    return int(blocked)


if __name__ == "__main__":
    raise SystemExit(main())
