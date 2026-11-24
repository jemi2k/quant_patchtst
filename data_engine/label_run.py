"""Freeze, generate, independently audit and report the approved historical label matrix."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_engine.config import MANIFEST_PATH, PARQUET_DIR, SYMBOLS
from data_engine.dib_configs import CONFIGS
from data_engine.ingest import parquet_content_hash
from data_engine.label_source import _file_identity, _sha256
from data_engine.labels import APPROVED_CONFIG


ROOT = Path("data/labels/approved-v2")
BARS = Path("data/bars")
MANIFEST = ROOT / "input_manifest.json"
PLAN = ROOT / "freeze.json"


def _load(path):
    return json.loads(path.read_bytes())


def _save(path, value, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        name = Path(temp.name)
        try:
            temp.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
            temp.flush()
            os.fsync(temp.fileno())
            if immutable:
                os.link(name, path)
            else:
                os.replace(name, path)
        finally:
            if name.exists():
                name.unlink()


def freeze():
    summary = _load(BARS / "matrix_summary.json")
    records = []
    for item in summary["combinations"]:
        asset, cfg, gen = item["symbol"], item["config"], item["generation"]
        directory = BARS / asset / cfg
        registry = _load(directory / "generations.json")
        base = directory / gen / f"{asset}-{cfg}-dib.parquet"
        if (item["status"] != "PASS" or registry["current"] != gen
                or (directory / "current").resolve() != base.parent.resolve()):
            raise ValueError(f"unverified generation {base}")
        artifacts, = [g["artifacts"] for g in registry["generations"] if g["id"] == gen]
        for filename, digest in artifacts.items():
            if _sha256(base.parent / filename) != digest:
                raise ValueError(f"bad registered artifact: {filename}")
        bars = pq.ParquetFile(base).metadata.num_rows
        if bars != item["bars"]:
            raise ValueError(f"DIB count differs from validated matrix: {base}")
        meta = _load(base.with_name(f"{asset}-{cfg}-dib-meta.json"))
        records.append(dict(asset=asset, dib_config=cfg, generation=gen, artifacts=artifacts,
                            bars=bars, candidate_samples=max(0, bars - 95), months=meta["input_months"],
                            closed_trades=item["input_rows"]-meta["tail"]["num_trades"]))
    if (len(records) != 35 or {(r["asset"],r["dib_config"]) for r in records}
            != {(a,c) for a in SYMBOLS for c in CONFIGS}):
        raise ValueError("incomplete DIB matrix")
    ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("xb") as output:
        output.write(Path(MANIFEST_PATH).read_bytes())
    _save(PLAN, dict(version=2, frozen_utc=datetime.now(timezone.utc).isoformat(),
                     parameters=asdict(APPROVED_CONFIG), manifest_sha256=_sha256(MANIFEST),
                     dynamic28k="dynema28k", entry="next_print", expiry="valid_zero",
                     multipliers={"long":[2,-1],"short":[-2,1]}, streams=records), immutable=True)


def _records():
    plan = _load(PLAN)
    if (plan["parameters"] != asdict(APPROVED_CONFIG) or plan["manifest_sha256"] != _sha256(MANIFEST)):
        raise ValueError("frozen run no longer matches parameters or manifest")
    return plan["streams"]


def verify_month(asset, name, manifest):
    path = Path(PARQUET_DIR) / asset / name
    entry = manifest[f"{asset}/{name[-15:-8]}"]
    identity = _file_identity(path)
    rows = pq.ParquetFile(path).metadata.num_rows
    if entry["status"] != "ok" or rows != entry["rows"]:
        raise ValueError(f"source coverage mismatch: {path}")
    digest = parquet_content_hash(str(path))
    if digest != entry["parquet_content_hash"] or _file_identity(path) != identity:
        raise ValueError(f"source changed or content hash mismatch: {path}")
    return dict(identity=identity, digest=digest, rows=rows)


def verify(asset):
    from concurrent.futures import ThreadPoolExecutor
    streams = [r for r in _records() if r["asset"] == asset]
    months = streams[0]["months"]
    if not all(r["months"] == months for r in streams):
        raise ValueError("monthly coverage differs across configurations")
    manifest = _load(MANIFEST)
    if {name[-15:-8] for name in months} != {key.split("/")[1] for key in manifest if key.startswith(asset+"/")}:
        raise ValueError("DIB month set differs from the frozen manifest")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(zip(months, pool.map(lambda name: verify_month(asset, name, manifest), months)))
    _save(ROOT / "sources" / (asset+".json"), dict(manifest_sha256=_sha256(MANIFEST),
          asset=asset, files=results, status="PASS"))


def check_cross_head_implications(parquet):
    """Reaching +/-2 sigma necessarily closed the opposite head at its one-sigma stop."""
    columns = [task + "_" + field for task in ("long", "short")
               for field in ("label", "reason", "end_ordinal")]
    for batch in parquet.iter_batches(columns=columns, batch_size=16384):
        fields = {key: np.array(batch.column(key).to_pylist(), dtype=object) for key in columns}
        for task, opposite in (("long", "short"), ("short", "long")):
            wins = fields[task + "_label"] == 1
            opposing_exits = fields[opposite + "_end_ordinal"][wins]
            if (not np.all(fields[opposite + "_reason"][wins] == "stop")
                    or not np.all(fields[opposite + "_label"][wins] == 0)
                    or any(value is None for value in opposing_exits)
                    or not np.all(opposing_exits <= fields[task + "_end_ordinal"][wins])):
                raise ValueError(f"{task} target without prior or simultaneous {opposite} stop")


def run_one(asset, cfg, stage):
    from data_engine.label_output import LABEL_VERSION, write_generation_labels
    record, = [r for r in _records() if (r["asset"], r["dib_config"]) == (asset, cfg)]
    generation = record["generation"]
    status = ROOT / "reports" / asset / (cfg + ".json")
    pending = ROOT / "staging" / asset / (cfg + ".parquet")
    output = ROOT / "datasets" / asset / cfg / (generation + "-labels.parquet")
    info = dict(asset=asset, dib_config=cfg, generation=generation,
                candidate_samples=record["candidate_samples"])
    try:
        registry = _load(BARS / asset / cfg / "generations.json")
        artifacts, = [g["artifacts"] for g in registry["generations"] if g["id"] == generation]
        if artifacts != record["artifacts"]:
            raise ValueError("generation registry differs from frozen checksums")
        cap = _load(ROOT / "sources" / (asset+".json"))
        timing = _load(ROOT.parent / "preflight" / (asset+".json"))["generations"][cfg]
        if timing["generation"] != generation or timing["time_inversion_in_closed_bars"]:
            raise ValueError(f"timestamp preflight failed: {timing}")
        params = dict(bars_root=BARS, trades_root=PARQUET_DIR, manifest_path=MANIFEST, verified_source=cap)
        if stage == "generate":
            if output.exists():
                raise FileExistsError(output)
            pending.parent.mkdir(parents=True, exist_ok=True)
            count = write_generation_labels(pending, asset=asset, dib_config=cfg, generation=generation,
                                            config=APPROVED_CONFIG, **params)
            if count != record["candidate_samples"]:
                raise ValueError(f"unexpected candidate count {count}")
            _save(status, dict(info, status="AWAITING_INDEPENDENT_AUDIT"))
        else:
            from data_engine.label_audit import audit_output
            if output.exists():
                raise FileExistsError(output)
            parquet = pq.ParquetFile(pending)
            spec = json.loads(parquet.schema_arrow.metadata[b"label_spec"])
            expected = dict(asset=asset, dib_config=cfg, generation=generation,
                            version=LABEL_VERSION, parameters=asdict(APPROVED_CONFIG),
                            manifest_sha256=_sha256(MANIFEST), max_trade_gap_ms=None, known_gaps=[],
                            barriers="log_target_2_stop_1", expiry="valid_zero", tape="completed_DIBs_only")
            if (parquet.metadata.num_rows != record["candidate_samples"]
                    or any(spec.get(k) != val for k, val in expected.items())):
                raise ValueError("staged labels differ from frozen experiment")
            check_cross_head_implications(parquet)
            audit = audit_output(pending, **params)
            digest = _sha256(pending)
            output.parent.mkdir(parents=True, exist_ok=True)
            os.link(pending, output)
            _save(status, dict(info, status="PASS", cross_head_implications="PASS", audit=audit, sha256=digest,
                               bytes=output.stat().st_size, output=str(output)))
            pending.unlink()
        print(f"{stage} {asset} {cfg} OK", flush=True)
    except Exception as exc:
        if not output.exists():
            _save(status, dict(info, status="FAIL", stage=stage, error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze","verify","generate","audit"])
    parser.add_argument("--asset", choices=SYMBOLS)
    parser.add_argument("--config", choices=list(CONFIGS))
    args = parser.parse_args()
    pa.set_cpu_count(2)
    if args.stage == "freeze":
        freeze()
    elif args.stage == "verify":
        verify(args.asset)
    else:
        run_one(args.asset, args.config, args.stage)
