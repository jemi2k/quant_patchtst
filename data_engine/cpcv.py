"""Build, validate and read the historical CPCV membership of frozen DIB decisions."""

import argparse
import csv
from datetime import date, datetime, timezone, timedelta
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_engine.cpcv_samples import load_stream, verify_sources
from data_engine.cpcv_split import GROUPS, PATHS, SPLITS, calendar_edges, prepare_stream, split_masks, validate_design
from data_engine.cpcv_support import count_cell, write_support
from data_engine.label_source import _sha256


EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
TASKS = ("long", "short")
FLAGS = ("train", "test", "purged", "embargoed")
IDENTITY = ("asset", "dib_config", "generation", "decision_bar_index")
STRUCTURAL = (*IDENTITY, "decision_time", "decision_ordinal", "input_first_bar_index", "input_last_bar_index",
              "input_first_time", "input_last_time", "complete_window")
END_FIELDS = ("label", "end_time", "end_ordinal", "end_bar_index")


def utc_ms(value):
    """A date is UTC midnight; a timestamp must specify its offset, at millisecond precision."""
    if len(value) == 10:
        stamp = datetime.combine(date.fromisoformat(value), datetime.min.time(), tzinfo=timezone.utc)
    else:
        stamp = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("timestamp requires a UTC offset")
    if stamp.microsecond % 1000:
        raise ValueError("holdout boundary must be an exact millisecond")
    return (stamp - EPOCH) // timedelta(milliseconds=1)


def _utc(value):
    return (EPOCH + timedelta(milliseconds=int(value))).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _values(table, name, fill=None):
    col = table[name]
    return (pc.fill_null(col, fill) if fill is not None else col).to_numpy()


def _pair_id(pair):
    return "G%d-G%d" % tuple(pair)


def _prepare(stream, edges):
    index = stream.index
    return prepare_stream(stream.close_times, _values(index, "decision_bar_index"),
                          _values(index, "long_valid"), _values(index, "short_valid"),
                          _values(index, "long_end_time", -1), _values(index, "short_end_time", -1),
                          edges, complete_window=_values(index, "complete_window"), dib_config=stream.record["dib_config"])


def _development_index(stream, prepared, bits, n):
    """Only pre-boundary decision and outcome data leave this function."""
    index = stream.index.slice(0, n)
    result = index.select(STRUCTURAL).append_column("group", pa.array(prepared.group[:n]))
    entry = index["entry_time"]
    before_cut = pc.less(entry, int(prepared.edges[-1]))
    deadline = prepared.decision_indices[:n] + 8
    deadline_known = deadline < len(prepared.close_times)
    deadline_known[deadline_known] &= prepared.close_times[deadline[deadline_known]] < prepared.edges[-1]
    for field in ("entry_time", "entry_ordinal"):
        result = result.append_column(field, pc.if_else(before_cut, index[field], pa.scalar(None, index[field].type)))
    for task in TASKS:
        for field in ("start_time", "start_ordinal"):
            key = task + "_" + field
            result = result.append_column(key, index[key])
        valid = pa.array(getattr(prepared, task + "_valid")[:n])
        censored = pa.array(getattr(prepared, task + "_censored")[:n])
        result = result.append_column(task + "_valid", valid).append_column(task + "_holdout_censored", censored)
        for field in END_FIELDS:
            key = task + "_" + field
            result = result.append_column(key, pc.if_else(valid, index[key], pa.scalar(None, index[key].type)))
        reason = task + "_reason"
        # Invalid source heads have no observation timestamp. Seal path-dependent
        # failure details when their full horizon was still unobserved at cutoff.
        uncertain = pc.and_(pa.array(~deadline_known), pc.is_in(index[reason], pa.array(
            ["data_gap", "no_entry", "incomplete_future", "invalid_barriers"])))
        visible_reason = pc.if_else(uncertain, pa.scalar("unresolved_at_cutoff"), index[reason])
        result = result.append_column(reason, pc.if_else(censored, pa.scalar("holdout_censored"), visible_reason))
    for flag in FLAGS:
        result = result.append_column(flag + "_splits", pa.array(bits[flag][:n]))
    return result.replace_schema_metadata(None)


def _configuration(root, cfg, records, edges):
    streams = {r["asset"]: load_stream(r) for r in records}
    prepared = {asset: _prepare(s, edges) for asset, s in streams.items()}
    bits, support, cells, labels = {}, [], {}, {}
    for asset, s in prepared.items():
        bits[asset] = {flag: np.zeros(len(s.group), dtype=np.uint32) for flag in FLAGS}
        # Group slices are consecutive even if two decisions share a close time.
        bounds = np.searchsorted(s.decision_time, edges, side="left")
        cells[asset] = {g: slice(bounds[g-1], bounds[g]) for g in GROUPS}
        support.append(count_cell(asset, cfg, None, 0, "holdout", complete=s.complete_window[bounds[-1]:]))
        index = streams[asset].index
        labels[asset] = {t + "_label": _values(index.slice(0, int(bounds[-1])), t + "_label", -1) for t in TASKS}

    for bit, pair in enumerate(SPLITS):
        flags = split_masks(prepared, pair)
        for asset, s in prepared.items():
            selection = flags[asset]
            for flag in FLAGS:
                bits[asset][flag][getattr(selection, flag)] |= np.uint32(1 << bit)
            for g, idx in cells[asset].items():
                support.append(count_cell(asset, cfg, _pair_id(pair), g, "test" if g in pair else "train",
                    complete=s.complete_window[idx], long_label=labels[asset]["long_label"][idx],
                    short_label=labels[asset]["short_label"][idx],
                    long_valid=s.long_valid[idx] | s.long_censored[idx], short_valid=s.short_valid[idx] | s.short_censored[idx],
                    long_censored=s.long_censored[idx], short_censored=s.short_censored[idx],
                    train=selection.train[idx], test=selection.test[idx],
                    purged=selection.purged[idx], embargoed=selection.embargoed[idx]))

    dev, hold = [], []
    for asset, s in prepared.items():
        stream = streams[asset]
        cut = int(np.searchsorted(s.decision_time, edges[-1]))
        dev.append(_development_index(stream, s, bits[asset], cut))
        hold.append(stream.index.slice(cut).select(STRUCTURAL).append_column(
            "group", pa.array(s.group[cut:])).replace_schema_metadata(None))
    for cohort, tables in (("development", dev), ("holdout", hold)):
        pq.write_table(pa.concat_tables(tables), root / cohort / (cfg + ".parquet"), compression="zstd",
                       row_group_size=32768)
    return support


def _method(edges, configs):
    return dict(version=1, holdout_start_ms=int(edges[-1]), holdout_start_utc=_utc(edges[-1]),
                configs=configs, group_anchor="earliest frozen DIB open across all streams",
                groups=[dict(id=g, start_ms=int(edges[g-1]), end_ms=int(edges[g]), start_utc=_utc(edges[g-1]),
                             end_utc=_utc(edges[g])) for g in GROUPS],
                splits=[dict(id=_pair_id(pair), bit=bit, test=list(pair), train=[g for g in GROUPS if g not in pair])
                        for bit, pair in enumerate(SPLITS)],
                paths={f"P{i}": [_pair_id(pair) for pair in path] for i, path in enumerate(PATHS, 1)},
                sample_key=list(IDENTITY), time_unit="UTC epoch milliseconds", group_intervals="[start,end)",
                input="96 consecutive gap-clean DIB log closes including decision, within one asset and generation",
                conditions=["shared dual-head PCGrad", "independent Long", "independent Short"],
                purge="closed interval intersection against pooled eligible test heads within this DIB configuration",
                embargo="any of the 96 actual input close times in either test calendar group or its test events",
                membership="unsigned 32-bit bitsets, one independent flag per lexicographic split, bit zero first",
                development_censor="disable each original valid head with end >= holdout_start; null its target and end",
                holdout="separate structural index; no outcomes in fold reports",
                runtime=dict(python=platform.python_version(), pyarrow=pa.__version__, numpy=np.__version__))


def _coverage_report(stage, rows):
    def columns(record, prefix=""):
        for name, value in record.items():
            key = prefix + name
            if isinstance(value, dict):
                yield from columns(value, key + "_")
            else:
                yield key, json.dumps(value) if isinstance(value, list) else value
                if key.endswith("_ms"):
                    yield key[:-3] + "_utc", _utc(value) if value is not None else None

    flattened = [dict(columns(record)) for record in rows]
    with (stage / "coverage_by_stream.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def build(root, records, provenance, holdout_ms, *, boundary_evidence=None, coverage=None):
    """Publish a new deterministic run directory. The caller provides verified source records."""
    from data_engine.cpcv_validate import validate_run_files

    validate_design()
    records = sorted(records, key=lambda r: (r["dib_config"], r["asset"]))
    if not records or len({(r["asset"],r["dib_config"]) for r in records}) != len(records):
        raise ValueError("expected unique input streams")
    configs = sorted({r["dib_config"] for r in records})
    root = Path(root)
    for protected in (Path("data/bars"), Path("data/labels"), Path("data/parquet"), Path("data/raw")):
        if protected.resolve() in (root.resolve(), *root.resolve().parents):
            raise ValueError("CPCV output cannot be published within the frozen sources")
    if root.exists():
        raise FileExistsError(root)
    edges = calendar_edges(min(r["history_start_ms"] for r in records), holdout_ms)
    if holdout_ms > max(r["last_close_ms"] for r in records):
        raise ValueError("holdout starts after all observed DIBs")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=".cpcv-") as temp:
        stage = Path(temp) / "run"
        for subdir in ("development", "holdout"):
            (stage / subdir).mkdir(parents=True)
        support = []
        for cfg in configs:
            support.extend(_configuration(stage, cfg, [r for r in records if r["dib_config"] == cfg], edges))
        reports = write_support(stage, support, boundaries=edges)
        if boundary_evidence is not None:
            if boundary_evidence["selected_holdout_ms"] != holdout_ms:
                raise ValueError("coverage decision differs from the configured cutoff")
            alternative, = [a for a in boundary_evidence["alternatives"] if a["holdout_ms"] == holdout_ms]
            coverage = alternative["streams"]
            _json(stage / "boundary_selection.json", boundary_evidence)
        if coverage is not None:
            _coverage_report(stage, coverage)
            with (stage / reports["report"]).open("a", encoding="utf-8") as handle:
                handle.write("\n## Holdout coverage\n\n[Development and holdout coverage for every asset and configuration]"
                             "(coverage_by_stream.csv).\n")
                if boundary_evidence is not None:
                    handle.write("\nThe [six-, nine- and twelve-month comparison](boundary_selection.json) "
                                 "uses timestamps, input continuity, available eight-bar horizons and validity masks.\n")
        result = validate_run_files(stage, edges, configs, records=records)
        _json(stage / "validation.json", result)
        manifest = _method(edges, configs)
        manifest.update(sources=records, source_receipts=provenance, support=reports,
                        validation="validation.json",
                        implementation_sha256={p.name: _sha256(p) for p in sorted(Path(__file__).parent.glob("cpcv*.py"))},
                        files={p.relative_to(stage).as_posix(): _sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()})
        _json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(_sha256(stage / "manifest.json") + "  manifest.json\n", encoding="ascii")
        verify_run(stage)
        os.rename(stage, root)
    return manifest


def _read_manifest(root):
    path = root / "manifest.json"
    if (root / "manifest.sha256").read_text() != _sha256(path) + "  manifest.json\n":
        raise ValueError("manifest checksum mismatch")
    manifest = json.loads(path.read_text())
    edges = calendar_edges(min(s["history_start_ms"] for s in manifest["sources"]), manifest["holdout_start_ms"])
    expected = _method(edges, manifest["configs"])
    for field in ("groups", "splits", "paths", "holdout_start_utc", "version"):
        if expected[field] != manifest[field]:
            raise ValueError(f"manifest has inconsistent {field}")
    return manifest


def verify_run(root):
    """Verify hashes of all published members and reject missing or extra files."""
    root = Path(root)
    manifest = _read_manifest(root)
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if files != set(manifest["files"]) | {"manifest.json", "manifest.sha256"}:
        raise ValueError("run directory does not match manifest file list")
    for name, digest in manifest["files"].items():
        if _sha256(root / name) != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
    return manifest


def _development(root, config, tasks):
    if tuple(tasks) not in (("long",), ("short",), TASKS):
        raise ValueError("invalid model tasks")
    root = Path(root)
    manifest = _read_manifest(root)
    if config not in manifest["configs"]:
        raise ValueError("unknown configuration")
    name = "development/" + config + ".parquet"
    if _sha256(root / name) != manifest["files"][name]:
        raise ValueError(f"artifact checksum mismatch: {name}")
    table = pq.ParquetFile(root / name).read()
    active = np.logical_or.reduce([_values(table, t + "_valid") for t in tasks])
    return table.filter(pa.array(active))


def read_fold(root, config, pair, role, *, tasks=TASKS):
    """Return eligible development rows with at least one requested task for a single condition."""
    pair = tuple(sorted(pair))
    if role not in ("train", "test") or pair not in SPLITS:
        raise ValueError("invalid split role or test pair")
    table = _development(root, config, tasks)
    mask = (_values(table, role + "_splits") & np.uint32(1 << SPLITS.index(pair))) != 0
    return table.filter(pa.array(mask))


def path_membership(root, config, path_number, *, tasks=TASKS):
    """Chronologically ordered keys and the designated OOS fold for a future prediction join."""
    if path_number not in range(1, 8):
        raise ValueError("path number must be 1 through 7")
    table = _development(root, config, tasks)
    table = table.filter(table["complete_window"])
    group = _values(table, "group")
    bits = _values(table, "test_splits")
    pair_ids = [""] * 9
    for pair in PATHS[path_number - 1]:
        members = (group == pair[0]) | (group == pair[1])
        if np.any((bits[members] & np.uint32(1 << SPLITS.index(pair))) == 0):
            raise ValueError("path row lacks its assigned OOS membership")
        pair_ids[pair[0]] = pair_ids[pair[1]] = _pair_id(pair)
    keys = table.select((*IDENTITY, "decision_time", "group", "long_valid", "short_valid"))
    keys = keys.append_column("oos_split", pc.take(pa.array(pair_ids), pa.array(group)))
    return keys.sort_by([(k, "ascending") for k in ("decision_time", "asset", "decision_bar_index")])


def main(argv=None):
    from data_engine.cpcv_coverage import coverage_rows, select_boundary
    from data_engine.config import DATA_END

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-start", required=True, type=utc_ms, help="UTC date or offset-bearing timestamp")
    parser.add_argument("--output", required=True, type=Path, help="new run directory")
    parser.add_argument("--document-selection", action="store_true",
                        help="verify that the explicit cutoff matches the coverage-based 6/9/12-month decision")
    args = parser.parse_args(argv)
    records, receipts = verify_sources()
    evidence = select_boundary(records, DATA_END) if args.document_selection else None
    coverage = coverage_rows(records, args.holdout_start) if evidence is None else None
    result = build(args.output, records, receipts, args.holdout_start, boundary_evidence=evidence, coverage=coverage)
    print(f"{len(result['sources'])} verified streams, {len(result['splits'])} splits and {len(result['paths'])} paths: {args.output}")


if __name__ == "__main__":
    main()
