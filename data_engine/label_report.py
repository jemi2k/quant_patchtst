"""Summarize the frozen label run using published audits and source certificates."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_engine.config import SYMBOLS
from data_engine.dib_configs import CONFIGS


DEFAULT_ROOT = Path("data/labels/approved-v2")
TASKS = ("long", "short")
REASONS = ("target", "stop", "expiry", "insufficient_history", "invalid_volatility",
           "invalid_barriers", "no_entry", "data_gap", "incomplete_future")


def _json(path):
    with path.open() as handle:
        return json.load(handle)


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def _publish(path, text):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=f".{path.name}-", delete=False) as out:
        name = out.name
        try:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            os.unlink(name)
            raise
    try:
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _asset_sources(root, asset, monthly, manifest, manifest_sha):
    source_path = root / "sources" / (asset + ".json")
    pre_path = root.parent / "preflight" / (asset + ".json")
    result = {"source_certificate": str(source_path), "preflight_report": str(pre_path),
              "source_status": "MISSING", "chronology_status": "MISSING"}
    try:
        source = _json(source_path)
        files = source["files"]
        if (source["asset"] != asset or source["manifest_sha256"] != manifest_sha
                or source["status"] != "PASS" or set(monthly) != set(files)):
            raise ValueError("certificate does not match the frozen asset, manifest or months")
        unresolved, duplicate_ids = set(), set()
        for name, mark in files.items():
            month = manifest[f"{asset}/{name[-15:-8]}"]
            if (month["status"] != "ok" or month["rows"] != mark["rows"]
                    or month["parquet_content_hash"] != mark["digest"]):
                raise ValueError(f"canonical certificate differs from manifest: {name}")
            unresolved.update(item["agg_trade_id"] for item in month.get("repair", {}).get("unresolved", []))
            duplicate_ids.update(item["agg_trade_id"] for item in month.get("kept_conflicts", []))
        result.update(source_status="PASS", verified_months=len(files),
                      source_rows=sum(item["rows"] for item in files.values()),
                      documented_duplicate_ids=sorted(duplicate_ids), unresolved_ids=sorted(unresolved))
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result.update(source_status="MISSING" if not source_path.exists() else "FAIL", source_error=str(exc))
    try:
        timing = _json(pre_path)
        if timing["asset"] != asset or timing["canonical_input_months"] != len(monthly):
            raise ValueError("preflight does not describe frozen asset/months")
        if result["source_status"] == "PASS" and timing["canonical_input_rows"] != result["source_rows"]:
            raise ValueError("preflight and certificate row counts disagree")
        result.update(chronology_status="PASS" if timing["first_inversion"] is None and
                      timing["prefix_rows_checked"] == timing["canonical_input_rows"] else "INVERSION_OR_INCOMPLETE",
                      scanned_prefix_rows=timing["prefix_rows_checked"], first_inversion=timing["first_inversion"])
        return result, timing
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result.update(chronology_status="MISSING" if not pre_path.exists() else "FAIL", chronology_error=str(exc))
        return result, {}


def _flatten(mapping, prefix=""):
    for key, value in mapping.items():
        key = prefix + key
        if isinstance(value, dict):
            yield from _flatten(value, key + "_")
        else:
            yield key, value


def _total(mask):
    return pc.sum(pc.cast(mask, pa.int64())).as_py() or 0


def _distribution(array):
    count = len(array) - array.null_count
    quantiles = pc.quantile(array, q=[.5, .95]).to_pylist() if count else [None, None]
    return {"observations": count, "min": pc.min(array).as_py(), "median": quantiles[0],
            "p95": quantiles[1], "max": pc.max(array).as_py(), "mean": pc.mean(array).as_py()}


def _metrics(path, audit, expected):
    if not isinstance(audit, dict) or not audit:
        raise ValueError("missing independent audit")
    columns = ["decision_time", "decision_bar_index", "entry_time", "entry_price", "vertical_bar_index"]
    columns.extend(task + "_" + name for task in TASKS for name in
                   ("label", "reason", "valid", "duration_bars", "duration_ms", "end_time"))
    table = pq.ParquetFile(path).read(columns=columns)
    if table.num_rows != expected:
        raise ValueError("published sample count differs from the freeze")
    entry_delay = pc.subtract(table["entry_time"], table["decision_time"])
    metrics = dict(audited_samples=expected, entries=len(entry_delay)-entry_delay.null_count,
                   entry_same_ms=_total(pc.equal(entry_delay, 0)),
                   valid_both=_total(pc.and_(table["long_valid"], table["short_valid"])),
                   valid_any=_total(pc.or_(table["long_valid"], table["short_valid"])))
    metrics["no_observed_entry"] = expected - metrics["entries"]
    metrics["neither_head_valid"] = expected - metrics["valid_any"]
    for outcome, name in ((1, "positive"), (0, "negative")):
        long_matches, short_matches = (pc.fill_null(pc.equal(table[task + "_label"], outcome), False)
                                       for task in TASKS)
        metrics["both_" + name] = _total(pc.and_(long_matches, short_matches))
        metrics["any_observed_" + name] = _total(pc.or_(long_matches, short_matches))
    for key, val in _distribution(entry_delay).items():
        metrics["entry_delay_ms_" + key] = val
    entry_prices = _distribution(table["entry_price"])
    if metrics["entries"] != entry_prices["observations"]:
        raise ValueError("observed entry price/time counts differ")
    metrics.update(("entry_price_" + key, val) for key, val in entry_prices.items())
    horizon = pc.subtract(table["vertical_bar_index"], table["decision_bar_index"])
    metrics.update(scheduled_horizon_bars_min=pc.min(horizon).as_py(),
                   scheduled_horizon_bars_max=pc.max(horizon).as_py())
    for task in TASKS:
        positive = _total(pc.equal(table[task + "_label"], 1))
        negative = _total(pc.equal(table[task + "_label"], 0))
        valid = _total(table[task + "_valid"])
        counts = {item["values"]: item["counts"] for item in pc.value_counts(table[task+"_reason"]).to_pylist()}
        if (set(counts)-set(REASONS) or sum(counts.values()) != expected or valid != positive+negative
                or positive != counts.get("target", 0) or negative != counts.get("stop", 0)+counts.get("expiry", 0)):
            raise ValueError(f"{task} label/mask/reason counts disagree")
        values = dict(positive=positive, negative=negative, valid=valid, invalid=expected-valid,
                      prevalence=(positive/valid if valid else None))
        values.update(("reason_" + reason, counts.get(reason, 0)) for reason in REASONS)
        expired = pc.equal(table[task + "_reason"], "expiry")
        for unit in ("ms", "bars"):
            duration = table[task + "_duration_" + unit]
            stats = _distribution(duration)
            if stats["observations"] != valid:
                raise ValueError(f"{task} duration count differs from valid label count")
            values.update(("duration_" + unit + "_" + stat, value) for stat, value in stats.items())
            expiry_stats = _distribution(pc.filter(duration, expired))
            if expiry_stats["observations"] != counts.get("expiry", 0):
                raise ValueError(f"{task} missing duration for observed expiry")
            values.update(("expiry_decision_to_end_" + unit + "_" + stat, value)
                          for stat, value in expiry_stats.items())
        expiry_after_entry = pc.filter(pc.subtract(table[task+"_end_time"], table["entry_time"]), expired)
        values.update(("expiry_entry_to_end_ms_" + stat, value)
                      for stat, value in _distribution(expiry_after_entry).items())
        observed = {item["values"]: item["counts"] for item in pc.value_counts(
            pc.drop_null(table[task + "_duration_bars"])).to_pylist()}
        last = metrics["scheduled_horizon_bars_max"]
        if last is not None:
            if any(bars < 1 or bars > last for bars in observed):
                raise ValueError(f"{task} event duration outside scheduled horizon")
            values.update((f"duration_{bars}_bar_count", observed.get(bars, 0)) for bars in range(1, last+1))
        metrics.update((task + "_" + key, val) for key, val in values.items())
    metrics.update(_flatten(audit, "audit_"))
    return metrics


def _stream(root, record, source, timing):
    asset, condition, generation = (record[k] for k in ("asset", "dib_config", "generation"))
    status_path = root / "reports" / asset / (condition + ".json")
    suffix = Path("datasets") / asset / condition / (generation + "-labels.parquet")
    artifact = root / suffix
    row = {k: record[k] for k in ("asset", "dib_config", "generation", "bars", "closed_trades",
                                   "candidate_samples")}
    row.update(output=None, sha256=None, bytes=None, audit_report=str(status_path),
               expected_output=str(artifact), source_status=source["source_status"],
               chronology_status=source["chronology_status"], status="MISSING", error=None,
               source_certificate=source["source_certificate"], preflight_report=source["preflight_report"])
    stream_times = timing.get("generations", {}).get(condition, {})
    row["time_inversion_in_closed_bars"] = stream_times.get("time_inversion_in_closed_bars")
    row["tail_rows"] = stream_times.get("tail_rows")
    try:
        if not status_path.is_file():
            row["error"] = "No per-stream report"
            return row
        status = _json(status_path)
        for key in ("asset", "dib_config", "generation", "candidate_samples"):
            if status[key] != row[key]:
                raise ValueError(f"report {key} differs from freeze")
        if status["status"] != "PASS":
            row.update(status=status["status"], error=status.get("error"), stage=status.get("stage"))
            return row
        if source["source_status"] != "PASS" or source["chronology_status"] not in ("PASS", "INVERSION_OR_INCOMPLETE"):
            raise ValueError("missing or failed input certificate/preflight")
        if (stream_times["generation"] != generation or stream_times["closed_rows"] != record["closed_trades"]
                or source["scanned_prefix_rows"] < record["closed_trades"]
                or stream_times["time_inversion_in_closed_bars"] is not False):
            raise ValueError("preflight closed-prefix coverage differs from frozen generation")
        if Path(status["output"]).parts[-len(suffix.parts):] != suffix.parts:
            raise ValueError("audited output path differs from frozen stream")
        if not artifact.is_file() or artifact.stat().st_size != status["bytes"] or _digest(artifact) != status["sha256"]:
            raise ValueError("published label file missing or differs from the audited digest")
        audit = status["audit"]
        if (audit["status"] != "PASS" or status["cross_head_implications"] != "PASS"
                or audit["verified_decisions"] != record["candidate_samples"]
                or audit["verified_closed_trades"] != record["closed_trades"]):
            raise ValueError("replay coverage or cross-head checks differ from the frozen stream")
        metrics = _metrics(artifact, audit, record["candidate_samples"])
        row.update(metrics, status="PASS", output=str(artifact), sha256=status["sha256"], bytes=status["bytes"])
    except (ValueError, KeyError, OSError, TypeError) as exc:
        row.update(status="FAIL", error=f"report verification: {exc}")
    return row


def _aggregate(rows):
    audited = [row for row in rows if row["status"] == "PASS"]
    fields = ["audited_samples", "valid_both", "valid_any", "neither_head_valid", "entries", "no_observed_entry",
              "entry_same_ms", "both_positive", "both_negative", "any_observed_positive", "any_observed_negative"]
    for task in TASKS:
        fields.extend(f"{task}_{key}" for key in ("positive", "negative", "valid", "invalid"))
        fields.extend(f"{task}_reason_{reason}" for reason in REASONS)
    result = {name: sum(row[name] for row in audited) if audited else None for name in fields}
    for task in TASKS:
        valid = result[f"{task}_valid"]
        result[f"{task}_prevalence"] = result[f"{task}_positive"] / valid if valid else None
    return result


def _csv(rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value) if isinstance(value, (tuple, list, dict)) else value
                         for key, value in row.items()})
    return buffer.getvalue()


def _count(value):
    return "—" if value is None else format(value, ",")


def _share(value):
    return "—" if value is None else format(value, ".2%")


def _gap_inventory(path, manifest_sha, freeze_sha):
    try:
        data = _json(path)
        if (data["status"] != "PASS" or data["manifest_sha256"] != manifest_sha
                or data["freeze_sha256"] != freeze_sha or set(data["assets"]) != set(SYMBOLS)):
            raise ValueError("gap inventory did not pass or differs from the frozen inputs")
        assets = {}
        for asset, entry in data["assets"].items():
            if (entry["asset"] != asset or set(entry["unique_break_positions_in_closed_prefix_by_config"]) != set(CONFIGS)
                    or entry["counts"]["time_inversion"] or entry["counts"]["invalid_id_order"]):
                raise ValueError(f"invalid gap inventory for {asset}")
            assets[asset] = {k: entry[k] for k in ("counts", "rows", "months", "unique_break_positions",
                             "skipped_id_values", "unique_break_positions_in_closed_prefix_by_config")}
        return dict(path=str(path), sha256=_digest(path), status="PASS", assets=assets)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return dict(path=str(path), status="MISSING" if not path.exists() else "FAIL", error=str(exc))


def _markdown(report):
    totals, rows = report["audited_totals"], report["streams"]
    lines = ["# Historical DIB label report", "",
             f'**{report["status"]}: {report["passed"]}/{len(rows)} frozen streams passed.** '
             f'Totals describe the {_count(totals["audited_samples"])} published, audited candidates only.',
             "", "| Head | Valid | Positive | Negative | Positive / valid |",
             "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        lines.append(f'| {task.title()} | {_count(totals[task+"_valid"])} | {_count(totals[task+"_positive"])} | '
                     f'{_count(totals[task+"_negative"])} | {_share(totals[task+"_prevalence"])} |')
    lines.extend(["", f'Both heads valid: {_count(totals["valid_both"])}; '
                  f'at least one valid: {_count(totals["valid_any"])}. '
                  'A negative is an observed stop or expiry. A censored label is missing, not negative.',
                  "", "| Asset | DIB | Status | Candidates¹ | Valid Long / Short | Long + | Short + |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f'| {row["asset"]} | {row["dib_config"]} | {row["status"]} | {_count(row["candidate_samples"])} | '
                     f'{_count(row.get("long_valid"))} / {_count(row.get("short_valid"))} | '
                     f'{_share(row.get("long_prevalence"))} | {_share(row.get("short_prevalence"))} |')
    lines.extend(["", "¹ Candidates are expected from the frozen DIB count. Rates use valid heads only. "
                  "Each DIB configuration contributes separately; totals do not count unique market opportunities."])
    issues = [row for row in rows if row["status"] != "PASS"]
    if issues:
        lines.extend(["", "## Unpublished or failed streams", ""])
        lines.extend(f'- {r["asset"]}/{r["dib_config"]}: {r["status"]} ({r.get("error") or "no successful audit"})'
                     for r in issues)
    lines.extend(["", "## Evidence", "",
                  f'- [Source gaps, permitted duplicates and unresolved records](gaps.json): '
                  f'{report["gap_inventory"]["status"]}. This file includes source ordinals and UTC times. '
                  'A source break may censor several labels; the per-head data_gap reason counts measure affected decisions.',
                  "- [Per-stream CSV](per_stream.csv): positive/negative counts, prevalence, reason counts, "
                  "durations and bar-count histograms, observed entry prices and delays, "
                  "decision-to-expiry (ms/bars) and entry-to-expiry (ms) "
                  "for observed expiries only, preflight, paths and Parquet SHA-256. "
                  "A blank is unavailable; a zero is an observed count.",
                  "- [Summary JSON](summary.json): frozen specification, asset input checks, per-stream evidence "
                  "and totals. Inspect the referenced individual audits for detail.",
                  "- [Frozen inputs](freeze.json) and [canonical input manifest](input_manifest.json). "
                  "A clean order scan does not establish the absence of every possible source outage."])
    return "\n".join(lines) + "\n"


def write_report(root=DEFAULT_ROOT):
    """Publish three aggregate reports, retaining one row per frozen stream even on failure."""
    root = Path(root)
    plan_path, manifest_path = root / "freeze.json", root / "input_manifest.json"
    plan, manifest = _json(plan_path), _json(manifest_path)
    manifest_sha = _digest(manifest_path)
    if manifest_sha != plan["manifest_sha256"]:
        raise ValueError("input manifest differs from the freeze")
    freeze_sha = _digest(plan_path)
    gaps = _gap_inventory(root / "gaps.json", manifest_sha, freeze_sha)
    rows, sources, preflight = [], {}, {}
    seen = set()
    for record in plan["streams"]:
        asset, condition = record["asset"], record["dib_config"]
        key = (asset, condition)
        if key in seen:
            raise ValueError(f"duplicate stream in freeze: {key}")
        seen.add(key)
        if asset not in sources:
            sources[asset], preflight[asset] = _asset_sources(root, asset, record["months"], manifest, manifest_sha)
        stream = _stream(root, record, sources[asset], preflight[asset])
        observed_gaps = gaps.get("assets", {}).get(asset, {})
        stream.update(gap_inventory=gaps["path"], gap_inventory_status=gaps["status"],
                      source_break_positions_in_closed_tape=observed_gaps.get(
                          "unique_break_positions_in_closed_prefix_by_config", {}).get(condition))
        if (stream["status"] == "PASS" and gaps["status"] == "PASS"
                and stream["source_break_positions_in_closed_tape"] != stream["audit_break_positions_in_closed_bars"]):
            stream.update(status="FAIL", error="gap inventory and label replay disagree on closed-tape breaks")
        rows.append(stream)
    required = {(asset, condition) for asset in SYMBOLS for condition in CONFIGS}
    if seen != required:
        raise ValueError(f"freeze does not describe the 35 required streams: missing={sorted(required-seen)}, "
                         f"extra={sorted(seen-required)}")
    counts = dict(Counter(row["status"] for row in rows))
    passed = counts.get("PASS", 0)
    complete = len(rows) == passed and gaps["status"] == "PASS"
    report = dict(status="FAIL" if counts.get("FAIL") or gaps["status"] == "FAIL" else "PASS" if complete else "INCOMPLETE",
                  expected=len(rows), passed=passed, statuses=counts, audited_totals=_aggregate(rows),
                  frozen_spec={key: val for key, val in plan.items() if key != "streams"},
                  freeze_sha256=freeze_sha, input_manifest=str(manifest_path), sources=sources, streams=rows,
                  gap_inventory=gaps)
    _publish(root / "summary.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    _publish(root / "per_stream.csv", _csv(rows))
    _publish(root / "REPORT.md", _markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    report = write_report(parser.parse_args().root)
    print(f'{report["status"]}: {report["passed"]}/{report["expected"]} audited datasets')
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
