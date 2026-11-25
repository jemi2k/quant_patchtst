"""Synthetic tests for the generation-pinned DIB/trade reader."""

import hashlib
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_engine.label_source import _file_identity, _sha256, iter_generation_bars
from data_engine.ingest import parquet_content_hash


class LabelSourceTests(unittest.TestCase):
    asset = "SOLUSDT"
    config = "fixed5k"
    generation = "gen-20000101T000000000000"
    duplicate_id = 926014272
    months = ("2025-06", "2025-07")
    cutoff = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trades_root = self.root / "trades"
        self.bars_root = self.root / "bars"
        config_dir = self.bars_root / self.asset / self.config
        gen = config_dir / self.generation
        gen.mkdir(parents=True)
        (config_dir / "current").symlink_to(self.generation)
        self.registry = config_dir / "generations.json"
        stem = f"{self.asset}-{self.config}-dib"
        self.parquet = gen / (stem + ".parquet")
        self.tail_path = gen / (stem + ".parquet.tail.json")
        self.meta_path = gen / (stem + "-meta.json")
        self.state_path = gen / (stem + ".parquet.state.npz")
        self.manifest_path = self.root / "manifest.json"
        (self.trades_root / self.asset).mkdir(parents=True)
        self.files = [self.trades_root / self.asset / f"{self.asset}-aggTrades-{month}.parquet"
                      for month in self.months]
        self.manifest = {f"{self.asset}/{month}": {"status": "ok", "rows": size}
                         for month, size in zip(self.months, (2, 5))}
        self.manifest[f"{self.asset}/2025-07"]["kept_conflicts"] = [
            {"agg_trade_id": self.duplicate_id, "underlying": [[1003, 1003], [1004, 1004]]}
        ]
        self.ids = [self.duplicate_id + delta for delta in (-3, -2, -1, 0, 0, 1, 2)]
        self.times = [self.cutoff + delta for delta in (-2, -1, 0, 1, 1, 3, 4)]
        self.prices = [100., 101., 98., 103., 90., 100., 150.]
        self.dib = {
            "open_time": [self.times[0], self.times[3]],
            "close_time": [self.times[2], self.times[5]],
            "open": [100., 103.], "high": [101., 103.], "low": [98., 90.],
            "close": [98., 100.], "num_trades": [3, 3],
        }
        self.tail = {"num_trades": 1, "open_time": self.times[-1], "close_time": self.times[-1],
                     "volume": 1., "quote_volume": 150., "imbalance": 150.}
        self.meta = {"symbol": self.asset, "config": {"id": self.config},
                     "input_months": [path.name for path in self.files], "input_files": 2,
                     "total_bars": 2, "tail": self.tail.copy()}
        np.savez(self.state_path, T=1)
        self.write_raw()
        self.write_artifacts()
        self.manifest_path.write_text(json.dumps(self.manifest))

    def write_raw(self, reseal=True):
        raw_trades = pa.table({
            "agg_trade_id": pa.array(self.ids, pa.int64()), "price": pa.array(self.prices, pa.float64()),
            "quantity": pa.array([1.] * 7), "first_trade_id": pa.array(range(1000, 1007), pa.int64()),
            "last_trade_id": pa.array(range(1000, 1007), pa.int64()),
            "transact_time": pa.array(self.times, pa.int64()), "is_buyer_maker": pa.array([False] * 7),
        })
        pq.write_table(raw_trades.slice(0, 2), self.files[0], row_group_size=1)
        pq.write_table(raw_trades.slice(2), self.files[1], row_group_size=2)
        if reseal:
            for path in self.files:
                month = path.stem.split("-aggTrades-")[1]
                self.manifest[f"{self.asset}/{month}"]["parquet_content_hash"] = parquet_content_hash(path)
            self.manifest_path.write_text(json.dumps(self.manifest))

    def write_artifacts(self):
        pq.write_table(pa.table(self.dib), self.parquet, row_group_size=1)
        self.meta_path.write_text(json.dumps(self.meta))
        self.tail_path.write_text(json.dumps(self.tail))
        artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (self.parquet, self.meta_path, self.tail_path, self.state_path)}
        self.registry.write_text(json.dumps({"current": self.generation, "generations": [
            {"id": self.generation, "artifacts": artifacts}]}))

    def read(self, generation=None, batch_size=2, max_trade_gap_ms=None, known_gaps=(), verified_source=None):
        return list(iter_generation_bars(
            self.asset, self.config, generation or self.generation, bars_root=self.bars_root,
            trades_root=self.trades_root, manifest_path=self.manifest_path,
            batch_size=batch_size, max_trade_gap_ms=max_trade_gap_ms, known_gaps=known_gaps,
            verified_source=verified_source))

    def test_source_certificate_is_bound_to_manifest_asset_and_file_identity(self):
        cert = dict(status="PASS", asset=self.asset, manifest_sha256=_sha256(self.manifest_path),
                    files={path.name: dict(rows=self.manifest[self.asset + "/" + month]["rows"],
                                          digest=parquet_content_hash(path), identity=_file_identity(path))
                           for month, path in zip(self.months, self.files)})
        self.assertEqual([len(bar.times) for bar in self.read(verified_source=cert)], [3, 3])
        for field in ("status", "asset", "manifest_sha256"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read(verified_source=dict(cert, **{field: "wrong"}))
        cert["files"][self.files[0].name]["identity"] = []
        with self.assertRaises(ValueError):
            self.read(verified_source=cert)

    def reject(self):
        with self.assertRaises((ValueError, OSError, RuntimeError)):
            self.read()

    def test_assignment_survives_file_row_group_and_batch_boundaries(self):
        for batch_size in (1, 2, 64):
            with self.subTest(batch_size=batch_size):
                bars = self.read(batch_size=batch_size)
                self.assertEqual(len(bars), 2)
                self.assertEqual([list(bar.prices) for bar in bars], [self.prices[:3], self.prices[3:6]])
                self.assertEqual([list(bar.times) for bar in bars], [self.times[:3], self.times[3:6]])
                self.assertFalse(any(bars[1].breaks))

    def test_reads_pinned_generation_when_current_moves(self):
        newer = "gen-20000102T000000000000"
        gen_dir = self.parquet.parent
        next_dir = gen_dir.parent / newer
        shutil.copytree(gen_dir, next_dir)
        new_bars = {"open_time": [self.times[0], self.times[2]], "close_time": [self.times[1], self.times[5]],
                    "open": [100., 98.], "high": [101., 103.], "low": [100., 90.],
                    "close": [101., 100.], "num_trades": [2, 4]}
        pq.write_table(pa.table(new_bars), next_dir / self.parquet.name)
        registry = json.loads(self.registry.read_text())
        registry["current"] = newer
        registry["generations"].append({"id": newer, "artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in next_dir.iterdir()}})
        self.registry.write_text(json.dumps(registry))
        current = self.registry.parent / "current"
        current.unlink()
        current.symlink_to(newer)
        self.assertEqual([len(bar.prices) for bar in self.read()], [3, 3])
        self.assertEqual([len(bar.prices) for bar in self.read(generation=newer)], [2, 4])
        with self.assertRaises((ValueError, OSError, RuntimeError)):
            self.read(generation="gen-unregistered")

    def test_digest_is_checked_for_each_label_source_artifact(self):
        for path in (self.parquet, self.meta_path, self.tail_path):
            with self.subTest(path=path.name):
                self.write_artifacts()
                registry = json.loads(self.registry.read_text())
                registry["generations"][0]["artifacts"][path.name] = "0" * 64
                self.registry.write_text(json.dumps(registry))
                self.reject()

    def test_metadata_identity_count_and_month_order(self):
        for key, wrong in (("symbol", "WRONG"), ("config", {"id": "fixed20k"}), ("total_bars", 3),
                           ("input_months", list(reversed(self.meta["input_months"])))):
            with self.subTest(key=key):
                previous, self.meta[key] = self.meta[key], wrong
                self.write_artifacts()
                self.reject()
                self.meta[key] = previous

    def test_manifest_missing_failed_or_wrong_row_count(self):
        key = f"{self.asset}/2025-06"
        entry = self.manifest[key]
        for replacement in (None, dict(entry, status="failed"), dict(entry, status="not_found"),
                            dict(entry, rows=3), {"status": "ok", "rows": 2}):
            with self.subTest(replacement=replacement):
                manifest = self.manifest.copy()
                if replacement is None:
                    del manifest[key]
                else:
                    manifest[key] = replacement
                self.manifest_path.write_text(json.dumps(manifest))
                self.reject()

    def test_missing_monthly_parquet(self):
        self.files[1].unlink()
        self.reject()

    def test_source_price_tamper_at_constant_record_count_and_unchanged_bars(self):
        self.dib = {"open_time": [self.times[0], self.times[5]], "close_time": [self.times[4], self.times[5]],
                    "open": [100., 100.], "high": [103., 100.], "low": [90., 100.],
                    "close": [90., 100.], "num_trades": [5, 1]}
        self.write_artifacts()
        self.assertEqual([len(bar.prices) for bar in self.read()], [5, 1])
        self.prices[1] = 100.5
        self.write_raw(reseal=False)
        self.reject()

    def test_nonconsecutive_months_even_when_manifest_and_metadata_agree(self):
        self.files[1].unlink()
        self.files[1] = self.files[1].with_name(self.files[1].name.replace("2025-07", "2025-08"))
        shift = 31 * 24 * 60 * 60 * 1000
        self.times[2:] = [time + shift for time in self.times[2:]]
        self.dib["open_time"][1] = self.times[3]
        self.dib["close_time"] = [self.times[2], self.times[5]]
        self.tail["open_time"] = self.tail["close_time"] = self.times[-1]
        self.meta["tail"] = self.tail.copy()
        self.meta["input_months"][1] = self.files[1].name
        self.manifest[f"{self.asset}/2025-08"] = self.manifest.pop(f"{self.asset}/2025-07")
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.write_raw()
        self.write_artifacts()
        self.reject()

    def test_bar_reconciliation_survives_resealed_artifact(self):
        changes = (("open", [100.5, 103.]), ("high", [100.5, 103.]), ("low", [97., 90.]),
                   ("close", [99., 100.]), ("open_time", [self.times[0] - 1, self.times[3]]),
                   ("close_time", [self.times[2] + 1, self.times[5]]), ("num_trades", [2, 4]))
        for name, wrong in changes:
            with self.subTest(field=name):
                previous, self.dib[name] = self.dib[name], wrong
                self.write_artifacts()
                self.reject()
                self.dib[name] = previous

    def test_tail_and_total_must_reconcile_even_if_both_json_files_agree(self):
        self.tail["num_trades"] = self.meta["tail"]["num_trades"] = 2
        self.write_artifacts()
        self.reject()

    def test_unlisted_duplicate_across_batches_is_rejected(self):
        self.manifest[f"{self.asset}/2025-07"]["kept_conflicts"] = []
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.reject()

    def test_trade_time_regression_at_bar_boundary_is_rejected(self):
        self.times[3] = self.times[2] - 1
        self.dib["open_time"][1] = self.times[3]
        self.write_raw()
        self.write_artifacts()
        self.reject()

    def test_regressing_id_across_month_boundary_is_rejected(self):
        self.ids[1] = self.ids[2] + 1
        self.write_raw()
        self.reject()

    def test_id_and_elapsed_gaps_mark_the_exact_rows(self):
        self.ids[0] -= 1
        self.ids[1] -= 1
        self.write_raw()
        bars = self.read(batch_size=1, max_trade_gap_ms=1)
        self.assertEqual(list(bars[0].breaks)[1:], [False, True])
        self.assertEqual(list(bars[1].breaks), [False, False, True])

    def test_unresolved_record_in_manifest_and_external_gap_protect_separate_rows(self):
        self.manifest[f"{self.asset}/2025-07"]["repair"] = {
            "unresolved": [{"agg_trade_id": self.ids[2], "reason": "synthetic uncertainty"}]}
        self.manifest_path.write_text(json.dumps(self.manifest))
        bars = self.read(known_gaps=[(self.cutoff + 2, self.cutoff + 3)])
        self.assertEqual(list(bars[0].breaks)[1:], [False, True])
        self.assertEqual(list(bars[1].breaks), [False, False, True])


if __name__ == "__main__":
    unittest.main()
