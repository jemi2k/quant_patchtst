"""End-to-end label publication on a miniature, generation-pinned price tape."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_engine.ingest import parquet_content_hash
from data_engine.label_output import write_generation_labels
from data_engine.labels import LabelConfig


class OutputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.kw = dict(asset="TEST", dib_config="fixed5k", generation="gen-fixture",
                       config=LabelConfig(), bars_root=root / "bars", trades_root=root / "trades",
                       manifest_path=root / "manifest.json", batch_size=13)
        self.destination = root / "out.parquet"
        self.gen = root / "bars/TEST/fixed5k/gen-fixture"
        self.gen.mkdir(parents=True)
        self.dib = self.gen / "TEST-fixed5k-dib.parquet"
        tape = root / "trades/TEST/TEST-aggTrades-2020-01.parquet"
        tape.parent.mkdir(parents=True)
        prices = [100. + i % 2 for i in range(96)] + [100., 101.5, 200.]
        times = np.arange(len(prices), dtype=np.int64) + 1577836800000
        pq.write_table(pa.table(dict(price=prices, transact_time=times,
                                     agg_trade_id=np.arange(len(times)))), tape)
        self.kw["manifest_path"].write_text(json.dumps({"TEST/2020-01": {
            "status": "ok", "rows": len(prices), "parquet_content_hash": parquet_content_hash(tape)}}))
        self.bar_data = dict(open_time=times[:97], close_time=list(times[:96]) + [times[97]],
                             open=prices[:97], high=prices[:96] + [101.5], low=prices[:97],
                             close=prices[:96] + [101.5], num_trades=[1] * 96 + [2])
        tail = {"num_trades": 1}
        self.dib.with_suffix(".parquet.tail.json").write_text(json.dumps(tail))
        (self.gen / "TEST-fixed5k-dib-meta.json").write_text(json.dumps(dict(
            symbol="TEST", config={"id": "fixed5k"}, input_files=1,
            input_months=[tape.name], total_bars=97, tail=tail)))
        self.seal()

    def seal(self):
        pq.write_table(pa.table(self.bar_data), self.dib)
        artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.gen.iterdir()}
        (self.gen.parent / "generations.json").write_text(json.dumps({
            "generations": [{"id": "gen-fixture", "artifacts": artifacts}]}))

    def test_final_file_preserves_null_mask_provenance_and_no_tail_outcome(self):
        self.assertEqual(write_generation_labels(self.destination, **self.kw), 2)
        output = pq.read_table(self.destination)
        first, last = output.to_pylist()
        self.assertEqual((first["short_label"], first["short_reason"]), (0, "stop"))
        self.assertIsNone(first["long_label"])
        self.assertIsNone(first["long_end_time"])
        for head in ("long", "short"):
            self.assertEqual(first[head + "_start_time"], first["decision_time"])
            self.assertEqual(first[head + "_start_ordinal"], 95)
        self.assertEqual((first["short_end_bar_index"], first["short_duration_ms"],
                          first["short_duration_bars"]), (96, 2, 1))
        self.assertIsNone(first["long_end_bar_index"])
        self.assertIsNone(first["long_duration_ms"])
        self.assertIsNone(first["long_duration_bars"])
        self.assertEqual((first["long_valid"], first["long_reason"]), (False, "incomplete_future"))
        self.assertIsNone(last["entry_price"])
        spec = json.loads(output.schema.metadata[b"label_spec"])
        self.assertEqual(spec["version"], 2)
        self.assertEqual(spec["parameters"]["horizon_bars"], 8)
        self.assertEqual(spec["generation"], "gen-fixture")
        self.assertEqual(spec["manifest_sha256"], hashlib.sha256(self.kw["manifest_path"].read_bytes()).hexdigest())
        previous = self.destination.read_bytes()
        with self.assertRaises(FileExistsError):
            write_generation_labels(self.destination, **self.kw)
        self.assertEqual(self.destination.read_bytes(), previous)

    def test_writer_rejects_unapproved_label_configuration(self):
        self.kw["config"] = LabelConfig(horizon_bars=7)
        with self.assertRaisesRegex(ValueError, "approved"):
            write_generation_labels(self.destination, **self.kw)
        self.assertFalse(self.destination.exists())

    def test_late_source_error_publishes_nothing(self):
        self.bar_data["high"][-1] = 200.
        self.seal()
        with self.assertRaises(ValueError):
            write_generation_labels(self.destination, **self.kw)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.destination.parent.glob(".labels-*")))


if __name__ == "__main__":
    unittest.main()
