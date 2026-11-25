"""Exercise the independent historical-label auditor on temporary synthetic tapes."""

import json
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_engine import test_label_output as fixtures
from data_engine.ingest import parquet_content_hash
from data_engine.label_audit import audit_output
from data_engine.label_output import write_generation_labels


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OutputTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path = self.fixture.destination
        self.kw = {key: self.fixture.kw[key] for key in ("bars_root", "trades_root", "manifest_path")}

    def audit(self):
        return audit_output(self.path, **self.kw)

    def generate(self, gap=False, tiny_sigma=False):
        bars = [[100. + idx % 2] for idx in range(96)]
        if tiny_sigma:
            bars = [[1.]] * 95 + [[np.nextafter(1., 2.)]]
        bars.append([100., 101.5 if gap else 103., 97., 100.])
        bars.extend([[100.]] * 10)
        closes = [len(prices) for prices in bars]
        tape_prices = [price for prices in bars for price in prices] + [200.]
        date = 1577836800000
        times = np.array([date + idx for idx, values in enumerate(bars) for _ in values]
                         + [date + len(bars)], dtype="int64")
        ids = np.arange(len(times))
        if gap:
            ids[98:] += 1
        path = self.kw["trades_root"] / "TEST/TEST-aggTrades-2020-01.parquet"
        pq.write_table(pa.table({"price": tape_prices, "transact_time": times, "agg_trade_id": ids}), path)
        self.kw["manifest_path"].write_text(json.dumps({"TEST/2020-01": {
            "status": "ok", "rows": len(times), "parquet_content_hash": parquet_content_hash(path)}}))
        self.fixture.bar_data = dict(
            open_time=date + np.arange(len(bars)), close_time=date + np.arange(len(bars)),
            open=[row[0] for row in bars], high=list(map(max, bars)), low=list(map(min, bars)),
            close=[row[-1] for row in bars], num_trades=closes)
        meta = self.fixture.gen / "TEST-fixed5k-dib-meta.json"
        info = json.loads(meta.read_text())
        info["total_bars"] = len(bars)
        meta.write_text(json.dumps(info))
        self.fixture.seal()
        write_generation_labels(self.path, **self.fixture.kw)
        return pq.read_table(self.path)

    def replace_rows(self, table, rows):
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), self.path)

    def test_valid_tied_tape_and_expiry_data(self):
        table = self.generate()
        first, second = table.to_pylist()[:2]
        self.assertEqual((first["long_reason"], first["short_reason"]), ("target", "stop"))
        self.assertEqual(second["long_reason"], "expiry")
        summary = self.audit()
        self.assertEqual(summary["verified_decisions"], 12)
        self.assertEqual(summary["verified_closed_trades"], 110)
        self.assertEqual(summary["break_positions_in_closed_bars"], 0)

    def test_plausible_alternative_first_touch_with_identical_timestamps_is_rejected(self):
        table = self.generate()
        rows = table.to_pylist()
        first = rows[0]
        first.update(long_label=0, long_reason="stop", long_end_ordinal=98,
                     short_label=1, short_reason="target", short_end_ordinal=98)
        self.replace_rows(table, rows)
        with self.assertRaises(ValueError):
            self.audit()

    def test_expiry_at_wrong_earlier_bar_even_if_reported_intervals_agree(self):
        table = self.generate()
        rows = table.to_pylist()
        expiry = rows[1]
        self.assertEqual(expiry["short_reason"], "expiry")
        for suffix in ("end_time", "end_bar_index", "end_ordinal", "duration_ms", "duration_bars"):
            expiry["short_" + suffix] -= 1
        self.replace_rows(table, rows)
        with self.assertRaises(ValueError):
            self.audit()

    def test_corrupted_identity_history_entry_levels_and_duration(self):
        table = self.generate()
        modifications = {"generation": "gen-other", "history_start_bar_index": 1,
                         "history_start_time": 1234, "decision_ordinal": 94,
                         "entry_time": 1235, "entry_ordinal": 98, "entry_price": 101.,
                         "sigma": .04, "long_target_price": 102.5, "short_stop_price": 101.5,
                         "long_start_time": 14, "short_start_ordinal": 94, "long_duration_ms": 25,
                         "short_duration_bars": 2, "short_valid": False}
        for field, wrong in modifications.items():
            with self.subTest(field=field):
                rows = table.to_pylist()
                self.assertNotEqual(rows[0][field], wrong)
                rows[0][field] = wrong
                self.replace_rows(table, rows)
                with self.assertRaises(ValueError):
                    self.audit()

    def test_dropped_repeated_and_reordered_rows_are_rejected(self):
        table = self.generate()
        rows = table.to_pylist()
        for variant in (rows[:-1], rows[:2] + rows[1:], rows[1:2] + rows[:1] + rows[2:]):
            with self.subTest(first=variant[0]["decision_bar_index"], count=len(variant)):
                self.replace_rows(table, variant)
                with self.assertRaises(ValueError):
                    self.audit()

    def test_invalid_head_end_values_and_invented_post_gap_outcome(self):
        table = self.generate(gap=True)
        rows = table.to_pylist()
        first = rows[0]
        self.assertEqual((first["long_reason"], first["short_reason"]), ("data_gap", "stop"))
        summary = self.audit()
        self.assertEqual(summary["break_positions_in_closed_bars"], 1)
        self.assertEqual(summary["completed_bars_with_breaks"], 1)
        fabricated = {key.replace("short_", "long_"): value for key, value in first.items()
                      if key.startswith("short_end_") or key.startswith("short_duration_")}
        for changes in ({"long_end_time": first["short_end_time"]},
                        dict(fabricated, long_valid=True, long_reason="stop", long_label=0)):
            with self.subTest(changes=changes):
                corrupt = table.to_pylist()
                corrupt[0].update(changes)
                self.replace_rows(table, corrupt)
                with self.assertRaises(ValueError):
                    self.audit()

    def test_wrong_type_and_modified_policy_or_provenance_rejected(self):
        table = self.generate()
        idx = table.schema.get_field_index("long_label")
        pq.write_table(table.set_column(idx, "long_label", table.column(idx).cast(pa.int32())), self.path)
        with self.assertRaises(ValueError):
            self.audit()
        for field, wrong in (("manifest_sha256", "f" * 64), ("version", 199), ("expiry", "valid_one")):
            with self.subTest(field=field):
                spec = json.loads(table.schema.metadata[b"label_spec"])
                spec[field] = wrong
                pq.write_table(table.replace_schema_metadata({b"label_spec": json.dumps(spec).encode()}),
                               self.path)
                with self.assertRaises(ValueError):
                    self.audit()

    def test_positive_scale_with_unrepresentable_barriers(self):
        table = self.generate(tiny_sigma=True)
        row = table.to_pylist()[0]
        self.assertGreater(row["sigma"], 0.)
        self.assertEqual(row["entry_price"], 100.)
        for head in ("long", "short"):
            self.assertEqual(row[head + "_reason"], "invalid_barriers")
            self.assertIsNone(row[head + "_target_price"])
            self.assertIsNone(row[head + "_stop_price"])
        self.assertEqual(self.audit()["verified_decisions"], len(table))


if __name__ == "__main__":
    unittest.main()
