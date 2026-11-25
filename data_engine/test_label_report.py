"""Synthetic reports exercise zero, missing, failed, and audited label cells."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from data_engine.label_report import CONFIGS, SYMBOLS, _aggregate, _csv, _gap_inventory, _metrics, _stream


class ReportTests(unittest.TestCase):
    def test_natural_counts_durations_and_nulls(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.parquet"
            data = dict(decision_time=[1, 2, 3], decision_bar_index=[95, 96, 97],
                        entry_time=[1, 5, None], entry_price=[100., 102., None], vertical_bar_index=[103, 104, 105],
                        long_label=[1, 0, None], long_valid=[True, True, False],
                        long_reason=["target", "expiry", "no_entry"],
                        long_duration_bars=[2, 8, None], long_duration_ms=[5, 12, None], long_end_time=[6, 14, None],
                        short_label=[0, None, None], short_valid=[True, False, False],
                        short_reason=["stop", "data_gap", "no_entry"],
                        short_duration_bars=[1, None, None], short_duration_ms=[0, None, None],
                        short_end_time=[1, None, None])
            pq.write_table(pa.table(data), path)
            result = _metrics(path, {"status": "PASS"}, 3)
        self.assertEqual((result["long_positive"], result["long_negative"], result["long_prevalence"]), (1, 1, .5))
        self.assertEqual((result["short_positive"], result["short_prevalence"]), (0, 0.))
        self.assertEqual((result["valid_both"], result["valid_any"], result["entries"]), (1, 2, 2))
        self.assertEqual((result["short_reason_data_gap"], result["short_reason_expiry"]), (1, 0))
        self.assertEqual((result["short_duration_ms_median"], result["long_duration_bars_max"]), (0, 8))
        self.assertEqual((result["entry_same_ms"], result["entry_delay_ms_max"]), (1, 3))
        self.assertEqual((result["entry_price_observations"], result["entry_price_median"]), (2, 101.))
        self.assertEqual((result["long_duration_8_bar_count"], result["short_duration_8_bar_count"]), (1, 0))
        self.assertEqual(result["long_expiry_decision_to_end_ms_median"], 12)
        self.assertEqual(result["long_expiry_decision_to_end_bars_median"], 8)
        self.assertEqual(result["long_expiry_entry_to_end_ms_median"], 9)
        self.assertEqual(result["short_expiry_decision_to_end_ms_observations"], 0)
        self.assertIsNone(result["short_expiry_decision_to_end_ms_median"])

    def test_failed_and_missing_streams_do_not_become_zeros(self):
        source = dict(source_status="PASS", chronology_status="PASS", source_certificate="cert", preflight_report="pre")
        record = dict(asset="TEST", dib_config="fixed5k", generation="gen-1", bars=99, closed_trades=100,
                      candidate_samples=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = _stream(root, record, source, {})
            path = root / "reports/TEST/fixed5k.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(record, status="FAIL", stage="audit", error="first-touch mismatch")))
            failed = _stream(root, record, source, {})
        self.assertEqual((missing["status"], failed["status"], failed["error"]), ("MISSING", "FAIL", "first-touch mismatch"))
        self.assertIsNone(_aggregate([missing, failed])["long_positive"])
        csv_rows = list(csv.DictReader(io.StringIO(_csv([missing, dict(failed, short_positive=0)]))))
        self.assertEqual((csv_rows[0]["short_positive"], csv_rows[1]["short_positive"]), ("", "0"))

    def test_gap_inventory_must_match_frozen_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gaps.json"
            self.assertEqual(_gap_inventory(path, "m", "f")["status"], "MISSING")
            body = dict(status="PASS", manifest_sha256="m", freeze_sha256="f",
                        assets={asset: dict(asset=asset, counts={"time_inversion": 0, "invalid_id_order": 0},
                                            rows=1, months=1, unique_break_positions=0, skipped_id_values=0,
                                            unique_break_positions_in_closed_prefix_by_config=dict.fromkeys(CONFIGS, 0))
                                for asset in SYMBOLS})
            path.write_text(json.dumps(body))
            self.assertEqual(_gap_inventory(path, "m", "f")["status"], "PASS")
            self.assertEqual(_gap_inventory(path, "m", "other")["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
