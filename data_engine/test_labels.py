"""Deterministic, in-memory checks for the DIB first-touch label contract."""

import math
import unittest

import numpy as np

from data_engine.labels import LabelConfig, TradeBar, iter_labels


def bar(index, observations, *, times=None, breaks=None):
    prices = np.asarray(observations, dtype=float)
    if times is None:
        times = np.arange(len(prices), dtype=np.int64) + 100 * index
    times = np.asarray(times, dtype=np.int64)
    breaks = np.zeros(len(prices), dtype=bool) if breaks is None else np.asarray(breaks, dtype=bool)
    return TradeBar(int(times[0]), int(times[-1]), float(prices[0]), float(prices.max()),
                    float(prices.min()), float(prices[-1]), prices, times, breaks)


def history(size=96):
    return [bar(index, [100 * math.exp(.01 * (index % 5) + .0002 * index)])
            for index in range(size)]


def records(bars, config=None):
    return list(iter_labels(iter(bars), asset="SYNTHETIC", dib_config="fixed5k",
                            generation="gen-synthetic", config=config or LabelConfig()))


def sigma_reference(prefix, config):
    closes = [item.close for item in prefix[-(config.volatility_window + 1):]]
    returns = [math.log(current / previous) for previous, current in zip(closes[:-1], closes[1:])]
    weights = [2 ** ((i - len(returns) + 1) / config.volatility_halflife) for i in range(len(returns))]
    mean = (sum(w * r for w, r in zip(weights, returns)) / sum(weights)
            if config.estimator == "ew_std" else 0.)
    return math.sqrt(sum(w * (r - mean) ** 2 for w, r in zip(weights, returns)) / sum(weights))


class LabelTests(unittest.TestCase):
    def assert_outcome(self, row, head, reason, label, end_time=None, end_ordinal=None):
        self.assertEqual(row[head + "_reason"], reason)
        self.assertEqual(row[head + "_label"], label)
        self.assertIs(row[head + "_valid"], label is not None)
        self.assertEqual(row[head + "_start_time"], row["decision_time"])
        self.assertEqual(row[head + "_start_ordinal"], row["decision_ordinal"])
        self.assertEqual(row[head + "_end_time"], end_time)
        self.assertEqual(row[head + "_end_ordinal"], end_ordinal)
        self.assertEqual(row[head + "_duration_ms"],
                         None if end_time is None else end_time - row["decision_time"])
        if label is None:
            self.assertIsNone(row[head + "_end_bar_index"])
            self.assertIsNone(row[head + "_duration_bars"])

    def test_finite_weighted_sigma_reference_and_future_independence(self):
        prefix = history()
        for estimator in ("ew_rms", "ew_std"):
            with self.subTest(estimator=estimator):
                config = LabelConfig(estimator=estimator)
                expected = sigma_reference(prefix, config)
                for future_price in (120., 500.):
                    decision = records(prefix + [bar(96, [110., future_price])], config)[0]
                    self.assertAlmostEqual(decision["sigma"], expected, places=12)
                    self.assertEqual(decision["entry_price"], 110.)
                    self.assertAlmostEqual(decision["long_target_price"], 110 * math.exp(2 * expected))
                    self.assertAlmostEqual(decision["long_stop_price"], 110 * math.exp(-expected))
                    self.assertAlmostEqual(decision["short_target_price"], 110 * math.exp(-2 * expected))
                    self.assertAlmostEqual(decision["short_stop_price"], 110 * math.exp(expected))
                    self.assertEqual(decision["vertical_bar_index"], 103)
                    self.assertEqual((decision["asset"], decision["dib_config"], decision["generation"]),
                                     ("SYNTHETIC", "fixed5k", "gen-synthetic"))

    def test_first_eligible_decision_and_volatility_warmup(self):
        self.assertEqual(records(history(95)), [])
        config = LabelConfig(volatility_window=96)
        first, second = records(history(97), config)
        self.assertEqual(first["decision_bar_index"], 95)
        self.assert_outcome(first, "long", "insufficient_history", None)
        self.assertIsNone(first["sigma"])
        self.assertEqual(second["decision_bar_index"], 96)
        self.assertEqual(second["history_start_bar_index"], 0)
        self.assertAlmostEqual(second["sigma"], sigma_reference(history(97), config))

    def test_same_timestamp_collision_uses_trade_order_and_future_rows_only(self):
        prefix = history()
        prefix[-1] = bar(95, [9999., 1., prefix[-1].close], times=[9498, 9499, 9500])
        for prices, wins in (([100., 125., 75., 100.], (1, 0)), ([100., 75., 125., 100.], (0, 1))):
            with self.subTest(prices=prices):
                future = bar(96, prices, times=[9500] * 4)
                decision = records(prefix + [future])[0]
                self.assertEqual(decision["entry_price"], 100.)
                self.assertEqual(decision["decision_time"], decision["entry_time"])
                self.assertEqual(decision["decision_ordinal"], 97)
                self.assertEqual(decision["entry_ordinal"], 98)
                for head, outcome in zip(("long", "short"), wins):
                    self.assert_outcome(decision, head, "target" if outcome else "stop",
                                        outcome, 9500, 99)
                    self.assertEqual(decision[head + "_duration_bars"], 1)
                    self.assertEqual(decision[head + "_end_bar_index"], 96)

    def test_delayed_entry_skips_prior_extremes_and_can_be_unavailable(self):
        prefix = history()
        future = bar(96, [50., 200., 108., 108.], times=[9500, 9501, 9502, 9503])
        config = LabelConfig(entry_delay_ms=2, horizon_bars=1)
        row = records(prefix + [future], config)[0]
        self.assertEqual((row["entry_time"], row["entry_price"], row["entry_ordinal"]), (9502, 108., 98))
        for head in ("long", "short"):
            self.assert_outcome(row, head, "expiry", 0, 9503, 99)
        row = records(prefix + [future], LabelConfig(entry_delay_ms=4, horizon_bars=1))[0]
        for head in ("long", "short"):
            self.assert_outcome(row, head, "no_entry", None)

    def test_inclusive_touches_on_expiry_observation_outrank_timeout(self):
        prefix = history()
        config = LabelConfig(horizon_bars=1)
        levels = records(prefix + [bar(96, [100., 100.])], config)[0]
        for head in ("long", "short"):
            for reason, label in (("target", 1), ("stop", 0)):
                with self.subTest(head=head, reason=reason):
                    price = levels[f"{head}_{reason}_price"]
                    row = records(prefix + [bar(96, [100., price])], config)[0]
                    self.assert_outcome(row, head, reason, label, 9601, 97)

    def test_first_possible_entry_at_deadline_is_not_a_trading_opportunity(self):
        row = records(history() + [bar(96, [100.])], LabelConfig(horizon_bars=1))[0]
        self.assertIsNone(row["entry_price"])
        for head in ("long", "short"):
            self.assert_outcome(row, head, "no_entry", None)

    def test_independent_head_intervals_and_both_negative(self):
        prefix, config = history(), LabelConfig()
        sigma = sigma_reference(prefix, config)
        prices = [100., 100 * math.exp(-1.5 * sigma), 100 * math.exp(1.5 * sigma)]
        row = records(prefix + [bar(96, prices)], config)[0]
        self.assert_outcome(row, "long", "stop", 0, 9601, 97)
        self.assert_outcome(row, "short", "stop", 0, 9602, 98)

    def test_heads_record_distinct_bar_durations(self):
        prefix, config = history(), LabelConfig(horizon_bars=2)
        middle = 100 * math.exp(1.5 * sigma_reference(prefix, config))
        row = records(prefix + [bar(96, [100., middle]), bar(97, [100.])], config)[0]
        self.assert_outcome(row, "short", "stop", 0, 9601, 97)
        self.assert_outcome(row, "long", "expiry", 0, 9700, 98)
        self.assertEqual((row["short_end_bar_index"], row["short_duration_bars"]), (96, 1))
        self.assertEqual((row["long_end_bar_index"], row["long_duration_bars"]), (97, 2))

    def test_expiry_and_off_by_one_at_horizon(self):
        prefix = history()
        config = LabelConfig(horizon_bars=2)
        row = records(prefix + [bar(96, [100., 100.]), bar(97, [100.]), bar(98, [500., 50.])], config)[0]
        for head in ("long", "short"):
            self.assert_outcome(row, head, "expiry", 0, 9700, 98)

    def test_truncation_preserves_resolved_head_and_masks_open_head(self):
        prefix, config = history(), LabelConfig()
        middle = 100 * math.exp(1.5 * sigma_reference(prefix, config))
        row = records(prefix + [bar(96, [100., middle])], config)[0]
        self.assert_outcome(row, "short", "stop", 0, 9601, 97)
        self.assert_outcome(row, "long", "incomplete_future", None)
        last = records(prefix, config)[-1]
        for head in ("long", "short"):
            self.assert_outcome(last, head, "no_entry", None)

    def test_gap_censors_pending_head_before_price_and_resets_history(self):
        prefix, config = history(), LabelConfig()
        middle = 100 * math.exp(1.5 * sigma_reference(prefix, config))
        gap = bar(96, [100., middle, 200.], breaks=[False, False, True])
        later = [bar(i, [100. + i % 3]) for i in range(97, 193)]
        rows = records(prefix + [gap] + later, config)
        self.assertEqual([row["decision_bar_index"] for row in rows], list(range(95, 193)))
        self.assert_outcome(rows[0], "short", "stop", 0, 9601, 97)
        self.assert_outcome(rows[0], "long", "data_gap", None)
        for row in rows[1:-1]:
            self.assertEqual(row["long_reason"], "insufficient_history")
        self.assertEqual(rows[-1]["history_start_bar_index"], 97)
        self.assertGreater(rows[-1]["sigma"], 0.)

    def test_gap_before_entry_and_decision_with_zero_sigma(self):
        row = records(history() + [bar(96, [100., 500.], breaks=[True, False])])[0]
        for head in ("long", "short"):
            self.assert_outcome(row, head, "data_gap", None)
        self.assertIsNone(row["entry_ordinal"])
        flat = [bar(i, [100.]) for i in range(96)]
        row = records(flat + [bar(96, [100., 500.])])[0]
        for head in ("long", "short"):
            self.assert_outcome(row, head, "invalid_volatility", None)

    def test_trade_validation_detects_time_regression_and_ohlc_mismatch(self):
        prefix = history()
        wrong = bar(96, [100., 101.])
        wrong_high = TradeBar(wrong.open_time, wrong.close_time, wrong.open, 300.,
                              wrong.low, wrong.close, wrong.prices, wrong.times, wrong.breaks)
        for data in ([wrong_high], [bar(96, [100., 101.], times=[9600, 9599])],
                     [bar(96, [100.], times=[9499])]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                records(prefix + data)


if __name__ == "__main__":
    unittest.main()
