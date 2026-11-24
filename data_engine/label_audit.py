"""Independent first-passage replay for the frozen, zero-delay historical label contract.

Only the verified DIB/aggregate-trade reader is shared with the writer. For
each future bar we keep record highs and lows up to its first continuity
break. Binary searches on these ladders recover exact earliest crossing rows.
"""

from collections import deque
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data_engine.label_source import iter_generation_bars, _sha256
from data_engine.label_output import label_schema


def _ladder(prices, times, offset):
    running = np.maximum.accumulate(prices)
    marks = np.r_[True, running[1:] > running[:-1]]
    indices = np.flatnonzero(marks)
    return prices[indices], times[indices], offset + indices


class PassageBar:
    def __init__(self, bar, index, offset):
        self.index = index
        self.first = (int(bar.open_time), offset)
        self.last = (int(bar.close_time), offset + len(bar.times) - 1)
        self.open = float(bar.open)
        self.has_gap = bool(bar.breaks.any())
        length = int(bar.breaks.argmax()) if self.has_gap else len(bar.times)
        self.entry_clean = length > 0
        self.up = self.down = None
        if length:
            tape = bar.prices[:length]
            self.up = _ladder(tape, bar.times[:length], offset)
            self.down = _ladder(-tape, bar.times[:length], offset)

    def touch(self, level, up):
        ladder = self.up if up else self.down
        threshold = level if up else -level
        if ladder is not None and ladder[0][-1] >= threshold:
            idx = int(ladder[0].searchsorted(threshold))
            return int(ladder[2][idx]), int(ladder[1][idx])
        return None


def _expected(decision, future, sigma, history_start, history_time):
    result = dict(decision_bar_index=decision.index, decision_time=decision.last[0],
                  decision_ordinal=decision.last[1], vertical_bar_index=decision.index + 8,
                  history_start_bar_index=history_start, history_start_time=history_time,
                  sigma=sigma, entry_time=None, entry_ordinal=None, entry_price=None)
    entry = None
    if sigma is None:
        common_reason = "insufficient_history"
    elif sigma == 0:
        common_reason = "invalid_volatility"
    elif not future:
        common_reason = "no_entry"
    elif not future[0].entry_clean:
        common_reason = "data_gap"
    else:
        common_reason = None
        entry = future[0].open
        result.update(entry_time=future[0].first[0], entry_ordinal=future[0].first[1], entry_price=entry)
    if entry is not None:
        try:
            lower, lower_stop, upper_stop, upper = (
                entry * math.exp(exponent * sigma) for exponent in (-2, -1, 1, 2))
            representable = 0 < lower < lower_stop < entry < upper_stop < upper < math.inf
        except OverflowError:
            representable = False
        if not representable:
            common_reason, entry = "invalid_barriers", None
    for name, direction in (("long", 1), ("short", -1)):
        target = stop = end_time = end_ordinal = end_index = label = None
        reason = common_reason
        if entry is not None:
            target = entry * math.exp(direction * 2 * sigma)
            stop = entry * math.exp(-direction * sigma)
            for bar in future:
                win = bar.touch(target, direction == 1)
                lose = bar.touch(stop, direction != 1)
                if win is not None or lose is not None:
                    won = lose is None or (win is not None and win[0] < lose[0])
                    label, reason = (1, "target") if won else (0, "stop")
                    end_ordinal, end_time = win if won else lose
                    end_index = bar.index
                    break
                if bar.has_gap:
                    reason = "data_gap"
                    break
            if reason is None:
                if len(future) == 8:
                    end_time, end_ordinal = future[-1].last
                    end_index, label, reason = future[-1].index, 0, "expiry"
                else:
                    reason = "incomplete_future"
        fields = dict(label=label, valid=label is not None, reason=reason, start_time=decision.last[0],
                      start_ordinal=decision.last[1], target_price=target, stop_price=stop,
                      end_time=end_time, end_ordinal=end_ordinal, end_bar_index=end_index,
                      duration_bars=None if end_index is None else end_index - decision.index,
                      duration_ms=None if end_time is None else end_time - decision.last[0])
        result.update((name + "_" + field, value) for field, value in fields.items())
    return result


def _rows(parquet):
    for batch in parquet.iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def audit_output(path, *, bars_root, trades_root, manifest_path, verified_source=None):
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if not schema.equals(label_schema(), check_metadata=False):
        raise ValueError("unexpected label schema")
    spec = json.loads((schema.metadata or {}).get(b"label_spec", b"{}"))
    policy = dict(version=2, parameters=dict(volatility_window=64, volatility_halflife=16.,
                  horizon_bars=8, entry_delay_ms=0, estimator="ew_rms"),
                  barriers="log_target_2_stop_1", expiry="valid_zero", tape="completed_DIBs_only",
                  time_unit="UTC epoch milliseconds", manifest_sha256=_sha256(Path(manifest_path)),
                  max_trade_gap_ms=None, known_gaps=[])
    if any(spec.get(key) != value for key, value in policy.items()):
        raise ValueError("historical label policy mismatch")
    identity = {key: spec[key] for key in ("asset", "dib_config", "generation")}
    bars = iter_generation_bars(**identity, bars_root=bars_root, trades_root=trades_root,
                                manifest_path=manifest_path, verified_source=verified_source)
    rows = _rows(parquet)
    windows, history = deque(), deque(maxlen=96)
    weight = np.asarray([2. ** ((age - 63) / 16.) for age in range(64)])
    weight /= weight.sum()
    offset = checked = gap_count = gap_bars = 0

    def compare():
        first, sigma, hist_index, hist_time = windows.popleft()
        if first.index < 95:
            return 0
        expected = _expected(first, [bar for bar, *_ in windows], sigma, hist_index, hist_time)
        expected.update(identity)
        actual = next(rows, None)
        if actual is None or expected.keys() != actual.keys():
            raise ValueError(f"missing label record or fields at {first.index}")
        for field, value in expected.items():
            saved = actual[field]
            if isinstance(value, float) and saved is not None:
                equal = math.isclose(saved, value, rel_tol=2e-13, abs_tol=0.)
            else:
                equal = saved == value
            if not equal:
                raise ValueError(f"{first.index}: {field}: expected {value}, found {saved}")
        return 1

    for index, raw in enumerate(bars):
        bar = PassageBar(raw, index, offset)
        offset += len(raw.times)
        if bar.has_gap:
            gap_count += int(raw.breaks.sum())
            gap_bars += 1
            history.clear()
        else:
            history.append((bar.index, bar.last[0], math.log(raw.close)))
        sigma = hist_index = hist_time = None
        if len(history) == 96:
            hist_index, hist_time, _ = history[0]
            log_prices = np.array([item[2] for item in history])[-65:]
            returns = log_prices[1:] - log_prices[:-1]
            sigma = math.sqrt(np.dot(weight, returns * returns))
        windows.append((bar, sigma, hist_index, hist_time))
        if len(windows) == 9:
            checked += compare()
    while windows:
        checked += compare()
    if next(rows, None) is not None or checked != parquet.metadata.num_rows:
        raise ValueError("extra or missing candidate rows")
    return dict(status="PASS", replay="record-extrema-first-passage-v1", verified_decisions=checked,
                verified_closed_trades=offset, break_positions_in_closed_bars=gap_count,
                completed_bars_with_breaks=gap_bars)
