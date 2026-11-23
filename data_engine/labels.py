"""Binary first-touch outcomes on completed DIBs. All times are UTC epoch ms."""

from collections import deque
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


INPUT_CLOSES = 96
TASKS = ("long", "short")


@dataclass(frozen=True)
class LabelConfig:
    """Research settings, in DIB event time unless marked ms."""

    volatility_window: int = 64
    volatility_halflife: float = 16.0
    horizon_bars: int = 8
    entry_delay_ms: int = 0
    estimator: str = "ew_rms"

    def __post_init__(self):
        for value, minimum in ((self.volatility_window, 2), (self.horizon_bars, 1),
                               (self.entry_delay_ms, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError("invalid volatility window, horizon or entry delay")
        if not math.isfinite(self.volatility_halflife) or self.volatility_halflife <= 0:
            raise ValueError("volatility_halflife must be finite and positive")
        if self.estimator not in ("ew_rms", "ew_std"):
            raise ValueError("estimator must be ew_rms or ew_std")


APPROVED_CONFIG = LabelConfig()


@dataclass(frozen=True)
class TradeBar:
    """One completed DIB and its consecutive archived price observations.

    breaks[j] means continuity is unverified immediately BEFORE observation j.
    The source adapter also marks observations identified as unreliable.
    """

    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    prices: np.ndarray
    times: np.ndarray
    breaks: np.ndarray

    def validate(self, previous_time=None):
        prices, times, breaks = self.prices, self.times, self.breaks
        if (prices.ndim != 1 or not prices.size or prices.shape != times.shape
                or prices.shape != breaks.shape or breaks.dtype != np.bool_
                or times.dtype.kind != "i"):
            raise ValueError("invalid trade array shapes or timestamp/break dtype")
        if not np.all(np.isfinite(prices)) or prices.min() <= 0:
            raise ValueError("nonpositive or nonfinite trade price")
        if (np.any(times[1:] < times[:-1])
                or (previous_time is not None and times[0] < previous_time)):
            raise ValueError("transaction time regressed; cannot certify first touch")
        if ((prices[0], prices.max(), prices.min(), prices[-1], times[0], times[-1])
                != (self.open, self.high, self.low, self.close, self.open_time, self.close_time)):
            raise ValueError("trade tape does not reproduce DIB OHLC/timestamps")


def _finish(row, task, reason, label=None, time=None, ordinal=None, bar_index=None):
    row.update({task + "_reason": reason, task + "_valid": label is not None,
                task + "_label": label, task + "_end_time": time,
                task + "_end_ordinal": ordinal, task + "_end_bar_index": bar_index,
                task + "_duration_ms": None if time is None else time - row["decision_time"],
                task + "_duration_bars": None if bar_index is None else bar_index - row["decision_bar_index"]})


def _invalidate_open(row, reason):
    for task in TASKS:
        if row[task + "_reason"] is None:
            _finish(row, task, reason)


def _advance(row, index, bar, offset, clean_end, config):
    start = 0
    if row["entry_ordinal"] is None:
        start = int(np.searchsorted(bar.times[:clean_end],
                                    row["decision_time"] + config.entry_delay_ms))
        entry_at_deadline = index == row["vertical_bar_index"] and start == len(bar.times) - 1
        if start < clean_end and not entry_at_deadline:
            entry, sigma = float(bar.prices[start]), row["sigma"]
            row.update(entry_ordinal=offset + start, entry_time=int(bar.times[start]),
                       entry_price=entry)
            try:
                levels = [entry * math.exp(k * sigma) for k in (-2, -1, 1, 2)]
            except OverflowError:
                levels = []
            if (not levels or not all(map(math.isfinite, levels))
                    or not 0 < levels[0] < levels[1] < entry < levels[2] < levels[3]):
                _invalidate_open(row, "invalid_barriers")
                return
            (row["short_target_price"], row["long_stop_price"],
             row["short_stop_price"], row["long_target_price"]) = levels
            start += 1
        else:
            start = clean_end

    prices = bar.prices[start:clean_end]
    for task in TASKS:
        if row[task + "_reason"] is not None:
            continue
        if prices.size:
            upper = row["long_target_price" if task == "long" else "short_stop_price"]
            lower = row["long_stop_price" if task == "long" else "short_target_price"]
            if bar.high >= upper or bar.low <= lower:
                touched = (prices >= upper) | (prices <= lower)
                hit = int(np.argmax(touched))
                if touched[hit]:
                    won = bool(prices[hit] >= upper) == (task == "long")
                    trade = start + hit
                    _finish(row, task, "target" if won else "stop", int(won),
                            int(bar.times[trade]), offset + trade, index)
                    continue
        if clean_end < len(bar.times):
            _finish(row, task, "data_gap")
        elif index == row["vertical_bar_index"]:
            if row["entry_ordinal"] is None:
                _finish(row, task, "no_entry")
            else:
                _finish(row, task, "expiry", 0, int(bar.close_time), offset + len(bar.times) - 1, index)


def iter_labels(bars, *, asset, dib_config, generation, config):
    """Yield ordered decision records, including censored rows, for one DIB stream.

    No row is emitted before the first 96 closes. Label fitting is unnecessary;
    every volatility estimate uses only closes up to the decision. Pending
    decisions hold metadata only; future trade arrays are processed one bar at a time.
    """
    lookback = max(INPUT_CLOSES, config.volatility_window + 1)
    closes = deque(maxlen=lookback)
    pending = deque()
    weights = np.exp2(-np.arange(config.volatility_window - 1, -1, -1)
                      / config.volatility_halflife)
    weights /= weights.sum()
    offset, last_time = 0, None
    for index, bar in enumerate(bars):
        bar.validate(last_time)
        last_time = int(bar.close_time)
        clean_end = int(np.argmax(bar.breaks)) if bar.breaks.any() else len(bar.times)
        for row in pending:
            if any(row[task + "_reason"] is None for task in TASKS):
                _advance(row, index, bar, offset, clean_end, config)
        offset += len(bar.times)
        if clean_end < len(bar.times):
            closes.clear()
        else:
            closes.append((index, last_time, math.log(bar.close)))
        if index >= INPUT_CLOSES - 1:
            row = dict(asset=asset, dib_config=dib_config, generation=generation,
                       decision_bar_index=index, decision_time=last_time, decision_ordinal=offset - 1,
                       history_start_time=None, history_start_bar_index=None,
                       entry_time=None, entry_ordinal=None, entry_price=None, sigma=None,
                       vertical_bar_index=index + config.horizon_bars)
            for task in TASKS:
                row[task + "_start_time"] = row["decision_time"]
                row[task + "_start_ordinal"] = row["decision_ordinal"]
                _finish(row, task, None)
                row[task + "_target_price"] = row[task + "_stop_price"] = None
            if len(closes) < lookback:
                _invalidate_open(row, "insufficient_history")
            else:
                row["history_start_bar_index"], row["history_start_time"], _ = closes[0]
                log_prices = np.fromiter((item[2] for item in closes), dtype=float)
                returns = np.diff(log_prices[-(config.volatility_window + 1):])
                if config.estimator == "ew_std":
                    returns -= np.dot(weights, returns)
                row["sigma"] = float(np.sqrt(np.dot(weights, returns * returns)))
                if not row["sigma"] > 0:
                    _invalidate_open(row, "invalid_volatility")
            pending.append(row)
        while pending and all(pending[0][task + "_reason"] is not None for task in TASKS):
            yield pending.popleft()
    for row in pending:
        _invalidate_open(row, "no_entry" if row["entry_ordinal"] is None else "incomplete_future")
        yield row
