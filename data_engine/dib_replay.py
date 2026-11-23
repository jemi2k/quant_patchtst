"""
data_engine/dib_replay.py
=========================
INDEPENDENT reference re-derivation of Dollar Imbalance Bars.

This module re-implements the DIB sampling algorithm from scratch (its own
numba kernel and its own EWMA functions - no imports from bars.py) so the
validator can verify the builder's outputs without trusting them:

  - walks the canonical input trades in file order
  - recomputes bar boundaries from the actual first threshold crossings
  - recomputes OHLC, timestamps, trade counts, base volume, quote volume,
    signed imbalance from the ACTUAL assigned trades
  - recomputes E_x (circular fast_ewma over the last `window` trade-level x_t)
    and the adaptive E_T (EWMA over the last `num_prev_bars` bar lengths,
    clamped to [min, max]) independently
  - reports input-ordering facts: decreasing agg ids, equal agg ids,
    large (>1s) transact_time inversions

The replay is numerically identical in design to bars.py's kernel (same
accumulation order), so results must match bit-for-bit on identical inputs;
the validator compares replay vs stored bars field by field.
"""

import glob
import os

import numpy as np
from numba import njit

from data_engine import config as data_config
from data_engine import dib_configs


@njit
def _replay_ewma_circular(buf, write_idx, window):
    """Independent fast_ewma over the circular buffer (alpha = 2/(window+1))."""
    alpha = 2.0 / (window + 1.0)
    decay = 1.0 - alpha
    inv_decay = 1.0 / decay
    num = 0.0
    den = 0.0
    w = decay ** (window - 1)
    for i in range(window):
        idx = write_idx + i
        if idx >= window:
            idx -= window
        num += buf[idx] * w
        den += w
        w *= inv_decay
    return num / den


@njit
def _replay_ewma_last(bar_len, n, num_prev_bars):
    """Independent EWMA over the last n bar lengths (alpha = 2/(npb+1))."""
    start = num_prev_bars - n
    decay = 1.0 - 2.0 / (num_prev_bars + 1.0)
    ewma_old = bar_len[start]
    weight = 1.0
    d = 1.0
    for i in range(1, n):
        d *= decay
        weight += d
        ewma_old = ewma_old * decay + bar_len[start + i]
    return ewma_old / weight


@njit
def _replay_batch(price, qty, buyer_maker, ttime, agg_id,
                  theta, T, E_T, E_x,
                  imb_buf, imb_write, imb_count,
                  bar_len, num_bars,
                  cur_open, cur_high, cur_low, cur_vol, cur_qvol, cur_open_time, cur_close_time,
                  o_open_time, o_close_time, o_open, o_high, o_low, o_close,
                  o_volume, o_qvolume, o_imbalance, o_ntrades,
                  o_E_T, o_E_x, o_threshold,
                  window, exp_ticks_init, min_bar_size, max_bar_size, num_prev_bars,
                  prev_id, id_decreasing, id_equal, equal_ids, n_equal_ids,
                  time_inversions, prev_tt):
    """Independent re-derivation of the DIB algorithm (fresh code, same formula)."""
    n = price.shape[0]
    n_bars = 0
    first = (T == 0)

    for i in range(n):
        # ordering facts
        if prev_id >= 0:
            if agg_id[i] < prev_id:
                id_decreasing += 1
            elif agg_id[i] == prev_id:
                id_equal += 1
                if n_equal_ids < equal_ids.shape[0]:
                    equal_ids[n_equal_ids] = agg_id[i]
                    n_equal_ids += 1
        prev_id = agg_id[i]
        if prev_tt >= 0 and ttime[i] < prev_tt - 1000:
            time_inversions += 1
        prev_tt = ttime[i]

        b = -1.0 if buyer_maker[i] else 1.0
        x = b * price[i] * qty[i]

        theta += x
        T += 1

        if first:
            cur_open = price[i]
            cur_high = price[i]
            cur_low = price[i]
            cur_open_time = ttime[i]
            cur_vol = 0.0
            cur_qvol = 0.0
            first = False
        else:
            if price[i] > cur_high:
                cur_high = price[i]
            if price[i] < cur_low:
                cur_low = price[i]
        cur_close = price[i]
        cur_close_time = ttime[i]
        cur_vol += qty[i]
        cur_qvol += price[i] * qty[i]

        imb_buf[imb_write] = x
        imb_write += 1
        if imb_write >= window:
            imb_write = 0
        imb_count += 1

        if imb_count == exp_ticks_init:
            E_x = _replay_ewma_circular(imb_buf, imb_write, window)

        if imb_count >= exp_ticks_init:
            threshold = E_T * abs(E_x)
            if abs(theta) > threshold:
                o_open_time[n_bars] = cur_open_time
                o_close_time[n_bars] = cur_close_time
                o_open[n_bars] = cur_open
                o_high[n_bars] = cur_high
                o_low[n_bars] = cur_low
                o_close[n_bars] = cur_close
                o_volume[n_bars] = cur_vol
                o_qvolume[n_bars] = cur_qvol
                o_imbalance[n_bars] = theta
                o_ntrades[n_bars] = T
                o_E_T[n_bars] = E_T
                o_E_x[n_bars] = E_x
                o_threshold[n_bars] = threshold
                n_bars += 1

                for j in range(num_prev_bars - 1):
                    bar_len[j] = bar_len[j + 1]
                bar_len[num_prev_bars - 1] = float(T)
                num_bars += 1
                nb = num_bars if num_bars < num_prev_bars else num_prev_bars
                E_T = _replay_ewma_last(bar_len, nb, num_prev_bars)
                if E_T > max_bar_size:
                    E_T = max_bar_size
                if E_T < min_bar_size:
                    E_T = min_bar_size

                E_x = _replay_ewma_circular(imb_buf, imb_write, window)

                theta = 0.0
                T = 0
                first = True
                cur_vol = 0.0
                cur_qvol = 0.0

    return (theta, T, E_T, E_x, imb_write, imb_count, num_bars,
            cur_open, cur_high, cur_low, cur_vol, cur_qvol, cur_open_time, cur_close_time, n_bars,
            prev_id, id_decreasing, id_equal, n_equal_ids, time_inversions, prev_tt)


def replay(symbol, cfg, parquet_dir=None, batch_size=None):
    """Independently re-derive bars for (symbol, cfg). Returns a dict with
    replayed bar arrays, the tail, and input-ordering facts."""
    from data_engine.bars import _resolve_params  # parameter resolution only
    p = _resolve_params(dict(cfg))
    window = p["expected_imbalance_window"]
    exp_init = p["exp_ticks_init"]
    npb = p["num_prev_bars"]
    batch_size = batch_size or p["batch_size"]
    min_bar = float(p["min_bar_size"])
    max_bar = float(p["max_bar_size"])

    import pyarrow.parquet as pq
    root = parquet_dir or data_config.PARQUET_DIR
    files = sorted(glob.glob(os.path.join(root, symbol, f"{symbol}-aggTrades-*.parquet")))
    if not files:
        raise ValueError(f"no monthly parquet for {symbol}")

    theta = 0.0
    T = 0
    E_T = float(p["E_T_init"])
    E_x = np.nan
    imb_buf = np.empty(window, dtype=np.float64)
    imb_write = 0
    imb_count = 0
    bar_len = np.zeros(npb, dtype=np.float64)
    num_bars = 0
    cur_open = 0.0
    cur_high = 0.0
    cur_low = 0.0
    cur_vol = 0.0
    cur_qvol = 0.0
    cur_open_time = 0
    cur_close_time = 0
    prev_id = -1
    prev_tt = -1
    id_decreasing = 0
    id_equal = 0
    equal_ids = np.empty(16, dtype=np.int64)
    n_equal_ids = 0
    time_inversions = 0

    cap = 1 << 24
    o_open_time = np.empty(cap, dtype=np.int64)
    o_close_time = np.empty(cap, dtype=np.int64)
    o_open = np.empty(cap, dtype=np.float64)
    o_high = np.empty(cap, dtype=np.float64)
    o_low = np.empty(cap, dtype=np.float64)
    o_close = np.empty(cap, dtype=np.float64)
    o_volume = np.empty(cap, dtype=np.float64)
    o_qvolume = np.empty(cap, dtype=np.float64)
    o_imbalance = np.empty(cap, dtype=np.float64)
    o_ntrades = np.empty(cap, dtype=np.int64)
    o_E_T = np.empty(cap, dtype=np.float64)
    o_E_x = np.empty(cap, dtype=np.float64)
    o_threshold = np.empty(cap, dtype=np.float64)
    chunks = []
    total_bars = 0
    total_rows = 0

    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_size):
            price = batch.column("price").to_numpy(zero_copy_only=False)
            qty = batch.column("quantity").to_numpy(zero_copy_only=False)
            bm = batch.column("is_buyer_maker").to_numpy(zero_copy_only=False)
            tt = batch.column("transact_time").to_numpy(zero_copy_only=False)
            aid = batch.column("agg_trade_id").to_numpy(zero_copy_only=False)
            total_rows += batch.num_rows
            (theta, T, E_T, E_x, imb_write, imb_count, num_bars,
             cur_open, cur_high, cur_low, cur_vol, cur_qvol, cur_open_time, cur_close_time, n_bars,
             prev_id, id_decreasing, id_equal, n_equal_ids, time_inversions, prev_tt) = _replay_batch(
                price, qty, bm, tt, aid, theta, T, E_T, E_x,
                imb_buf, imb_write, imb_count, bar_len, num_bars,
                cur_open, cur_high, cur_low, cur_vol, cur_qvol, cur_open_time, cur_close_time,
                o_open_time, o_close_time, o_open, o_high, o_low, o_close,
                o_volume, o_qvolume, o_imbalance, o_ntrades,
                o_E_T, o_E_x, o_threshold,
                window, exp_init, min_bar, max_bar, npb,
                prev_id, id_decreasing, id_equal, equal_ids, n_equal_ids,
                time_inversions, prev_tt)
            if n_bars > 0:
                chunks.append((
                    o_open_time[:n_bars].copy(), o_close_time[:n_bars].copy(),
                    o_open[:n_bars].copy(), o_high[:n_bars].copy(), o_low[:n_bars].copy(), o_close[:n_bars].copy(),
                    o_volume[:n_bars].copy(), o_qvolume[:n_bars].copy(), o_imbalance[:n_bars].copy(),
                    o_ntrades[:n_bars].copy(), o_E_T[:n_bars].copy(), o_E_x[:n_bars].copy(), o_threshold[:n_bars].copy()))
                total_bars += n_bars
                if total_bars > cap * 0.9:
                    raise RuntimeError("replay output cap exceeded")

    if chunks:
        bars = {
            "open_time": np.concatenate([c[0] for c in chunks]),
            "close_time": np.concatenate([c[1] for c in chunks]),
            "open": np.concatenate([c[2] for c in chunks]),
            "high": np.concatenate([c[3] for c in chunks]),
            "low": np.concatenate([c[4] for c in chunks]),
            "close": np.concatenate([c[5] for c in chunks]),
            "volume": np.concatenate([c[6] for c in chunks]),
            "quote_volume": np.concatenate([c[7] for c in chunks]),
            "imbalance": np.concatenate([c[8] for c in chunks]),
            "num_trades": np.concatenate([c[9] for c in chunks]),
            "E_T": np.concatenate([c[10] for c in chunks]),
            "E_x": np.concatenate([c[11] for c in chunks]),
            "threshold": np.concatenate([c[12] for c in chunks]),
        }
    else:
        bars = {k: np.empty(0) for k in
                ("open_time", "close_time", "open", "high", "low", "close",
                 "volume", "quote_volume", "imbalance", "num_trades", "E_T", "E_x", "threshold")}

    tail = {"num_trades": int(T), "volume": float(cur_vol), "quote_volume": float(cur_qvol),
            "imbalance": float(theta), "open_time": int(cur_open_time), "close_time": int(cur_close_time)}
    return {"bars": bars, "tail": tail, "input_rows": total_rows, "files": len(files),
            "id_decreasing": int(id_decreasing), "id_equal": int(id_equal),
            "equal_ids": [int(x) for x in equal_ids[:n_equal_ids]],
            "time_inversions": int(time_inversions)}
