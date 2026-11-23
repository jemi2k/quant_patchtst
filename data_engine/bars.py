"""
data_engine/bars.py  (v2)
=========================
Dollar Imbalance Bar (DIB) engine - faithful port of the AFML / mlfinlab EMA
dollar imbalance bars (Advances in Financial Machine Learning, ch. 2.3).

Sampling rule (per asset):
  - sign             b_t = -1 if is_buyer_maker else +1   (Binance native aggressor,
                                                          intentionally retained instead of the tick rule)
  - dollar imbalance x_t = b_t * price_t * quantity_t
  - cumulative       theta = sum(x_t)
  - emit a bar when  |theta| > E_T * |E_x|                (STRICT '>', reference source)

  E_x = EWMA over the most recent `expected_imbalance_window` trade-level
        imbalance observations (mlfinlab fast_ewma.ewma semantics).
        DOCUMENTED ADAPTATION: E_x is refreshed at each DIB close (the
        reference refreshes per tick bar).
  E_T = EWMA over the last `num_prev_bars` completed bar lengths,
        clamped to [min_bar_size, max_bar_size].
        DOCUMENTED ADAPTATION: the dynamic configuration caps E_T at 28,000
        (project-specific stability constraint; the reference default is
        unconstrained).

Warm-up is causal: E_x is NaN until `exp_ticks_init` trades have been observed,
so no bar can form before that point; the warm-up trades belong to the first
bar. No statistics are precomputed and reprocessed.

EOF behavior: the final incomplete bar is NOT force-closed. Its remaining
trade count, base volume, quote volume, signed imbalance, and timestamp range
are recorded in a `.tail.json` file, and the full continuation state
(accumulators + EWMA buffers) is saved to a `.state.npz` file so that a future
run can resume exactly where this one stopped.

Every completed bar records E_T, E_x, and the closing threshold
(= E_T * |E_x|) used at emission, for all configurations.

Configuration matrix: data_engine/dib_configs.py (single source of truth).

Run:
    python -m data_engine.bars --test                      # EWMA reference checks
    python -m data_engine.bars --symbols ETHUSDT           # default config (dynema28k)
    python -m data_engine.bars --all --config fixed10k
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit

from data_engine import config as data_config
from data_engine import dib_configs

# Backward-compatible aliases (values come from dib_configs).
EXP_NUM_TICKS_INIT = dib_configs.EXP_NUM_TICKS_INIT
EXPECTED_IMBALANCE_WINDOW = dib_configs.EXPECTED_IMBALANCE_WINDOW
NUM_PREV_BARS = dib_configs.NUM_PREV_BARS
MAX_BAR_SIZE = dib_configs.CONFIGS["dynema28k"]["max_bar_size"]
BATCH_SIZE = dib_configs.BATCH_SIZE
BAR_BUFFER = dib_configs.BAR_BUFFER

BAR_SCHEMA = pa.schema([
    pa.field("open_time", pa.int64()),
    pa.field("close_time", pa.int64()),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("volume", pa.float64()),
    pa.field("quote_volume", pa.float64()),
    pa.field("imbalance", pa.float64()),
    pa.field("num_trades", pa.int64()),
    pa.field("E_T", pa.float64()),
    pa.field("E_x", pa.float64()),
    pa.field("threshold", pa.float64()),
])


# --- EWMA helpers (mlfinlab fast_ewma semantics: y = sum_i arr[i]*(1-a)^(n-1-i) / sum_j (1-a)^j,
#     a = 2/(window+1)) -----------------------------------------------------------------

@njit
def _ewma_arr(arr, n, window):
    """Reproduce mlfinlab fast_ewma.ewma(arr, window) over arr[0..n-1]."""
    alpha = 2.0 / (window + 1.0)
    decay = 1.0 - alpha
    if n <= 0:
        return 0.0
    ewma_old = arr[0]
    weight = 1.0
    d = 1.0
    for i in range(1, n):
        d *= decay
        weight += d
        ewma_old = ewma_old * decay + arr[i]
    return ewma_old / weight


@njit
def _ewma_circular(buf, write_idx, window):
    """EWMA over a full circular buffer (oldest element at buf[write_idx])."""
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
def _ewma_last_n(bar_len, n, num_prev_bars):
    """EWMA of the last n bar lengths stored at the tail of bar_len (buffer of
    size num_prev_bars), span = num_prev_bars -> alpha = 2/(num_prev_bars+1)."""
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


# --- Core kernel (state carried across batches and months) ---------------------------

@njit
def _dib_batch(price, qty, buyer_maker, ttime,
               theta, T, E_T, E_x,
               imb_buf, imb_write, imb_count,
               bar_len, num_bars,
               cur_open, cur_high, cur_low, cur_vol, cur_qvol, cur_open_time, cur_close_time,
               o_open_time, o_close_time, o_open, o_high, o_low, o_close,
               o_volume, o_qvolume, o_imbalance, o_ntrades,
               o_E_T, o_E_x, o_threshold,
               window, exp_ticks_init, min_bar_size, max_bar_size, num_prev_bars):
    """Process one batch of trades; emits bars. Carries all EWMA/accumulator state."""
    n = price.shape[0]
    n_bars = 0
    first = (T == 0)

    for i in range(n):
        b = -1.0 if buyer_maker[i] else 1.0
        x = b * price[i] * qty[i]          # dollar imbalance observation

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

        # push the trade-level imbalance into the circular buffer
        imb_buf[imb_write] = x
        imb_write += 1
        if imb_write >= window:
            imb_write = 0
        imb_count += 1

        # warm-up: compute the first E_x once exp_ticks_init trades observed
        if imb_count == exp_ticks_init:
            E_x = _ewma_circular(imb_buf, imb_write, window)

        # bar sampling (only after warm-up)
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

                # E_T = EWMA of the last num_prev_bars bar lengths
                for j in range(num_prev_bars - 1):
                    bar_len[j] = bar_len[j + 1]
                bar_len[num_prev_bars - 1] = float(T)
                num_bars += 1
                nb = num_bars if num_bars < num_prev_bars else num_prev_bars
                E_T = _ewma_last_n(bar_len, nb, num_prev_bars)
                if E_T > max_bar_size:
                    E_T = max_bar_size
                if E_T < min_bar_size:
                    E_T = min_bar_size

                # E_x = EWMA of the last `window` trades
                E_x = _ewma_circular(imb_buf, imb_write, window)

                theta = 0.0
                T = 0
                first = True
                cur_vol = 0.0
                cur_qvol = 0.0

    return (theta, T, E_T, E_x, imb_write, imb_count, num_bars,
            cur_open, cur_high, cur_low, cur_vol, cur_qvol, cur_open_time, cur_close_time, n_bars)


# --- reference reproduction (pure Python) for the deterministic test ------------------

def _ref_ewma(arr, window):
    """Pure-Python reproduction of mlfinlab fast_ewma.ewma(arr, window)."""
    alpha = 2.0 / (window + 1.0)
    n = len(arr)
    ewma_old = arr[0]
    weight = 1.0
    for i in range(1, n):
        weight += (1.0 - alpha) ** i
        ewma_old = ewma_old * (1.0 - alpha) + arr[i]
    return ewma_old / weight


def _test_ewma():
    rng = np.random.RandomState(0)
    seq = rng.randn(30000)

    # (1) contiguous EWMA vs reference, across several (n, window)
    for n, window in [(1, 3), (2, 3), (3, 3), (10, 10), (1000, 1000), (10000, 10000)]:
        arr = seq[:n]
        ref = _ref_ewma(arr, window)
        got = _ewma_arr(arr, n, window)
        assert abs(ref - got) < 1e-12, f"n={n} window={window}: ref={ref} got={got}"

    # (2) circular EWMA vs contiguous EWMA (wrapped write index)
    window = 10000
    buf = np.empty(window, dtype=np.float64)
    write = 0
    for x in seq:
        buf[write] = x
        write = (write + 1) % window
    contig = seq[len(seq) - window:].copy()
    ref = _ref_ewma(contig, window)
    got_circ = _ewma_circular(buf, write, window)
    assert abs(ref - got_circ) < 1e-12, f"circular: ref={ref} got={got_circ}"

    # (3) E_T EWMA over last n bar lengths, num_prev_bars = 3
    bl = np.array([10.0, 20.0, 30.0])
    assert abs(_ewma_last_n(bl, 3, 3) - _ref_ewma(bl, 3)) < 1e-12

    # (4) parametrized num_prev_bars = 5
    bl5 = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert abs(_ewma_last_n(bl5, 5, 5) - _ref_ewma(bl5, 5)) < 1e-12
    assert abs(_ewma_last_n(bl5, 2, 5) - _ref_ewma(bl5[-2:], 5)) < 1e-12

    print("EWMA test PASSED: Numba _ewma_arr / _ewma_circular / _ewma_last_n match mlfinlab fast_ewma")


# --- file / flush / build ------------------------------------------------------------

def _list_monthly_files(symbol, parquet_dir=None):
    root = parquet_dir or data_config.PARQUET_DIR
    pattern = os.path.join(root, symbol, f"{symbol}-aggTrades-*.parquet")
    return sorted(glob.glob(pattern))


def _flush(writer, cols):
    arrays = [np.concatenate(c) for c in cols]
    writer.write_table(pa.Table.from_arrays(arrays, schema=BAR_SCHEMA))


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_generation(symbol, cid, bars_dir=None):
    """Resolve the CURRENT generation of (symbol, config) through the `current`
    symlink. Returns the base parquet path of the current generation, or None if
    no valid generation exists. Readers must resolve all four artifacts through
    this pointer so they always observe one complete generation."""
    bars_dir = bars_dir or dib_configs.OUT_DIR
    cur = os.path.join(bars_dir, symbol, cid, "current")
    if not os.path.islink(cur):
        return None
    gen_dir = os.path.join(bars_dir, symbol, cid, os.readlink(cur))
    base = os.path.join(gen_dir, f"{symbol}-{cid}-dib.parquet")
    return base if os.path.exists(base) else None


def verify_input(symbol, parquet_dir=None, manifest_path=None, expected_months=None):
    """Pre-build input verification: file existence, schema/dtypes, chronological
    ordering across monthly files (via parquet footer statistics - cheap), the
    documented duplicate-id pair (SOLUSDT), and STRICT coverage: every expected
    month (from the manifest - including months recorded as failed/not_found -
    or from an explicit expected_months list) must be present and processable.
    Raises ValueError on violation. A missing manifest is a failure, never a
    silent skip."""
    files = _list_monthly_files(symbol, parquet_dir=parquet_dir)
    if not files:
        raise ValueError(f"no monthly parquet for {symbol}")

    want_types = {"agg_trade_id": "int64", "price": "double", "quantity": "double",
                  "first_trade_id": "int64", "last_trade_id": "int64",
                  "transact_time": "int64", "is_buyer_maker": "bool"}
    prev_max_id = None
    prev_max_tt = None
    prev_month = None
    for f in files:
        pf = pq.ParquetFile(f)
        sch = pf.schema_arrow
        for name, wt in want_types.items():
            if name not in sch.names:
                raise ValueError(f"{f}: missing column {name}")
            if str(sch.field(name).type) != wt:
                raise ValueError(f"{f}: column {name} has type {sch.field(name).type}, expected {wt}")
        id_idx = sch.names.index("agg_trade_id")
        tt_idx = sch.names.index("transact_time")
        min_id = 2 ** 62
        max_id = -1
        min_tt = 2 ** 62
        max_tt = -1
        for rg in range(pf.metadata.num_row_groups):
            st_id = pf.metadata.row_group(rg).column(id_idx).statistics
            st_tt = pf.metadata.row_group(rg).column(tt_idx).statistics
            min_id = min(min_id, int(st_id.min))
            max_id = max(max_id, int(st_id.max))
            min_tt = min(min_tt, int(st_tt.min))
            max_tt = max(max_tt, int(st_tt.max))
        if prev_max_id is not None and min_id <= prev_max_id:
            raise ValueError(f"{os.path.basename(f)}: agg ids overlap/regress vs {prev_month} "
                             f"(min {min_id} <= prev max {prev_max_id})")
        if prev_max_tt is not None and min_tt < prev_max_tt:
            raise ValueError(f"{os.path.basename(f)}: transact_time overlaps {prev_month} "
                             f"(min {min_tt} < prev max {prev_max_tt})")
        prev_max_id, prev_max_tt, prev_month = max_id, max_tt, os.path.basename(f)

    # STRICT expected-month coverage. Expected months come from the manifest
    # (ALL entries for the symbol, regardless of status - a month recorded as
    # failed or not_found still counts as expected and its absence must fail)
    # or from an explicit expected_months override (tests / custom dirs).
    if expected_months is not None:
        expected = sorted(set(expected_months))
    elif manifest_path is not None or parquet_dir is None:
        manifest_path = manifest_path or data_config.MANIFEST_PATH
        if not os.path.exists(manifest_path):
            raise ValueError(
                f"manifest {manifest_path} is missing - expected coverage cannot be established")
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        keys = sorted(k for k in manifest if k.startswith(symbol + "/"))
        if not keys:
            raise ValueError(f"manifest records no months for {symbol}")
        expected = sorted(k.split("/", 1)[1] for k in keys)
    else:
        expected = None  # custom input dir without explicit months: file-set check only

    if expected is not None:
        have = sorted(os.path.basename(f).split("-aggTrades-")[1][:7] for f in files)
        missing = sorted(set(expected) - set(have))
        extra = sorted(set(have) - set(expected))
        if missing:
            raise ValueError(f"{symbol}: MISSING expected months {missing}")
        if extra:
            raise ValueError(f"{symbol}: unexpected extra months {extra}")

    # documented duplicate-id pair must be preserved (no dedup by agg id).
    # Enforced whenever a 2025-07 input exists (always for the canonical dir);
    # custom/synthetic input dirs without that month skip the check.
    if symbol == "SOLUSDT":
        target = [f for f in files if "aggTrades-2025-07" in f]
        if target:
            cnt = 0
            for b in pq.ParquetFile(target[0]).iter_batches(batch_size=1 << 20, columns=["agg_trade_id"]):
                aid = b.column("agg_trade_id").to_numpy(zero_copy_only=False)
                cnt += int((aid == 926014272).sum())
            if cnt != 2:
                raise ValueError(f"SOLUSDT: expected exactly 2 rows with agg_trade_id 926014272, found {cnt}")
        elif parquet_dir is None:
            raise ValueError("SOLUSDT: missing 2025-07 input for duplicate-id check")

    return {"files": len(files), "first": os.path.basename(files[0]), "last": os.path.basename(files[-1])}


def _resolve_params(cfg):
    """Merge a configuration dict with the frozen methodology defaults and
    return the fully resolved parameter set (recorded in the run metadata)."""
    p = dict(cfg)
    p.setdefault("expected_imbalance_window", dib_configs.EXPECTED_IMBALANCE_WINDOW)
    p.setdefault("exp_ticks_init", dib_configs.EXP_NUM_TICKS_INIT)
    p.setdefault("num_prev_bars", dib_configs.NUM_PREV_BARS)
    p.setdefault("batch_size", dib_configs.BATCH_SIZE)
    p.setdefault("bar_buffer", dib_configs.BAR_BUFFER)
    dib_configs.validate_methodology()
    if p["exp_ticks_init"] < p["expected_imbalance_window"]:
        raise ValueError("exp_ticks_init must be >= expected_imbalance_window "
                         "(circular EWMA buffer must be full at first E_x)")
    return p


def build_dib(symbol, cfg=None, bars_dir=None, parquet_dir=None):
    """Build DIBs for one asset under one configuration.

    Publication model (interruption-safe):
      - the four artifacts (parquet / .tail.json / .state.npz / -meta.json) are
        built inside a staging directory
      - a new generation directory gen-<id> is created under
        {bars_dir}/{SYM}/{cfg_id}/ and the staged artifacts are moved into it
      - publication is a SINGLE atomic switch: the `current` symlink inside
        {bars_dir}/{SYM}/{cfg_id}/ is atomically replaced to point at the new
        generation (tmp symlink + os.replace)
      - previous generations are NEVER deleted automatically; readers always
        resolve all four artifacts through the `current` pointer (see
        resolve_generation), so they observe either the complete old
        generation or the complete new one, never a mixture
      - generation identifiers and per-artifact metadata are recorded in
        {bars_dir}/{SYM}/{cfg_id}/generations.json
      - failures or interruptions at any stage leave `current` untouched;
        abandoned staging directories are cleared at the start of the next build

    Returns the meta dict (includes "generation" and "published").
    """
    import datetime as _dt
    cfg = _resolve_params(dict(cfg if cfg is not None else dib_configs.get_config(dib_configs.DEFAULT_CONFIG)))
    dib_configs.check_environment()
    window = cfg["expected_imbalance_window"]
    exp_init = cfg["exp_ticks_init"]
    npb = cfg["num_prev_bars"]
    batch_size = cfg["batch_size"]
    bar_buffer = cfg["bar_buffer"]
    E_T_init = float(cfg["E_T_init"])
    min_bar = float(cfg["min_bar_size"])
    max_bar = float(cfg["max_bar_size"])

    if bars_dir is None:
        bars_dir = dib_configs.OUT_DIR
    cid_dir = os.path.join(bars_dir, symbol, cfg["id"])
    os.makedirs(cid_dir, exist_ok=True)
    artifact_name = f"{symbol}-{cfg['id']}-dib.parquet"

    input_check = verify_input(symbol, parquet_dir=parquet_dir)
    files = _list_monthly_files(symbol, parquet_dir=parquet_dir)

    # state (carried across batches and months; also the resume state)
    theta = 0.0
    T = 0
    E_T = E_T_init
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

    # Versioned staging: all four artifacts are built into a staging directory,
    # then moved into a new generation directory and published via ONE atomic
    # `current`-symlink switch. Previous generations are preserved.
    stage_dir = os.path.join(bars_dir, symbol, f".staging-{cfg['id']}")
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir)
    staged = os.path.join(stage_dir, artifact_name)

    writer = None
    try:
        writer = pq.ParquetWriter(staged, BAR_SCHEMA, compression="zstd")
        max_bars = batch_size
        o_open_time = np.empty(max_bars, dtype=np.int64)
        o_close_time = np.empty(max_bars, dtype=np.int64)
        o_open = np.empty(max_bars, dtype=np.float64)
        o_high = np.empty(max_bars, dtype=np.float64)
        o_low = np.empty(max_bars, dtype=np.float64)
        o_close = np.empty(max_bars, dtype=np.float64)
        o_volume = np.empty(max_bars, dtype=np.float64)
        o_qvolume = np.empty(max_bars, dtype=np.float64)
        o_imbalance = np.empty(max_bars, dtype=np.float64)
        o_ntrades = np.empty(max_bars, dtype=np.int64)
        o_E_T = np.empty(max_bars, dtype=np.float64)
        o_E_x = np.empty(max_bars, dtype=np.float64)
        o_threshold = np.empty(max_bars, dtype=np.float64)

        cols = [[] for _ in range(13)]
        buffered = 0

        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=batch_size):
                price = batch.column("price").to_numpy(zero_copy_only=False)
                qty = batch.column("quantity").to_numpy(zero_copy_only=False)
                bm = batch.column("is_buyer_maker").to_numpy(zero_copy_only=False)
                tt = batch.column("transact_time").to_numpy(zero_copy_only=False)

                (theta, T, E_T, E_x, imb_write, imb_count, num_bars,
                 cur_open, cur_high, cur_low, cur_vol, cur_qvol, cur_open_time, cur_close_time, n_bars) = _dib_batch(
                    price, qty, bm, tt, theta, T, E_T, E_x,
                    imb_buf, imb_write, imb_count, bar_len, num_bars,
                    cur_open, cur_high, cur_low, cur_vol, cur_qvol, cur_open_time, cur_close_time,
                    o_open_time, o_close_time, o_open, o_high, o_low, o_close,
                    o_volume, o_qvolume, o_imbalance, o_ntrades,
                    o_E_T, o_E_x, o_threshold,
                    window, exp_init, min_bar, max_bar, npb)

                if n_bars > 0:
                    cols[0].append(o_open_time[:n_bars].copy())
                    cols[1].append(o_close_time[:n_bars].copy())
                    cols[2].append(o_open[:n_bars].copy())
                    cols[3].append(o_high[:n_bars].copy())
                    cols[4].append(o_low[:n_bars].copy())
                    cols[5].append(o_close[:n_bars].copy())
                    cols[6].append(o_volume[:n_bars].copy())
                    cols[7].append(o_qvolume[:n_bars].copy())
                    cols[8].append(o_imbalance[:n_bars].copy())
                    cols[9].append(o_ntrades[:n_bars].copy())
                    cols[10].append(o_E_T[:n_bars].copy())
                    cols[11].append(o_E_x[:n_bars].copy())
                    cols[12].append(o_threshold[:n_bars].copy())
                    buffered += n_bars
                    if buffered >= bar_buffer:
                        _flush(writer, cols)
                        cols = [[] for _ in range(13)]
                        buffered = 0

        if buffered > 0:
            _flush(writer, cols)
        writer.close()
        writer = None

        # structural self-check before publishing
        total_bars = pq.ParquetFile(staged).metadata.num_rows

        # --- unclosed tail accounting (no force-close) ---
        tail = {
            "num_trades": int(T),
            "volume": float(cur_vol),
            "quote_volume": float(cur_qvol),
            "imbalance": float(theta),
            "open_time": int(cur_open_time),
            "close_time": int(cur_close_time),
        }
        with open(staged + ".tail.json", "w") as fh:
            json.dump(tail, fh, indent=2)

        # --- continuation state (resume-capable) ---
        np.savez(staged + ".state.npz",
                 theta=np.float64(theta), T=np.int64(T), E_T=np.float64(E_T), E_x=np.float64(E_x),
                 imb_buf=imb_buf.astype(np.float64), imb_write=np.int64(imb_write), imb_count=np.int64(imb_count),
                 bar_len=bar_len.astype(np.float64), num_bars=np.int64(num_bars),
                 cur_open=np.float64(cur_open), cur_high=np.float64(cur_high), cur_low=np.float64(cur_low),
                 cur_vol=np.float64(cur_vol), cur_qvol=np.float64(cur_qvol),
                 cur_open_time=np.int64(cur_open_time), cur_close_time=np.int64(cur_close_time))

        import platform
        import numba
        import pyarrow
        meta = {
            "symbol": symbol,
            "config": cfg,
            "resolved_parameters": {
                "exp_ticks_init": exp_init,
                "expected_imbalance_window": window,
                "num_prev_bars": npb,
                "E_T_init": E_T_init,
                "min_bar_size": min_bar,
                "max_bar_size": max_bar,
                "batch_size": batch_size,
                "bar_buffer": bar_buffer,
            },
            "input_check": input_check,
            "input_files": len(files),
            "input_months": [os.path.basename(f) for f in files],
            "total_bars": int(total_bars),
            "tail": tail,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "numba": numba.__version__,
                "pyarrow": pyarrow.__version__,
            },
        }
        with open(staged.replace(".parquet", "-meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    except KeyboardInterrupt:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    # --- publish the complete generation -------------------------------------
    # One atomic switch: create the generation directory, move the staged
    # artifacts into it, then atomically repoint the `current` symlink.
    # `current` is the ONLY thing readers resolve; it is replaced in a single
    # os.replace of a tmp symlink, so readers observe either the complete old
    # generation or the complete new one - never a mixture.
    gen_id = f"gen-{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
    gen_dir = os.path.join(cid_dir, gen_id)
    try:
        os.makedirs(gen_dir)
        for src in (staged, staged + ".tail.json", staged + ".state.npz",
                    staged.replace(".parquet", "-meta.json")):
            os.replace(src, os.path.join(gen_dir, os.path.basename(src)))
        # atomic pointer switch
        cur = os.path.join(cid_dir, "current")
        tmp_link = os.path.join(cid_dir, ".current.tmp")
        if os.path.lexists(tmp_link):
            os.remove(tmp_link)
        os.symlink(gen_id, tmp_link)
        os.replace(tmp_link, cur)
        # record the generation (atomic write)
        gens_path = os.path.join(cid_dir, "generations.json")
        gens = {"generations": [], "current": gen_id}
        if os.path.exists(gens_path):
            with open(gens_path) as fh:
                gens = json.load(fh)
        gens.setdefault("generations", []).append({
            "id": gen_id,
            "artifacts": {
                os.path.basename(src): sha256_file(os.path.join(gen_dir, os.path.basename(src)))
                for src in (staged, staged + ".tail.json", staged + ".state.npz",
                            staged.replace(".parquet", "-meta.json"))
            },
        })
        gens["current"] = gen_id
        with open(gens_path + ".tmp", "w") as fh:
            json.dump(gens, fh, indent=2)
        os.replace(gens_path + ".tmp", gens_path)
    except Exception:
        # the new generation was never referenced by `current`; remove it.
        # The previous `current` pointer is untouched, so readers keep seeing
        # the complete previous generation.
        shutil.rmtree(gen_dir, ignore_errors=True)
        tmp_link = os.path.join(cid_dir, ".current.tmp")
        try:
            if os.path.isdir(tmp_link):
                shutil.rmtree(tmp_link, ignore_errors=True)
            elif os.path.lexists(tmp_link):
                os.remove(tmp_link)
        except OSError:
            pass
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    shutil.rmtree(stage_dir, ignore_errors=True)
    meta["generation"] = gen_id
    meta["published"] = os.path.join(gen_dir, artifact_name)
    print(f"{symbol} [{cfg['id']}]: {total_bars:,} bars, tail {tail['num_trades']:,} trades "
          f"-> {meta['published']} (gen {gen_id})")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Dollar Imbalance Bar engine")
    parser.add_argument("--test", action="store_true", help="run the EWMA reference test")
    parser.add_argument("--symbols", nargs="*", help="assets (default: all)")
    parser.add_argument("--all", action="store_true", help="process all assets")
    parser.add_argument("--config", default=dib_configs.DEFAULT_CONFIG,
                        help=f"configuration id from dib_configs (default: {dib_configs.DEFAULT_CONFIG})")
    args = parser.parse_args()

    if args.test:
        _test_ewma()
        return

    cfg = dib_configs.get_config(args.config)
    symbols = data_config.SYMBOLS if (args.all or not args.symbols) else args.symbols
    for sym in symbols:
        build_dib(sym, cfg)


if __name__ == "__main__":
    main()
