"""
data_engine/dib_configs.py
==========================
Single source of truth for the Dollar Imbalance Bar configuration matrix.

Five research configurations:
  1. fixed5k    - Fixed expected ticks E_T = 5,000
  2. fixed10k   - Fixed expected ticks E_T = 10,000
  3. fixed20k   - Fixed expected ticks E_T = 20,000
  4. fixed28k   - Fixed expected ticks E_T = 28,000
  5. dynema28k  - Dynamic EMA E_T, capped at 28,000 (reference condition)

Shared, frozen methodology parameters (do not change without a new config id):
  - aggressor sign: b_t = -1 if is_buyer_maker else +1 (Binance native)
  - signed dollar flow: x_t = b_t * price_t * quantity_t
  - bar closes when |theta| > E_T * |E_x|  (STRICT greater-than, reference source)
  - E_x: EWMA over the most recent EXPECTED_IMBALANCE_WINDOW trade-level x_t
        values, mlfinlab fast_ewma semantics (alpha = 2/(window+1)); refreshed
        at each DIB close (documented adaptation)
  - warm-up: causal; first E_x at exactly EXP_NUM_TICKS_INIT trades observed;
        warm-up trades belong to the first bar
  - E_T (dynamic): initialized to EXP_NUM_TICKS_INIT, then EWMA over the last
        NUM_PREV_BARS completed bar lengths (alpha = 2/(NUM_PREV_BARS+1)),
        clamped to [MIN_BAR_SIZE, MAX_BAR_SIZE]
  - E_T (fixed): pinned at E_T_INIT (implemented as min=max clamp; the EWMA
        update is computed and discarded - identical to the verified behavior)

Fixed configurations pin E_T via min = max = E_T_init, so the threshold is
T0 * |E_x| exactly, from the very first bar (verified behavior preserved).
"""

# --- Frozen methodology parameters ------------------------------------------------
EXP_NUM_TICKS_INIT = 20_000        # warm-up length AND initial E_T for the dynamic config
EXPECTED_IMBALANCE_WINDOW = 10_000  # E_x EWMA window (trades)
NUM_PREV_BARS = 3                  # E_T EWMA window (bars)

BATCH_SIZE = 500_000               # parquet iter_batches row count
BAR_BUFFER = 10_000                # bars buffered before flushing to parquet

# --- Configuration matrix ----------------------------------------------------------
CONFIGS = {
    "fixed5k": {
        "id": "fixed5k",
        "name": "Fixed E_T = 5,000",
        "E_T_init": 5_000.0,
        "min_bar_size": 5_000.0,
        "max_bar_size": 5_000.0,
        "dynamic": False,
    },
    "fixed10k": {
        "id": "fixed10k",
        "name": "Fixed E_T = 10,000",
        "E_T_init": 10_000.0,
        "min_bar_size": 10_000.0,
        "max_bar_size": 10_000.0,
        "dynamic": False,
    },
    "fixed20k": {
        "id": "fixed20k",
        "name": "Fixed E_T = 20,000",
        "E_T_init": 20_000.0,
        "min_bar_size": 20_000.0,
        "max_bar_size": 20_000.0,
        "dynamic": False,
    },
    "fixed28k": {
        "id": "fixed28k",
        "name": "Fixed E_T = 28,000",
        "E_T_init": 28_000.0,
        "min_bar_size": 28_000.0,
        "max_bar_size": 28_000.0,
        "dynamic": False,
    },
    "dynema28k": {
        "id": "dynema28k",
        "name": "Dynamic EMA E_T, capped at 28,000",
        "E_T_init": float(EXP_NUM_TICKS_INIT),  # 20,000
        "min_bar_size": 0.0,
        "max_bar_size": 28_000.0,
        "dynamic": True,
    },
}

# Default configuration for the plain `python -m data_engine.bars` entry point.
DEFAULT_CONFIG = "dynema28k"

OUT_DIR = "data/bars"  # outputs: {OUT_DIR}/{SYM}/{SYM}-{id}-dib.parquet (+ .tail.json, .state.npz, -meta.json)


def get_config(cid):
    if cid not in CONFIGS:
        raise ValueError(f"unknown DIB configuration {cid!r}; known: {sorted(CONFIGS)}")
    return CONFIGS[cid]


def validate_methodology():
    """Invariant checks on the frozen methodology parameters (called at build time)."""
    if EXP_NUM_TICKS_INIT < EXPECTED_IMBALANCE_WINDOW:
        raise ValueError(
            f"EXP_NUM_TICKS_INIT ({EXP_NUM_TICKS_INIT}) must be >= "
            f"EXPECTED_IMBALANCE_WINDOW ({EXPECTED_IMBALANCE_WINDOW}): the circular "
            f"EWMA buffer must be full when the first E_x is computed")
    if NUM_PREV_BARS < 1:
        raise ValueError("NUM_PREV_BARS must be >= 1")
    for cid, cfg in CONFIGS.items():
        if cfg["min_bar_size"] > cfg["max_bar_size"]:
            raise ValueError(f"{cid}: min_bar_size > max_bar_size")
        if not (cfg["min_bar_size"] <= cfg["E_T_init"] <= cfg["max_bar_size"]):
            raise ValueError(f"{cid}: E_T_init outside [min, max]")
        if not cfg["dynamic"] and cfg["min_bar_size"] != cfg["max_bar_size"]:
            raise ValueError(f"{cid}: fixed config must pin E_T (min == max)")


# Tested dependency set (see requirements-dib.txt). The numba/numpy/llvmlite
# combination is the only one exercised by the test suite; a mismatch fails the
# build rather than silently producing potentially divergent float behavior.
TESTED_ENVIRONMENT = {
    "numpy": "1.19.2",
    "numba": "0.51.2",
    "llvmlite": "0.34.0",
    "pyarrow": "17.0.0",
    "python": "3.8.5",
}


def check_environment():
    """Hard-fail on numba/numpy/llvmlite mismatch (float/JIT behavior),
    warn on pyarrow/python drift. Also forces an actual JIT compilation round
    (import alone does not guarantee the toolchain works)."""
    import platform
    import warnings

    import numpy
    import numba
    try:
        import llvmlite
        llvm_version = llvmlite.__version__
    except ImportError:
        llvm_version = "missing"
    try:
        import pyarrow
        pa_version = pyarrow.__version__
    except ImportError:
        pa_version = "missing"
    if numba.__version__ != TESTED_ENVIRONMENT["numba"]:
        raise RuntimeError(
            f"numba {numba.__version__} is untested; the pinned, tested version is "
            f"{TESTED_ENVIRONMENT['numba']} (see requirements-dib.txt)")
    if numpy.__version__ != TESTED_ENVIRONMENT["numpy"]:
        raise RuntimeError(
            f"numpy {numpy.__version__} is untested with numba {numba.__version__}; "
            f"the pinned, tested pair is numpy {TESTED_ENVIRONMENT['numpy']} + "
            f"numba {TESTED_ENVIRONMENT['numba']}")
    if llvm_version != TESTED_ENVIRONMENT["llvmlite"]:
        raise RuntimeError(
            f"llvmlite {llvm_version} is untested with numba {numba.__version__}; "
            f"the pinned, tested pair is llvmlite {TESTED_ENVIRONMENT['llvmlite']} + "
            f"numba {TESTED_ENVIRONMENT['numba']}")
    if pa_version != TESTED_ENVIRONMENT["pyarrow"]:
        warnings.warn(f"pyarrow {pa_version} differs from the tested {TESTED_ENVIRONMENT['pyarrow']}")
    if platform.python_version() != TESTED_ENVIRONMENT["python"]:
        warnings.warn(f"python {platform.python_version()} differs from the tested {TESTED_ENVIRONMENT['python']}")
    # force an actual JIT compilation + execution round
    from numba import njit
    @njit
    def _env_probe(a, b):
        s = 0.0
        for i in range(a.shape[0]):
            s += a[i] * b
        return s
    import numpy as _np
    probe = _env_probe(_np.array([1.0, 2.0, 3.0]), 2.0)
    if abs(probe - 12.0) > 1e-12:
        raise RuntimeError("numba JIT probe produced an unexpected result")
