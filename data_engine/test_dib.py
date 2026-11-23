"""
data_engine/test_dib.py
=======================
Synthetic-data tests for the DIB engine and validator.

Run: python -m data_engine.test_dib   (uses the tested environment's interpreter)

Covers:
  T1  EWMA reference checks (mlfinlab fast_ewma parity, incl. num_prev_bars=5)
  T2  fixed-config closure points, strict threshold, E_T pinning, diagnostics
  T3  dynamic E_T update rule (recomputed from bar lengths, incl. 28k cap)
  T4  tail accounting at EOF + continuation state
  T5  warm-up invariant assertion (window > exp_ticks_init must raise)
  T6  duplicate aggregate-id rows: both executions counted, nothing dropped
  T7  validator PASS on a valid synthetic output (all correctness checks)
  T8  validator detects deliberately introduced errors (6 corruption modes)
  T9  schema / dtypes / nulls on synthetic output
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_engine import bars as bars_mod
from data_engine import dib_configs
from data_engine import dib_replay
from data_engine import dib_validate

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"[PASS] {name}")
    else:
        FAIL.append(name)
        print(f"[FAIL] {name}: {detail}")


def make_input_parquet(parquet_dir, symbol, rows):
    """rows: list of (agg_trade_id, price, quantity, first_trade_id, last_trade_id,
    transact_time, is_buyer_maker)."""
    os.makedirs(os.path.join(parquet_dir, symbol), exist_ok=True)
    t = pa.table({
        "agg_trade_id": pa.array([r[0] for r in rows], pa.int64()),
        "price": pa.array([r[1] for r in rows], pa.float64()),
        "quantity": pa.array([r[2] for r in rows], pa.float64()),
        "first_trade_id": pa.array([r[3] for r in rows], pa.int64()),
        "last_trade_id": pa.array([r[4] for r in rows], pa.int64()),
        "transact_time": pa.array([r[5] for r in rows], pa.int64()),
        "is_buyer_maker": pa.array([r[6] for r in rows], pa.bool_()),
    })
    pq.write_table(t, os.path.join(parquet_dir, symbol, f"{symbol}-aggTrades-2000-01.parquet"))


def build_synthetic(symbol, rows, cfg, out_dir):
    parquet_dir = tempfile.mkdtemp(prefix="dibtest_in_")
    make_input_parquet(parquet_dir, symbol, rows)
    meta = bars_mod.build_dib(symbol, cfg, bars_dir=out_dir, parquet_dir=parquet_dir)
    base = bars_mod.resolve_generation(symbol, cfg["id"], out_dir)
    return parquet_dir, base, meta


def make_gen_tree(tmp, symbol, cid, src_base):
    """Copy a built generation into a fresh bars tree (gen-1 + current symlink)
    for corruption tests; returns the gen base path."""
    gen_dir = os.path.join(tmp, symbol, cid, "gen-1")
    os.makedirs(gen_dir, exist_ok=True)
    base = os.path.join(gen_dir, f"{symbol}-{cid}-dib.parquet")
    shutil.copy(src_base, base)
    shutil.copy(src_base + ".tail.json", base + ".tail.json")
    shutil.copy(src_base + ".state.npz", base + ".state.npz")
    shutil.copy(src_base.replace(".parquet", "-meta.json"), base.replace(".parquet", "-meta.json"))
    os.symlink("gen-1", os.path.join(tmp, symbol, cid, "current"))
    return base


def seq_trades(n, qty=1.0, price=1.0, start_id=1, start_tt=1000):
    rows = []
    for i in range(n):
        rows.append((start_id + i, price, qty, start_id + i, start_id + i, start_tt + i, False))
    return rows


def main():
    # ---------------- T1: EWMA reference ----------------
    try:
        bars_mod._test_ewma()
        check("T1_ewma_reference", True)
    except Exception as exc:
        check("T1_ewma_reference", False, str(exc))

    work = tempfile.mkdtemp(prefix="dibtest_")
    try:
        # ---------------- T2: fixed config, deterministic closures ----------------
        # constant x_t = +1 (buy qty=1 @ price=1); fixed E_T=10; window=5; init=5
        # E_x = 1.0 -> threshold = 10; strict '>' -> bars close at trades 11, 22, ...
        cfg = {"id": "testfixed", "name": "test fixed 10",
               "E_T_init": 10.0, "min_bar_size": 10.0, "max_bar_size": 10.0,
               "dynamic": False, "expected_imbalance_window": 5, "exp_ticks_init": 5}
        parquet_dir, out_path, meta = build_synthetic("SOLUSDT", seq_trades(55), cfg, work)
        bars = pq.read_table(out_path)
        nt = bars.column("num_trades").to_numpy()
        et = bars.column("E_T").to_numpy()
        thr = bars.column("threshold").to_numpy()
        imb = bars.column("imbalance").to_numpy()
        check("T2_fixed_bar_count", bars.num_rows == 5, f"bars={bars.num_rows}")
        check("T2_fixed_bar_lengths", bool((nt == 11).all()), f"lengths={nt.tolist()}")
        check("T2_fixed_E_T_pinned", bool((et == 10.0).all()), f"E_T={et.tolist()}")
        check("T2_fixed_threshold", bool((np.abs(thr - 10.0) < 1e-12).all()), f"thr={thr.tolist()}")
        check("T2_fixed_imbalance", bool((np.abs(imb - 11.0) < 1e-12).all()), f"imb={imb.tolist()}")
        tail = json.load(open(out_path + ".tail.json"))
        check("T2_fixed_tail_zero", tail["num_trades"] == 0, f"tail={tail}")

        # ---------------- T3: dynamic E_T update rule + cap ----------------
        # varying x values -> varying bar lengths; verify stored E_T equals a manual
        # recomputation from bar lengths (EWMA window=3, alpha=0.5, capped at 28)
        rows = []
        xvals = [2.0] * 6 + [1.0] * 20 + [4.0] * 8 + [1.0] * 30   # forces varied bar lengths
        for i, x in enumerate(xvals):
            rows.append((i + 1, 1.0, x, i + 1, i + 1, 1000 + i, False))
        cfg3 = {"id": "testdyn", "name": "test dynamic",
                "E_T_init": 10.0, "min_bar_size": 0.0, "max_bar_size": 28.0,
                "dynamic": True, "expected_imbalance_window": 5, "exp_ticks_init": 5}
        parquet_dir3, out_path3, meta3 = build_synthetic("SOLUSDT", rows, cfg3, work)
        bars3 = pq.read_table(out_path3)
        nt3 = bars3.column("num_trades").to_numpy().astype(float)
        et3 = bars3.column("E_T").to_numpy()
        thr3 = bars3.column("threshold").to_numpy()
        ex3 = bars3.column("E_x").to_numpy()
        # manual E_T recomputation: the E_T stored for bar k is the value USED to
        # close it = E_T_init for bar 1, then the post-close EWMA over preceding
        # completed bar lengths (window=3, alpha=0.5), capped at 28
        manual = []
        acc = []
        for k, L in enumerate(nt3):
            if k == 0:
                v = 10.0  # E_T_init
            else:
                acc.append(nt3[k - 1])
                nb = min(len(acc), 3)
                if nb == 1:
                    v = acc[-1]
                elif nb == 2:
                    v = (acc[-2] * 0.5 + acc[-1]) / 1.5
                else:
                    v = (acc[-3] * 0.5 ** 2 + acc[-2] * 0.5 + acc[-1]) / (0.5 ** 2 + 0.5 + 1)
                v = min(max(v, 0.0), 28.0)
            manual.append(v)
        ok = all(abs(a - b) < 1e-9 for a, b in zip(et3, manual))
        check("T3_dynamic_E_T_update", ok, f"stored={et3.tolist()} manual={manual}")
        ok_thr = all(abs(a - b * abs(c)) < 1e-9 for a, b, c in zip(thr3, et3, ex3))
        check("T3_dynamic_threshold_eq_ET_absEX", ok_thr)
        ok_replay = all(abs(i) > t for i, t in zip(bars3.column("imbalance").to_numpy(), thr3))
        check("T3_dynamic_threshold_replay", ok_replay)
        check("T3_dynamic_cap", bool((et3 <= 28.0).all()))

        # ---------------- T4: tail accounting + continuation state ----------------
        rows_t = seq_trades(30, start_id=5000, start_tt=5000)
        parquet_dir4, out_path4, meta4 = build_synthetic("SOLUSDT", rows_t, cfg, work)
        tail4 = json.load(open(out_path4 + ".tail.json"))
        bars4 = pq.read_table(out_path4)
        nt4 = bars4.column("num_trades").to_numpy()
        check("T4_tail_partitioning", int(nt4.sum()) + tail4["num_trades"] == 30,
              f"bars={int(nt4.sum())} tail={tail4['num_trades']}")
        state = np.load(out_path4 + ".state.npz")
        check("T4_tail_state_consistent", int(state["T"]) == tail4["num_trades"]
              and float(state["theta"]) == tail4["imbalance"]
              and float(state["cur_vol"]) == tail4["volume"])
        check("T4_tail_volume", tail4["volume"] == 30 - int(nt4.sum()),
              f"tail volume {tail4['volume']} vs trades {tail4['num_trades']}")

        # ---------------- T5: warm-up invariant ----------------
        cfg_bad = {"id": "testbad", "name": "bad warmup", "E_T_init": 10.0,
                   "min_bar_size": 10.0, "max_bar_size": 10.0, "dynamic": False,
                   "expected_imbalance_window": 50, "exp_ticks_init": 10}
        raised = False
        try:
            build_synthetic("SOLUSDT", seq_trades(60), cfg_bad, work)
        except ValueError as exc:
            raised = "exp_ticks_init" in str(exc)
        check("T5_warmup_invariant_raises", raised)

        # ---------------- T6: duplicate agg id rows ----------------
        # base ids 926014262..926014303: index 10 = 926014272, so the dup pair
        # lands exactly at the documented id with monotonic surroundings
        # (..., 271, 272, 272, 274, ...) and increasing timestamps
        rows_dup = seq_trades(42, start_id=926014262, start_tt=7000)
        rows_dup[10] = (926014272, 1.0, 5.0, 7001, 7001, 7010, True)
        rows_dup[11] = (926014272, 1.0, 7.0, 7003, 7003, 7011, False)
        parquet_dir6, out_path6, meta6 = build_synthetic("SOLUSDT", rows_dup, cfg, work)
        r6 = dib_validate.validate_one("SOLUSDT", cfg["id"], bars_dir=work, parquet_dir=parquet_dir6)
        ok6 = all(c["ok"] for c in r6["checks"])
        check("T6_duplicate_id_validator_passes", ok6,
              [c for c in r6["checks"] if not c["ok"]])

        # ---------------- T7: validator PASS on valid output ----------------
        r7 = dib_validate.validate_one("SOLUSDT", cfg["id"], bars_dir=work, parquet_dir=parquet_dir6)
        ok7 = all(c["ok"] for c in r7["checks"])
        check("T7_validator_pass_valid", ok7, [c for c in r7["checks"] if not c["ok"]])

        # ---------------- T8: validator detects deliberate corruption ----------------
        # fresh valid build for corruption tests
        parquet_dir8, out_path8, meta8 = build_synthetic("SOLUSDT", rows_dup, cfg, work)
        r8 = dib_validate.validate_one("SOLUSDT", cfg["id"], bars_dir=work, parquet_dir=parquet_dir8)
        assert all(c["ok"] for c in r8["checks"]), "corruption baseline must pass"

        def m_inflate(t):
            i = t.schema.get_field_index("imbalance")
            arr = t.column("imbalance").to_numpy().copy()
            arr[0] *= 1e6
            return t.set_column(i, "imbalance", pa.array(arr))

        def m_partition(t):
            i = t.schema.get_field_index("num_trades")
            arr = t.column("num_trades").to_numpy().copy()
            arr[0] -= 1
            return t.set_column(i, "num_trades", pa.array(arr))

        def m_volume(t):
            i = t.schema.get_field_index("volume")
            arr = t.column("volume").to_numpy().copy()
            arr[0] += 1e6
            return t.set_column(i, "volume", pa.array(arr))

        def m_ohlc(t):
            i = t.schema.get_field_index("low")
            lo = t.column("low").to_numpy().copy()
            hi = t.column("high").to_numpy().copy()
            lo[0] = hi[0] + 100
            return t.set_column(i, "low", pa.array(lo))

        def m_chrono(t):
            i = t.schema.get_field_index("open_time")
            ot = t.column("open_time").to_numpy().copy()
            ot[0], ot[1] = ot[1], ot[0]
            return t.set_column(i, "open_time", pa.array(ot))

        def copy_sidecars(src_base, dst_base):
            shutil.copy(src_base + ".tail.json", dst_base + ".tail.json")
            shutil.copy(src_base + ".state.npz", dst_base + ".state.npz")
            shutil.copy(src_base.replace(".parquet", "-meta.json"), dst_base.replace(".parquet", "-meta.json"))

        def run_corrupt(mod_fn, failing_name, label):
            tmp = tempfile.mkdtemp(prefix="dibtest_corrupt_")
            base = make_gen_tree(tmp, "SOLUSDT", cfg["id"], out_path8)
            pq.write_table(mod_fn(pq.read_table(out_path8)), base)
            r = dib_validate.validate_one("SOLUSDT", cfg["id"], bars_dir=tmp, parquet_dir=parquet_dir8)
            names_failed = [c["name"] for c in r["checks"] if not c["ok"]]
            check(f"T8_detect_{label}", failing_name in names_failed, f"failed checks: {names_failed}")
            shutil.rmtree(tmp, ignore_errors=True)

        def m_zero_imbalance(t):
            i = t.schema.get_field_index("imbalance")
            arr = t.column("imbalance").to_numpy().copy()
            arr[0] = 0.0
            return t.set_column(i, "imbalance", pa.array(arr))

        run_corrupt(m_inflate, "signed_imbalance_conservation", "inflated_imbalance")
        run_corrupt(m_zero_imbalance, "threshold_replay", "zeroed_imbalance")
        run_corrupt(m_partition, "partitioning", "broken_partition")
        run_corrupt(m_volume, "base_volume_conservation", "broken_volume")
        run_corrupt(m_ohlc, "ohlc_sanity", "broken_ohlc")
        run_corrupt(m_chrono, "bar_chronology", "broken_chronology")

        # missing tail file (state and meta present)
        tmp = tempfile.mkdtemp(prefix="dibtest_corrupt_")
        base = make_gen_tree(tmp, "SOLUSDT", cfg["id"], out_path8)
        os.remove(base + ".tail.json")
        r = dib_validate.validate_one("SOLUSDT", cfg["id"], bars_dir=tmp, parquet_dir=parquet_dir8)
        names_failed = [c["name"] for c in r["checks"] if not c["ok"]]
        check("T8_detect_missing_tail", "artifacts" in names_failed, f"failed checks: {names_failed}")
        shutil.rmtree(tmp, ignore_errors=True)

        # ---------------- T9: schema/dtypes/nulls ----------------
        r9 = dib_validate.validate_one("SOLUSDT", cfg["id"], bars_dir=work, parquet_dir=parquet_dir6)
        c9 = {c["name"]: c for c in r9["checks"]}
        check("T9_schema", c9.get("schema_columns", {}).get("ok", False))
        check("T9_no_nulls", c9.get("no_nulls", {}).get("ok", False))

        # ---------------- T10: independent replay equality (randomized input) ----
        rng = np.random.RandomState(42)
        n = 5000
        qty = np.round(rng.rand(n) * 5 + 0.1, 4)
        price = np.round(rng.rand(n) * 3 + 98.0, 2)
        bm = rng.rand(n) > 0.5
        rows_rnd = [(i + 1, float(price[i]), float(qty[i]), i + 1, i + 1, 8000 + i, bool(bm[i]))
                    for i in range(n)]
        parquet_dir10, out_path10, meta10 = build_synthetic("SOLUSDT", rows_rnd, cfg3, work)
        r10 = dib_validate.validate_one("SOLUSDT", cfg3["id"], bars_dir=work, parquet_dir=parquet_dir10)
        c10 = {c["name"]: c for c in r10["checks"]}
        replay_checks = ["replay_ok", "replay_bar_count", "replay_num_trades", "replay_open_time",
                         "replay_close_time", "replay_ohlc", "replay_volumes", "replay_imbalance",
                         "replay_E_T", "replay_E_x", "replay_threshold", "replay_tail", "input_ordering"]
        ok10 = all(c10.get(name, {}).get("ok", False) for name in replay_checks)
        check("T10_independent_replay_equality", ok10,
              [name for name in replay_checks if not c10.get(name, {}).get("ok", True)])

        # ---------------- T11: actual cap activation ----------------
        # 40 alternating +1/-1 trades keep |theta| near 0 (no close), then a
        # 33-trade buy run: bar 1 closes at trade 44 (threshold 10*|E_x| ~ 3.5),
        # post-close E_T = min(44, 28) = 28, and bar 2 must close at trade 73
        # using the CAPPED E_T = 28 -> the stored E_T column shows 28.0
        rows_cap = []
        for i in range(40):
            rows_cap.append((i + 1, 1.0, 1.0, i + 1, i + 1, 9000 + i, (i % 2 == 1)))
        for i in range(33):
            j = 40 + i
            rows_cap.append((j + 1, 1.0, 1.0, j + 1, j + 1, 9040 + i, False))
        cfg_cap = {"id": "testcap", "name": "test cap 28", "E_T_init": 10.0,
                   "min_bar_size": 0.0, "max_bar_size": 28.0, "dynamic": True,
                   "expected_imbalance_window": 5, "exp_ticks_init": 5}
        parquet_dir11, out_path11, meta11 = build_synthetic("SOLUSDT", rows_cap, cfg_cap, work)
        bars11 = pq.read_table(out_path11)
        et11 = bars11.column("E_T").to_numpy()
        check("T11_cap_activation", bool((et11 <= 28.0).all() and (et11 == 28.0).any()),
              f"E_T values {et11.tolist()}")

        # ---------------- T12: cross-file ordering ----------------
        parquet_dir12 = tempfile.mkdtemp(prefix="dibtest_x2_")
        make_input_parquet(parquet_dir12, "SOLUSDT",
                           [(i + 1, 1.0, 1.0, i + 1, i + 1, 10000 + i, False) for i in range(20)])
        # second file: contiguous ids/times
        pq.write_table(pa.table({
            "agg_trade_id": pa.array([21 + i for i in range(20)], pa.int64()),
            "price": pa.array([1.0] * 20, pa.float64()),
            "quantity": pa.array([1.0] * 20, pa.float64()),
            "first_trade_id": pa.array([21 + i for i in range(20)], pa.int64()),
            "last_trade_id": pa.array([21 + i for i in range(20)], pa.int64()),
            "transact_time": pa.array([10020 + i for i in range(20)], pa.int64()),
            "is_buyer_maker": pa.array([False] * 20, pa.bool_()),
        }), os.path.join(parquet_dir12, "SOLUSDT", "SOLUSDT-aggTrades-2000-02.parquet"))
        cfg12 = dict(cfg)
        cfg12["id"] = "testx2"
        bars_mod.build_dib("SOLUSDT", cfg12, bars_dir=work, parquet_dir=parquet_dir12)
        out_path12 = bars_mod.resolve_generation("SOLUSDT", cfg12["id"], work)
        r12 = dib_validate.validate_one("SOLUSDT", cfg12["id"], bars_dir=work, parquet_dir=parquet_dir12)
        ok12 = all(c["ok"] for c in r12["checks"])
        check("T12_cross_file_ordering_valid", ok12, [c for c in r12["checks"] if not c["ok"]])
        # negative: second file with REGRESSING ids must fail verify_input
        pq.write_table(pa.table({
            "agg_trade_id": pa.array([5 + i for i in range(20)], pa.int64()),
            "price": pa.array([1.0] * 20, pa.float64()),
            "quantity": pa.array([1.0] * 20, pa.float64()),
            "first_trade_id": pa.array([5 + i for i in range(20)], pa.int64()),
            "last_trade_id": pa.array([5 + i for i in range(20)], pa.int64()),
            "transact_time": pa.array([10020 + i for i in range(20)], pa.int64()),
            "is_buyer_maker": pa.array([False] * 20, pa.bool_()),
        }), os.path.join(parquet_dir12, "SOLUSDT", "SOLUSDT-aggTrades-2000-02.parquet"))
        raised12 = False
        try:
            bars_mod.verify_input("SOLUSDT", parquet_dir=parquet_dir12)
        except ValueError as exc:
            raised12 = "overlap" in str(exc) or "regress" in str(exc)
        check("T12_cross_file_ordering_regression_detected", raised12)

        # ---------------- T13: failure-safe atomic publication ----------------
        # valid build first
        cfg13 = dict(cfg)
        cfg13["id"] = "testatomic"
        parquet_dir13, out_path13, meta13 = build_synthetic(
            "SOLUSDT", seq_trades(40, start_id=20000, start_tt=20000), cfg13, work)
        good_bytes = open(out_path13, "rb").read()
        # failed rebuild (bad cfg) to the SAME path: must raise, preserve the
        # existing output, and leave no partial files
        raised13 = False
        try:
            bars_mod.build_dib("SOLUSDT", cfg_bad, bars_dir=work, parquet_dir=parquet_dir13)
        except ValueError:
            raised13 = True
        still_intact = open(out_path13, "rb").read() == good_bytes
        stage_left = os.path.exists(os.path.join(work, "SOLUSDT", ".staging-testatomic"))
        check("T13_atomic_failed_build_raises", raised13)
        check("T13_atomic_existing_output_preserved", still_intact)
        check("T13_atomic_no_partial_files", not stage_left, f"staging dir left: {stage_left}")

        # ---------------- T14: assignment corruption with totals preserved ------
        # swap two bars' num_trades (totals unchanged) -> replay must detect it
        def m_swap_assignment(t):
            i = t.schema.get_field_index("num_trades")
            arr = t.column("num_trades").to_numpy().copy()
            arr[0], arr[1] = arr[1], arr[0]
            return t.set_column(i, "num_trades", pa.array(arr))

        tmp14 = tempfile.mkdtemp(prefix="dibtest_corrupt14_")
        base14 = make_gen_tree(tmp14, "SOLUSDT", cfg["id"], out_path8)
        pq.write_table(m_swap_assignment(pq.read_table(out_path8)), base14)
        r14 = dib_validate.validate_one("SOLUSDT", cfg["id"], bars_dir=tmp14, parquet_dir=parquet_dir8)
        c14 = {c["name"]: c for c in r14["checks"]}
        swap_detected = not c14["replay_num_trades"]["ok"]
        partition_still_passes = c14["partitioning"]["ok"]
        check("T14_detect_swapped_assignment", swap_detected,
              f"replay_num_trades ok={c14['replay_num_trades']['ok']}")
        check("T14_partitioning_unaffected_by_swap", partition_still_passes)
        shutil.rmtree(tmp14, ignore_errors=True)

        # ---------------- T15: explicit JIT compilation + kernel equivalence ----
        try:
            dib_configs.check_environment()
            check("T15_environment_jit_probe", True)
        except Exception as exc:
            check("T15_environment_jit_probe", False, str(exc))
        import llvmlite
        import numba
        print(f"   env: numpy={np.__version__} numba={numba.__version__} "
              f"llvmlite={llvmlite.__version__} pyarrow={pq.__version__ if hasattr(pq,'__version__') else '?'}")
        # direct kernel-vs-kernel equivalence on identical small inputs
        rng15 = np.random.RandomState(7)
        n15 = 120
        pr15 = np.round(rng15.rand(n15) * 3 + 98.0, 2)
        qt15 = np.round(rng15.rand(n15) * 5 + 0.1, 4)
        bm15 = rng15.rand(n15) > 0.5
        tt15 = (30000 + np.arange(n15) * 10).astype(np.int64)
        aid15 = (40000 + np.arange(n15)).astype(np.int64)

        def _kernel_state():
            w, ei, npb_ = 5, 5, 3
            return (0.0, 0, 10.0, np.nan, np.empty(w), 0, 0, np.zeros(npb_), 0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0, 0), w, ei, npb_

        def _call_builder(price, qty, bm, tt):
            st, w, ei, npb_ = _kernel_state()
            outs = [np.empty(128) for _ in range(13)]
            r = bars_mod._dib_batch(price, qty, bm, tt, *st, *outs, w, ei, 0.0, 10.0, npb_)
            n = r[14]
            return [o[:n] for o in outs], list(r[0:7]), list(r[7:14])

        def _call_replay(price, qty, bm, tt, aid):
            st, w, ei, npb_ = _kernel_state()
            outs = [np.empty(128) for _ in range(13)]
            eqids = np.empty(16, dtype=np.int64)
            r = dib_replay._replay_batch(price, qty, bm, tt, aid, *st, *outs,
                                         w, ei, 0.0, 10.0, npb_,
                                         -1, 0, 0, eqids, 0, -1, 0)
            n = r[14]
            return [o[:n] for o in outs], list(r[0:7]), list(r[7:14])

        ob, sb, cb = _call_builder(pr15, qt15, bm15, tt15)
        or_, sr, cr = _call_replay(pr15, qt15, bm15, tt15, aid15)
        bars_eq = all(np.allclose(a, b, rtol=1e-12) and len(a) == len(b) for a, b in zip(ob, or_))
        state_eq = all(abs(a - b) < 1e-12 for a, b in zip(sb, sr))
        cur_eq = all(abs(a - b) < 1e-12 for a, b in zip(cb, cr))
        check("T15_kernel_equivalence", bars_eq and state_eq and cur_eq,
              f"bars_eq={bars_eq} state_eq={state_eq} cur_eq={cur_eq}")

        # ---------------- T16: strict coverage (failed months count) ------------
        fake_manifest = os.path.join(work, "fake_manifest.json")
        with open(fake_manifest, "w") as fh:
            json.dump({"TESTSYM/2000-01": {"status": "ok"},
                       "TESTSYM/2000-02": {"status": "failed"},
                       "TESTSYM/2000-03": {"status": "not_found"}}, fh)
        parquet_dir16 = tempfile.mkdtemp(prefix="dibtest_cov_")
        make_input_parquet(parquet_dir16, "TESTSYM", seq_trades(30, start_id=1, start_tt=100))
        raised16a = False
        try:
            bars_mod.verify_input("TESTSYM", parquet_dir=parquet_dir16, manifest_path=fake_manifest)
        except ValueError as exc:
            raised16a = "MISSING" in str(exc)
        check("T16_failed_month_counts_as_expected", raised16a)
        raised16b = False
        try:
            bars_mod.verify_input("TESTSYM", parquet_dir=parquet_dir16,
                                  manifest_path=os.path.join(work, "no_such_manifest.json"))
        except ValueError as exc:
            raised16b = "missing" in str(exc)
        check("T16_missing_manifest_fails", raised16b)

        # ---------------- T17: time inversions are blocking ---------------------
        cfg17 = dict(cfg)
        cfg17["id"] = "testinv"
        rows_inv = seq_trades(40, start_id=30000, start_tt=30000)
        # trade 20 drops from tt 30020 to 28000: a 2s inversion vs the previous
        # trade's 30019 - well beyond the 1s blocking threshold
        rows_inv[20] = (rows_inv[20][0], rows_inv[20][1], rows_inv[20][2],
                        rows_inv[20][3], rows_inv[20][4], 28000, rows_inv[20][6])
        parquet_dir17, out_path17, meta17 = build_synthetic("SOLUSDT", rows_inv, cfg17, work)
        r17 = dib_validate.validate_one("SOLUSDT", cfg17["id"], bars_dir=work, parquet_dir=parquet_dir17)
        c17 = {c["name"]: c for c in r17["checks"]}
        check("T17_time_inversion_blocking", not c17["input_ordering"]["ok"],
              f"input_ordering detail: {c17['input_ordering']['detail']}")

        # ---------------- T18: corrupted tail fields detected -------------------
        cfg18 = dict(cfg)
        cfg18["id"] = "testtail"
        parquet_dir18, out_path18, meta18 = build_synthetic(
            "SOLUSDT", seq_trades(45, start_id=50000, start_tt=50000), cfg18, work)

        def tail_corrupt(which, expect_fail):
            tmp = tempfile.mkdtemp(prefix="dibtest_tail_")
            base = make_gen_tree(tmp, "SOLUSDT", cfg18["id"], out_path18)
            t = json.load(open(base + ".tail.json"))
            if which == "qvol":
                t["quote_volume"] += 1e6
            elif which == "otime":
                t["open_time"] += 999999
            elif which == "ctime":
                t["close_time"] += 999999
            with open(base + ".tail.json", "w") as fh:
                json.dump(t, fh)
            r = dib_validate.validate_one("SOLUSDT", cfg18["id"], bars_dir=tmp, parquet_dir=parquet_dir18)
            failed = [c["name"] for c in r["checks"] if not c["ok"]]
            check(f"T18_detect_tail_{which}", expect_fail in failed, f"failed: {failed}")
            shutil.rmtree(tmp, ignore_errors=True)

        tail_corrupt("qvol", "replay_tail")
        tail_corrupt("otime", "replay_tail")
        tail_corrupt("ctime", "replay_tail")

        # ---------------- T19: missing state artifact ---------------------------
        tmp19 = tempfile.mkdtemp(prefix="dibtest_nostate_")
        base19 = make_gen_tree(tmp19, "SOLUSDT", cfg18["id"], out_path18)
        os.remove(base19 + ".state.npz")
        r19 = dib_validate.validate_one("SOLUSDT", cfg18["id"], bars_dir=tmp19, parquet_dir=parquet_dir18)
        failed19 = [c["name"] for c in r19["checks"] if not c["ok"]]
        check("T19_detect_missing_state", "artifacts" in failed19, f"failed: {failed19}")
        shutil.rmtree(tmp19, ignore_errors=True)

        # ---------------- T20: publication failure preserves all four artifacts --
        # Build into a dedicated bars dir; make the dir read-only so the rebuild
        # fails during staging/publication; all four previous artifacts must
        # survive byte-identical and no staging dir may remain.
        cfg20 = dict(cfg)
        cfg20["id"] = "testpub"
        parquet_dir20 = tempfile.mkdtemp(prefix="dibtest_pubin_")
        make_input_parquet(parquet_dir20, "SOLUSDT", seq_trades(45, start_id=60000, start_tt=60000))
        out_dir20 = os.path.join(work, "SOLUSDT-pubtest")
        os.makedirs(out_dir20, exist_ok=True)
        bars_mod.build_dib("SOLUSDT", cfg20, bars_dir=out_dir20, parquet_dir=parquet_dir20)
        out_path20 = bars_mod.resolve_generation("SOLUSDT", cfg20["id"], out_dir20)
        snap20 = {
            "parquet": open(out_path20, "rb").read(),
            ".tail.json": open(out_path20 + ".tail.json", "rb").read(),
            ".state.npz": open(out_path20 + ".state.npz", "rb").read(),
            "meta": open(out_path20.replace(".parquet", "-meta.json"), "rb").read(),
        }
        os.chmod(os.path.join(out_dir20, "SOLUSDT"), 0o555)  # read-only: staging cannot be created
        raised20 = False
        try:
            bars_mod.build_dib("SOLUSDT", cfg20, bars_dir=out_dir20, parquet_dir=parquet_dir20)
        except (OSError, ValueError):
            raised20 = True
        os.chmod(os.path.join(out_dir20, "SOLUSDT"), 0o755)
        intact20 = (open(out_path20, "rb").read() == snap20["parquet"]
                    and open(out_path20 + ".tail.json", "rb").read() == snap20[".tail.json"]
                    and open(out_path20 + ".state.npz", "rb").read() == snap20[".state.npz"]
                    and open(out_path20.replace(".parquet", "-meta.json"), "rb").read() == snap20["meta"])
        stage20_gone = not os.path.exists(os.path.join(out_dir20, ".staging-testpub"))
        check("T20_publish_failure_raises", raised20)
        check("T20_publish_previous_generation_preserved", intact20)
        check("T20_publish_staging_cleaned", stage20_gone)

        # ---------------- T21: replay across the actual 500,000-row boundary ----
        cfg21 = dict(cfg)
        cfg21["id"] = "testbig"
        parquet_dir21, out_path21, meta21 = build_synthetic(
            "SOLUSDT", seq_trades(1_000_001, start_id=100000, start_tt=100000), cfg21, work)
        r21 = dib_validate.validate_one("SOLUSDT", cfg21["id"], bars_dir=work, parquet_dir=parquet_dir21)
        c21 = {c["name"]: c for c in r21["checks"]}
        ok21 = all(c["ok"] for c in r21["checks"])
        bars21 = pq.read_table(out_path21)
        n21 = bars21.num_rows
        nt21 = bars21.column("num_trades").to_numpy()
        boundary_span = (n21 == 1_000_001 // 11 and int(nt21.sum()) + r21["characterization"]["tail_trades"] == 1_000_001
                         and nt21[1_000_001 // 11 // 2] == 11)
        check("T21_replay_across_500k_boundary", ok21 and boundary_span,
              f"ok={ok21} bars={n21} boundary_span={boundary_span} "
              f"failed={[c['name'] for c in r21['checks'] if not c['ok']]}")

        # ---------------- T22: OHLC-swap assignment detection -------------------
        # randomized prices/quantities so every bar has DISTINCT OHLC (regular
        # patterns make the swap invisible - the validator would correctly pass)
        cfg22 = dict(cfg)
        cfg22["id"] = "testohlcswap"
        rng22 = np.random.RandomState(22)
        n22 = 400
        rows22 = [(i + 1, float(np.round(rng22.rand() * 3 + 98.0, 2)),
                   float(np.round(rng22.rand() * 5 + 0.1, 4)), i + 1, i + 1,
                   70000 + i, bool(rng22.rand() > 0.5)) for i in range(n22)]
        parquet_dir22, out_path22, meta22 = build_synthetic("SOLUSDT", rows22, cfg22, work)

        def m_swap_ohlc(t):
            for cname in ("open", "high", "low", "close"):
                i = t.schema.get_field_index(cname)
                arr = t.column(cname).to_numpy().copy()
                arr[0], arr[1] = arr[1], arr[0]
                t = t.set_column(i, cname, pa.array(arr))
            return t

        tmp22 = tempfile.mkdtemp(prefix="dibtest_ohlcswap_")
        base22 = make_gen_tree(tmp22, "SOLUSDT", cfg22["id"], out_path22)
        pq.write_table(m_swap_ohlc(pq.read_table(out_path22)), base22)
        r22 = dib_validate.validate_one("SOLUSDT", cfg22["id"], bars_dir=tmp22, parquet_dir=parquet_dir22)
        failed22 = [c["name"] for c in r22["checks"] if not c["ok"]]
        check("T22_detect_swapped_ohlc", "replay_ohlc" in failed22, f"failed: {failed22}")
        shutil.rmtree(tmp22, ignore_errors=True)

        # ---------------- T23: failures at multiple publication stages ----------
        # Readers must always resolve one complete generation (old or new).
        cfg23 = dict(cfg)
        cfg23["id"] = "testgen"
        parquet_dir23, out_path23, meta23 = build_synthetic(
            "SOLUSDT", seq_trades(45, start_id=80000, start_tt=80000), cfg23, work)
        gen1 = meta23["generation"]
        cid_dir23 = os.path.join(work, "SOLUSDT", cfg23["id"])

        def gen_snapshot(base):
            snap = {s: open(base + s, "rb").read()
                    for s in ("", ".tail.json", ".state.npz")}
            snap["meta"] = open(base.replace(".parquet", "-meta.json"), "rb").read()
            return snap

        snap_gen1 = gen_snapshot(out_path23)

        # (a) staging-initialization failure: squat the staging path with a FILE
        blocker23 = os.path.join(work, "SOLUSDT", ".staging-testgen")
        with open(blocker23, "w") as fh:
            fh.write("x")
        raised23a = False
        try:
            bars_mod.build_dib("SOLUSDT", cfg23, bars_dir=work, parquet_dir=parquet_dir23)
        except OSError:
            raised23a = True
        os.remove(blocker23)
        cur23a = bars_mod.resolve_generation("SOLUSDT", cfg23["id"], work)
        intact23a = (os.path.basename(os.path.dirname(cur23a)) == gen1
                     and gen_snapshot(cur23a) == snap_gen1)
        check("T23_stage_init_failure", raised23a and intact23a,
              f"raised={raised23a} old_gen_intact={intact23a}")

        # (b) switch-stage failure: squat .current.tmp with a DIRECTORY
        tmpblock = os.path.join(cid_dir23, ".current.tmp")
        os.makedirs(tmpblock)
        raised23b = False
        try:
            bars_mod.build_dib("SOLUSDT", cfg23, bars_dir=work, parquet_dir=parquet_dir23)
        except OSError:
            raised23b = True
        cur23b = bars_mod.resolve_generation("SOLUSDT", cfg23["id"], work)
        intact23b = (os.path.basename(os.path.dirname(cur23b)) == gen1
                     and gen_snapshot(cur23b) == snap_gen1)
        leftover_tmp = os.path.lexists(tmpblock)
        check("T23_switch_stage_failure", raised23b and intact23b and not leftover_tmp,
              f"raised={raised23b} old_gen_intact={intact23b} tmp_left={leftover_tmp}")

        # (c) successful rebuild publishes gen-2 and PRESERVES gen-1 on disk
        meta23c = bars_mod.build_dib("SOLUSDT", cfg23, bars_dir=work, parquet_dir=parquet_dir23)
        gen2 = meta23c["generation"]
        cur23c = bars_mod.resolve_generation("SOLUSDT", cfg23["id"], work)
        gen1_still_there = os.path.exists(os.path.join(cid_dir23, gen1,
                                                       "SOLUSDT-testgen-dib.parquet"))
        switched = os.path.basename(os.path.dirname(cur23c)) == gen2
        check("T23_gen2_published_gen1_preserved", switched and gen1_still_there
              and gen_snapshot(out_path23) == snap_gen1,
              f"switched={switched} gen1_preserved={gen1_still_there}")

        # readers resolve the NEW generation as one complete set
        r23 = dib_validate.validate_one("SOLUSDT", cfg23["id"], bars_dir=work, parquet_dir=parquet_dir23)
        ok23 = all(c["ok"] for c in r23["checks"])
        check("T23_reader_sees_complete_new_generation", ok23,
              [c for c in r23["checks"] if not c["ok"]])

    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{PASS.__len__()} passed, {FAIL.__len__()} failed")
    if FAIL:
        print("FAILED:", FAIL)
    return 1 if FAIL else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
