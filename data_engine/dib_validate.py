"""
data_engine/dib_validate.py
===========================
Validation battery for Dollar Imbalance Bar outputs.

For every (symbol, config):
  CORRECTNESS CHECKS (blocking; exit code reflects these only):
    - schema: expected 13 columns, dtypes, zero nulls
    - partitioning: sum(num_trades) + tail.num_trades == input trade count
    - base-volume conservation: sum(volume) + tail.volume == input sum(quantity)
    - quote-volume conservation: sum(quote_volume) + tail.quote_volume == input sum(price*qty)
    - signed-imbalance conservation: sum(imbalance) + tail.imbalance == input sum(b*p*q)
    - threshold replay: |imbalance| > threshold and threshold == E_T*|E_x| for every bar
    - OHLC sanity: low <= open, close <= high; volume > 0; num_trades >= 1
    - chronology: open_time <= close_time; bars ordered; no gaps between bars
    - tail structure: fields present and consistent
    - duplicate aggregate-id treatment: documented duplicate ids present in the
      input and fully partitioned (nothing dropped)
    - metadata consistency: recorded parameters match the config matrix
  CHARACTERIZATION (report-only; no pass/fail targets):
    bars/day, trades/bar percentiles, bar durations, E_T/E_x distributions,
    corr(bar index, T), tail accounting.

Run:
    python -m data_engine.dib_validate [--symbols ...] [--configs ...]
    python -m data_engine.dib_validate --bars-dir <dir> --parquet-dir <dir>
"""

import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_engine import config as data_config
from data_engine import dib_configs

EXPECTED_COLUMNS = [
    ("open_time", "int64"), ("close_time", "int64"),
    ("open", "double"), ("high", "double"), ("low", "double"), ("close", "double"),
    ("volume", "double"), ("quote_volume", "double"), ("imbalance", "double"),
    ("num_trades", "int64"),
    ("E_T", "double"), ("E_x", "double"), ("threshold", "double"),
]

KNOWN_DUPLICATE_IDS = {"SOLUSDT": {926014272}}  # documented upstream duplicate agg id


def _rel_tol(x):
    return 1e-7 * max(1.0, abs(x))


def _input_totals(symbol, parquet_dir):
    """Stream the canonical input parquets: rows, sum(qty), sum(p*q), sum(signed p*q),
    and (informational) duplicate agg ids."""
    totals = {"rows": 0, "qty": 0.0, "pq": 0.0, "signed_pq": 0.0}
    dup_ids = {}
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(parquet_dir, symbol, f"{symbol}-aggTrades-*.parquet")))
    if not files:
        raise ValueError(f"no input parquet for {symbol} under {parquet_dir}")
    prev_id = None
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=1 << 20,
                                 columns=["agg_trade_id", "price", "quantity", "is_buyer_maker"]):
            n = b.num_rows
            totals["rows"] += n
            pr = b.column("price").to_numpy(zero_copy_only=False)
            qt = b.column("quantity").to_numpy(zero_copy_only=False)
            bm = b.column("is_buyer_maker").to_numpy(zero_copy_only=False)
            aid = b.column("agg_trade_id").to_numpy(zero_copy_only=False)
            totals["qty"] += float(qt.sum())
            totals["pq"] += float((pr * qt).sum())
            sign = np.where(bm, -1.0, 1.0)
            totals["signed_pq"] += float((sign * pr * qt).sum())
            if n > 1:
                eq = np.where(aid[1:] == aid[:-1])[0]
                for i in eq.tolist():
                    dup_ids[int(aid[i + 1])] = dup_ids.get(int(aid[i + 1]), 0) + 1
            if prev_id is not None and int(aid[0]) == prev_id:
                dup_ids[prev_id] = dup_ids.get(prev_id, 0) + 1
            prev_id = int(aid[-1])
    return totals, dup_ids


def validate_one(symbol, cid, bars_dir=None, parquet_dir=None):
    bars_dir = bars_dir or dib_configs.OUT_DIR
    parquet_dir = parquet_dir or data_config.PARQUET_DIR
    cfg = dib_configs.get_config(cid) if cid in dib_configs.CONFIGS else None

    # resolve all four artifacts through the current-generation pointer
    from data_engine.bars import resolve_generation
    base = resolve_generation(symbol, cid, bars_dir)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    # --- load artifacts ---
    if base is None or not os.path.exists(base):
        check("artifacts", False, f"no current generation for {symbol}/{cid} "
              f"(resolve_generation -> {base})")
        return {"symbol": symbol, "config": cid, "checks": checks, "characterization": {}}
    tail_path = base + ".tail.json"
    state_path = base + ".state.npz"
    meta_path = base.replace(".parquet", "-meta.json")
    if not os.path.exists(tail_path):
        check("artifacts", False, "missing .tail.json")
    if not os.path.exists(state_path):
        check("artifacts", False, "missing .state.npz")
    if not os.path.exists(meta_path):
        check("artifacts", False, "missing -meta.json")
    if not all(c["ok"] for c in checks):
        return {"symbol": symbol, "config": cid, "checks": checks, "characterization": {}}

    bars = pq.read_table(base)
    n_bars = bars.num_rows
    tail = json.load(open(tail_path))
    try:
        np.load(state_path)
        check("state_loads", True)
    except Exception as exc:
        check("state_loads", False, str(exc))
    meta = json.load(open(meta_path))

    # --- schema / nulls ---
    names = [(f.name, str(f.type)) for f in bars.schema]
    check("schema_columns", names == EXPECTED_COLUMNS, f"got {names[:4]}...")
    nulls = {c: bars.column(c).null_count for c in bars.column_names}
    check("no_nulls", all(v == 0 for v in nulls.values()), {k: v for k, v in nulls.items() if v})

    # --- input totals ---
    try:
        totals, dup_ids = _input_totals(symbol, parquet_dir)
    except ValueError as exc:
        check("input_read", False, str(exc))
        return {"symbol": symbol, "config": cid, "checks": checks, "characterization": {}}
    check("input_read", True, f"{totals['rows']:,} rows")

    nt = bars.column("num_trades").to_numpy().astype(np.int64)
    vol = bars.column("volume").to_numpy()
    qvol = bars.column("quote_volume").to_numpy()
    imb = bars.column("imbalance").to_numpy()
    et = bars.column("E_T").to_numpy()
    ex = bars.column("E_x").to_numpy()
    thr = bars.column("threshold").to_numpy()
    ot = bars.column("open_time").to_numpy()
    ct = bars.column("close_time").to_numpy()
    op = bars.column("open").to_numpy()
    hi = bars.column("high").to_numpy()
    lo = bars.column("low").to_numpy()
    cl = bars.column("close").to_numpy()

    # --- partitioning / conservation ---
    tail_trades = int(tail.get("num_trades", 0))
    check("partitioning",
          int(nt.sum()) + tail_trades == totals["rows"],
          f"bar trades {int(nt.sum()):,} + tail {tail_trades:,} vs input {totals['rows']:,}")
    check("base_volume_conservation",
          abs(float(vol.sum()) + float(tail.get("volume", 0.0)) - totals["qty"]) <= _rel_tol(totals["qty"]),
          f"bars+tail {float(vol.sum()) + float(tail.get('volume', 0.0)):.4f} vs input {totals['qty']:.4f}")
    check("quote_volume_conservation",
          abs(float(qvol.sum()) + float(tail.get("quote_volume", 0.0)) - totals["pq"]) <= _rel_tol(totals["pq"]),
          f"bars+tail {float(qvol.sum()) + float(tail.get('quote_volume', 0.0)):.4f} vs input {totals['pq']:.4f}")
    check("signed_imbalance_conservation",
          abs(float(imb.sum()) + float(tail.get("imbalance", 0.0)) - totals["signed_pq"]) <= _rel_tol(totals["signed_pq"]),
          f"bars+tail {float(imb.sum()) + float(tail.get('imbalance', 0.0)):.4f} vs input {totals['signed_pq']:.4f}")

    # --- threshold replay ---
    if n_bars > 0:
        replayed_ok = True
        bad_rows = []
        for i in range(n_bars):
            if not (abs(imb[i]) > thr[i] and abs(thr[i] - et[i] * abs(ex[i])) <= 1e-9 * max(1.0, abs(thr[i]))):
                replayed_ok = False
                if len(bad_rows) < 5:
                    bad_rows.append(i)
        check("threshold_replay", replayed_ok, f"violations at bar indices {bad_rows}" if bad_rows else "")

        # --- OHLC / chronology ---
        check("ohlc_sanity",
              bool((lo <= op).all() and (lo <= cl).all() and (cl <= hi).all() and (op <= hi).all()
                   and (vol > 0).all() and (nt >= 1).all()),
              "low<=open,close<=high and volume>0 and num_trades>=1")
        check("bar_chronology",
              bool((ot <= ct).all() and (np.diff(ot) >= 0).all() and (ct[:-1] <= ot[1:]).all()),
              "open<=close, ordered opens, close_k <= open_{k+1}")
    else:
        check("threshold_replay", True, "no bars emitted")
        check("ohlc_sanity", True, "no bars emitted")
        check("bar_chronology", True, "no bars emitted")

    # --- tail structure ---
    tail_ok = (tail_trades >= 0 and "volume" in tail and "quote_volume" in tail
               and "imbalance" in tail and "open_time" in tail and "close_time" in tail)
    if tail_ok and tail_trades > 0:
        tail_ok = tail_ok and tail["volume"] > 0 and tail["close_time"] >= tail["open_time"]
    check("tail_structure", tail_ok, f"tail trades {tail_trades:,}")

    # --- duplicate aggregate-id treatment ---
    # If a documented duplicate id is present in the input, both executions must
    # be there (exactly one adjacent equal-id pair) and nothing may be dropped
    # (guaranteed by partitioning). Absent from the input: informational only.
    known = KNOWN_DUPLICATE_IDS.get(symbol, set())
    present = {k: v for k, v in dup_ids.items() if k in known}
    if present:
        check("duplicate_id_treatment",
              all(v == 1 for v in present.values()),
              f"documented duplicate ids found in input: {present} (each needs exactly one equal-id pair)")
    else:
        check("duplicate_id_treatment", True, "no documented duplicate ids in this input")

    # --- metadata consistency ---
    mp = meta.get("resolved_parameters", {})
    if cfg is not None:
        meta_ok = (mp.get("E_T_init") == cfg["E_T_init"] and mp.get("min_bar_size") == cfg["min_bar_size"]
                   and mp.get("max_bar_size") == cfg["max_bar_size"]
                   and mp.get("exp_ticks_init") == dib_configs.EXP_NUM_TICKS_INIT
                   and mp.get("expected_imbalance_window") == dib_configs.EXPECTED_IMBALANCE_WINDOW
                   and mp.get("num_prev_bars") == dib_configs.NUM_PREV_BARS
                   and meta.get("total_bars") == n_bars)
    else:
        # custom/synthetic config: require the recorded parameters to be complete
        meta_ok = all(k in mp for k in ("E_T_init", "min_bar_size", "max_bar_size",
                                        "exp_ticks_init", "expected_imbalance_window",
                                        "num_prev_bars")) and meta.get("total_bars") == n_bars
    check("metadata_consistency", meta_ok,
          f"meta total_bars={meta.get('total_bars')} vs {n_bars}")

    # --- independent replay comparison -------------------------------------------
    # Re-derive bars from the actual input trades with a separate implementation
    # (data_engine/dib_replay.py) and compare every field. This verifies bar
    # assignment, first-threshold-crossing boundaries, OHLC/volumes/imbalance
    # against the ACTUAL assigned trades, and the adaptive E_x/E_T updates -
    # without trusting the builder's stored diagnostics.
    try:
        from data_engine import dib_replay
        cfg_from_meta = {
            "E_T_init": float(mp["E_T_init"]), "min_bar_size": float(mp["min_bar_size"]),
            "max_bar_size": float(mp["max_bar_size"]),
            "expected_imbalance_window": int(mp["expected_imbalance_window"]),
            "exp_ticks_init": int(mp["exp_ticks_init"]), "num_prev_bars": int(mp["num_prev_bars"]),
        }
        rep = dib_replay.replay(symbol, cfg_from_meta, parquet_dir=parquet_dir)
        check("replay_ok", True, f"replayed {len(rep['bars']['num_trades']):,} bars from {rep['input_rows']:,} trades")
    except Exception as exc:
        check("replay_ok", False, str(exc))
        return {"symbol": symbol, "config": cid, "checks": checks, "characterization": {}}

    rb = rep["bars"]
    if len(rb["num_trades"]) == 0:
        check("replay_bar_count", n_bars == 0, f"replay 0 bars vs stored {n_bars}")
    else:
        rtol = 1e-9
        def cmp(a_stored, a_replay):
            return (len(a_stored) == len(a_replay)
                    and bool(np.allclose(a_stored, a_replay, rtol=rtol, atol=1e-12)))

        check("replay_bar_count", n_bars == len(rb["num_trades"]),
              f"stored {n_bars} vs replayed {len(rb['num_trades'])}")
        check("replay_num_trades", cmp(nt, rb["num_trades"]),
              "bar assignments differ from first-threshold-crossing boundaries")
        check("replay_open_time", cmp(ot, rb["open_time"]))
        check("replay_close_time", cmp(ct, rb["close_time"]))
        check("replay_ohlc", cmp(op, rb["open"]) and cmp(hi, rb["high"])
              and cmp(lo, rb["low"]) and cmp(cl, rb["close"]))
        check("replay_volumes", cmp(vol, rb["volume"])
              and cmp(qvol, rb["quote_volume"]))
        check("replay_imbalance", cmp(imb, rb["imbalance"]))
        check("replay_E_T", cmp(et, rb["E_T"]), "adaptive E_T update diverges")
        check("replay_E_x", cmp(ex, rb["E_x"]), "E_x recalculation diverges")
        check("replay_threshold", cmp(thr, rb["threshold"]),
              "closing thresholds diverge from the independent recalculation")
        # complete tail comparison: all six fields, replayed vs stored
        rt = rep["tail"]
        tail_vol_ok = abs(float(rt["volume"]) - float(tail.get("volume", 0.0))) <= _rel_tol(totals["qty"])
        tail_qvol_ok = abs(float(rt["quote_volume"]) - float(tail.get("quote_volume", 0.0))) <= _rel_tol(totals["pq"])
        tail_imb_ok = abs(float(rt["imbalance"]) - float(tail.get("imbalance", 0.0))) <= _rel_tol(totals["signed_pq"])
        tail_time_ok = (int(rt["open_time"]) == int(tail.get("open_time", -1))
                        and int(rt["close_time"]) == int(tail.get("close_time", -1)))
        check("replay_tail",
              int(rt["num_trades"]) == tail_trades and tail_vol_ok and tail_qvol_ok
              and tail_imb_ok and tail_time_ok,
              f"replay tail {rt} vs stored {tail} "
              f"(vol_ok={tail_vol_ok} qvol_ok={tail_qvol_ok} imb_ok={tail_imb_ok} time_ok={tail_time_ok})")

    # input ordering facts from the replay: no decreasing agg ids; equal-id pairs
    # allowed only for the documented duplicate ids; transact_time inversions
    # EXCEEDING one second are BLOCKING (sub-second jitter is a documented
    # source-specific allowance; equal timestamps are acceptable).
    equal_ids = set(rep.get("equal_ids", []))
    equal_ok = all(e in known for e in equal_ids)
    check("input_ordering",
          rep["id_decreasing"] == 0 and equal_ok and rep["time_inversions"] == 0,
          f"decreasing_ids={rep['id_decreasing']} equal_ids={sorted(equal_ids)} "
          f"(documented={sorted(known) if known else 'none'}); "
          f"time_inversions>1s={rep['time_inversions']} (blocking if >0)")

    # --- characterization (report-only, no targets) ---
    char = {
        "bars": int(n_bars),
        "input_rows": totals["rows"],
        "tail_trades": tail_trades,
        "tail_volume": float(tail.get("volume", 0.0)),
        "tail_quote_volume": float(tail.get("quote_volume", 0.0)),
        "tail_imbalance": float(tail.get("imbalance", 0.0)),
    }
    if n_bars > 0:
        days = (ot.max() - ot.min()) / 86400000
        char["bars_per_day"] = (n_bars / days) if days > 0 else None
        char["trades_per_bar"] = {
            "p25": float(np.percentile(nt, 25)), "median": float(np.median(nt)),
            "p75": float(np.percentile(nt, 75)), "p95": float(np.percentile(nt, 95)),
            "max": float(nt.max()),
        }
        char["duration_hours"] = {
            "median": float(np.median((ct - ot) / 3.6e6)),
            "p95": float(np.percentile((ct - ot) / 3.6e6, 95)),
            "max": float((ct - ot).max() / 3.6e6),
        }
        mp_min = mp.get("min_bar_size", 0.0)
        mp_max = mp.get("max_bar_size", 0.0)
        char["E_T"] = {"median": float(np.median(et)), "p95": float(np.percentile(et, 95)),
                       "max": float(et.max()),
                       "frac_at_min": float((et <= mp_min * 1.001).mean()) if mp_min > 0 else None,
                       "frac_at_max": float((et >= mp_max * 0.999).mean())}
        char["abs_E_x"] = {"median": float(np.median(np.abs(ex))),
                           "p95": float(np.percentile(np.abs(ex), 95)),
                           "max": float(np.abs(ex).max())}
        char["corr_bar_index_vs_T"] = float(np.corrcoef(np.arange(n_bars), nt)[0, 1]) if n_bars > 1 else None

    return {"symbol": symbol, "config": cid, "checks": checks, "characterization": char}


def main():
    ap = argparse.ArgumentParser(description="DIB output validation battery")
    ap.add_argument("--symbols", nargs="*", default=data_config.SYMBOLS)
    ap.add_argument("--configs", nargs="*", default=list(dib_configs.CONFIGS))
    ap.add_argument("--bars-dir", default=dib_configs.OUT_DIR)
    ap.add_argument("--parquet-dir", default=data_config.PARQUET_DIR)
    ap.add_argument("--report", default="data/dib_validation_report.json")
    args = ap.parse_args()

    results = {}
    all_ok = True
    for sym in args.symbols:
        for cid in args.configs:
            r = validate_one(sym, cid, bars_dir=args.bars_dir, parquet_dir=args.parquet_dir)
            results[f"{sym}/{cid}"] = r
            ok = all(c["ok"] for c in r["checks"])
            all_ok = all_ok and ok
            print(f"[{'PASS' if ok else 'FAIL'}] {sym}/{cid}")
            for c in r["checks"]:
                if not c["ok"]:
                    print(f"    - {c['name']}: {c['detail']}")

    with open(args.report, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"report: {args.report}")
    print("overall:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
