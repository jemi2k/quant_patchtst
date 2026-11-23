"""
data_engine/bars_sensitivity.py  (v2)
=====================================
Diagnostic experiments AND the unified DIB construction runner.

The bar-construction kernel lives in data_engine/bars.py (single implementation,
no duplication); the configuration matrix lives in data_engine/dib_configs.py.

Experiment 1 - lower-bound sensitivity (two-sided bounded dynamic E_T):
    E_T_min in {0, 100, 500, 1000, 2000, 5000, 7000, 9000}, E_T_max = 10,000.
    Diagnostic only; outputs are temporary and removed after reporting.

Experiment 2 - constant-E_T baseline:
    T0 in {5000, 10000, 20000, 28000} (min = max = T0 pins E_T).
    Diagnostic only; outputs are temporary and removed after reporting.

Unified matrix runner (persistent outputs):
    python -m data_engine.bars_sensitivity --run-matrix [--symbols ...] [--configs ...]
    For every (symbol, config): builds bars under
        data/bars/{SYM}/{SYM}-{config_id}-dib.parquet
        + .tail.json, .state.npz, -meta.json
    and writes per-run logs to logs/dib_{SYM}_{config_id}.log
"""

import argparse
import logging
import os
import shutil
import tempfile

import numpy as np
import pyarrow.parquet as pq

from data_engine import config as data_config
from data_engine import dib_configs
from data_engine.bars import build_dib, resolve_generation

logger = logging.getLogger("data_engine.bars_sensitivity")

PROGRESS_EVERY = 5_000_000  # progress prints are handled by the kernel caller; kept for logs


def _exp1_config(mn):
    return {
        "id": f"exp1_min{mn:g}",
        "name": f"Experiment 1: min={mn:g}, max=10,000",
        "E_T_init": float(dib_configs.EXP_NUM_TICKS_INIT),
        "min_bar_size": float(mn),
        "max_bar_size": 10_000.0,
        "dynamic": True,
    }


def _exp2_config(t0):
    return {
        "id": f"exp2_T0{t0:g}",
        "name": f"Experiment 2: constant E_T = {t0:g}",
        "E_T_init": float(t0),
        "min_bar_size": float(t0),
        "max_bar_size": float(t0),
        "dynamic": False,
    }


def report(path, label, min_bar_size, max_bar_size):
    """Characterize a built bars parquet. Robust to 0 and 1 bar."""
    t = pq.read_table(path)
    n = t.num_rows
    if n == 0:
        print(f"===== {label} =====")
        print("total bars: 0 (no bars emitted)")
        print()
        return {"label": label, "n": 0}

    nt = t.column("num_trades").to_numpy().astype(float)
    ot = t.column("open_time").to_numpy()
    ct = t.column("close_time").to_numpy()
    E_T = t.column("E_T").to_numpy()
    E_x = t.column("E_x").to_numpy()
    durs = (ct - ot) / 3600000.0
    ax = np.abs(E_x)
    med_ET = float(np.median(E_T))
    has_min = min_bar_size > 0

    days = (ot.max() - ot.min()) / 86400000
    bpd = (n / days) if days > 0 else np.nan

    print(f"===== {label} =====")
    print(f"total bars: {n:,}")
    print(f"bars/day: {bpd:.2f}  (over {days:.0f} days)" if days > 0 else "bars/day: n/a (single bar)")
    print(f"num_trades: p25={np.percentile(nt,25):,.0f}  median={np.median(nt):,.0f}  p75={np.percentile(nt,75):,.0f}  p95={np.percentile(nt,95):,.0f}  max={nt.max():,.0f}")
    print(f"duration(h): median={np.median(durs):.2f}  p95={np.percentile(durs,95):.2f}  max={durs.max():.2f}")
    print(f"E_T: median={np.median(E_T):,.0f}  p95={np.percentile(E_T,95):,.0f}  max={E_T.max():,.0f}")
    if has_min:
        print(f"fraction E_T at min: {(E_T <= min_bar_size*1.001).mean()*100:.1f}%")
    print(f"fraction E_T at max: {(E_T >= max_bar_size*0.999).mean()*100:.1f}%")
    print(f"|E_x|: median={np.median(ax):,.0f}  p95={np.percentile(ax,95):,.0f}  max={ax.max():,.0f}")
    if n > 1:
        print(f"corr(bar_index, T): {np.corrcoef(np.arange(n), nt)[0,1]:.3f}")
    else:
        print("corr(bar_index, T): n/a (single bar)")
    print(f"median(T)/median(E_T): {np.median(nt)/med_ET:.2f}")
    print()
    return {"label": label, "n": n, "bpd": bpd, "medT": float(np.median(nt)),
            "p95T": float(np.percentile(nt, 95)), "maxT": float(nt.max()),
            "medDur": float(np.median(durs)), "maxDur": float(durs.max()),
            "corr": float(np.corrcoef(np.arange(n), nt)[0, 1]) if n > 1 else np.nan,
            "medET": med_ET,
            "fracMin": float((E_T <= min_bar_size * 1.001).mean()) if has_min else np.nan,
            "fracMax": float((E_T >= max_bar_size * 0.999).mean()),
            "ratio": float(np.median(nt) / med_ET)}


def _run_temporary(symbol, cfg, label, min_bar_size, max_bar_size):
    """Build one diagnostic configuration into a temp dir, report, clean up."""
    tmpdir = tempfile.mkdtemp(prefix="dib_sens_")
    try:
        build_dib(symbol, cfg, bars_dir=tmpdir)
        base = resolve_generation(symbol, cfg["id"], tmpdir)
        return report(base, label, min_bar_size, max_bar_size)
    except ValueError as exc:
        # e.g. empty input glob - report as a failed diagnostic, not a crash
        print(f"===== {label} =====")
        print(f"FAILED: {exc}")
        print()
        return {"label": label, "n": None, "error": str(exc)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _summary(rows):
    print(f"{'label':>22} {'bars':>10} {'/day':>8} {'medT':>10} {'p95T':>10} {'maxT':>11} {'medDur':>7} {'maxDur':>8} {'corr':>6} {'medE_T':>9} {'%min':>6} {'%max':>6} {'medT/medE_T':>12}")
    for r in rows:
        if r.get("n") is None:
            print(f"{r['label']:>22} {'FAILED':>10}")
            continue
        if r["n"] == 0:
            print(f"{r['label']:>22} {0:>10}")
            continue
        fm = f"{r['fracMin']*100:>5.1f}" if r['fracMin'] == r['fracMin'] else "   N/A"
        corr = f"{r['corr']:>6.3f}" if r['corr'] == r['corr'] else "   N/A"
        bpd = f"{r['bpd']:>8.2f}" if r['bpd'] == r['bpd'] else "     N/A"
        print(f"{r['label']:>22} {r['n']:>10,} {bpd} {r['medT']:>10,.0f} {r['p95T']:>10,.0f} {r['maxT']:>11,.0f} {r['medDur']:>7.2f} {r['maxDur']:>8.1f} {corr} {r['medET']:>9,.0f} {fm:>6} {r['fracMax']*100:>5.1f} {r['ratio']:>12.2f}")


def run_matrix(symbols, config_ids):
    """Unified runner: builds AND auto-validates every (symbol, config).

    Records build + validation status for all combinations, prints a final
    summary, writes data/bars/matrix_summary.json, and returns a nonzero exit
    status unless every required combination built and validated successfully.
    """
    import json

    from data_engine import dib_validate

    dib_configs.validate_methodology()
    dib_configs.check_environment()
    rows = []
    failures = 0

    for sym in symbols:
        for cid in config_ids:
            cfg = dib_configs.get_config(cid)
            row = {"symbol": sym, "config": cid, "output": None,
                   "generation": None, "build": "PENDING", "validation": "PENDING"}

            log_path = os.path.join("logs", f"dib_{sym}_{cid}.log")
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(fh)
            try:
                logger.info("start %s [%s]", sym, cid)
                meta = build_dib(sym, cfg)
                row["build"] = "OK"
                row["bars"] = meta["total_bars"]
                row["tail_trades"] = meta["tail"]["num_trades"]
                row["generation"] = meta["generation"]
                row["output"] = meta["published"]
                logger.info("built %s [%s]: %s bars, tail %s trades, gen %s",
                            sym, cid, meta["total_bars"], meta["tail"]["num_trades"], meta["generation"])
            except Exception as exc:
                row["build"] = f"FAILED: {type(exc).__name__}: {str(exc)[:200]}"
                logger.exception("build failed %s [%s]", sym, cid)
            else:
                try:
                    vr = dib_validate.validate_one(sym, cid)
                    ok = all(c["ok"] for c in vr["checks"])
                    row["validation"] = "PASS" if ok else "FAIL"
                    if not ok:
                        row["failed_checks"] = [c["name"] for c in vr["checks"] if not c["ok"]]
                    logger.info("validation %s [%s]: %s", sym, cid, row["validation"])
                except Exception as exc:
                    row["validation"] = f"ERROR: {type(exc).__name__}: {str(exc)[:200]}"
                    logger.exception("validation failed %s [%s]", sym, cid)
            finally:
                logger.removeHandler(fh)
                fh.close()

            if row["build"] != "OK" or row["validation"] != "PASS":
                failures += 1
            rows.append(row)
            print(f"[{row['build'] if row['build']=='OK' else 'FAIL'} / "
                  f"{row['validation']}] {sym} {cid} -> {row['output']} (gen {row['generation']})")

    # final summary
    os.makedirs(dib_configs.OUT_DIR, exist_ok=True)
    with open(os.path.join(dib_configs.OUT_DIR, "matrix_summary.json"), "w") as fh:
        json.dump({"combinations": rows,
                   "total": len(rows), "passed": len(rows) - failures, "failed": failures}, fh, indent=2)

    print("\n===== MATRIX SUMMARY =====")
    for r in rows:
        status = "PASS" if (r["build"] == "OK" and r["validation"] == "PASS") else "FAIL"
        detail = r.get("failed_checks") or (r["build"] if r["build"] != "OK" else r["validation"])
        print(f"  [{status}] {r['symbol']:10} {r['config']:10} bars={r.get('bars', '-'):>10} "
              f"tail={r.get('tail_trades', '-'):>10} {r['output']}  {detail if status=='FAIL' else ''}")
    print(f"TOTAL: {len(rows)} combinations, {len(rows) - failures} passed, {failures} failed")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="DIB diagnostic experiments and unified construction runner")
    parser.add_argument("--symbol", default="ETHUSDT", help="symbol for the diagnostic experiments")
    parser.add_argument("--experiment", choices=["1", "2", "both"], default="both",
                        help="which diagnostic experiment to run")
    parser.add_argument("--run-matrix", action="store_true",
                        help="build persistent bars for every (symbol, config)")
    parser.add_argument("--symbols", nargs="*", help="matrix symbols (default: all seven)")
    parser.add_argument("--configs", nargs="*", help="matrix config ids (default: all five)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.run_matrix:
        symbols = args.symbols or data_config.SYMBOLS
        config_ids = args.configs or list(dib_configs.CONFIGS)
        import sys
        sys.exit(run_matrix(symbols, config_ids))

    if args.experiment in ("1", "both"):
        print("\n########## Experiment 1: lower-bound sensitivity (E_T_max = 10,000) ##########")
        rows = []
        for mn in [0.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 7000.0, 9000.0]:
            cfg = _exp1_config(mn)
            rows.append(_run_temporary(args.symbol, cfg, cfg["name"], mn, 10_000.0))
        _summary(rows)
    if args.experiment in ("2", "both"):
        print("\n########## Experiment 2: constant-E_T baseline ##########")
        rows = []
        for t0 in [5000.0, 10000.0, 20000.0, 28000.0]:
            cfg = _exp2_config(t0)
            rows.append(_run_temporary(args.symbol, cfg, cfg["name"], t0, t0))
        _summary(rows)


if __name__ == "__main__":
    main()
