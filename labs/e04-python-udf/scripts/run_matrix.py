"""Smoke (1M rows, 3 iters + 2 warmups) for all options; full (10M rows, 5 iters) for top-2 options per UDF kind."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *

KINDS = ["scalar", "pandas", "arrow"]; BODIES = ["cheap", "expensive"]; THREADS = [1, 8, 32]
OUT = os.path.join(LAB, "results")

def save(name, meta, cfg, runs):
    json.dump({**meta, "config": cfg, "runs": runs}, open(os.path.join(OUT, name), "w"), indent=1)

def group(opts, kind, body, th, rows, iters, warm, timeout=3000):
    """Run one (kind, body, threads) cell for several options back-to-back under ONE bench-lock hold."""
    import subprocess
    code = ("import sys,json;sys.path.insert(0,%r);from common import *;"
            "print(json.dumps([run_cfg(o,%r,%r,%d,%d,%d,%d,%d) for o in %r]))") % (os.path.dirname(os.path.abspath(__file__)), kind, body, th, rows, iters, warm, timeout, list(opts))
    p = subprocess.run(["flock", "/home/gopal/krishiv-data/.bench.lock", sys.executable, "-c", code],
                       env={**os.environ, "E04_LOCKED": "1"}, capture_output=True, text=True)
    return json.loads(p.stdout.strip().splitlines()[-1])

def smoke():
    meta = metadata(); runs = []
    cfg = dict(rows=1_000_000, iters=3, warmups=2, batch_rows=8192, threads=THREADS)
    for kind in KINDS:
        for body in BODIES:
            for th in THREADS:
                opts = "ABCD" if (kind == "scalar" or th == 1) else "ABD"
                for r in group(opts, kind, body, th, 1_000_000, 3, 2):
                    runs.append(r)
                    print(r["option"], kind, body, th, r.get("status"), round(r.get("rows_per_s_median", 0)), flush=True)
                save("smoke.json", meta, cfg, runs)
    return runs

def full(smoke_runs):
    meta = metadata(); runs = []
    cfg = dict(rows=10_000_000, iters=5, warmups=2, batch_rows=8192, threads=THREADS)
    sel = {}
    for kind in KINDS:
        for body in BODIES:
            best = {}
            for r in smoke_runs:
                if r["kind"] == kind and r["body"] == body and r.get("status") == "ok":
                    best[r["option"]] = max(best.get(r["option"], 0), r["rows_per_s_median"])
            top = sorted(best, key=lambda o: -best[o])[:2]
            sel[f"{kind}/{body}"] = {"ranking_by_smoke_best_rows_per_s": best, "top2": top}
            extra = [o for o in ("A", "D") if o not in top and kind != "scalar" or (o == "D" and o not in top)]
            for opt in top + [e for e in extra if e not in top]:
                for th in THREADS:
                    r = group([opt], kind, body, th, 10_000_000, 5, 2)[0]
                    r["selected_as"] = "top2" if opt in top else "extra (needed for D-overhead decision)"
                    runs.append(r)
                    print("FULL", opt, kind, body, th, r.get("status"), round(r.get("rows_per_s_median", 0)), flush=True)
                    save("full.json", meta, {**cfg, "selection": sel}, runs)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        full(json.load(open(os.path.join(OUT, "smoke.json")))["runs"])
    else:
        smoke()
