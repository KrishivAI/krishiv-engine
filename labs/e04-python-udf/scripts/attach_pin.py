"""Finding: naive per-call Python::attach (no pinned thread state) vs pinned worker threads, 1 thread, arrow cheap, 1M rows."""
import json, os, sys, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import common
from common import *
out = []
for pin in (True, False):
    for opt in "AB":
        a = host_args(opt, "arrow", "cheap", 1, 1_000_000, 3, 2) + ([] if pin else ["--no-pin"])
        p = subprocess.run(a, env=env_for(opt), capture_output=True, text=True)
        r = json.loads(p.stdout.strip().splitlines()[-1]); r["pinned_thread_state"] = pin
        r["loadavg"] = open("/proc/loadavg").read().split()[:3]
        out.append(r); print(opt, pin, round(r["rows_per_s_median"]/1e6, 1), "M rows/s p50", r["batch_latency_ms_p50"], flush=True)
json.dump({**metadata(), "runs": out}, open(os.path.join(LAB, "results", "attach_pin.json"), "w"), indent=1)
