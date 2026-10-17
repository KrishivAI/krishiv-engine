"""Crash blast radius: UDF segfaults (ctypes.string_at(0)) on a poison batch. Does the Rust host survive?"""
import json, os, signal, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

out = {**metadata(), "test": "ctypes.string_at(0) in UDF on batch 51 of 101; threads=1", "results": {}}
for opt in "ABCD":
    kind = "scalar" if opt == "C" else "arrow"
    p = subprocess.run(["flock","/home/gopal/krishiv-data/.bench.lock"]+host_args(opt, kind, "cheap", 1) + ["--crashtest"], env=env_for(opt), capture_output=True, text=True, timeout=300)
    lines = p.stdout.strip().splitlines()
    rc = p.returncode
    out["results"][opt] = {
        "host_returncode": rc,
        "host_died_by_signal": (signal.Signals(rc-128).name if rc > 128 else signal.Signals(-rc).name if rc < 0 else None)  # flock wrapper reports 128+sig,
        "host_survived": rc == 0 and any(l.startswith("PHASE3") for l in lines),
        "stdout": lines,
        "stderr_tail": p.stderr[-300:],
    }
    print(opt, rc, lines[-3:])
json.dump(out, open(os.path.join(LAB, "results", "crash.json"), "w"), indent=1)
