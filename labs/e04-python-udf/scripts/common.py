import json, os, platform, subprocess, sys, time
LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def venv_py(v):
    return os.path.join(LAB, f".venv{v}", "bin", "python")

def binary(opt):
    return os.path.join(LAB, "target314t" if opt == "B" else "target314", "release", "e04-python-udf")

def env_for(opt):
    v = "314t" if opt == "B" else "314"
    py = venv_py(v)
    base = subprocess.check_output([py, "-c", "import sys;print(sys.base_prefix)"], text=True).strip()
    sp = subprocess.check_output([py, "-c", "import site;print(site.getsitepackages()[0])"], text=True).strip()
    e = dict(os.environ)
    e["PYTHONHOME"] = base
    e["PYTHONPATH"] = f"{sp}:{os.path.join(LAB, 'python')}"
    e.pop("PYTHON_GIL", None)
    return e

def host_args(opt, kind, body, threads, rows=1_000_000, iters=3, warm=2):
    a = [binary(opt), "--option", opt, "--kind", kind, "--body", body, "--threads", str(threads),
         "--rows", str(rows), "--iters", str(iters), "--warmups", str(warm),
         "--python", venv_py("314"), "--worker", os.path.join(LAB, "python", "worker.py")]
    return a

def run_cfg(opt, kind, body, threads, rows, iters, warm, timeout=1800):
    t0 = time.time()
    # Shared machine: other processes compete for the CPUs. Wait briefly for a quieter moment, record load either way.
    FL = ["flock", "/home/gopal/krishiv-data/.bench.lock"]
    for _ in range(24):  # retry (wait) while starting loadavg > 8; give up after ~2 min and flag
        if float(open("/proc/loadavg").read().split()[0]) <= 8:
            break
        time.sleep(5)
    load0 = open("/proc/loadavg").read().split()[:3]
    try:
        p = subprocess.run(([] if os.environ.get("E04_LOCKED") else FL) + host_args(opt, kind, body, threads, rows, iters, warm), env=env_for(opt),
                           capture_output=True, text=True, timeout=timeout)
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if lines:
            r = json.loads(lines[-1])
        else:
            r = {"status": "crashed", "returncode": p.returncode, "stderr_tail": p.stderr[-400:]}
        r["returncode"] = p.returncode
    except subprocess.TimeoutExpired:
        r = {"status": "timeout"}
    r.update(option=opt, kind=kind, body=body, threads=threads, harness_wall_s=round(time.time() - t0, 1),
             loadavg_before=load0, load_gt_8_at_start=float(load0[0]) > 8, oversubscribed=threads > 12, loadavg_after=open("/proc/loadavg").read().split()[:3])
    if r.get("status") == "ok":
        r["cpu_us_per_krow"] = r["cpu_s_timed"] / (r["rows_per_iter"] * r["iters"]) * 1e9
    return r

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL, cwd=LAB).strip()
    except Exception:
        return ""

def metadata():
    def pv(v):
        return json.loads(subprocess.check_output([venv_py(v), "-c",
            "import sys,json,pyarrow,pandas,numpy;print(json.dumps(dict(python=sys.version,gil_enabled=sys._is_gil_enabled(),pyarrow=pyarrow.__version__,pandas=pandas.__version__,numpy=numpy.__version__)))"], text=True))
    return {
        "environment": "shared-laptop (i7-9750H, 12 HW threads, concurrent experiments; relative numbers only)",
        "machine": {"cpu": sh("lscpu | grep 'Model name' | sed 's/.*: *//'"), "logical_cpus": os.cpu_count(),
                    "mem_total_gb": round(int(sh("awk '/MemTotal/{print $2}' /proc/meminfo") or 0) / 1048576, 1),
                    "kernel": platform.release(), "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2")},
        "python_3.14": pv("314"), "python_3.14t": pv("314t"),
        "rust": sh("rustc --version"), "pyo3": "0.30.1", "arrow_rs": "60.0.0",
        "commit": sh("git rev-parse HEAD"), "dirty": bool(sh("git status --porcelain -- .")),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
