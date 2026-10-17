"""Memory visibility: can the host account Python allocations? Run under 3.14 and 3.14t (argv[1]=label).
Also (label 'D') shows host-side visibility/enforcement for a separate worker process."""
import ctypes, json, os, sys, tracemalloc, subprocess, resource
MB = 256

def rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

def inproc():
    import numpy as np, pyarrow as pa
    res = {}
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    tracemalloc.start()
    keep = []
    def probe(name, fn):
        r0, (t0, _), a0 = rss(), tracemalloc.get_traced_memory(), pa.total_allocated_bytes()
        keep.append(fn())
        r1, (t1, _), a1 = rss(), tracemalloc.get_traced_memory(), pa.total_allocated_bytes()
        res[name] = {"alloc_mb": MB, "rss_delta_mb": round((r1 - r0) / 2**20), "tracemalloc_delta_mb": round((t1 - t0) / 2**20),
                     "pyarrow_pool_delta_mb": round((a1 - a0) / 2**20)}
    probe("python bytearray", lambda: bytearray(MB * 2**20))
    probe("numpy (np.ones)", lambda: np.ones(MB * 2**17))
    def pab():
        b = pa.allocate_buffer(MB * 2**20); ctypes.memset(b.address, 1, MB * 2**20); return b
    probe("pyarrow native buffer (pa.allocate_buffer, Arrow memory pool)", pab)
    def cm():
        p = libc.malloc(MB * 2**20); ctypes.memset(p, 1, MB * 2**20); return p
    probe("ctypes libc malloc+memset (native ext proxy)", cm)
    return res

def worker_probe():
    """Option D view: host reads a worker's RSS from /proc and can cap it with RLIMIT_AS / cgroups."""
    code = ("import sys,os,time\nimport numpy as np\nprint(os.getpid(),flush=True)\nsys.stdin.readline()\n"
            "a=np.ones(256*2**17)\nprint('ok',flush=True)\nsys.stdin.readline()\n"
            "try:\n  b=np.ones(2048*2**17); print('alloc2G-succeeded',flush=True)\nexcept MemoryError:\n  print('MemoryError',flush=True)\nsys.stdin.readline()\n")
    def lim():
        resource.setrlimit(resource.RLIMIT_AS, (1536 * 2**20, 1536 * 2**20))
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, preexec_fn=lim)
    pid = int(p.stdout.readline())
    def hrss():
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("VmRSS"): return int(l.split()[1]) // 1024
    r0 = hrss()
    p.stdin.write("\n"); p.stdin.flush(); a = p.stdout.readline().strip(); r1 = hrss()
    p.stdin.write("\n"); p.stdin.flush(); b = p.stdout.readline().strip()
    p.stdin.write("\n"); p.stdin.flush(); p.wait()
    return {"worker_rss_mb_before": r0, "worker_rss_mb_after_256MB_alloc": r1, "host_sees_delta_mb": r1 - r0,
            "over_limit_alloc_result_with_RLIMIT_AS_1.5GB": b, "host_pid_alive_after": True}

if __name__ == "__main__":
    label = sys.argv[1]
    out = {"label": label, "python": sys.version, "gil_enabled": sys._is_gil_enabled()}
    out["in_process_allocation_visibility"] = inproc()
    if label == "3.14":
        out["isolated_worker_visibility"] = worker_probe()
    print(json.dumps(out, indent=1))
