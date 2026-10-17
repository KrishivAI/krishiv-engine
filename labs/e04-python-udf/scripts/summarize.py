import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import LAB
def table(fn):
    d = json.load(open(os.path.join(LAB, "results", fn)))
    rows = {}
    for r in d["runs"]:
        rows.setdefault((r["kind"], r["body"], r["threads"]), {})[r["option"]] = r
    lines = ["| kind | body | thr | A GIL | B free-thr | C subinterp | D procs+shm | D vs A | D vs B |", "|---|---|---|---|---|---|---|---|---|"]
    def f(r):
        if r is None: return "-"
        if r.get("status") != "ok": return r.get("status", "-")
        return f"{r['rows_per_s_median']/1e6:.2f}M (p50 {r['batch_latency_ms_p50']:.2f} / p99 {r['batch_latency_ms_p99']:.2f} ms)"
    def ov(a, b):
        if not a or not b or a.get("status") != "ok" or b.get("status") != "ok": return "-"
        return f"{(1 - a['rows_per_s_median']/b['rows_per_s_median'])*100:+.1f}%"
    for k in sorted(rows, key=lambda k: (["scalar","pandas","arrow"].index(k[0]), k[1], k[2])):
        o = rows[k]
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {f(o.get('A'))} | {f(o.get('B'))} | {f(o.get('C'))} | {f(o.get('D'))} | {ov(o.get('D'), o.get('A'))} | {ov(o.get('D'), o.get('B'))} |")
    return "\n".join(lines), d
if __name__ == "__main__":
    print(table(sys.argv[1])[0])
