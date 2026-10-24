"""db-benchmark style groupby (``h2o-groupby``) and join (``h2o-join``) suites.

Data: ``h2o_data.py`` (``krishiv-datasets h2o-groupby|h2o-join``) under ``$KRISHIV_DATA/h2o/<version>/<label>/``.
Questions: ``bench/queries/h2o-groupby/q01..q10.sql`` and ``bench/queries/h2o-join/q01..q05.sql`` (the benchmark's
question set, written as portable SQL over the tables ``x`` / ``small`` / ``medium`` / ``big``).

Answer checksums first: like the benchmark's own ``chk`` column, every question has a checksum query, the question
wrapped as ``SELECT COUNT(*), SUM(c1), SUM(c2) ... FROM (<question>)``. Each engine runs it once, untimed, and its
row (the answer's row count and the sums of the checked columns) must equal DuckDB's before the question is timed;
a mismatch marks the question failed and ``run`` exits non-zero. Results are never materialised in Python: the timed
call is ``Engine.timed_run``, which returns only the row count.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from . import compare, h2o_data
from .adapters import Dataset, Engine, Query, make_engine
from .adapters.base import check_query
from .errors import error_fields
from .paths import data_root, fmt_sf, queries_dir, results_dir

SUITES = {"h2o-groupby": "groupby", "h2o-join": "join"}
SIZES = {"0.5gb": 10**7, "5gb": 10**8, "50gb": 10**9}  # rows; the benchmark's names for the data sizes
SIZE_GB = {"0.5gb": 0.5, "5gb": 5.0, "50gb": 50.0}
DEFAULT_KEYS = 100  # K of the headline groupby data (G1_<rows>_1e2_0_0)
MAX_STORED_ERROR = 300

# Columns whose sums form the checksum (the benchmark's ``chk``), per question.
CHK_COLUMNS: dict[str, dict[str, list[str]]] = {
    "h2o-groupby": {
        "q01": ["v1"],
        "q02": ["v1"],
        "q03": ["v1", "v3"],
        "q04": ["v1", "v2", "v3"],
        "q05": ["v1", "v2", "v3"],
        "q06": ["median_v3", "sd_v3"],
        "q07": ["range_v1_v2"],
        "q08": ["largest2_v3"],
        "q09": ["r2"],
        "q10": ["v3", "cnt"],
    },
    "h2o-join": {q: ["v1", "v2"] for q in ("q01", "q02", "q03", "q04", "q05")},
}


def is_h2o(suite: str) -> bool:
    return suite in SUITES


def check_sql(sql: str, columns: list[str]) -> str:
    """The checksum query of a question: its row count and the sum of each checked column."""
    body = sql.strip().rstrip(";").strip()
    sums = ", ".join(f"SUM({c}) AS {c}" for c in columns)
    return f"SELECT COUNT(*) AS n_rows, {sums} FROM ({body}) AS ans"


def load_queries(suite: str, engine: str, only: list[str] | None = None) -> list[Query]:
    """``bench/queries/<suite>/qNN.sql`` (override: ``<engine>/qNN.sql``) with the checksum SQL attached."""
    if suite not in CHK_COLUMNS:
        raise ValueError(f"not an h2o suite: {suite!r}")
    d = queries_dir(suite)
    out = []
    for f in sorted(d.glob("q*.sql")):
        if only and f.stem not in only:
            continue
        if f.stem not in CHK_COLUMNS[suite]:
            raise ValueError(f"{suite}/{f.stem}: no checksum columns defined")
        ov = d / engine / f.name
        sql = (ov if ov.exists() else f).read_text().strip()
        out.append(Query(f.stem, sql, suite, check_sql(sql, CHK_COLUMNS[suite][f.stem])))
    return out


# ---------------------------------------------------------------- datasets
def label_for(
    suite: str, rows: int, k: int = DEFAULT_KEYS, na_pct: int = 0, sorted_: bool = False
) -> str:  # the upstream file-name stem
    if suite == "h2o-groupby":
        return h2o_data.G1Params(rows, k, na_pct, sorted_).label
    return h2o_data.J1Params(rows, na_pct, sorted_).label


def dataset_root(
    suite: str,
    size: str,
    version: str = h2o_data.GENERATOR_VERSION,
    k: int = DEFAULT_KEYS,
    na_pct: int = 0,
    sorted_: bool = False,
) -> Path:
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; choose from {', '.join(SIZES)}")
    return data_root() / "h2o" / version / label_for(suite, SIZES[size], k, na_pct, sorted_)


def resolve_dataset(root: Path, size_gb: float) -> Dataset:
    """The ``Dataset`` of a generated directory (its ``*.parquet`` tables), or a SystemExit saying how to make it."""
    tables = sorted(p.stem for p in root.glob("*.parquet"))
    if not tables:
        raise SystemExit(f"no dataset at {root}; generate it with `uv run krishiv-datasets h2o-groupby|h2o-join ...`")
    return Dataset(root.name, size_gb, root, tables)


def dataset_provenance(root: Path) -> dict[str, Any]:
    man = h2o_data.read_manifest(root)
    if man is None:
        return {"generator": None, "manifest_sha256": None, "content_sha256": None}
    return {
        "generator": f"{man['generator']} {man['generator_version']} (numpy {man['libs']['numpy']})",
        "generator_version": man["generator_version"],
        "params": man["params"],
        "manifest_sha256": man["content_sha256"],
        "content_sha256": man["content_sha256"],
        "rows": {t: v["rows"] for t, v in man["tables"].items()},
    }


# ---------------------------------------------------------------- reference checksums (DuckDB)
def check_sql_sha(check_sql: str) -> str:
    """Short digest identifying a checksum query's text (cache key per question)."""
    return hashlib.sha256(check_sql.encode()).hexdigest()[:16]


def reference_checksums(ds: Dataset, queries: list[Query], memory_limit: str = "12GB") -> dict[str, list]:
    """DuckDB's checksum row per question, cached under ``results/h2o-ref`` keyed by the dataset's content digest.

    A cached row is reused only when its stored checksum-SQL digest matches the question's current checksum SQL,
    so editing a question (or its checksum columns) recomputes that question instead of reusing a stale reference.
    Legacy cache entries without a SQL digest are ignored and rewritten.
    """
    import duckdb

    man = h2o_data.read_manifest(ds.root)
    cache = None
    if man:
        cache = results_dir() / "h2o-ref" / f"{ds.name}-{man['content_sha256'][:16]}-duckdb{duckdb.__version__}.json"
    want = {q.id: check_sql_sha(q.check_sql or q.sql) for q in queries}
    stored: dict[str, dict] = {}
    if cache and cache.exists():
        raw = json.loads(cache.read_text())
        if isinstance(raw, dict):
            for qid, e in raw.items():
                if (
                    isinstance(e, dict)
                    and isinstance(e.get("row"), list)
                    and isinstance(e.get("check_sql_sha256"), str)
                ):
                    stored[qid] = e
    have: dict[str, list] = {
        q.id: stored[q.id]["row"] for q in queries if q.id in stored and stored[q.id]["check_sql_sha256"] == want[q.id]
    }
    missing = [q for q in queries if q.id not in have]
    if missing:
        eng = make_engine("duckdb", memory_limit=memory_limit)
        eng.setup(ds)
        try:
            for q in missing:
                (row,) = eng.run(check_query(q))
                have[q.id] = list(row)
                stored[q.id] = {"row": have[q.id], "check_sql_sha256": want[q.id]}
        finally:
            eng.teardown()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(stored, indent=1, sort_keys=True))
    return {q.id: have[q.id] for q in queries}


def compare_checksums(got: list, ref: list) -> tuple[bool, str]:
    """Row count exact, sums within the harness float tolerance (summation order differs between engines)."""
    if len(got) != len(ref):
        return False, f"checksum has {len(got)} values, expected {len(ref)}"
    if not compare.row_equal(got, ref):
        return False, f"checksum {got} != reference {ref}"
    return True, "checksum equals the reference"


# ---------------------------------------------------------------- run
def run_h2o(
    engine_name: str,
    suite: str,
    root: Path,
    size_gb: float,
    iterations: int = 3,
    warmup: int = 1,
    only: list[str] | None = None,
    engine_kwargs: dict | None = None,
    timeout_s: float | None = 600,
) -> tuple[Path, int]:
    """Runs the suite on one engine; returns (result JSON, number of failed questions)."""
    from .runner import engine_record, git_info, machine_info

    if iterations < 1 or warmup < 0:
        raise SystemExit("--iterations must be >= 1 and --warmup >= 0")
    ds = resolve_dataset(root, size_gb)
    queries = load_queries(suite, engine_name, only)
    if not queries:
        raise SystemExit(f"no question matches {only}")
    # The reference is always DuckDB's own question text: an engine override must produce the same
    # answer, never define the reference (a wrong override would otherwise agree with itself).
    refs = reference_checksums(ds, load_queries(suite, "duckdb", only))
    eng: Engine = make_engine(engine_name, **(engine_kwargs or {}))
    started = dt.datetime.now(dt.UTC)
    record: dict[str, Any] = {
        "schema": 2,
        "started_utc": started.isoformat(),
        "machine": machine_info(),
        "git": {"repo": "krishiv-engine (harness lives in bench/)", **git_info()},
        "engine": engine_record(eng, engine_name),
        "dataset": {"name": ds.name, "scale_factor": size_gb, "path": str(root), "tables": ds.tables}
        | dataset_provenance(root),
        "suite": suite,
        "iterations": iterations,
        "warmup": warmup,
        "timeout_s": timeout_s,
        "answers_available": True,
        "answer_reference": "duckdb",
        "queries": {},
    }
    failed = 0
    eng.setup(ds)
    pool = cf.ThreadPoolExecutor(max_workers=1)

    def call(fn, *args):
        """``fn(*args)`` on the worker thread with the per-call timeout (the engine is cancelled on expiry)."""
        nonlocal pool
        fut = pool.submit(fn, *args)
        try:
            return fut.result(timeout=timeout_s)
        except cf.TimeoutError:
            eng.cancel()
            with contextlib.suppress(Exception):
                fut.result(timeout=30)
            pool.shutdown(wait=False, cancel_futures=True)
            pool = cf.ThreadPoolExecutor(max_workers=1)
            raise TimeoutError(f"exceeded {timeout_s}s") from None

    try:
        for q in queries:
            entry: dict[str, Any] = {"iterations": iterations}
            record["queries"][q.id] = entry
            try:
                (row,) = call(eng.run, check_query(q))
                ok, msg = compare_checksums(list(row), refs[q.id])
                entry["chk"] = list(row)
                entry["chk_reference"] = refs[q.id]
                entry["answer_check"] = {"status": "pass" if ok else "fail", "detail": msg}
                if not ok:
                    failed += 1
                    print(f"  {q.id}: ANSWER MISMATCH (not timed): {msg[:200]}")
                    continue
                n_rows, times, meas = int(row[0]), [], []
                for i in range(warmup + iterations):
                    eng.begin_measure()
                    try:
                        n, secs = call(eng.timed_run, q)
                    finally:
                        m = eng.end_measure()
                    if n != n_rows:
                        raise RuntimeError(f"timed run returned {n} rows, the checksum run {n_rows}")
                    if i >= warmup:
                        times.append(secs)
                        meas.append(m)
            except Exception as e:  # recorded, the run continues with the next question
                failed += 1
                entry.update(error_fields(e))
                print(f"  {q.id}: ERROR {entry['error'][:120]}")
                continue
            entry.update(
                times_s=times,
                p50_s=statistics.median(times),
                min_s=min(times),
                max_s=max(times),
                mean_s=statistics.fmean(times),
                rows_returned=n_rows,
                peak_rss_bytes=_max([m.peak_rss_bytes for m in meas]),
                baseline_rss_bytes=_max([m.baseline_rss_bytes for m in meas]),
                cpu_seconds_median=_med([m.cpu_seconds for m in meas]),
                spill_bytes=_max([m.spill_bytes for m in meas]),
            )
            print(f"  {q.id}: p50={entry['p50_s'] * 1000:10.1f} ms rows={n_rows} chk=pass")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        eng.teardown()
    record["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    record["failed_queries"] = failed
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{started.strftime('%Y%m%dT%H%M%SZ')}_{engine_name}_{ds.name}_sf{fmt_sf(size_gb)}.json"
    f.write_text(json.dumps(record, indent=2))
    return f, failed


def _max(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# ---------------------------------------------------------------- report
def render(results: dict[str, dict], baseline: str | None = None) -> str:
    """Markdown table: p50 per engine and question, ratio to ``baseline``, checksum verdicts and failures."""
    engines = list(results)
    if not engines:
        raise SystemExit("no results to report")
    baseline = baseline if baseline in results else engines[0]
    first = results[baseline]
    ds = first["dataset"]
    qids = sorted({q for r in results.values() for q in r["queries"]})
    lines = [
        f"# Krishiv benchmark report: {first['suite']} {ds['name']}",
        "",
        f"- Dataset: `{ds['name']}` ({ds.get('generator') or 'no manifest'}), content sha256 "
        f"`{(ds.get('content_sha256') or 'n/a')[:16]}`",
        f"- {first['iterations']} timed + {first['warmup']} warm-up runs per question; median (p50) shown. Every "
        "engine's answer checksum must equal DuckDB's before a question is timed.",
        "",
        "| Engine | Version | Result file |",
        "|---|---|---|",
    ]
    lines += [f"| {e} | {results[e]['engine']['version']} | `{results[e]['_file']}` |" for e in engines]
    others = [e for e in engines if e != baseline]
    lines += [
        "",
        "| question | "
        + " | ".join(f"{e} p50 ms" for e in engines)
        + " | "
        + " | ".join(f"{e}/{baseline}" for e in others)
        + " |",
        "|---|" + "---:|" * (len(engines) + len(others)),
    ]
    ratios: dict[str, list[float]] = {e: [] for e in others}
    bad, skipped = [], []
    for q in qids:
        cells, rat = [], []
        per = {e: results[e]["queries"].get(q, {}) for e in engines}
        for e, d in per.items():
            if "p50_s" in d:
                cells.append(f"{d['p50_s'] * 1000:,.1f}")
            else:
                cells.append("FAIL" if d else "-")
                if d:
                    why = d.get("error") or d.get("answer_check", {}).get("detail", "failed")
                    bad.append(f"- `{q}` **{e}**: {str(why)[:200]}")
                else:
                    skipped.append(f"- `{q}` **{e}**: no result (question skipped or from a different run)")
        b = per[baseline].get("p50_s")
        for e in others:
            x = per[e].get("p50_s")
            if b and x:
                ratios[e].append(x / b)
                rat.append(f"{x / b:.2f}x")
            else:
                rat.append("-")
        lines.append(f"| {q} | " + " | ".join(cells) + " | " + " | ".join(rat) + " |")
    geo = [f"{e}: **{statistics.geometric_mean(r):.2f}x** over {len(r)} questions" for e, r in ratios.items() if r]
    lines += [
        "",
        f"Ratios are engine / `{baseline}` (>1 = slower). Geomean " + ("; ".join(geo) if geo else "n/a") + ".",
    ]
    lines += ["", "## Correctness", ""]
    if bad:
        lines += ["Questions that failed the checksum comparison or errored (never timed):", "", *bad]
    if skipped:
        lines += [
            "Questions with no result for an engine (skipped or from a different run; not verified):",
            "",
            *skipped,
        ]
    if not bad and not skipped:
        lines.append(f"All {len(qids)} questions: every engine's answer checksum equals the DuckDB reference.")
    return "\n".join(lines) + "\n"


def write_report(
    suite: str, size_gb: float, baseline: str | None, files: list[Path] | None, dataset: str | None
) -> Path:
    from .report import latest_results

    if files:
        res = {}
        for f in files:
            r = json.loads(Path(f).read_text())
            res[r["engine"]["name"]] = r | {"_file": Path(f).name}
    else:
        res = latest_results(suite, size_gb, dataset)
    out = results_dir() / f"report_{dataset or suite}_sf{fmt_sf(size_gb)}.md"
    out.write_text(render(res, baseline))
    return out


__all__ = ["CHK_COLUMNS", "SIZES", "SUITES", "is_h2o", "run_h2o"]
