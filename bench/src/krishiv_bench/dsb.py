"""DSB (Microsoft, derived from TPC-DS): a many-join, skewed, correlated decision-support benchmark (P2-T141).

Used for the join-enumerator decision (P2-T113) in place of JOB/IMDB, whose data terms are non-commercial.

Licences (evidence in ``krishiv-internal/research/2026-10-24-join-benchmark-licences.md``):

* the ``microsoft/dsb`` repository is MIT, but its generator source (``code/tools``, derived from the TPC-DS tools) and
  the query templates carry the TPC "Legal Notice" and ship with the TPC End User License Agreement v2.2
  (``code/TPC_DS_EULA.txt``): a restricted, revocable licence to use the software on systems under the user's control;
  redistribution only with the EULA and legend; performance results must be "clearly identified as not being
  comparable to TPC Benchmark Results";
* so this repository contains **none** of it. ``fetch_source`` checks out the pinned upstream commit into
  ``$KRISHIV_DATA/cache/dsb/<commit>/``, ``build_tools`` compiles ``dsdgen`` / ``dsqgen`` there (gcc and make only: the
  repository ships the generated parser sources), and the query texts are instantiated at generation time into the
  dataset directory. Nothing is redistributed; results are DSB results, not TPC-DS results.

Dataset: ``$KRISHIV_DATA/dsb/sf<N>-dbgen-<GENERATOR_VERSION>/`` (the ``dsb@<version>`` dataset name, see ``paths``):
``<table>.parquet`` for the 24 tables, ``queries/<id>.sql``, ``oracle.rows.json.gz`` (DuckDB's answers) and
``MANIFEST.json`` (licences, source commit, build flags, checksums). A dataset is immutable: it is written under a
temporary name and renamed into place, and an existing one is never touched.

Determinism: ``dsdgen`` has a fixed default seed. The upstream C code reads uninitialised locals, so its output depends
on the compiler unless those are zeroed; the tools are built with ``-ftrivial-auto-var-init=zero`` (GCC >= 12),
which makes -O0 and -O2 builds produce identical bytes. The manifest records the SHA-256 of every raw ``.dat`` file (the
dataset identity, independent of DuckDB), the SHA-256 of every Parquet file (it also depends on the DuckDB version,
recorded) and an order-independent row hash per table.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import compare
from .adapters import Query
from .paths import data_root, dataset_dir

REPO_URL = "https://github.com/microsoft/dsb"
COMMIT = "ec9a156ceee923db1114cbe388f6183b53d49787"
GENERATOR_VERSION = "v1"
DATASET_NAME = f"dsb@{GENERATOR_VERSION}"
DEFAULT_SEED = 1
LICENCE = {
    "benchmark": "DSB (Microsoft), derived from TPC-DS; not comparable to published TPC-DS results",
    "repository": "MIT (https://github.com/microsoft/dsb/blob/main/LICENSE)",
    "generator_and_templates": "TPC End User License Agreement v2.2 (code/TPC_DS_EULA.txt in the DSB repository)",
    "redistribution": "none: generator, templates and queries are fetched at run time, never stored in this repo",
    "evidence": "krishiv-internal/research/2026-10-24-join-benchmark-licences.md",
}
# gcc 15 rejects the K&R-isms of the upstream sources; the zeroing flag is what makes the output reproducible.
TOOL_CFLAGS = (
    "-O2 -w -std=gnu99 -fcommon -ftrivial-auto-var-init=zero -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 "
    "-Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion -Wno-incompatible-pointer-types"
)
TEMPLATE_DIRS = {"agg": "agg_queries", "mb": "multi_block_queries", "spj": "spj_queries"}
MANIFEST = "MANIFEST.json"
ORACLE = "oracle.rows.json.gz"
DIALECT = "postgres"
_TYPES = {
    "char": "VARCHAR",
    "varchar": "VARCHAR",
    "integer": "INTEGER",
    "int": "INTEGER",
    "date": "DATE",
    "time": "TIME",
}


# ---------------------------------------------------------------- source and tools
def cache_dir() -> Path:
    return data_root() / "cache" / "dsb" / COMMIT


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch_source() -> Path:
    """The pinned upstream commit, checked out under the cache; verified by commit hash (content addressed)."""
    src = cache_dir() / "src"
    if (src / ".git").is_dir():
        head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        if head.strip() == COMMIT:
            return src
        shutil.rmtree(src)
    src.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-q"], src)
    _run(["git", "fetch", "-q", "--depth", "1", REPO_URL, COMMIT], src)
    _run(["git", "checkout", "-q", "FETCH_HEAD"], src)
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        raise RuntimeError(f"fetched {head}, expected {COMMIT}")
    return src


def _build_stamp() -> dict[str, str]:
    """Current tool-build configuration: flags plus the compiler version (cache identity)."""
    out = subprocess.run(["gcc", "--version"], capture_output=True, text=True, check=True).stdout
    gcc = out.splitlines()[0].strip() if out.splitlines() else "unknown"
    return {"gcc": gcc, "tool_cflags": TOOL_CFLAGS}


def build_tools(src: Path) -> Path:
    """Builds ``dsdgen`` and ``dsqgen`` from ``<src>/code/tools``; returns the build directory holding both."""
    build = cache_dir() / "build"
    stamp = build / ".krishiv-build-stamp.json"
    for tool in ("gcc", "make"):
        if shutil.which(tool) is None:
            sys.exit(f"`{tool}` not found: building the DSB generator needs gcc and make")
    if (build / "dsdgen").is_file() and (build / "dsqgen").is_file():
        try:
            recorded = json.loads(stamp.read_text())
        except (OSError, ValueError):
            recorded = None
        if recorded == _build_stamp():
            return build
    shutil.rmtree(build, ignore_errors=True)
    shutil.copytree(src / "code" / "tools", build)
    # The parser sources are generated and shipped; make would rerun yacc/lex (not installed)
    # if the grammar looked newer.
    now = time.time()
    for name in ("qgen.y", "tokenizer.l"):
        os.utime(build / name, (now - 3600, now - 3600))
    for name in ("y.tab.c", "y.tab.h", "tokenizer.c", "lex.yy.c"):
        if (build / name).exists():
            os.utime(build / name, (now, now))
    cmd = ["make", "-f", "Makefile.suite", "OS=LINUX", f"LINUX_CFLAGS={TOOL_CFLAGS}", "dsdgen", "dsqgen"]
    subprocess.run(cmd, cwd=build, check=True, stdout=subprocess.DEVNULL)
    if not (build / "dsdgen").is_file() or not (build / "dsqgen").is_file():
        raise RuntimeError("DSB tool build produced no dsdgen/dsqgen")
    stamp.write_text(json.dumps(_build_stamp(), sort_keys=True) + "\n")
    return build


# ---------------------------------------------------------------- schema and conversion
def parse_schema(ddl: str) -> dict[str, list[tuple[str, str]]]:
    """``create table`` statements of the upstream DDL as table -> [(column, DuckDB type)] (dbgen_version excluded)."""
    schema: dict[str, list[tuple[str, str]]] = {}
    for m in re.finditer(r"create\s+table\s+(\w+)\s*\((.*?)\n\)\s*;", ddl, re.S | re.I):
        name, body = m.group(1).lower(), m.group(2)
        if name == "dbgen_version":
            continue
        cols = []
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line or line.lower().startswith("primary key"):
                continue
            t = re.match(r"(\w+)\s+(\w+)\s*(\(\s*\d+\s*(?:,\s*\d+\s*)?\))?", line)
            if t is None:
                raise ValueError(f"cannot parse column {line!r} of {name}")
            col, typ, args = t.group(1).lower(), t.group(2).lower(), (t.group(3) or "").replace(" ", "")
            if typ == "decimal":
                duck = f"DECIMAL{args}"
            elif typ in _TYPES:
                duck = _TYPES[typ]
            else:
                raise ValueError(f"unsupported type {typ!r} in {name}.{col}")
            cols.append((col, duck))
        schema[name] = cols
    return schema


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _sql_str(p: Path | str) -> str:
    return str(p).replace("'", "''")


def convert_table(dat: Path, cols: list[tuple[str, str]], out: Path, threads: int) -> dict[str, Any]:
    """``dat`` (pipe separated, no trailing delimiter, Latin-1, empty = NULL) -> Parquet; returns the table's record."""
    import duckdb

    con = duckdb.connect()
    try:
        con.execute(f"SET threads={threads}")
        spec = ", ".join(f"'{c}': '{t}'" for c, t in cols)
        con.execute(
            f"COPY (SELECT * FROM read_csv('{_sql_str(dat)}', delim='|', header=false, quote='', escape='', "
            f"encoding='latin-1', columns={{{spec}}})) TO '{_sql_str(out)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"
        )
        rows, rowhash = con.execute(
            f"SELECT count(*), coalesce(sum(h::HUGEINT), 0)::VARCHAR FROM "
            f"(SELECT hash(t) AS h FROM read_parquet('{_sql_str(out)}') t)"
        ).fetchone()  # ty: ignore[not-iterable]
    finally:
        con.close()
    return {"file": out.name, "rows": rows, "row_hash_sum": rowhash, "file_sha256": sha256_file(out)}


# ---------------------------------------------------------------- queries
def template_files(src: Path) -> dict[str, Path]:
    """Query id -> template path, e.g. ``agg_q013``, ``mb_q001``, ``spj_q013`` (the number is upstream's)."""
    out: dict[str, Path] = {}
    for cls, d in TEMPLATE_DIRS.items():
        for f in sorted((src / "query_templates_pg" / d).glob("query*.tpl")):
            num = re.match(r"query(\d+)", f.name)
            assert num is not None, f.name
            out[f"{cls}_q{num.group(1)}"] = f
    return out


def generate_queries(build: Path, src: Path, scale: int, seed: int, out: Path) -> dict[str, str]:
    """Instantiates every template once with ``dsqgen`` (PostgreSQL dialect, fixed seed); returns id -> SQL."""
    out.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    for qid, tpl in template_files(src).items():
        work = out / qid
        work.mkdir(exist_ok=True)
        cmd = [
            str(build / "dsqgen"),
            "-DIRECTORY", str(tpl.parent),
            "-TEMPLATE", tpl.name,
            "-DIALECT", DIALECT,
            "-OUTPUT_DIR", str(work),
            "-SCALE", str(scale),
            "-RNGSEED", str(seed),
            "-STREAMS", "1",
            "-QUIET", "Y",
        ]  # fmt: skip
        subprocess.run(cmd, cwd=build, check=True, stdout=subprocess.DEVNULL)
        text = (work / "query_0.sql").read_text().strip()
        if not text:
            raise RuntimeError(f"dsqgen produced no text for {tpl.name}")
        texts[qid] = text + "\n"
        shutil.rmtree(work)
    return texts


def relation_count(sql: str) -> int | None:
    """Base-table references in the statement, subqueries and repeated aliases included (the join-order search size)."""
    import duckdb

    try:
        plan_json = duckdb.connect().execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]  # ty: ignore[not-subscriptable]
    except duckdb.Error:
        return None
    if '"error":true' in plan_json:  # json_serialize_sql reports parse errors in the JSON, it does not raise
        return None
    return len(re.findall(r'"type":"BASE_TABLE"', plan_json))


def count_distribution(counts: dict[str, int | None]) -> dict[int, int]:
    dist: dict[int, int] = {}
    for n in counts.values():
        if n is not None:
            dist[n] = dist.get(n, 0) + 1
    return dict(sorted(dist.items()))


# ---------------------------------------------------------------- oracle
def run_oracle(root: Path, queries: dict[str, str], tables: list[str], timeout_s: float = 600) -> dict[str, Any]:
    """Runs every query in DuckDB over the dataset's Parquet and stores the answers as ``oracle.rows.json.gz``."""
    import duckdb

    con = duckdb.connect()
    for t in tables:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{_sql_str(root / (t + '.parquet'))}')")
    rows_by_q: dict[str, list] = {}
    record: dict[str, Any] = {}
    for qid, sql in queries.items():
        t0 = time.perf_counter()
        timer = None
        try:
            import threading

            timer = threading.Timer(timeout_s, con.interrupt)
            timer.start()
            rows = con.execute(sql).fetchall()
        except duckdb.Error as e:
            record[qid] = {"error": str(e).splitlines()[0][:300]}
            print(f"[dsb] oracle {qid}: ERROR {record[qid]['error'][:100]}", flush=True)
            continue
        finally:
            if timer is not None:
                timer.cancel()
        rows_by_q[qid] = rows
        # order-independent: DuckDB may return rows without a top-level ORDER BY, or ties, in any order
        blob = "\n".join(sorted(json.dumps([compare.enc(c) for c in r]) for r in rows)).encode()
        record[qid] = {
            "rows": len(rows),
            "rows_sha256": hashlib.sha256(blob).hexdigest(),
            "seconds": round(time.perf_counter() - t0, 3),
        }
    con.close()
    save_oracle(root / ORACLE, rows_by_q)
    return record


def save_oracle(path: Path, rows_by_query: dict[str, list]) -> None:
    """``compare.save_rows`` format, gzip header without a timestamp so the file's checksum is reproducible."""
    payload = json.dumps({q: [[compare.enc(c) for c in r] for r in rows] for q, rows in rows_by_query.items()})
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as f:
        f.write(payload.encode())


def load_oracle(root: Path) -> dict[str, list[tuple]]:
    f = root / ORACLE
    return compare.load_rows(f) if f.is_file() else {}


# ---------------------------------------------------------------- generation
def generate(scale: int, seed: int = DEFAULT_SEED, threads: int = 1, oracle: bool = True, force: bool = False) -> Path:
    """Generates ``dsb@<version>`` at ``scale`` (an integer, as ``dsdgen`` requires) and returns its directory."""
    if scale < 1:
        raise ValueError("DSB scale must be an integer >= 1 (dsdgen)")
    final = dataset_dir(DATASET_NAME, scale)
    if final.exists() and any(final.iterdir()):
        if not force:
            print(f"[dsb] {final} exists, skipping (datasets are immutable)")
            return final
        raise FileExistsError(f"{final} already exists; datasets are immutable (use a new GENERATOR_VERSION)")
    src = fetch_source()
    build = build_tools(src)
    schema = parse_schema((src / "scripts" / "create_tables.sql").read_text())
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{final.name}.partial-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    raw = tmp / "raw"
    raw.mkdir(parents=True)
    try:
        print(f"[dsb] dsdgen -scale {scale} (single process, minutes per GB)", flush=True)
        subprocess.run(
            [str(build / "dsdgen"), "-scale", str(scale), "-dir", str(raw), "-terminate", "N", "-force", "-quiet", "Y"],
            cwd=build, check=True, stdout=subprocess.DEVNULL,
        )  # fmt: skip
        tables: dict[str, Any] = {}
        raw_sha: dict[str, str] = {}
        for f in sorted(raw.iterdir()):
            if f.name != "dbgen_version.dat":  # holds the generation timestamp
                raw_sha[f.name] = sha256_file(f)
        for name, cols in schema.items():
            dat = raw / f"{name}.dat"
            if not dat.is_file():
                raise RuntimeError(f"dsdgen did not write {dat.name}")
            print(f"[dsb] parquet {name}", flush=True)
            tables[name] = {
                **convert_table(dat, cols, tmp / f"{name}.parquet", threads),
                "raw_sha256": raw_sha[dat.name],
            }
        shutil.rmtree(raw)
        qdir = tmp / "queries"
        texts = generate_queries(build, src, scale, seed, tmp / ".qgen")
        shutil.rmtree(tmp / ".qgen")
        qdir.mkdir()
        for qid, text in texts.items():
            (qdir / f"{qid}.sql").write_text(text)
        counts = {qid: relation_count(sql) for qid, sql in texts.items()}
        queries = {
            qid: {"sha256": hashlib.sha256(texts[qid].encode()).hexdigest(), "relations": counts[qid]} for qid in texts
        }
        oracle_rec: dict[str, Any] = {}
        if oracle:
            print(f"[dsb] oracle: {len(texts)} queries in DuckDB", flush=True)
            oracle_rec = run_oracle(tmp, texts, list(schema))
            for qid, rec in oracle_rec.items():
                queries[qid]["oracle"] = rec
        import duckdb

        manifest = {
            "schema": 1,
            "dataset": "dsb",
            "generator_version": GENERATOR_VERSION,
            "scale": scale,
            "query_seed": seed,
            "licence": LICENCE,
            "source": {"repo": REPO_URL, "commit": COMMIT, "tool_cflags": TOOL_CFLAGS, "dialect": DIALECT},
            "libs": {"duckdb": duckdb.__version__},
            "parquet_threads": threads,
            "tables": tables,
            "queries": queries,
            "relation_count_distribution": count_distribution(counts),
            "oracle_file": ORACLE if oracle else None,
            "oracle_file_sha256": sha256_file(tmp / ORACLE) if oracle else None,
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if final.exists():
            final.rmdir()
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def read_manifest(directory: Path) -> dict[str, Any] | None:
    f = directory / MANIFEST
    return json.loads(f.read_text()) if f.is_file() else None


def verify(directory: Path) -> list[str]:
    """Recomputes file checksums against the manifest; returns the problems found (empty = verified)."""
    man = read_manifest(directory)
    if man is None:
        return [f"{directory}: no {MANIFEST}"]
    problems = []
    for name, rec in man["tables"].items():
        f = directory / rec["file"]
        if not f.is_file():
            problems.append(f"{name}: {f} is missing")
        elif sha256_file(f) != rec["file_sha256"]:
            problems.append(f"{name}: Parquet sha256 differs from the manifest")
    for qid, rec in man["queries"].items():
        f = directory / "queries" / f"{qid}.sql"
        if not f.is_file():
            problems.append(f"{qid}: query file is missing")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != rec["sha256"]:
            problems.append(f"{qid}: query text differs from the manifest")
    if man.get("oracle_file"):
        f = directory / man["oracle_file"]
        if not f.is_file():
            problems.append(f"{f.name} is missing")
        elif sha256_file(f) != man["oracle_file_sha256"]:
            problems.append(f"{f.name}: sha256 differs from the manifest")
    return problems


# ---------------------------------------------------------------- harness hooks
def load_queries(root: Path, only: list[str] | None = None) -> list[Query]:
    """The dataset's instantiated queries (``queries/<id>.sql``), in id order."""
    d = root / "queries"
    files = sorted(d.glob("*.sql"))
    if not files:
        raise SystemExit(f"no DSB queries under {d}; run `uv run krishiv-datasets dsb --sf <N>`")
    return [Query(f.stem, f.read_text(), "dsb") for f in files if not only or f.stem in only]
