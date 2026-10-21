"""ClickBench query texts: fetched from a pinned upstream commit, verified, cached, adapted at run time.

The ClickBench queries are licensed CC BY-NC-SA 4.0 (ClickHouse/ClickBench), so this repository does not contain or
redistribute them. ``load`` downloads the two upstream files this harness uses from a pinned commit, checks their
SHA-256, and caches them under ``$KRISHIV_DATA/clickbench/queries/<commit>/`` (like the dataset itself):

* ``duckdb-parquet/queries.sql``: used for every engine except ClickHouse;
* ``clickhouse/queries.sql``: used for ClickHouse.

Both are one query per line, in the order q01..q43. The adaptations the harness applies (``adapt``) are
transformations of that text, listed in ``bench/README.md``:

1. The upstream DuckDB file spells string length ``STRLEN(x)``; every engine here gets ``length(x)``.
2. The dataset's ``EventDate`` is a day count and ``EventTime`` epoch seconds (the raw Parquet columns, no view), so
   date-string bounds on ``EventDate`` become day numbers and the minute extraction / truncation is integer
   arithmetic on ``EventTime``.
3. Queries whose ``LIMIT`` result is not fully ordered upstream get tie-break sort keys (``TIE_BREAK``) so that all
   engines return the same rows.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from .paths import data_root

COMMIT = "0e10b8434eef001b43156b860a4e2e3665a35d51"
LICENSE = "CC BY-NC-SA 4.0"
RAW_URL = "https://raw.githubusercontent.com/ClickHouse/ClickBench/{commit}/{path}"
# engine-agnostic source name -> (path in the upstream repository, SHA-256 of the file at COMMIT)
PINS: dict[str, tuple[str, str]] = {
    "duckdb-parquet": (
        "duckdb-parquet/queries.sql",
        "e84ea26e1ed13a1dc1e46e76ddc9e49ead67a0444cdd5ae92085daad269c5508",
    ),
    "clickhouse": ("clickhouse/queries.sql", "a7d6673357348ee9680443216b6f26f30d1dce9f313b419d38502417b2c2a219"),
}
EXPECTED_QUERIES = 43

# Extra sort keys appended to the final ORDER BY (an ORDER BY is added when the query has none).
TIE_BREAK: dict[str, list[str]] = {
    "q11": ["MobilePhoneModel"],
    "q18": ["UserID", "SearchPhrase"],
    "q19": ["UserID", "m", "SearchPhrase"],
    "q22": ["SearchPhrase"],
    "q23": ["SearchPhrase"],
    "q25": ["SearchPhrase"],
    "q31": ["SearchEngineID", "ClientIP"],
    "q32": ["WatchID", "ClientIP"],
    "q33": ["WatchID", "ClientIP"],
    "q39": ["URL"],
    "q40": ["TraficSourceID", "SearchEngineID", "AdvEngineID", "Src", "Dst"],
    "q41": ["URLHash", "EventDate"],
}

Fetch = Callable[[str], bytes]


def source_name(engine: str) -> str:
    return "clickhouse" if engine == "clickhouse" else "duckdb-parquet"


def _http_get(url: str, attempts: int = 3, timeout_s: float = 60.0) -> bytes:
    """GET with a timeout; retries only transient failures (5xx, connection errors), never a 4xx."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as r:  # fixed https URL
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if i == attempts - 1:
                raise
        time.sleep(2.0 * (i + 1))
    raise AssertionError("unreachable")


def _verified(data: bytes, expected: str, what: str) -> bytes:
    got = hashlib.sha256(data).hexdigest()
    if got != expected:
        raise ValueError(f"{what}: SHA-256 {got} does not match the pinned {expected}")
    return data


def cache_dir(commit: str = COMMIT) -> Path:
    return data_root() / "clickbench" / "queries" / commit[:12]


def fetch_file(
    name: str, fetch: Fetch = _http_get, pins: Mapping[str, tuple[str, str]] = PINS, commit: str = COMMIT
) -> str:
    """The verified upstream file for source ``name`` (cached; a corrupt cache entry is fetched again)."""
    path, sha = pins[name]
    dest = cache_dir(commit) / f"{name}.sql"
    with contextlib.suppress(OSError):
        data = dest.read_bytes()
        if hashlib.sha256(data).hexdigest() == sha:
            return data.decode()
    data = _verified(fetch(RAW_URL.format(commit=commit, path=path)), sha, f"ClickBench {path} at {commit[:12]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return data.decode()


def _day_number(match: re.Match[str]) -> str:
    try:
        d = dt.date.fromisoformat(match.group(2))
    except ValueError as e:
        raise ValueError(f"invalid date literal in a ClickBench query: {match.group(2)!r}") from e
    return f"EventDate {match.group(1)} {(d - dt.date(1970, 1, 1)).days}"


def adapt(sql: str, qid: str, tie_break: Mapping[str, list[str]] = TIE_BREAK) -> str:
    sql = re.sub(r"\bSTRLEN\(", "length(", sql)
    sql = sql.replace("toDateTime(EventTime)", "EventTime")
    sql = re.sub(r"EventDate\s*(>=|<=)\s*'(\d{4}-\d{2}-\d{2})'", _day_number, sql)
    sql = re.sub(r"extract\(minute FROM EventTime\)", "CAST(FLOOR(EventTime % 3600 / 60) AS BIGINT)", sql)
    sql = re.sub(r"DATE_TRUNC\('minute', EventTime\)", "EventTime - EventTime % 60", sql)
    keys = tie_break.get(qid)
    if keys:
        head, sep, tail = sql.rstrip().rstrip(";").rpartition(" LIMIT ")
        if not sep:
            raise ValueError(f"{qid}: cannot add sort keys to a query without LIMIT")
        add = ", ".join(keys)
        head = f"{head}, {add}" if re.search(r"\bORDER BY\b", head) else f"{head} ORDER BY {add}"
        sql = f"{head} LIMIT {tail};"
    return sql


def load(
    engine: str,
    fetch: Fetch = _http_get,
    pins: Mapping[str, tuple[str, str]] = PINS,
    commit: str = COMMIT,
    expected: int = EXPECTED_QUERIES,
) -> dict[str, str]:
    """``{qNN: adapted SQL}`` for ``engine``."""
    lines = [ln for ln in fetch_file(source_name(engine), fetch, pins, commit).splitlines() if ln.strip()]
    if len(lines) != expected:
        raise ValueError(f"ClickBench query file has {len(lines)} queries, expected {expected}")
    return {f"q{i:02d}": adapt(ln, f"q{i:02d}") for i, ln in enumerate(lines, 1)}
