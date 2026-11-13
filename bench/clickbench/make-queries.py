#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Writes ClickBench's ``queries.sql`` for Krishiv to stdout: the upstream file, adapted to Spark SQL.

The 43 ClickBench queries are licensed CC BY-NC-SA 4.0 (ClickHouse/ClickBench), so this repository does not carry
their text. This script downloads the ``duckdb-parquet/queries.sql`` of a pinned upstream commit, checks its SHA-256,
applies the adaptations listed in ``README.md`` and prints one query per line (q01..q43). Standard library only.

    ./make-queries.py > queries.sql
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import sys
import urllib.request

COMMIT = "0e10b8434eef001b43156b860a4e2e3665a35d51"
PATH = "duckdb-parquet/queries.sql"
SHA256 = "e84ea26e1ed13a1dc1e46e76ddc9e49ead67a0444cdd5ae92085daad269c5508"
URL = f"https://raw.githubusercontent.com/ClickHouse/ClickBench/{COMMIT}/{PATH}"
EXPECTED = 43

# q29's domain extraction: the upstream (DuckDB, RE2) text and its Spark SQL form. Spark's string literals unescape
# backslashes (so `\\.` is a literal dot pattern) and its replacement refers to group 1 as `$1`.
Q29_UPSTREAM = r"REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1')"
Q29_SPARK = r"REGEXP_REPLACE(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '$1')"


def _day_number(m: re.Match[str]) -> str:
    d = dt.date.fromisoformat(m.group(2))
    return f"EventDate {m.group(1)} {(d - dt.date(1970, 1, 1)).days}"


def adapt(sql: str) -> str:
    """One upstream query, in Spark SQL for the raw ``hits.parquet`` columns (see README, "Adaptations")."""
    sql = re.sub(r"\bSTRLEN\(", "length(", sql)  # 1
    sql = sql.replace("toDateTime(EventTime)", "EventTime")  # 2
    sql = re.sub(r"EventDate\s*(>=|<=)\s*'(\d{4}-\d{2}-\d{2})'", _day_number, sql)  # 3
    sql = re.sub(r"extract\(minute FROM EventTime\)", "CAST(FLOOR(EventTime % 3600 / 60) AS BIGINT)", sql)  # 4
    sql = re.sub(r"DATE_TRUNC\('minute', EventTime\)", "EventTime - EventTime % 60", sql)  # 5
    if Q29_UPSTREAM in sql:  # 6
        sql = sql.replace(Q29_UPSTREAM, Q29_SPARK)
    return sql


def main() -> int:
    with urllib.request.urlopen(URL, timeout=60) as r:  # fixed https URL
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        print(f"{PATH} at {COMMIT[:12]}: SHA-256 {got} is not the pinned {SHA256}", file=sys.stderr)
        return 1
    lines = [ln for ln in data.decode().splitlines() if ln.strip()]
    if len(lines) != EXPECTED:
        print(f"expected {EXPECTED} queries, found {len(lines)}", file=sys.stderr)
        return 1
    if sum(Q29_UPSTREAM in ln for ln in lines) != 1:
        print("the upstream REGEXP_REPLACE call changed; update Q29_UPSTREAM", file=sys.stderr)
        return 1
    sys.stdout.write("\n".join(adapt(ln) for ln in lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
