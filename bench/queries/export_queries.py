"""Regenerate vendored query texts from DuckDB's tpch/tpcds extensions (see bench/README.md provenance).

    uv run python queries/export_queries.py
Writes queries/tpch/qNN.sql, queries/tpcds/qNN.sql and queries/tpch/expected_rows_sf1.json
(row counts from DuckDB's bundled official SF1 answer sets).
"""
import json
from pathlib import Path

import duckdb

here = Path(__file__).parent
con = duckdb.connect()
for ext, fn in (("tpch", "tpch_queries"), ("tpcds", "tpcds_queries")):
    con.execute(f"INSTALL {ext}; LOAD {ext}")
    d = here / ext
    d.mkdir(exist_ok=True)
    for nr, q in con.execute(f"SELECT query_nr, query FROM {fn}() ORDER BY 1").fetchall():
        (d / f"q{nr:02d}.sql").write_text(q.rstrip() + "\n")
exp = {}
for nr, ans in con.execute("SELECT query_nr, answer FROM tpch_answers() WHERE scale_factor=1 ORDER BY 1").fetchall():
    exp[f"q{nr:02d}"] = len([l for l in ans.splitlines() if l.strip()]) - 1  # minus header
(here / "tpch" / "expected_rows_sf1.json").write_text(json.dumps(exp, indent=1) + "\n")
