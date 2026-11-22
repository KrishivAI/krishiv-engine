#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001
"""Record Spark's `LIKE` / `ILIKE` over `UTF8_LCASE` (SQL-40) from real Spark runs.

Spark source: `Like` / `ILike` in `sql/catalyst/.../expressions/regexpExpressions.scala` at each tag
(`inputTypes` is `StringTypeBinaryLcase`; the pattern is compiled with CASE_INSENSITIVE | UNICODE_CASE for
`UTF8_LCASE`, and `ILike` is `Like(Lower(l), Lower(r))`). Every expected value below comes from running the
query in an `apache/spark:<ver>` container, never from the source. Each case records the output column name
(`SELECT <expr>` with no alias), the `simpleString` type, the value rendered as text (or the error class,
SQLSTATE and first message line). Column-operand cases read a temporary view built from VALUES.

Usage (from the repo root; the containers are only read from, never stopped):

    python tests/scripts/gen_like_lcase_goldens.py 4.3.0=p3-1-spark43 4.2.0=s8-sql7-spark420 \
        4.1.0=p31-cws-spark-4.1.0 4.0.0=p31-cws-spark-4.0.0

Writes `tests/fixtures/spark-<ver>/like-lcase.json` (format_version 1). Spark 3.5 has no collations.
`--runner` (internal) is the part that executes inside the container under `spark-submit`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

L = "collate utf8_lcase"
SIGMA, FINAL, CAPS = "σ", "ς", "Σ"

# (id, sql). `SELECT <expr>` is used unaliased so the column name is recorded.
CASES: list[tuple[str, str]] = [
    ("plain_like", f"SELECT 'APPLE' {L} LIKE 'app%'"),
    ("plain_like_false", f"SELECT 'xyz' {L} LIKE 'APP%'"),
    ("plain_ilike", f"SELECT 'APPLE' {L} ILIKE 'app%'"),
    ("null_like", f"SELECT CAST(NULL AS STRING) {L} LIKE 'a%'"),
    ("final_sigma_word", f"SELECT 'Α{CAPS}' {L} LIKE 'α{FINAL}'"),
    ("sigma_caps", f"SELECT '{SIGMA}' {L} LIKE '{CAPS}'"),
    ("final_caps", f"SELECT '{FINAL}' {L} LIKE '{CAPS}'"),
    ("final_sigma", f"SELECT '{FINAL}' {L} LIKE '{SIGMA}'"),
    ("sigma_final", f"SELECT '{SIGMA}' {L} LIKE '{FINAL}'"),
    ("ilike_final_caps", f"SELECT '{FINAL}' {L} ILIKE '{CAPS}'"),
    ("ilike_final_sigma", f"SELECT '{FINAL}' {L} ILIKE '{SIGMA}'"),
    ("long_s", f"SELECT 'ſ' {L} LIKE 's'"),
    ("s_long", f"SELECT 's' {L} LIKE 'ſ'"),
    ("ilike_long_s", f"SELECT 'ſ' {L} ILIKE 's'"),
    ("dotless_i", f"SELECT 'ı' {L} LIKE 'i'"),
    ("i_dotless", f"SELECT 'i' {L} LIKE 'ı'"),
    ("ilike_dotless_i", f"SELECT 'ı' {L} ILIKE 'i'"),
    ("idot_like_i", f"SELECT 'İ' {L} LIKE 'i'"),
    ("i_like_idot", f"SELECT 'i' {L} LIKE 'İ'"),
    ("idot_like_idot", f"SELECT 'İ' {L} LIKE 'İ'"),
    ("micro_like", f"SELECT 'µ' {L} LIKE 'μ'"),
    ("micro_ilike", f"SELECT 'µ' {L} ILIKE 'μ'"),
    ("idot_ilike_i", f"SELECT 'İ' {L} ILIKE 'i'"),
    ("idot_ilike_i_dot", f"SELECT 'İ' {L} ILIKE 'i̇'"),
    ("i_dot_ilike_idot", f"SELECT 'i̇' {L} ILIKE 'İ'"),
    ("idot_like_i_dot", f"SELECT 'İ' {L} LIKE 'i̇'"),
    ("xidoty_like_xiy", f"SELECT 'xİy' {L} LIKE 'xiy'"),
    ("xidoty_like_x_y", f"SELECT 'xİy' {L} LIKE 'x_y'"),
    ("xidoty_like_x__y", f"SELECT 'xİy' {L} LIKE 'x__y'"),
    ("not_like", f"SELECT 'APPLE' {L} NOT LIKE 'app%'"),
    ("not_like_long_s", f"SELECT 'ſ' {L} NOT LIKE 's'"),
    ("not_ilike", f"SELECT 'APPLE' {L} NOT ILIKE 'app%'"),
    ("not_paren_like", f"SELECT NOT ('ſ' {L} LIKE 's')"),
    ("like_func", f"SELECT like('ſ' {L}, 's')"),
    ("like_func_escape", f"SELECT like('50%' {L}, '50/%' {L}, '/')"),
    ("ilike_func", f"SELECT ilike('ς' {L}, '{CAPS}')"),
    ("like_func_col_collate", f"SELECT like('AB' {L}, 'a%' {L})"),
    ("escape_percent", f"SELECT '50%' {L} LIKE '50\\%'"),
    ("escape_percent_no_match", f"SELECT '500' {L} LIKE '50\\%'"),
    ("escape_percent_case", f"SELECT 'AB%' {L} LIKE 'ab\\%'"),
    ("escape_clause", f"SELECT '50%' {L} LIKE '50/%' ESCAPE '/'"),
    ("escape_clause_ilike", f"SELECT 'AB%' {L} ILIKE 'ab/%' ESCAPE '/'"),
    ("pattern_collated", f"SELECT 'Ab' {L} LIKE 'a%' {L}"),
    ("operand_pattern_collated", f"SELECT 'Ab' LIKE 'a%' {L}"),
    ("pattern_collate_fn", f"SELECT 'Ab' {L} LIKE collate('a%', 'utf8_lcase')"),
    ("ilike_name", f"SELECT 'Ab' {L} ILIKE 'a%' {L}"),
    ("underscore_counts_code_points", f"SELECT 'ß' {L} LIKE '_'"),
    ("sharp_s_ss", f"SELECT 'ß' {L} LIKE 'ss'"),
    ("unicode_ci_like", "SELECT 'APPLE' collate unicode_ci LIKE 'app%'"),
    ("unicode_ci_ilike", "SELECT 'APPLE' collate unicode_ci ILIKE 'app%'"),
    ("binary_like", "SELECT 'APPLE' collate utf8_binary LIKE 'app%'"),
    ("binary_ilike", "SELECT 'APPLE' collate utf8_binary ILIKE 'app%'"),
]

# Column operand + column pattern: a view with both columns collated UTF8_LCASE, ordered by id.
COL_ROWS = [
    (1, f"{CAPS}X", f"{SIGMA}%"),
    (2, "ſ", "S"),
    (3, "İ", "i"),
    (4, "İ", "i̇"),
    (5, "x", "X"),
    (6, "i̇", "İ"),
    (7, f"{FINAL}", f"{CAPS}"),
    (8, "µ", "μ"),
    (9, "ı", "I"),
    (10, "50%", "50\\%"),
    (11, "500", "50\\%"),
    (12, None, "a"),
    (13, "a", None),
]
SETUP = [
    "CREATE OR REPLACE TEMPORARY VIEW t AS SELECT id, collate(c, 'utf8_lcase') AS c, "
    "collate(p, 'utf8_lcase') AS p FROM VALUES "
    + ", ".join(
        f"({i}, " + ", ".join("NULL" if v is None else "'" + v.replace("'", "''") + "'" for v in (c, p)) + ")"
        for i, c, p in COL_ROWS
    )
    + " AS v(id, c, p)"
]
COL_CASES: list[tuple[str, str]] = [
    ("col_like", "SELECT id, c LIKE p FROM t ORDER BY id"),
    ("col_ilike", "SELECT id, c ILIKE p FROM t ORDER BY id"),
    ("col_not_like", "SELECT id, c NOT LIKE p FROM t ORDER BY id"),
    ("col_not_ilike", "SELECT id, c NOT ILIKE p FROM t ORDER BY id"),
    ("col_like_func", "SELECT id, like(c, p) FROM t ORDER BY id"),
    ("col_ilike_func", "SELECT id, ilike(c, p) FROM t ORDER BY id"),
    ("col_not_paren", "SELECT id, NOT (c LIKE p) FROM t ORDER BY id"),
    ("col_literal_pattern", "SELECT id, c LIKE 'σ%' FROM t ORDER BY id"),
    ("col_literal_pattern_ilike", "SELECT id, c ILIKE 'S' FROM t ORDER BY id"),
    ("col_operand_literal", f"SELECT id, 'ΑΣ' {L} LIKE p FROM t ORDER BY id"),
]


def runner(setup: list[str], cases: list[list[str]]) -> None:
    """Runs inside the container (spark-submit); prints one JSON document."""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    for s in setup:
        spark.sql(s)
    out = []
    for cid, sql in cases:
        rec = {"id": cid, "sql": sql}
        try:
            df = spark.sql(sql)
            rec["columns"] = df.columns
            rec["types"] = [f.dataType.simpleString() for f in df.schema.fields]
            rec["rows"] = [[None if v is None else str(v) for v in r] for r in df.collect()]
        except Exception as e:
            je = getattr(e, "_origin", None) or getattr(e, "java_exception", None)
            err = {}
            if je is not None:
                for g in ("getCondition", "getErrorClass"):
                    try:
                        err["class"] = str(getattr(je, g)())
                        break
                    except Exception:
                        continue
                try:
                    err["sqlState"] = str(je.getSqlState())
                    err["params"] = {str(k): str(v) for k, v in sorted(je.getMessageParameters().items())}
                except Exception:
                    pass
                err["message"] = str(je.getMessage()).split("\n")[0]
            else:
                err["message"] = str(e).split("\n")[0]
            rec["error"] = err
        out.append(rec)
    print("@@JSON@@" + json.dumps({"version": spark.version, "results": out}, ensure_ascii=True))


def main() -> None:
    if sys.argv[1] == "--runner":
        runner(json.loads(sys.argv[2]), json.loads(sys.argv[3]))
        return
    me = Path(__file__).resolve()
    root = me.parents[2]
    cases = [[a, b] for a, b in CASES + COL_CASES]
    for spec in sys.argv[1:]:
        ver, ctr = spec.split("=")
        # A per-run path: concurrent runs must not replace each other's script.
        remote = f"/tmp/gen_like_lcase_{uuid.uuid4().hex}.py"
        subprocess.run(["docker", "cp", str(me), f"{ctr}:{remote}"], check=True)
        try:
            p = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-e",
                    "PYTHONIOENCODING=utf-8",
                    "-e",
                    "LC_ALL=C.UTF-8",
                    ctr,
                    "/opt/spark/bin/spark-submit",
                    "--conf",
                    "spark.driver.extraJavaOptions=-Dfile.encoding=UTF-8",
                    remote,
                    "--runner",
                    json.dumps(SETUP, ensure_ascii=True),
                    json.dumps(cases, ensure_ascii=True),
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        finally:
            subprocess.run(["docker", "exec", ctr, "rm", "-f", remote], check=False)
        line = next(x for x in p.stdout.splitlines() if x.startswith("@@JSON@@"))
        doc = json.loads(line[len("@@JSON@@") :])
        assert doc["version"] == ver, (doc["version"], ver)
        doc = {
            "format_version": 1,
            "generator": "tests/scripts/gen_like_lcase_goldens.py",
            "spark_version": doc["version"],
            "spark_tag": f"v{ver}",
            "setup": SETUP,
            "results": doc["results"],
        }
        d = root / "tests/fixtures" / f"spark-{ver}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "like-lcase.json").write_text(json.dumps(doc, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
        print(ver, len(doc["results"]), "cases")


if __name__ == "__main__":
    main()
