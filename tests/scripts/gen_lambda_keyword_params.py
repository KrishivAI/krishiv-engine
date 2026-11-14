#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the lambda-parameter keyword fixture (SQL-3).

Spark reads every keyword as an identifier unless keywords are enforced, so a lambda parameter may be any
keyword (`(left, right) -> ...` is Spark's own `array_sort` example). For every Spark 4.2 keyword
(`tests/fixtures/spark-4.2.0/sql-keywords.json`) and every body shape below, a REAL Spark run parses
`SELECT transform(array(1), <lambda>)` with the session SQL parser (default settings) and the fixture records
how many references in the lambda body Spark's `AstBuilder` made lambda variables (`lambda 'K` in the parsed
plan, the parameter list excluded), or the error condition when it does not parse. A keyword Spark reads as
something else in the body (`true`, `null`, `current_path`) counts 0 there.

Usage (from repo root; a throw-away container, removed afterwards):

    docker run -d --rm --name krishiv-lambda-probe apache/spark:4.2.0 sleep infinity
    python3 tests/scripts/gen_lambda_keyword_params.py --container krishiv-lambda-probe
    docker stop krishiv-lambda-probe
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gen_function_matrix import container_image, spark_shell

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_lambda_keyword_params.py"
KEYWORDS = Path("tests/fixtures/spark-4.2.0/sql-keywords.json")

# `K` is the keyword under test, `y` a second parameter.
SHAPES = [
    "(K, y) -> K - y",
    "(K, y) -> K",
    "(y, K) -> y + K",
    "K -> K - 1",
    "K -> K",
    "(K, y) -> K IS NULL",
    "(K, y) -> K.a",
    "(K, y) -> K[0]",
    "(K, y) -> -K",
    "(K, y) -> K < y",
    "(K, y) -> K AND y",
    "(K, y) -> K || y",
    "(y, K) -> CASE WHEN y < K THEN K ELSE K END",
    "(y, K) -> CASE WHEN K THEN y WHEN K THEN y END",
    "(y, K) -> y[K]",
    "(y, K) -> y <=> K",
    "(y, K) -> y - K",
    "(y, K) -> abs(K)",
    "(y, K) -> K OR y",
    "(y, K) -> K = y",
]

SCRIPT = """
import scala.io.Source
val p = spark.sessionState.sqlParser
println("@@VERSION " + spark.version)
Source.fromFile("/tmp/krishiv_cases.jsonl").getLines().foreach { q =>
  val r = try { "OK " + p.parsePlan(q).toString.replace('\\n', ' ') } catch {
    case e: org.apache.spark.sql.catalyst.parser.ParseException => "ERR " + e.getCondition
  }
  println("@@R " + r)
}
println("@@END")
System.exit(0)
"""


def lambda_sql(shape: str, keyword: str) -> str:
    return "SELECT transform(array(1), " + re.sub(r"\bK\b", keyword, shape) + ")"


def body_refs(plan: str, keyword: str) -> int:
    # The parameter list prints the parameter once more (`lambda 'K, lambda 'y, false`).
    return len(re.findall(r"lambda '" + re.escape(keyword) + r"\b", plan)) - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="krishiv-lambda-probe")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/lambda-keyword-params.json")
    args = ap.parse_args()

    keywords = [k["text"] for k in json.loads(KEYWORDS.read_text(encoding="utf-8"))["keywords"]]
    cases = [(k, s) for k in keywords for s in SHAPES]
    data = "".join(lambda_sql(s, k) + "\n" for k, s in cases)
    lines = spark_shell(args.container, SCRIPT, data)
    version = next(line.split(" ", 1)[1] for line in lines if line.startswith("@@VERSION "))
    results = [line[4:] for line in lines if line.startswith("@@R ")]
    if len(results) != len(cases):
        raise SystemExit(f"spark-shell printed {len(results)} results for {len(cases)} cases")
    refs: dict[str, list[int | str]] = {k: [] for k in keywords}
    for (k, _), r in zip(cases, results, strict=True):
        refs[k].append(body_refs(r[3:], k) if r.startswith("OK ") else r[4:])
    head = {
        "spark_version": version,
        "image": container_image(args.container),
        "generator": GENERATOR,
        "notes": "SELECT transform(array(1), <shape with K replaced by the keyword>) parsed by Spark's session "
        "SQL parser with default settings; per keyword, one entry per shape: the number of lambda-variable "
        "references to the keyword in the body, or the parse error condition.",
        "shapes": SHAPES,
    }
    # One line per keyword keeps the fixture reviewable.
    out = [f'{{"format_version": {FORMAT_VERSION},']
    out += [f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()]
    out.append(' "body_refs": {')
    out.append(",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in refs.items()))
    out.append(" }\n}")
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases against Spark {version} -> {args.out}")


if __name__ == "__main__":
    main()
