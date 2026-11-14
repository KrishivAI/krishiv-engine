#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  (the embedded Scala probe and Spark source quotes keep their lines)
"""Generate the TABLESAMPLE / Sample fixture (SQL-14).

A REAL Spark run (local[1]) records, for `TABLESAMPLE (x PERCENT | BUCKET x OUT OF y) [REPEATABLE (seed)]`:
the parsed `Sample(lb, ub, seed, method)` node, the sampled `id` lists over several relation shapes
(one-split range, a 4-split range, a subquery, VALUES, one and three Parquet files, a nested sample, a
filter above vs below the sample), the first 10 `nextDouble()` of `XORShiftRandom(seed)` for the seeds in
SEEDS, the error rows for the bad values, and what an unseeded sample does. Every sampled list is then
re-derived with the pure-Python port in `tablesample_ref.py`; the script fails when the two disagree.

Spark 4.2 reads `TABLESAMPLE` only after a table name, a subquery or a parenthesised VALUES: after a table
function (`FROM range(0, 10) TABLESAMPLE (...)`) it is read as a table alias and fails (see `errors`).

Usage (from repo root; a throw-away container, removed afterwards):

    docker run -d --rm --name krishiv-tablesample-probe apache/spark:4.2.0 sleep infinity
    python3 tests/scripts/gen_tablesample.py --container krishiv-tablesample-probe
    docker stop krishiv-tablesample-probe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from gen_function_matrix import container_image, spark_shell
from tablesample_ref import XORShiftRandom, bernoulli_cell, hash_seed

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_tablesample.py"
SEEDS = [0, 7, 42, -1, 1 << 40]
LONG_MAX = (1 << 63) - 1
VALUES20 = "(VALUES " + ",".join(f"({i})" for i in range(1, 21)) + ")"

# (name, spark confs applied for the run, sql, expected partitions of the sampled relation or None)
S = "TABLESAMPLE"
CASES: list[tuple[str, str, str]] = []


def add(name: str, sql: str, conf: str = "-") -> None:
    CASES.append((name, conf, sql))


for rel_name, rel in [("view", "t"), ("subquery", "(SELECT id FROM range(0, 1000))")]:
    add(f"{rel_name}_10pct_seed7", f"SELECT id FROM {rel} {S} (10 PERCENT) REPEATABLE (7)")
    add(f"{rel_name}_50pct_seed0", f"SELECT id FROM {rel} {S} (50 PERCENT) REPEATABLE (0)")
    add(f"{rel_name}_1pct_seed42", f"SELECT id FROM {rel} {S} (1 PERCENT) REPEATABLE (42)")
    add(f"{rel_name}_bucket3of10_seed7", f"SELECT id FROM {rel} {S} (BUCKET 3 OUT OF 10) REPEATABLE (7)")
add("view_0pct_seed7", f"SELECT id FROM t {S} (0 PERCENT) REPEATABLE (7)")
add("view_100pct_seed7", f"SELECT id FROM t {S} (100 PERCENT) REPEATABLE (7)")
add("view_bucket0of10_seed7", f"SELECT id FROM t {S} (BUCKET 0 OUT OF 10) REPEATABLE (7)")
add("view_10pct_seed2p40", f"SELECT id FROM t {S} (10 PERCENT) REPEATABLE (1099511627776)")
add("view_10pct_seedLongMax", f"SELECT id FROM t {S} (10 PERCENT) REPEATABLE ({LONG_MAX})")
add("view_fraction_0p5pct_seed1", f"SELECT id FROM t {S} (0.5 PERCENT) REPEATABLE (1)")
add("view_pct_over100_within_eps", f"SELECT id FROM t {S} (100.0000001 PERCENT) REPEATABLE (7)")
add("view_alias_after_sample", f"SELECT x.id FROM t {S} (10 PERCENT) REPEATABLE (7) x")
add("view_bernoulli_keyword_seed3", f"SELECT id FROM t {S} BERNOULLI (10 PERCENT) REPEATABLE (3)")
# The RNG sees the rows below the Sample: a filter above it keeps the sampled rows that pass, a filter in a
# subquery below it changes which rows are drawn for.
add("filter_above_sample", f"SELECT id FROM t {S} (10 PERCENT) REPEATABLE (7) WHERE id % 2 = 0")
add("filter_below_sample", f"SELECT id FROM (SELECT id FROM t WHERE id % 2 = 0) {S} (10 PERCENT) REPEATABLE (7)")
add("project_below_sample", f"SELECT x FROM (SELECT id + 1 AS x FROM t) {S} (10 PERCENT) REPEATABLE (7)")
add(
    "nested_50pct_seed7_twice",
    f"SELECT id FROM (SELECT id FROM t {S} (50 PERCENT) REPEATABLE (7)) {S} (50 PERCENT) REPEATABLE (7)",
)
add("values20_50pct_seed7", f"SELECT col1 FROM {VALUES20} {S} (50 PERCENT) REPEATABLE (7)")
add("values20_30pct_seed0", f"SELECT col1 FROM {VALUES20} {S} (30 PERCENT) REPEATABLE (0)")
add("values20_bucket5of10_seed42", f"SELECT col1 FROM {VALUES20} {S} (BUCKET 5 OUT OF 10) REPEATABLE (42)")
# A 4-split range: partition i uses seed + i, ids 250*i .. 250*i+249; checked with and without whole-stage codegen.
for conf in ("-", "wholeStage=false"):
    sfx = "" if conf == "-" else "_nocodegen"
    add(f"range4_10pct_seed7{sfx}", f"SELECT id FROM t4 {S} (10 PERCENT) REPEATABLE (7)", conf)
    add(f"range4_50pct_seed0{sfx}", f"SELECT id FROM t4 {S} (50 PERCENT) REPEATABLE (0)", conf)
    add(f"range4_10pct_seedLongMax{sfx}", f"SELECT id FROM t4 {S} (10 PERCENT) REPEATABLE ({LONG_MAX})", conf)
add(
    "range4_via_subquery_10pct_seed7",
    f"SELECT id FROM (SELECT * FROM range(0, 1000, 1, 4)) {S} (10 PERCENT) REPEATABLE (7)",
)
add("range4_dataframe_api_equivalent", "SELECT id FROM t4 TABLESAMPLE (10 PERCENT) REPEATABLE (7)")
add("parquet1_10pct_seed7", f"SELECT id FROM pq1 {S} (10 PERCENT) REPEATABLE (7)")
add("parquet1_allcols_10pct_seed7", f"SELECT id, b, s FROM pq1 {S} (10 PERCENT) REPEATABLE (7)")
add("parquet3files_10pct_seed7", f"SELECT id FROM pq3 {S} (10 PERCENT) REPEATABLE (7)")
add(
    "parquet1_union_all_10pct_seed7",
    f"SELECT id FROM (SELECT * FROM pq1 UNION ALL SELECT * FROM pq1) {S} (10 PERCENT) REPEATABLE (7)",
)
add("parquet1_count_10pct_seed7", f"SELECT id FROM pq1 {S} (10 PERCENT) REPEATABLE (7) ORDER BY id")
add("parquet_big_1pct_seed7", f"SELECT id FROM big {S} (1 PERCENT) REPEATABLE (7)")

ERRORS = [
    f"SELECT * FROM t {S} (101 PERCENT)",
    f"SELECT * FROM t {S} (-1 PERCENT)",
    f"SELECT * FROM t {S} (BUCKET 11 OUT OF 10)",
    f"SELECT * FROM t {S} (BUCKET 3 OUT OF 0)",
    f"SELECT * FROM t {S} (BUCKET 3 OUT OF 10 ON id)",
    f"SELECT * FROM t {S} (BUCKET 3 OUT OF 10 ON rand())",
    f"SELECT * FROM t {S} (10 PERCENT) REPEATABLE (1.5)",
    f"SELECT * FROM t {S} (10 PERCENT) REPEATABLE (-3)",
    f"SELECT * FROM t {S} (10 PERCENT) REPEATABLE (99999999999999999999)",
    f"SELECT * FROM t {S} (10 BYTES)",
    f"SELECT * FROM t {S} (10B)",
    f"SELECT * FROM t {S} (10)",
    f"SELECT * FROM t {S} (id PERCENT)",
    f"SELECT * FROM t {S} SYSTEM (10 PERCENT) REPEATABLE (3)",
    f"SELECT * FROM range(0, 1000) {S} (10 PERCENT)",
    f"SELECT * FROM t x {S} (10 PERCENT)",
    f"SELECT * FROM t a JOIN t b ON a.id = b.id {S} (50 PERCENT)",
]

# Parses that succeed with a surprising tree (no golden rows).
QUIRKS = [
    f"SELECT * FROM t {S} (10 PERCENT) REPEATABLE (abc)",
    f"SELECT * FROM t {S} (10 ROWS) REPEATABLE (3)",
    f"SELECT * FROM t {S} (10 ROWS)",
    f"SELECT * FROM t {S} (10 PERCENT)",
    f"SELECT * FROM t {S} SYSTEM (10 PERCENT)",
    f"SELECT * FROM t {S} (BUCKET 0 OUT OF 10)",
]

SCRIPT = r"""
import scala.io.Source
val cls = Class.forName("org.apache.spark.util.random.XORShiftRandom")
def mk(s: Long): java.util.Random = cls.getConstructor(classOf[Long]).newInstance(java.lang.Long.valueOf(s)).asInstanceOf[java.util.Random]
val hcls = Class.forName("org.apache.spark.util.random.XORShiftRandom$")
println("@@VERSION " + spark.version)
println("@@CONF defaultParallelism=" + spark.sparkContext.defaultParallelism + " shuffle.partitions=" + spark.conf.get("spark.sql.shuffle.partitions") +
  " files.maxPartitionBytes=" + spark.conf.get("spark.sql.files.maxPartitionBytes") + " files.openCostInBytes=" + spark.conf.get("spark.sql.files.openCostInBytes") +
  " codegen.wholeStage=" + spark.conf.get("spark.sql.codegen.wholeStage") + " master=" + spark.sparkContext.master + " java=" + System.getProperty("java.version"))
spark.range(0, 1000).createOrReplaceTempView("t")
spark.range(0, 1000, 1, 4).createOrReplaceTempView("t4")
spark.range(0, 1000).selectExpr("id", "id * 2 AS b", "cast(id AS string) AS s").write.mode("overwrite").parquet("/tmp/pq1")
spark.range(0, 1000).selectExpr("id", "id * 2 AS b").repartition(3).write.mode("overwrite").parquet("/tmp/pq3")
spark.range(0, 2000000).selectExpr("id", "rand(1) AS r").write.mode("overwrite").parquet("/tmp/big")
spark.read.parquet("/tmp/pq1").createOrReplaceTempView("pq1")
spark.read.parquet("/tmp/pq3").createOrReplaceTempView("pq3")
spark.read.parquet("/tmp/big").createOrReplaceTempView("big")
val p = spark.sessionState.sqlParser
def esc(s: String) = s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
for (s <- Seq(0L, 7L, 42L, -1L, 1L << 40)) {
  val r = mk(s)
  val d = (0 until 10).map(_ => r.nextDouble())
  println("@@RNG " + s + "\t" + hcls.getMethod("hashSeed", classOf[Long]).invoke(hcls.getField("MODULE$").get(null), java.lang.Long.valueOf(s)) + "\t" +
    d.map(x => java.lang.Long.toHexString(java.lang.Double.doubleToLongBits(x))).mkString(",") + "\t" + d.map(_.toString).mkString(","))
}
val lines = Source.fromFile("/tmp/krishiv_cases.jsonl").getLines().toList
for (l <- lines) {
  val a = l.split("\t", 4)
  val (kind, name, conf, sql) = (a(0), a(1), a(2), a(3))
  kind match {
    case "case" =>
      if (conf == "wholeStage=false") spark.conf.set("spark.sql.codegen.wholeStage", "false")
      val parsed = p.parsePlan(sql).toString.split("\n").map(_.trim).filter(_.contains("Sample")).headOption.getOrElse("-")
      val df = spark.sql(sql)
      val parts = df.rdd.getNumPartitions
      val ids = df.collect().map(r => r.get(0).toString.toLong).sorted
      val plan = df.queryExecution.executedPlan.toString.split("\n").map(_.trim).filter(_.contains("Sample")).mkString(" | ")
      println("@@CASE " + name + "\t" + parts + "\t" + parsed + "\t" + esc(plan) + "\t" + ids.mkString(","))
      spark.conf.set("spark.sql.codegen.wholeStage", "true")
    case "error" =>
      val r = try { p.parsePlan(sql); spark.sql(sql); "OK\t\t\t" } catch {
        case e: org.apache.spark.sql.catalyst.parser.ParseException => "ParseException\t" + e.getCondition + "\t" + e.getSqlState + "\t" + esc(e.getMessage)
        case e: Throwable => e.getClass.getName + "\t\t\t" + esc(String.valueOf(e.getMessage))
      }
      println("@@ERR " + esc(sql) + "\t" + r)
    case "quirk" =>
      val r = try { p.parsePlan(sql).toString.split("\n").map(_.trim).mkString(" ") } catch { case e: Throwable => "ERR " + e.getClass.getName }
      println("@@QUIRK " + esc(sql) + "\t" + esc(r))
  }
}
// Unseeded: the seed is drawn once per physical plan (SampleExec), so one Dataset executed twice repeats itself.
val q = "SELECT id FROM t TABLESAMPLE (10 PERCENT)"
val runs = (0 until 20).map(_ => spark.sql(q).collect().map(_.getLong(0)).mkString(","))
val one = spark.sql(q); val twice = one.collect().map(_.getLong(0)).toSeq == one.collect().map(_.getLong(0)).toSeq
println("@@UNSEEDED " + runs.distinct.size + "\t" + runs.size + "\t" + twice + "\t" + esc(spark.sql(q).queryExecution.analyzed.toString.split("\n").filter(_.contains("Sample")).mkString("|")))
println("@@END")
System.exit(0)
"""


def parse_lines(lines: list[str], tag: str) -> list[list[str]]:
    return [line[len(tag) + 3 :].split("\t") for line in lines if line.startswith(f"@@{tag} ")]


def unesc(s: str) -> str:
    return s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="krishiv-tablesample-probe")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/tablesample.json")
    args = ap.parse_args()

    data = "".join(f"case\t{n}\t{c}\t{s}\n" for n, c, s in CASES)
    data += "".join(f"error\t-\t-\t{s}\n" for s in ERRORS)
    data += "".join(f"quirk\t-\t-\t{s}\n" for s in QUIRKS)
    lines = spark_shell(args.container, SCRIPT, data)
    version = next(line.split(" ", 1)[1] for line in lines if line.startswith("@@VERSION "))
    conf = next(line.split(" ", 1)[1] for line in lines if line.startswith("@@CONF "))

    rng: dict[str, dict] = {}
    for seed, hashed, bits, dec in parse_lines(lines, "RNG"):
        ref = XORShiftRandom(int(seed))
        mine = [struct.pack(">d", ref.next_double()).hex().lstrip("0") for _ in range(10)]
        assert mine == bits.split(","), f"reference RNG disagrees with Spark for seed {seed}"
        assert hash_seed(int(seed)) == int(hashed) & ((1 << 64) - 1)
        rng[seed] = {"hash_seed": int(hashed), "first10_double_bits_hex": bits.split(","), "first10": dec.split(",")}

    cases: list[dict[str, Any]] = []
    for name, parts, parsed, plan, ids in parse_lines(lines, "CASE"):
        sorted_ids = [int(x) for x in ids.split(",")] if ids else []
        cases.append(
            {
                "name": name,
                "partitions": int(parts),
                "parsed_sample": parsed,
                "executed_sample_nodes": unesc(plan),
                "count": len(sorted_ids),
                "ids": sorted_ids,
            }
        )
    expected_names = [n for n, _, _ in CASES]
    got_names = [c["name"] for c in cases]
    assert got_names == expected_names, (
        f"case order or count differs; missing {[n for n in expected_names if n not in got_names]}"
    )
    by_sql = {n: s for n, _, s in CASES}

    # Self-check: derive the rows with the reference port for the shapes where the partition layout is known.
    def check(name: str, rows_per_part: list[list[int]], lb: float, ub: float, seed: int) -> None:
        got = sorted(bernoulli_cell(rows_per_part, lb, ub, seed))
        want = next(c["ids"] for c in cases if c["name"] == name)
        assert got == want, f"reference sampler disagrees with Spark on {name}: {len(got)} vs {len(want)}"

    one = [list(range(1000))]
    four = [list(range(250 * i, 250 * i + 250)) for i in range(4)]
    check("view_10pct_seed7", one, 0.0, 0.1, 7)
    check("subquery_50pct_seed0", one, 0.0, 0.5, 0)
    check("view_1pct_seed42", one, 0.0, 0.01, 42)
    check("view_bucket3of10_seed7", one, 0.0, 3 / 10, 7)
    check("view_10pct_seed2p40", one, 0.0, 0.1, 1 << 40)
    check("view_10pct_seedLongMax", one, 0.0, 0.1, LONG_MAX)
    check("values20_50pct_seed7", [list(range(1, 21))], 0.0, 0.5, 7)
    check("values20_bucket5of10_seed42", [list(range(1, 21))], 0.0, 0.5, 42)
    for sfx in ("", "_nocodegen"):
        check(f"range4_10pct_seed7{sfx}", four, 0.0, 0.1, 7)
        check(f"range4_50pct_seed0{sfx}", four, 0.0, 0.5, 0)
        # seed + index wraps as a signed 64-bit long.
        check(f"range4_10pct_seedLongMax{sfx}", four, 0.0, 0.1, LONG_MAX)
    check("parquet1_10pct_seed7", one, 0.0, 0.1, 7)
    check("filter_below_sample", [[i for i in range(1000) if i % 2 == 0]], 0.0, 0.1, 7)
    check("parquet1_union_all_10pct_seed7", [list(range(1000)), list(range(1000))], 0.0, 0.1, 7)

    for c in cases:
        if c["name"] == "parquet_big_1pct_seed7":
            ids = c.pop("ids")
            c["ids_first50"], c["ids_last50"] = ids[:50], ids[-50:]
            c["ids_sha256_comma_joined"] = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    for c in cases:
        c["sql"] = by_sql[c["name"]]
        c["confs"] = {n: cf for n, cf, _ in CASES}[c["name"]]

    errors = [
        {
            "sql": unesc(sql),
            "thrown": kind,
            "condition": cond or None,
            "sqlstate": st if st not in ("", "null") else None,
            "message": unesc(msg),
        }
        for sql, kind, cond, st, msg in parse_lines(lines, "ERR")
    ]
    quirks = [{"sql": unesc(sql), "parsed": unesc(tree)} for sql, tree in parse_lines(lines, "QUIRK")]
    distinct, runs, twice, analyzed = parse_lines(lines, "UNSEEDED")[0]
    out = {
        "format_version": FORMAT_VERSION,
        "spark_version": version,
        "image": container_image(args.container),
        "generator": GENERATOR,
        "run_conf": conf,
        "notes": "Real Spark run, local[1]. `ids` are the sorted first column of each query. `parsed_sample` is the "
        "Sample node from the session SQL parser (lowerBound, upperBound, withReplacement, [seed], method). "
        "`executed_sample_nodes` are the Sample lines of the executed plan. `rng` holds XORShiftRandom(seed) values.",
        "rng": rng,
        "cases": cases,
        "errors": errors,
        "quirks": quirks,
        "unseeded": {
            "sql": "SELECT id FROM t TABLESAMPLE (10 PERCENT)",
            "runs": int(runs),
            "distinct_results": int(distinct),
            "same_dataset_twice_equal": twice == "true",
            "analyzed_sample_node": unesc(analyzed),
            "rule": "SampleExec.resolvedSeed = seed.getOrElse((math.random * 1000).toLong), drawn when the physical node is constructed",
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases, {len(errors)} errors against Spark {version} -> {args.out}")


if __name__ == "__main__":
    main()
