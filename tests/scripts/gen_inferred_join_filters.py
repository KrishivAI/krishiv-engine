#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the inferred-join-filters fixture (perf-2 T072b unit 3).

Ground truth for Spark's `InferFiltersFromConstraints` as seen in the optimized plan. Every value comes
from a REAL Spark 4.2.0 run (pyspark 4.2.0, local mode), never from docs or memory. Only plans are
needed, so every table is created EMPTY (no data files) in an in-memory catalog.

For each query the optimized plan is computed twice, once normally and once with
`spark.sql.optimizer.excludedRules=...InferFiltersFromConstraints` (so the predicates left are the
ones the query wrote, after predicate pushdown). Setting `spark.sql.constraintPropagationEnabled=false`
was tried first and does NOT suppress the inferred filters in the optimized plan (Spark 4.2.0).
For every scan (leaf of the optimized plan, including scalar-subquery and EXISTS plans) the conjuncts of
the Filter directly above it are split into:

  written       (a) present with the rule excluded
  isnotnull     (b) inferred `isnotnull(attribute)`
  equivalence   (c) inferred, `attr <cmp> literal` / `attr IN (literals)` / `NOT` of those: a constant or
                    IN predicate moved through an equivalence class of `=` keys
  substitution  (d) any other inferred predicate: substitution into other expressions (casts, arithmetic,
                    attribute = attribute, ...); Krishiv does not infer these

Scans are keyed by `<relation>#<occurrence>` in plan traversal order (children first, then the node's
expression subqueries). Column names are printed without expression ids.

Usage (run on a machine that may run Spark, see krishiv-internal CLAUDE.md "Spark oracle"; local mode):

    env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' \
      python tests/scripts/gen_inferred_join_filters.py --root . \
        --out tests/fixtures/spark-4.2.0/inferred-join-filters.json

`--root` is the engine repo root (reads bench/queries/tpch and tests/spark-golden/v4.2.0/inputs).
Output is deterministic and contains no local paths or hostnames.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

FORMAT_VERSION = 1
EXCLUDED = "spark.sql.optimizer.excludedRules"
RULE = "org.apache.spark.sql.catalyst.optimizer.InferFiltersFromConstraints"

# Exact TPC-H types as tpchgen writes them (Parquet at sf1: int64 keys, int32 small ints,
# decimal(15,2) money, date32, string). Spark's file sources make every column nullable on read,
# so the tables are declared nullable, which is what a Parquet scan looks like to the optimizer.
TPCH = {
    "lineitem": "l_orderkey bigint, l_partkey bigint, l_suppkey bigint, l_linenumber int, "
    "l_quantity decimal(15,2), l_extendedprice decimal(15,2), l_discount decimal(15,2), "
    "l_tax decimal(15,2), l_returnflag string, l_linestatus string, l_shipdate date, "
    "l_commitdate date, l_receiptdate date, l_shipinstruct string, l_shipmode string, "
    "l_comment string",
    "orders": "o_orderkey bigint, o_custkey bigint, o_orderstatus string, o_totalprice decimal(15,2), "
    "o_orderdate date, o_orderpriority string, o_clerk string, o_shippriority int, o_comment string",
    "customer": "c_custkey bigint, c_name string, c_address string, c_nationkey bigint, c_phone string, "
    "c_acctbal decimal(15,2), c_mktsegment string, c_comment string",
    "part": "p_partkey bigint, p_name string, p_mfgr string, p_brand string, p_type string, p_size int, "
    "p_container string, p_retailprice decimal(15,2), p_comment string",
    "partsupp": "ps_partkey bigint, ps_suppkey bigint, ps_availqty int, ps_supplycost decimal(15,2), ps_comment string",
    "supplier": "s_suppkey bigint, s_name string, s_address string, s_nationkey bigint, s_phone string, "
    "s_acctbal decimal(15,2), s_comment string",
    "nation": "n_nationkey bigint, n_name string, n_regionkey bigint, n_comment string",
    "region": "r_regionkey bigint, r_name string, r_comment string",
}

# Small tables for the negative / contrast cases.
SMALL = {
    "na": "a int, x int",
    "nb": "b int, y int",
    "nc": "c int, z int",
    "ns": "s string, w int",
}

# (name, why, sql). Observed plans are recorded; nothing here is an expectation.
NEGATIVE = [
    (
        "left_outer_filter_on_null_supplying_side",
        "b = 5 below the right side of LEFT OUTER ON a = b must not become a = 5 on the preserved side",
        "SELECT * FROM na LEFT OUTER JOIN (SELECT * FROM nb WHERE b = 5) q ON a = b",
    ),
    (
        "left_outer_where_on_preserved_side",
        "a = 5 on the preserved side moves to the null-supplying side as b = 5",
        "SELECT * FROM na LEFT OUTER JOIN nb ON a = b WHERE a = 5",
    ),
    (
        "left_outer_on_clause_literal_right",
        "b = 5 in the ON clause of a LEFT OUTER (right-only condition) stays on the right",
        "SELECT * FROM na LEFT OUTER JOIN nb ON a = b AND b = 5",
    ),
    (
        "left_outer_on_clause_literal_left",
        "a = 5 in the ON clause of a LEFT OUTER (left-only condition) must not filter the preserved side",
        "SELECT * FROM na LEFT OUTER JOIN nb ON a = b AND a = 5",
    ),
    (
        "right_outer_filter_on_null_supplying_side",
        "mirror: a = 5 below the left side of RIGHT OUTER ON a = b must not become b = 5 on the preserved side",
        "SELECT * FROM (SELECT * FROM na WHERE a = 5) q RIGHT OUTER JOIN nb ON a = b",
    ),
    (
        "right_outer_where_on_preserved_side",
        "b = 5 on the preserved (right) side moves to the left as a = 5",
        "SELECT * FROM na RIGHT OUTER JOIN nb ON a = b WHERE b = 5",
    ),
    (
        "full_outer",
        "FULL OUTER infers nothing on either side",
        "SELECT * FROM na FULL OUTER JOIN nb ON a = b",
    ),
    (
        "full_outer_with_side_filter",
        "FULL OUTER with filters inside the inputs: nothing is moved across",
        "SELECT * FROM (SELECT * FROM na WHERE a = 5) q FULL OUTER JOIN (SELECT * FROM nb WHERE y > 1) r ON a = b",
    ),
    (
        "cast_equality_inner",
        "cast(a as string) = s: isnotnull on both attributes (null-intolerant cast), literal not moved",
        "SELECT * FROM na JOIN ns ON CAST(a AS STRING) = s",
    ),
    (
        "cast_equality_literal_on_attr",
        "cast(a as string) = s with a = 5: whether s = '5' (a cast form) is inferred",
        "SELECT * FROM na JOIN ns ON CAST(a AS STRING) = s WHERE a = 5",
    ),
    (
        "cast_equality_literal_on_string",
        "cast(a as string) = s with s = '5': whether a = 5 is inferred",
        "SELECT * FROM na JOIN ns ON CAST(a AS STRING) = s WHERE s = '5'",
    ),
    (
        "null_safe_key_inner",
        "<=> keys: no isnotnull and nothing moved",
        "SELECT * FROM na JOIN nb ON a <=> b",
    ),
    (
        "null_safe_key_with_literal",
        "<=> keys with a > 5: only isnotnull(a) / a > 5, nothing for b",
        "SELECT * FROM na JOIN nb ON a <=> b WHERE a > 5",
    ),
    (
        "inner_equality_key",
        "positive control: isnotnull on both keys",
        "SELECT * FROM na JOIN nb ON a = b",
    ),
    (
        "inner_literal_through_equality",
        "positive control: a = 5 moves to b = 5",
        "SELECT * FROM na JOIN nb ON a = b WHERE a = 5",
    ),
    (
        "inner_in_list_through_equality",
        "positive control: a IN (1,2,3) moves to b IN (1,2,3)",
        "SELECT * FROM na JOIN nb ON a = b WHERE a IN (1, 2, 3)",
    ),
    (
        "inner_range_through_equality",
        "positive control: a > 5 AND a < 10 moves to b",
        "SELECT * FROM na JOIN nb ON a = b WHERE a > 5 AND a < 10",
    ),
    (
        "inner_transitive_chain",
        "equivalence class a = b = c: a = 5 reaches b and c",
        "SELECT * FROM na JOIN nb ON a = b JOIN nc ON b = c WHERE a = 5",
    ),
    (
        "inner_substitution_into_expression",
        "category (d): a + 1 > 5 becomes b + 1 > 5",
        "SELECT * FROM na JOIN nb ON a = b WHERE a + 1 > 5",
    ),
    (
        "inner_non_deterministic",
        "non-deterministic predicates are never moved",
        "SELECT * FROM na JOIN nb ON a = b WHERE a > rand() * 10",
    ),
    (
        "inner_or_predicate",
        "an OR over the key: moved through the equivalence as a whole",
        "SELECT * FROM na JOIN nb ON a = b WHERE a = 1 OR a = 2",
    ),
    (
        "inner_is_null_filter",
        "a IS NULL with a = b: contradiction with isnotnull(a)",
        "SELECT * FROM na JOIN nb ON a = b WHERE a IS NULL",
    ),
    (
        "inner_non_equi_condition",
        "non-equi join condition a < b: isnotnull on both, no equivalence",
        "SELECT * FROM na JOIN nb ON a < b",
    ),
    (
        "inner_expression_key",
        "key is an expression a + 1 = b: isnotnull(a), isnotnull(b)",
        "SELECT * FROM na JOIN nb ON a + 1 = b WHERE b = 5",
    ),
    (
        "left_semi",
        "LEFT SEMI infers for both sides like an inner join",
        "SELECT * FROM na LEFT SEMI JOIN nb ON a = b WHERE a = 5",
    ),
    (
        "left_semi_right_literal",
        "LEFT SEMI with the literal on the right side",
        "SELECT * FROM na LEFT SEMI JOIN (SELECT * FROM nb WHERE b = 5) q ON a = b",
    ),
    (
        "left_anti",
        "LEFT ANTI infers on the right side only",
        "SELECT * FROM na LEFT ANTI JOIN nb ON a = b WHERE a = 5",
    ),
    (
        "left_anti_right_literal",
        "LEFT ANTI with the literal on the right side must not reach the left",
        "SELECT * FROM na LEFT ANTI JOIN (SELECT * FROM nb WHERE b = 5) q ON a = b",
    ),
    (
        "exists_subquery",
        "EXISTS rewrites to a left semi join",
        "SELECT * FROM na WHERE EXISTS (SELECT 1 FROM nb WHERE b = a) AND a = 5",
    ),
    (
        "not_exists_subquery",
        "NOT EXISTS rewrites to a left anti join",
        "SELECT * FROM na WHERE NOT EXISTS (SELECT 1 FROM nb WHERE b = a) AND a = 5",
    ),
    (
        "in_subquery",
        "IN (subquery) rewrites to a left semi join",
        "SELECT * FROM na WHERE a IN (SELECT b FROM nb) AND a = 5",
    ),
    (
        "cross_join_with_where_equality",
        "comma join with the equality in WHERE becomes an inner join",
        "SELECT * FROM na, nb WHERE a = b AND a = 5",
    ),
    (
        "cross_join_no_condition",
        "cross join: no join condition, so nothing is inferred across",
        "SELECT * FROM (SELECT * FROM na WHERE a = 5) q CROSS JOIN nb",
    ),
]

GOLDEN_FILES = [
    "inner-join.sql",
    "outer-join.sql",
    "cross-join.sql",
    "natural-join.sql",
    "using-join.sql",
    "join-empty-relation.sql",
    "join-resolution-edge-cases.sql",
    "join-lateral.sql",
    "postgreSQL/join.sql",
    "udf/udf-outer-join.sql",
    "udf/udf-natural-join.sql",
]

CMP = {
    "EqualTo",
    "EqualNullSafe",
    "GreaterThan",
    "GreaterThanOrEqual",
    "LessThan",
    "LessThanOrEqual",
}
SELECT_START = re.compile(r"^\s*\(*\s*(select|with|from|values|table)\b", re.I)
ID_RE = re.compile(r"#\d+L?")


def strip_ids(s: str) -> str:
    return ID_RE.sub("", str(s))


def split_statements(text: str) -> list[str]:
    """Split a SQLQueryTestSuite input file on ';' outside quotes; drop `--` comment lines."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("--")]
    out, cur, quote, i = [], [], None, 0
    src = "\n".join(lines)
    while i < len(src):
        ch = src[i]
        if quote:
            cur.append(ch)
            if ch == "\\" and i + 1 < len(src):
                cur.append(src[i + 1])
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
            cur.append(ch)
        elif ch == ";":
            stmt = "".join(cur).strip()
            if stmt:
                out.append(stmt)
            cur = []
        else:
            cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


class Jvm:
    def __init__(self, spark):
        self.spark = spark
        self.jvm = spark._jvm
        self.conv = self.jvm.scala.jdk.javaapi.CollectionConverters

    def seq(self, scala_seq):
        return list(self.conv.asJava(scala_seq))

    @staticmethod
    def cls(o) -> str:
        return str(o.getClass().getSimpleName())

    def kids(self, e):
        return self.seq(e.children())

    def conjuncts(self, e):
        if self.cls(e) == "And":
            return self.conjuncts(e.left()) + self.conjuncts(e.right())
        return [e]

    def classify_inferred(self, e) -> str:
        """(b) isnotnull / (c) equivalence / (d) substitution for an inferred conjunct."""
        c = self.cls(e)
        if c == "IsNotNull":
            return "isnotnull"
        inner = e
        while self.cls(inner) == "Not":
            inner = self.kids(inner)[0]
        c = self.cls(inner)
        ks = self.kids(inner)
        if c in CMP and len(ks) == 2:
            kinds = sorted(self.cls(k) for k in ks)
            if kinds == ["AttributeReference", "Literal"]:
                return "equivalence"
        if (
            c == "In"
            and ks
            and self.cls(ks[0]) == "AttributeReference"
            and all(self.cls(k) == "Literal" for k in ks[1:])
        ):
            return "equivalence"
        if c == "InSet":
            ks = self.kids(inner)
            if len(ks) == 1 and self.cls(ks[0]) == "AttributeReference":
                return "equivalence"
        return "substitution"

    def scans(self, plan):
        """List of (label, [conjunct expr]) for every leaf, in traversal order (incl. subqueries)."""
        out: list = []

        def label(leaf) -> str:
            s = strip_ids(leaf.simpleString(200))
            m = re.match(r"^(Relation\s+\S+?)\[", s)
            if m:
                return m.group(1)
            first = s.split("[", 1)[1].split(",")[0].rstrip("]") if "[" in s else ""
            return f"{s.split(' ')[0]}[{first},..]"

        def walk(node):
            ch = self.seq(node.children())
            for c in ch:
                if not self.seq(c.children()):
                    conds = []
                    if self.cls(node) == "Filter":
                        conds = self.conjuncts(node.condition())
                    out.append((label(c), conds))
            for c in ch:
                if self.seq(c.children()):
                    walk(c)
            for sub in self.seq(node.subqueries()):
                if not self.seq(sub.children()):
                    out.append((label(sub), []))
                walk(sub)

        if not self.seq(plan.children()):
            out.append((label(plan), []))
        walk(plan)
        return out

    def joins(self, plan):
        out = []

        def walk(node):
            if self.cls(node) == "Join":
                cond = node.condition()
                out.append(
                    {
                        "type": str(node.joinType().sql()),
                        "condition": strip_ids(cond.get().toString()) if cond.isDefined() else None,
                    }
                )
            for c in self.seq(node.children()):
                walk(c)
            for sub in self.seq(node.subqueries()):
                walk(sub)

        walk(plan)
        return out


def optimized(spark, sql: str, propagate: bool):
    if propagate:
        spark.conf.unset(EXCLUDED)
    else:
        spark.conf.set(EXCLUDED, RULE)
    df = spark.sql(sql)
    return df._jdf.queryExecution().optimizedPlan()


def analyze_query(spark, jv: Jvm, sql: str) -> dict:
    """Plans with propagation on and off; the four categories per scan."""
    try:
        plan_on = optimized(spark, sql, True)
        plan_off = optimized(spark, sql, False)
        on, off = jv.scans(plan_on), jv.scans(plan_off)
        res = {
            "joins": jv.joins(plan_on),
            "optimized_plan": strip_ids(plan_on.treeString()).rstrip("\n").splitlines(),
            "scans": {},
        }
    finally:
        spark.conf.unset(EXCLUDED)
    occ: dict[str, int] = {}
    occ_off: dict[str, list] = {}
    for lbl, conds in off:
        occ_off.setdefault(lbl, []).append({strip_ids(c.toString()) for c in conds})
    problems = []
    for lbl, conds in on:
        n = occ.get(lbl, 0)
        occ[lbl] = n + 1
        key = f"{lbl}#{n + 1}"
        written_set = occ_off.get(lbl, [])
        written_set = written_set[n] if n < len(written_set) else None
        entry = {"written": [], "isnotnull": [], "equivalence": [], "substitution": []}
        if written_set is None:
            problems.append(f"{key}: scan only present with the rule enabled")
            written_set = set()
        seen = set()
        for c in conds:
            text = strip_ids(c.toString())
            seen.add(text)
            if text in written_set:
                entry["written"].append(text)
            else:
                entry[jv.classify_inferred(c)].append(text)
        for w in sorted(written_set - seen):
            problems.append(f"{key}: written predicate dropped by the rule: {w}")
        for k in entry:
            entry[k].sort()
        res["scans"][key] = entry
    if problems:
        res["problems"] = problems
    return res


def run_golden_file(spark, jv: Jvm, path: Path, tag: str, idx: int) -> dict:
    """Execute a SQLQueryTestSuite input; plan every SELECT. INSERT statements are skipped (plans need no data)."""
    for t in list(spark.catalog.listTables()):
        if t.isTemporary:
            spark.catalog.dropTempView(t.name)
    spark.sql("RESET")  # also clears excludedRules
    db = f"g{idx}"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
    spark.sql(f"USE {db}")
    queries, skipped = [], []
    for n, stmt in enumerate(split_statements(path.read_text()), 1):
        head = stmt.lstrip()[:60].replace("\n", " ")
        if re.match(r"^\s*insert\b", stmt, re.I):
            skipped.append({"n": n, "reason": "insert skipped", "head": head})
        elif SELECT_START.match(stmt):
            try:
                r = analyze_query(spark, jv, stmt)
                r.update(n=n, sql=stmt)
                queries.append(r)
            except Exception as e:
                skipped.append({"n": n, "reason": error_head(e), "head": head})
        else:
            try:
                spark.sql(stmt)
            except Exception as e:
                skipped.append({"n": n, "reason": error_head(e), "head": head})
    spark.sql("USE default")
    return {"file": f"{tag}/{path.name}", "queries": queries, "skipped": skipped}


def error_head(e: Exception) -> str:
    m = re.search(r"\[([A-Z_][A-Z0-9_.]+)\]", str(e))
    return m.group(1) if m else type(e).__name__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="engine repo root")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--parts",
        default="tpch,tpch_nn,negative,golden",
        help="comma list of: tpch, tpch_nn (the TPC-H queries over non-nullable relations), negative, "
        "golden (all golden files) or golden:<i> (index into GOLDEN_FILES); partial outputs are merged "
        "with --merge",
    )
    ap.add_argument("--merge", nargs="*", help="merge these partial fixtures into --out and exit")
    args = ap.parse_args()
    root = Path(args.root)
    if args.merge:
        merged = None
        for f in args.merge:
            part = json.loads(Path(f).read_text())
            if merged is None:
                merged = part
            else:
                merged["tpch"].update(part["tpch"])
                merged["tpch_nonnull"].update(part["tpch_nonnull"])
                merged["negative"].update(part["negative"])
                merged["golden_join_files"].extend(part["golden_join_files"])
        if merged is None:
            raise SystemExit("--merge needs at least one fixture")
        merged["golden_join_files"].sort(
            key=lambda g: [Path(f).name for f in GOLDEN_FILES].index(g["file"].rsplit("/", 1)[1])
        )
        Path(args.out).write_text(json.dumps(merged, indent=1) + "\n")
        return
    parts = set(args.parts.split(","))

    from pyspark.sql import SparkSession

    wh = tempfile.mkdtemp(prefix="inferred-join-filters-")
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("gen_inferred_join_filters")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.sql.warehouse.dir", wh)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    jv = Jvm(spark)
    for name, cols in {**TPCH, **SMALL}.items():
        spark.sql(f"CREATE TABLE {name} ({cols}) USING parquet")

    fixture = {
        "format_version": FORMAT_VERSION,
        "spark": spark.version,
        "categories": {
            "written": "(a) conjuncts present with InferFiltersFromConstraints excluded "
            "(spark.sql.optimizer.excludedRules)",
            "isnotnull": "(b) inferred isnotnull(attribute)",
            "equivalence": "(c) inferred attr <cmp> literal, attr IN (literals), NOT of those",
            "substitution": "(d) any other inferred predicate (not inferred by Krishiv)",
        },
        "ansi_enabled": spark.conf.get("spark.sql.ansi.enabled"),
        "tpch": {},
        "tpch_nonnull": {},
        "negative": {},
        "golden_join_files": [],
    }
    for i in range(1, 23) if "tpch" in parts else []:
        sql = (root / f"bench/queries/tpch/q{i:02d}.sql").read_text().strip().rstrip(";")
        fixture["tpch"][f"q{i:02d}"] = analyze_query(spark, jv, sql)
    if "tpch_nn" in parts:
        # Same queries over empty NON-nullable relations (shadowing temp views over an RDD, so the
        # optimizer cannot fold them like a LocalRelation): shows which inferred isnotnull exist only
        # because Spark's file sources report every column nullable.
        from pyspark.sql.types import StructField, StructType

        for name, cols in TPCH.items():
            fields = StructType.fromDDL(cols).fields
            nn = StructType([StructField(f.name, f.dataType, False) for f in fields])
            spark.createDataFrame(spark.sparkContext.emptyRDD(), nn).createOrReplaceTempView(name)
        for i in range(1, 23):
            sql = (root / f"bench/queries/tpch/q{i:02d}.sql").read_text().strip().rstrip(";")
            fixture["tpch_nonnull"][f"q{i:02d}"] = analyze_query(spark, jv, sql)
        for name in TPCH:
            spark.catalog.dropTempView(name)
    for name, why, sql in NEGATIVE if "negative" in parts else []:
        r = analyze_query(spark, jv, sql)
        r.update(why=why, sql=sql)
        fixture["negative"][name] = r
    base = root / "tests/spark-golden/v4.2.0/inputs"
    for i, rel in enumerate(GOLDEN_FILES):
        p = base / rel
        if ("golden" in parts or f"golden:{i}" in parts) and p.exists():
            fixture["golden_join_files"].append(run_golden_file(spark, jv, p, "v4.2.0/inputs", i))
    spark.stop()
    Path(args.out).write_text(json.dumps(fixture, indent=1, sort_keys=False) + "\n")


if __name__ == "__main__":
    main()
