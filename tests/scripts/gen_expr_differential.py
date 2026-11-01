#!/usr/bin/env python3
"""Expression-level differential fixture against Spark 4.2 (P2-T101).

Generates seeded random scalar expressions (depth <= 4) over the typed table `t` below, runs them on a
REAL Spark in local mode (never from memory) and writes the fixture `crates/krishiv-testkit` replays
(`tests/expr_differential.rs`, `src/expr_diff.rs`). Every sub-expression of a root is a case of its
own, so a mismatching root is narrowed to its smallest mismatching sub-expression without running
Spark again. Usage (the JDK path is the machine's; Spark needs Java 17 or 21):

    env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' \
        python -I tests/scripts/gen_expr_differential.py --roots 300 --seed 20261101 \
        > tests/fixtures/spark-4.2.0/expr-differential.json

`--roots N` is the budget (10,000 nightly), `--seed` makes the run repeatable. Spark logs go to stderr.
Each case holds the column's Spark type, the rows as Spark's `HiveResult` prints them, or the Spark
error class; `ansi` says which `spark.sql.ansi.enabled` the case ran under.
"""

import argparse
import json
import random
import sys

# The table every expression reads: edge values of each type, NULLs, one row per edge. Strings avoid
# tab and newline, which `HiveResult` rows are split on.
INTS = [0, 1, -1, 7, None, 2147483647, -2147483648, 3, 100, -5, 42, None]
LONGS = [0, 1, -1, 9223372036854775807, None, -9223372036854775808, 5, -300, 2147483648, 12, None, 77]
DOUBLES = [0.0, -0.0, 1.5, -2.25, None, float("nan"), float("inf"), float("-inf"), 1e308, 3.14159, 0.1, 100.0]
DECIMALS = [
    "0.00",
    "1.50",
    "-2.25",
    None,
    "99999999.99",
    "-99999999.99",
    "0.01",
    "10.00",
    "3.33",
    None,
    "7.07",
    "-0.50",
]
STRINGS = ["", "a", "Abc", None, "hello world", " pad ", "ß", "日本", "12", "-3.5", "AbCdEf", "x,y,z"]
BOOLS = [True, False, None, True, False, None, True, True, False, False, None, True]
DATES = [
    "1970-01-01",
    "2000-02-29",
    "1999-12-31",
    None,
    "2024-03-31",
    "0001-01-01",
    "9999-12-31",
    "2023-01-31",
    "1900-03-01",
    "2021-07-04",
    None,
    "2038-01-19",
]

COLS = {
    "int": "i",
    "bigint": "l",
    "double": "d",
    "decimal": "m",
    "string": "s",
    "boolean": "b",
    "date": "dt",
}
NUMERIC = ["int", "bigint", "decimal", "double"]
INTEGRAL = ["int", "bigint"]
RANK = {"int": 0, "bigint": 1, "decimal": 2, "double": 3}
SQL_TYPE = {
    "int": "INT",
    "bigint": "BIGINT",
    "double": "DOUBLE",
    "decimal": "DECIMAL(10,2)",
    "string": "STRING",
    "boolean": "BOOLEAN",
    "date": "DATE",
}
LITERALS = {
    "int": ["0", "1", "-1", "2", "7", "2147483647", "-2147483648", "100"],
    "bigint": ["0L", "1L", "-1L", "9223372036854775807L", "-9223372036854775808L", "3000000000L"],
    "double": ["0.0D", "1.5D", "-2.5D", "1e308D", "CAST('NaN' AS DOUBLE)", "0.1D", "1e-300D"],
    "decimal": ["0.00BD", "1.50BD", "-2.25BD", "99999999.99BD", "0.01BD", "123456789.12BD"],
    "string": ["''", "'a'", "'Abc'", "'hello world'", "' pad '", "'12'", "'-3.5'", "'x,y'", "'é'"],
    "boolean": ["true", "false"],
    "date": ["DATE'1970-01-01'", "DATE'2000-02-29'", "DATE'9999-12-31'", "DATE'2024-03-31'"],
}


def lit(v, kind):
    if v is None:
        return "NULL"
    if kind == "boolean":
        return "true" if v else "false"
    if kind in ("string", "date"):
        return "'" + v.replace("'", "''") + "'" if kind == "string" else f"DATE'{v}'"
    if kind == "double":
        if v != v:
            return "CAST('NaN' AS DOUBLE)"
        if v in (float("inf"), float("-inf")):
            return f"CAST('{'-' if v < 0 else ''}Infinity' AS DOUBLE)"
        return f"{v!r}D"
    if kind == "decimal":
        return f"{v}BD"
    return str(v) + ("L" if kind == "bigint" else "")


def setup_sql():
    cols = [
        ("id", "int"),
        ("i", "int"),
        ("l", "bigint"),
        ("d", "double"),
        ("m", "decimal"),
        ("s", "string"),
        ("b", "boolean"),
        ("dt", "date"),
    ]
    data = [list(range(12)), INTS, LONGS, DOUBLES, DECIMALS, STRINGS, BOOLS, DATES]
    rows = []
    for r in range(12):
        cells = []
        for (_, kind), col in zip(cols, data, strict=True):
            v = col[r]
            cells.append(
                "CAST(NULL AS " + SQL_TYPE.get(kind, "INT") + ")"
                if v is None
                else f"CAST({lit(v, kind)} AS {SQL_TYPE.get(kind, 'INT')})"
            )
        rows.append("(" + ", ".join(cells) + ")")
    names = ", ".join(n for n, _ in cols)
    return f"CREATE OR REPLACE TEMPORARY VIEW t AS SELECT * FROM VALUES {', '.join(rows)} AS t({names})"


def common(a, b):
    """The numeric family both operands coerce to (decimal with double is double)."""
    return a if RANK[a] >= RANK[b] else b


class Node:
    __slots__ = ("ansi", "kids", "kind", "sql")

    def __init__(self, sql, kind, kids=()):
        self.sql, self.kind, self.kids = sql, kind, list(kids)


# name -> (argument kinds, result kind or callable on the argument kinds, template). An argument kind
# of "N" is any numeric, "I" integral; a template is filled with the argument SQL.
def result_div(a, b):
    if "double" in (a, b):
        return "double"
    return "decimal" if "decimal" in (a, b) else "double"


FUNCS = []


def f(args, ret, template):
    FUNCS.append((args, ret, template))


for op in ("+", "-", "*", "%"):
    f(("N", "N"), lambda a, b: common(a, b), f"({{0}} {op} {{1}})")
f(("N", "N"), result_div, "({0} / {1})")
f(("ID", "ID"), lambda a, b: "bigint", "({0} DIV {1})")
f(("N", "N"), lambda a, b: common(a, b), "pmod({0}, {1})")
f(("N",), lambda a: a, "abs({0})")
f(("N",), lambda a: a, "(- {0})")
f(("N",), lambda a: "decimal" if a == "decimal" else "bigint", "floor({0})")
f(("N",), lambda a: "decimal" if a == "decimal" else "bigint", "ceil({0})")
f(("N",), lambda a: "double", "sqrt({0})")
f(("N",), lambda a: "double", "ln({0})")
f(("N",), lambda a: "double", "exp({0})")
f(("N", "N"), lambda a, b: "double", "power({0}, {1})")
f(("N",), lambda a: "double", "signum({0})")
f(("N",), lambda a: a, "round({0}, 1)")
f(("N",), lambda a: a, "round({0}, 0)")
f(("N",), lambda a: a, "round({0}, -1)")
f(("I", "I"), lambda a, b: common(a, b), "({0} & {1})")
f(("I", "I"), lambda a, b: common(a, b), "({0} | {1})")
f(("I", "I"), lambda a, b: common(a, b), "({0} ^ {1})")
f(("I",), lambda a: a, "(~ {0})")
f(("N", "N"), lambda a, b: common(a, b), "greatest({0}, {1})")
f(("N", "N"), lambda a, b: common(a, b), "least({0}, {1})")
for op in ("=", "<>", "<", "<=", ">", ">=", "<=>"):
    f(("N", "N"), lambda a, b: "boolean", f"({{0}} {op} {{1}})")
    f(("string", "string"), lambda: "boolean", f"({{0}} {op} {{1}})")
    f(("date", "date"), lambda: "boolean", f"({{0}} {op} {{1}})")
f(("N", "N", "N"), lambda a, b, c: "boolean", "({0} BETWEEN {1} AND {2})")
f(("N",), lambda a: "boolean", "({0} IN (0, 1, 7))")
f(("string",), lambda: "boolean", "({0} IN ('a', 'Abc', ''))")
f(("double",), lambda: "boolean", "isnan({0})")
f(("boolean", "boolean"), lambda: "boolean", "({0} AND {1})")
f(("boolean", "boolean"), lambda: "boolean", "({0} OR {1})")
f(("boolean",), lambda: "boolean", "(NOT {0})")
f(("any",), lambda a: "boolean", "({0} IS NULL)")
f(("any",), lambda a: "boolean", "({0} IS NOT NULL)")
f(("boolean", "same", "same"), lambda c, a, b: a, "if({0}, {1}, {2})")
f(("same", "same"), lambda a, b: a, "coalesce({0}, {1})")
f(("same", "same"), lambda a, b: a, "nullif({0}, {1})")
f(("boolean", "same", "same"), lambda c, a, b: a, "(CASE WHEN {0} THEN {1} ELSE {2} END)")
f(("string",), lambda: "int", "length({0})")
f(("string",), lambda: "string", "upper({0})")
f(("string",), lambda: "string", "lower({0})")
f(("string",), lambda: "string", "trim({0})")
f(("string",), lambda: "string", "ltrim({0})")
f(("string",), lambda: "string", "reverse({0})")
f(("string",), lambda: "string", "initcap({0})")
f(("string",), lambda: "int", "ascii({0})")
f(("string",), lambda: "string", "substring({0}, 2, 3)")
f(("string",), lambda: "string", "substring({0}, -2)")
f(("string",), lambda: "string", "left({0}, 2)")
f(("string",), lambda: "string", "repeat({0}, 2)")
f(("string",), lambda: "string", "lpad({0}, 6, '*')")
f(("string", "string"), lambda: "string", "concat({0}, {1})")
f(("string", "string"), lambda: "string", "concat_ws(',', {0}, {1})")
f(("string", "string", "string"), lambda: "string", "replace({0}, {1}, {2})")
f(("string", "string"), lambda: "int", "instr({0}, {1})")
f(("string", "string"), lambda: "boolean", "contains({0}, {1})")
f(("string", "string"), lambda: "boolean", "startswith({0}, {1})")
f(("string",), lambda: "boolean", "({0} LIKE '%b%')")
f(("date", "int"), lambda: "date", "date_add({0}, {1})")
f(("date", "int"), lambda: "date", "date_sub({0}, {1})")
f(("date", "int"), lambda: "date", "add_months({0}, {1})")
f(("date", "date"), lambda: "int", "datediff({0}, {1})")
f(("date",), lambda: "int", "year({0})")
f(("date",), lambda: "int", "month({0})")
f(("date",), lambda: "int", "dayofweek({0})")
f(("date",), lambda: "int", "quarter({0})")
f(("date",), lambda: "date", "last_day({0})")
f(("any",), lambda a: "string", "CAST({0} AS STRING)")
for k in ("int", "bigint", "double", "decimal", "boolean", "date"):
    f(("any",), lambda a, k=k: k, "CAST({0} AS " + SQL_TYPE[k] + ")")
    f(("any",), lambda a, k=k: k, "try_cast({0} AS " + SQL_TYPE[k] + ")")


class Gen:
    def __init__(self, rng):
        self.rng = rng

    def leaf(self, kind):
        if self.rng.random() < 0.6:
            return Node(COLS[kind], kind)
        return Node(self.rng.choice(LITERALS[kind]), kind)

    def pick_kind(self, spec, same):
        if spec == "N":
            return self.rng.choice(NUMERIC)
        if spec == "I":
            return self.rng.choice(INTEGRAL)
        if spec == "ID":
            return self.rng.choice(["int", "bigint", "decimal"])
        if spec == "same":
            return same
        if spec == "any":
            return self.rng.choice(list(COLS))
        return spec

    def expr(self, depth, kind=None):
        """An expression of `kind` (any when None) of depth at most `depth`."""
        if depth == 0 or self.rng.random() < 0.15:
            return self.leaf(kind or self.rng.choice(list(COLS)))
        for _ in range(30):
            args, ret, template = self.rng.choice(FUNCS)
            same = kind or self.rng.choice(list(COLS))
            kinds = [self.pick_kind(spec, same) for spec in args]
            out = ret(*kinds) if ret.__code__.co_argcount else ret()
            if "same" in args:
                out = same
            if kind is not None and out != kind:
                continue
            kids = [self.expr(depth - 1, k) for k in kinds]
            return Node(template.format(*[k.sql for k in kids]), out, kids)
        return self.leaf(kind or self.rng.choice(list(COLS)))


def walk(node, seen, order):
    """Post-order, de-duplicated; returns the node's case key."""
    key = node.sql
    kid_keys = [walk(k, seen, order) for k in node.kids]
    if key not in seen:
        seen[key] = {"sql": node.sql, "kids": kid_keys}
        order.append(key)
    return key


def error_class(e):
    try:
        cls = e.getCondition()
        if cls:
            return cls
    except Exception:
        pass
    return type(e).__name__


def run_batch(spark, hive, exprs):
    """{sql: (schema, rows) or ("error", class)} for the expressions, bisecting on failure."""
    out = {}
    if not exprs:
        return out
    cols = ", ".join(f"{e} AS c{n}" for n, e in enumerate(exprs))
    try:
        df = spark.sql(f"SELECT {cols} FROM t ORDER BY id")
        seq = hive(df._jdf.queryExecution().executedPlan())
        lines = [seq.apply(n) for n in range(seq.size())]
        fields = df.schema.fields
        rows = [ln.split("\t") for ln in lines]
        if any(len(r) != len(exprs) for r in rows):
            raise ValueError("cell count")
        for n, e in enumerate(exprs):
            out[e] = {"schema": fields[n].dataType.simpleString(), "rows": [r[n] for r in rows]}
        return out
    except Exception as e:
        if len(exprs) == 1:
            out[exprs[0]] = {"error": error_class(e)}
            return out
    mid = len(exprs) // 2
    out.update(run_batch(spark, hive, exprs[:mid]))
    out.update(run_batch(spark, hive, exprs[mid:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("expr-differential")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    hive = spark._jvm.org.apache.spark.sql.execution.HiveResult.hiveResultString
    spark.sql(setup_sql())
    rng = random.Random(args.seed)
    gen = Gen(rng)
    groups = {True: ([], {}), False: ([], {})}
    roots = []
    for n in range(args.roots):
        ansi = n % 5 != 4
        order, seen = groups[ansi]
        node = gen.expr(rng.randint(1, args.depth))
        roots.append((ansi, walk(node, seen, order)))
    cases = []
    for ansi in (True, False):
        spark.conf.set("spark.sql.ansi.enabled", "true" if ansi else "false")
        order, seen = groups[ansi]
        results = {}
        for lo in range(0, len(order), args.batch):
            results.update(run_batch(spark, hive, order[lo : lo + args.batch]))
        root_keys = {k for a, k in roots if a == ansi}
        for key in order:
            case = {"sql": seen[key]["sql"], "ansi": ansi, "kids": seen[key]["kids"], "root": key in root_keys}
            case.update(results[key])
            cases.append(case)
    json.dump({"spark": spark.version, "seed": args.seed, "setup": setup_sql(), "cases": cases}, sys.stdout, indent=0)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
