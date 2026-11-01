#!/usr/bin/env python3
"""Seeded random queries over the TPC-H schema, answered by DuckDB (P2-T110a).

SQLancer-style generator: filter/projection, grouped aggregates with HAVING, 2-4 table joins along the
foreign keys (inner and left), EXISTS / NOT EXISTS / IN / scalar subqueries, DISTINCT, UNION ALL and CTEs,
with predicates drawn from values in the data and NULLs introduced by `NULLIF`, `CASE` and left joins. Each
query's rows are DuckDB's answer, compared by Krishiv as a multiset (`query_oracle.rs`). Usage:

    uv run --project bench python -I tests/scripts/gen_query_oracle.py \
        --data crates/krishiv-exec/tests/fixtures/tpch-sf0.01-parquet --count 120 --seed 20261101 \
        > tests/fixtures/query-oracle/queries.json

Aggregates avoid `avg` over decimals (Spark rounds it to the scale plus 4, DuckDB returns a double) and the
queries use no `/`, so both engines' answers are the same numbers. Cells are text: `NULL`, integers and
decimals as written, dates ISO.
"""

import argparse
import datetime
import decimal
import json
import os
import random
import sys

import duckdb

TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]
# (table, column, table, column): the TPC-H foreign keys.
FKS = [
    ("orders", "o_custkey", "customer", "c_custkey"),
    ("lineitem", "l_orderkey", "orders", "o_orderkey"),
    ("lineitem", "l_partkey", "part", "p_partkey"),
    ("lineitem", "l_suppkey", "supplier", "s_suppkey"),
    ("partsupp", "ps_partkey", "part", "p_partkey"),
    ("partsupp", "ps_suppkey", "supplier", "s_suppkey"),
    ("customer", "c_nationkey", "nation", "n_nationkey"),
    ("supplier", "s_nationkey", "nation", "n_nationkey"),
    ("nation", "n_regionkey", "region", "r_regionkey"),
]
MAX_ROWS = 300


def kind_of(duck_type):
    t = duck_type.upper()
    if t.startswith("DECIMAL"):
        return "dec"
    if t in ("BIGINT", "INTEGER", "HUGEINT", "SMALLINT"):
        return "int"
    return "date" if t == "DATE" else "str"


class Schema:
    def __init__(self, con):
        self.cols = {}
        self.sample = {}
        for t in TABLES:
            self.cols[t] = [(r[0], kind_of(r[1])) for r in con.execute(f"DESCRIBE {t}").fetchall()]
            for c, _ in self.cols[t]:
                vals = con.execute(f"SELECT DISTINCT {c} FROM {t} USING SAMPLE 40 ROWS").fetchall()
                self.sample[(t, c)] = [v[0] for v in vals if v[0] is not None and len(str(v[0])) <= 24] or [None]


def lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, datetime.date):
        return f"DATE '{v.isoformat()}'"
    return str(v)


class Gen:
    def __init__(self, schema, rng):
        self.s, self.rng = schema, rng

    def col(self, alias_of, kinds=None):
        """A (qualified name, kind, table, column) of one of the aliased tables."""
        choices = [(f"{a}.{c}", k, t, c) for a, t in alias_of for c, k in self.s.cols[t] if kinds is None or k in kinds]
        return self.rng.choice(choices)

    def pred(self, alias_of, depth=2):
        r = self.rng
        if depth > 0 and r.random() < 0.4:
            op = r.choice(["AND", "OR"])
            return f"({self.pred(alias_of, depth - 1)} {op} {self.pred(alias_of, depth - 1)})"
        if depth > 0 and r.random() < 0.1:
            return f"(NOT {self.pred(alias_of, depth - 1)})"
        name, kind, t, c = self.col(alias_of)
        v = r.choice(self.s.sample[(t, c)])
        form = r.random()
        if v is None or form < 0.08:
            return f"{name} IS {r.choice(['', 'NOT '])}NULL"
        if kind == "str" and form < 0.35:
            return f"{name} LIKE {lit(str(v)[: r.randint(1, 3)] + '%')}"
        if form < 0.5:
            w = r.choice(self.s.sample[(t, c)])
            lo, hi = sorted([v, w], key=lambda x: (x is None, x))
            return f"{name} BETWEEN {lit(lo)} AND {lit(hi)}"
        if form < 0.65:
            vals = r.sample(self.s.sample[(t, c)], min(3, len(self.s.sample[(t, c)])))
            return f"{name} IN ({', '.join(lit(x) for x in vals)})"
        if form < 0.75 and kind in ("int", "dec"):
            return f"NULLIF({name}, {lit(v)}) {r.choice(['=', '<>', '>'])} {lit(r.choice(self.s.sample[(t, c)]))}"
        return f"{name} {r.choice(['=', '<>', '<', '<=', '>', '>='])} {lit(v)}"

    def proj(self, alias_of, n):
        out = []
        for i in range(n):
            name, kind, t, c = self.col(alias_of)
            r = self.rng.random()
            if kind == "dec" and r < 0.2:
                other = self.col(alias_of, ["dec"])[0]
                out.append(f"{name} * {other} AS c{i}")
            elif r < 0.3:
                out.append(f"CASE WHEN {self.pred(alias_of, 1)} THEN {name} END AS c{i}")
            elif r < 0.4:
                v = self.rng.choice(self.s.sample[(t, c)])
                out.append(f"COALESCE(NULLIF({name}, {lit(v)}), {name}) AS c{i}")
            else:
                out.append(f"{name} AS c{i}")
        return out

    def on_filter(self, alias, table, kind):
        """A filter in the ON clause of an outer join: the foreign keys are complete, so only this
        leaves unmatched rows (NULL-extended) for the join type to matter."""
        if kind != "LEFT JOIN" or self.rng.random() < 0.4:
            return ""
        return f" AND {self.pred([(alias, table)], 1)}"

    def joins(self, n):
        """A join tree of n tables along the foreign keys: (FROM text, [(alias, table)])."""
        r = self.rng
        edge = r.choice(FKS)
        tabs = [("a0", edge[0]), ("a1", edge[2])]
        kind = r.choice(["JOIN", "JOIN", "LEFT JOIN"])
        text = f"{edge[0]} a0 {kind} {edge[2]} a1 ON a0.{edge[1]} = a1.{edge[3]}{self.on_filter('a1', edge[2], kind)}"
        while len(tabs) < n:
            options = []
            for a, t in tabs:
                for x in FKS:
                    if x[0] == t and x[2] not in dict((tt, 1) for _, tt in tabs):
                        options.append((a, x[1], x[2], x[3]))
                    if x[2] == t and x[0] not in dict((tt, 1) for _, tt in tabs):
                        options.append((a, x[3], x[0], x[1]))
            if not options:
                break
            a, mine, other, theirs = r.choice(options)
            nxt = f"a{len(tabs)}"
            jk = r.choice(["JOIN", "LEFT JOIN"])
            text += f" {jk} {other} {nxt} ON {a}.{mine} = {nxt}.{theirs}{self.on_filter(nxt, other, jk)}"
            tabs.append((nxt, other))
        return text, tabs

    def agg(self, alias_of, n):
        r = self.rng
        keys = [self.col(alias_of)[0] for _ in range(r.randint(0, 2))]
        keys = list(dict.fromkeys(keys))
        items = [f"{k} AS g{i}" for i, k in enumerate(keys)]
        for i in range(n):
            f = r.choice(["count", "sum", "min", "max", "avg", "cnt_star", "cnt_distinct"])
            if f == "cnt_star":
                items.append(f"count(*) AS a{i}")
                continue
            if f in ("sum", "avg"):
                name = self.col(alias_of, ["int", "dec"] if f == "sum" else ["int"])[0]
            else:
                name = self.col(alias_of)[0]
            f = "count(DISTINCT " if f == "cnt_distinct" else f + "("
            items.append(f"{f}{name}) AS a{i}")
        return keys, items

    def query(self):
        r = self.rng
        shape = r.choice(["filter", "agg", "join", "joinagg", "exists", "scalar", "distinct", "union", "cte"])
        t = r.choice(TABLES)
        a1 = [("a0", t)]
        if shape == "filter":
            return f"SELECT {', '.join(self.proj(a1, r.randint(1, 4)))} FROM {t} a0 WHERE {self.pred(a1)}"
        if shape == "agg":
            keys, items = self.agg(a1, r.randint(1, 3))
            tail = f" GROUP BY {', '.join(keys)}" if keys else ""
            having = f" HAVING count(*) > {r.randint(0, 3)}" if keys and r.random() < 0.4 else ""
            return f"SELECT {', '.join(items)} FROM {t} a0 WHERE {self.pred(a1)}{tail}{having}"
        if shape in ("join", "joinagg"):
            text, tabs = self.joins(r.randint(2, 4))
            where = f" WHERE {self.pred(tabs)}" if r.random() < 0.8 else ""
            if shape == "join":
                return f"SELECT {', '.join(self.proj(tabs, r.randint(1, 4)))} FROM {text}{where}"
            keys, items = self.agg(tabs, r.randint(1, 3))
            tail = f" GROUP BY {', '.join(keys)}" if keys else ""
            return f"SELECT {', '.join(items)} FROM {text}{where}{tail}"
        if shape == "exists":
            e = r.choice(FKS)
            outer, mine, inner, theirs = (e[0], e[1], e[2], e[3]) if r.random() < 0.5 else (e[2], e[3], e[0], e[1])
            inner_pred = self.pred([("b0", inner)], 1)
            neg = r.choice(["", "NOT "])
            form = r.choice(["exists", "in"])
            oa = [("a0", outer)]
            if form == "exists":
                sub = f"{neg}EXISTS (SELECT 1 FROM {inner} b0 WHERE b0.{theirs} = a0.{mine} AND {inner_pred})"
            else:
                sub = f"a0.{mine} {neg}IN (SELECT b0.{theirs} FROM {inner} b0 WHERE {inner_pred})"
            return f"SELECT {', '.join(self.proj(oa, r.randint(1, 3)))} FROM {outer} a0 WHERE {sub}"
        if shape == "scalar":
            name, _, _, c = self.col(a1, ["int", "dec"])
            agg = r.choice(["max", "min"])
            return (
                f"SELECT {', '.join(self.proj(a1, r.randint(1, 3)))} FROM {t} a0 WHERE {name} "
                f"{r.choice(['=', '<', '>='])} (SELECT {agg}({c}) FROM {t} s0 WHERE {self.pred([('s0', t)], 1)})"
            )
        if shape == "distinct":
            return f"SELECT DISTINCT {', '.join(self.proj(a1, r.randint(1, 3)))} FROM {t} a0 WHERE {self.pred(a1)}"
        if shape == "union":
            cols = [self.col(a1)[0] for _ in range(r.randint(1, 3))]
            # two filters of one table over the same columns, UNION ALL (and sometimes UNION)
            sel = ", ".join(cols)
            op = r.choice(["UNION ALL", "UNION"])
            return f"SELECT {sel} FROM {t} a0 WHERE {self.pred(a1)} {op} SELECT {sel} FROM {t} a0 WHERE {self.pred(a1)}"
        keys, items = self.agg(a1, r.randint(1, 2))
        tail = f" GROUP BY {', '.join(keys)}" if keys else ""
        inner = f"SELECT {', '.join(items)} FROM {t} a0 WHERE {self.pred(a1)}{tail}"
        return f"WITH q AS ({inner}) SELECT * FROM q"


def cell(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, decimal.Decimal | int | str):
        return str(v)
    if isinstance(v, datetime.date):
        return v.isoformat()
    return repr(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    con = duckdb.connect()
    for t in TABLES:
        path = os.path.join(args.data, f"{t}.parquet")
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{path}')")
    rng = random.Random(args.seed)
    gen = Gen(Schema(con), rng)
    out = []
    seen = set()
    tries = 0
    while len(out) < args.count and tries < args.count * 20:
        tries += 1
        sql = gen.query()
        if sql in seen:
            continue
        seen.add(sql)
        try:
            rows = con.execute(sql).fetchall()
        except duckdb.Error as e:
            sys.stderr.write(f"skipped ({e}): {sql}\n")
            continue
        if len(rows) > MAX_ROWS or (not rows and rng.random() < 0.8):
            continue
        out.append({"sql": sql, "rows": [[cell(v) for v in row] for row in rows]})
    json.dump({"duckdb": duckdb.__version__, "seed": args.seed, "queries": out}, sys.stdout, indent=0)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
