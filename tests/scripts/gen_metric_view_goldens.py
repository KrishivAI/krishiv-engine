#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the metric-view goldens (P3 4.2-MV, design note `design/P3-metric-views.md` section 6; ADR 0008).

Nothing is transcribed: every outcome below is produced by a real Spark run. For each release a one-off
throw-away container (images pinned by digest below, named `o2-mv-spark<ver>`) runs this same file under
`spark-submit` in `--probe` mode (classic PySpark, local mode, the in-memory session catalog). The probe runs every
case and prints one JSON document, which the host side writes to
`tests/fixtures/spark-<version>/metric-views/cases.json`.

A case is a list of steps run in a fresh `SparkSession.newSession()` (own temporary views, current database and
SQL configs) and a fresh database `mv_<case>` (plus `mv_<case>_b` where a case needs two), dropped afterwards.
Step kinds:

- `sql`: `spark.sql(text)`; records the schema (name, `simpleString` type, nullability, column metadata when not
  empty) and the rows of `.collect()`, or the error and the phase it was raised in (`sql`: parse/analysis or an
  eager command; `collect`: planning/execution). Rows are sorted by the generator unless the text has an
  `ORDER BY` (`"row_order": "spark"` vs `"sorted"`).
- `conf`: `spark.conf.set(key, value)`.
- `api`: a DataFrame-API step (`table_schema`, `table_collect`, `union_self`, `df_measure`, `list_tables`,
  `via_temp`).

An error records the JVM exception class, the condition (error class), SQLSTATE, message parameters, query
context and `getMessage()` (the full message text), plus the cause chain (class and message). Normalised, and
only these: the `Created Time` / `Last Access` values of DESCRIBE, and expression ids (`#123`) in EXPLAIN text.

`match` per step is the generator's reading of how Krishiv must compare (design note section 5): `exact`;
`exact_before_plan` for analysis errors whose `getMessage()` appends the analyzed plan (compare the text up to and
including `; line N pos M`, and the parameters); or `class_sqlstate_prefix` for Spark internal errors and for
SnakeYAML scanner messages inside `INVALID_METRIC_VIEW_YAML` (only the prefix before the scanner text is compared).

Usage (from the repo root; needs docker; a few minutes per release):

    python3 tests/scripts/gen_metric_view_goldens.py [--only 4.2.1,4.3.0]
    python3 tests/scripts/gen_metric_view_goldens.py --rematch   # recompute `match` only, no Spark run

The 4.1.0, 4.0.0 and 3.5.7 runs execute only the cases marked `gate` (the version refusals of section 0).
The probe part runs inside the image's own Python (3.8 for 3.5.7), so it avoids newer syntax at run time.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SELF = Path(__file__).resolve()
# The probe runs from a bind mount one level deep, where the repo root does not exist.
ROOT = SELF.parents[2] if len(SELF.parents) > 2 else SELF.parent
FORMAT_VERSION = 1

# version -> (container name, image pinned by digest, fixture dir, gate-only)
RUNS = {
    "4.2.1": (
        "o2-mv-spark421",
        "apache/spark:4.2.1@sha256:0cb70cf0193ec81c93c6c47078208512bbe659dd9dc89535dd2773826f0ea2bf",
        "spark-4.2.1",
        False,
    ),
    "4.3.0": (
        "o2-mv-spark430",
        "apache/spark:4.3.0@sha256:3dedc5c8665a6f472e2f0a3a828207a79a30785090ef325d3af219cb90a73d63",
        "spark-4.3.0",
        False,
    ),
    "4.1.0": (
        "o2-mv-spark41x",
        "apache/spark:4.1.0@sha256:370ada682945240b41cc2785aeb0b12c1b276b5bb46fc3930ba6732726ed6836",
        "spark-4.1.0",
        True,
    ),
    "4.0.0": (
        "o2-mv-g2-spark400",
        "apache/spark:4.0.0@sha256:a89782d90529a623fc4471cdddb0f9c32d6eed10a81b256a80207b23c3b1df00",
        "spark-4.0.0",
        True,
    ),
    "3.5.7": (
        "o2-mv-spark35",
        "apache/spark:3.5.7@sha256:936ff39fd63e2bb5ed064f0fbe1518198473f1cdbfa2f863d087a9a8e58116ba",
        "spark-3.5.7",
        True,
    ),
}
SPARK_CONF = {
    "spark.sql.catalogImplementation": "in-memory",
    "spark.sql.warehouse.dir": "/tmp/o2mv-warehouse",
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.shuffle.partitions": "2",
    "spark.ui.enabled": "false",
}

# ---------------------------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------------------------

ROWS = [
    ("region_1", "product_1", 80, 5.0),
    ("region_1", "product_2", 70, 10.0),
    ("REGION_1", "product_3", 60, 15.0),
    ("REGION_1", "product_4", 50, 20.0),
    ("region_2", "product_1", 40, 25.0),
    ("region_2", "product_2", 30, 30.0),
    ("REGION_2", "product_3", 20, 35.0),
    ("REGION_2", "product_4", 10, 40.0),
]
BASE_DIMS = [("region", "region"), ("product", "product"), ("region_upper", "upper(region)")]
BASE_MEAS = [("count_sum", "sum(count)"), ("price_avg", "avg(price)")]


def sql(text: str, setup: bool = False) -> dict:
    step: dict = {"kind": "sql", "sql": text}
    if setup:
        step["setup"] = True
    return step


def conf(key: str, value: str) -> dict:
    return {"kind": "conf", "key": key, "value": value}


def api(op: str, **args: str) -> dict:
    return {"kind": "api", "op": op, "args": dict(sorted(args.items()))}


def table(name: str = "test_table") -> list:
    values = ", ".join(f"('{r}', '{p}', {c}, {pr})" for r, p, c, pr in ROWS)
    return [
        sql(f"CREATE TABLE {name} (region STRING, product STRING, count INT, price DOUBLE) USING parquet", True),
        sql(f"INSERT INTO {name} VALUES {values}", True),
    ]


def yaml(
    source: str = "test_table",
    filt: str | None = None,
    dims: list | None = None,
    meas: list | None = None,
    version: str = "0.1",
) -> str:
    dims = BASE_DIMS if dims is None else dims
    meas = BASE_MEAS if meas is None else meas
    out = [f"version: {version}", f"source: {source}"]
    if filt is not None:
        out.append(f"filter: {filt}")
    if dims:
        out.append("dimensions:")
        out += [f"  - name: {n}\n    expr: {e}" for n, e in dims]
    if meas:
        out.append("measures:")
        out += [f"  - name: {n}\n    expr: {e}" for n, e in meas]
    return "\n".join(out)


def create(body: str, name: str = "mv", head: str = "CREATE VIEW", clauses: str = "WITH METRICS LANGUAGE YAML") -> str:
    return f"{head} {name} {clauses} AS $$\n{body}\n$$"


def mv(body: str | None = None, name: str = "mv") -> dict:
    return sql(create(yaml() if body is None else body, name))


CASES: list = []


def case(name: str, group: str, ref: str, steps: list, gate: bool = False, tables: bool = True) -> None:
    CASES.append(
        {"name": name, "group": group, "note_ref": ref, "gate": gate, "steps": (table() if tables else []) + steps}
    )


def suite_cases() -> None:
    """The MetricViewSuite tests (v4.2.0 sql/core/src/test/.../execution/MetricViewSuite.scala), as SQL."""
    ref = "6 MetricViewSuite"
    q2 = "SELECT measure(count_sum), measure(price_avg) FROM mv"
    for label, source in [("asset", "test_table"), ("sql", "SELECT * FROM test_table")]:
        case(
            f"suite_source_type_{label}",
            "suite",
            ref,
            [
                mv(yaml(source=source)),
                sql(q2),
                sql(q2 + " WHERE region_upper = 'REGION_1'"),
                sql("SELECT sum(count), avg(price) FROM test_table WHERE upper(region) = 'REGION_1'"),
            ],
        )
    case(
        "suite_where_clause",
        "suite",
        ref,
        [mv(yaml(filt="product = 'product_1'")), sql(q2), sql(q2 + " WHERE region_upper = 'REGION_1'")],
    )
    case(
        "suite_dimensions_and_measures",
        "suite",
        ref,
        [
            mv(),
            sql("SELECT region, product, measure(count_sum), measure(price_avg) FROM mv GROUP BY region, product"),
            sql("SELECT region_upper FROM mv GROUP BY 1"),
            sql("SELECT measure(count_sum) FROM mv"),
        ],
    )
    case("suite_source_column_unresolved", "suite", ref, [mv(), sql("SELECT sum(count) FROM mv")])
    case(
        "suite_order_by_limit",
        "suite",
        ref,
        [
            mv(),
            sql("SELECT region, measure(count_sum) FROM mv GROUP BY region ORDER BY 2 DESC"),
            sql("SELECT product, measure(price_avg) FROM mv GROUP BY product ORDER BY 2 ASC LIMIT 2"),
        ],
    )
    case(
        "suite_complex_where",
        "suite",
        ref,
        [
            mv(),
            sql(
                "SELECT region, product, measure(count_sum) FROM mv WHERE region_upper = 'REGION_1' "
                "AND product IN ('product_1', 'product_2') GROUP BY region, product"
            ),
            sql(q2 + " WHERE region_upper LIKE 'REGION_%' AND product <> 'product_4'"),
        ],
    )
    case(
        "suite_filter_and_query_where",
        "suite",
        ref,
        [
            mv(yaml(filt="product IN ('product_1', 'product_2')")),
            sql("SELECT region, measure(count_sum) FROM mv WHERE region_upper = 'REGION_1' GROUP BY region"),
        ],
    )
    multi = [
        ("count_sum", "sum(count)"),
        ("count_avg", "avg(count)"),
        ("count_max", "max(count)"),
        ("count_min", "min(count)"),
        ("price_sum", "sum(price)"),
    ]
    case(
        "suite_multiple_measures",
        "suite",
        ref,
        [
            mv(yaml(dims=[("region", "region")], meas=multi)),
            sql(
                "SELECT measure(count_sum), measure(count_avg), measure(count_max), measure(count_min), "
                "measure(price_sum) FROM mv"
            ),
            sql("SELECT region, measure(count_sum), measure(count_max), measure(price_sum) FROM mv GROUP BY region"),
        ],
    )
    case(
        "suite_case_dimension",
        "suite",
        ref,
        [
            mv(
                yaml(
                    dims=[
                        ("region", "region"),
                        ("region_category", "CASE WHEN region = 'region_1' THEN 'Group A' ELSE 'Group B' END"),
                    ],
                    meas=[("count_sum", "sum(count)")],
                )
            ),
            sql("SELECT region_category, measure(count_sum) FROM mv GROUP BY region_category"),
        ],
    )
    case(
        "suite_arithmetic_measures",
        "suite",
        ref,
        [
            mv(
                yaml(
                    dims=[("region", "region")],
                    meas=[("total_revenue", "sum(count * price)"), ("avg_revenue", "avg(count * price)")],
                )
            ),
            sql("SELECT measure(total_revenue), measure(avg_revenue) FROM mv"),
            sql("SELECT region, measure(total_revenue) FROM mv GROUP BY region ORDER BY 2 DESC, 1"),
        ],
    )
    case(
        "suite_group_by_dimension_expr",
        "suite",
        ref,
        [
            mv(),
            sql(
                "SELECT region_upper, product, measure(count_sum) FROM mv GROUP BY region_upper, product "
                "ORDER BY region_upper, product"
            ),
        ],
    )
    case(
        "suite_where_or",
        "suite",
        ref,
        [mv(), sql(q2 + " WHERE region = 'region_1' OR product = 'product_1'")],
    )
    case(
        "suite_dimension_only_multi",
        "suite",
        ref,
        [mv(), sql("SELECT region_upper, product FROM mv GROUP BY region_upper, product")],
    )
    case("suite_select_star_fails", "suite", ref, [mv(), sql("SELECT * FROM mv")])
    case(
        "suite_sql_source_complex",
        "suite",
        ref,
        [
            mv(yaml(source="SELECT region, product, count, price FROM test_table WHERE count > 20")),
            sql(q2),
            sql("SELECT region, measure(count_sum) FROM mv GROUP BY region"),
        ],
    )
    case(
        "suite_measure_without_group_by",
        "suite",
        ref,
        [mv(), sql("SELECT measure(count_sum) FROM mv"), sql(q2)],
    )
    case(
        "suite_combine_dimension_where",
        "suite",
        ref,
        [
            mv(),
            sql(
                "SELECT product, measure(count_sum) FROM mv WHERE region = 'region_1' "
                "AND region_upper = 'REGION_1' GROUP BY product"
            ),
        ],
    )
    case(
        "suite_count_distinct",
        "suite",
        ref,
        [
            mv(
                yaml(
                    dims=[("region", "region")],
                    meas=[("product_count", "count(distinct product)"), ("count_sum", "sum(count)")],
                )
            ),
            sql("SELECT region, measure(product_count), measure(count_sum) FROM mv GROUP BY region"),
        ],
    )
    case(
        "suite_union_self_dataframe",
        "suite",
        "4.7 df.union(df)",
        [
            mv(),
            api(
                "union_self",
                sql="SELECT region, measure(count_sum) AS total_count, measure(price_avg) AS avg_price "
                "FROM mv GROUP BY region",
            ),
        ],
    )


DOLLAR = "$$\n" + yaml() + "\n$$"


def parse_cases() -> None:
    ref = "1 check order"
    y = DOLLAR
    for label, clauses in [
        ("comment", "WITH METRICS LANGUAGE YAML COMMENT 'a' COMMENT 'b'"),
        ("tblproperties", "WITH METRICS LANGUAGE YAML TBLPROPERTIES ('a'='1') TBLPROPERTIES ('b'='2')"),
        ("language", "WITH METRICS LANGUAGE YAML LANGUAGE YAML"),
        ("with_metrics", "WITH METRICS WITH METRICS LANGUAGE YAML"),
    ]:
        case(f"parse_duplicate_{label}", "parse", ref, [sql(f"CREATE VIEW mv {clauses} AS {y}")], tables=False)
    case(
        "parse_duplicate_before_replace_check",
        "parse",
        ref,
        [sql(f"CREATE OR REPLACE VIEW IF NOT EXISTS mv WITH METRICS WITH METRICS LANGUAGE YAML AS {y}")],
        tables=False,
    )
    case(
        "parse_if_not_exists_and_replace",
        "parse",
        ref,
        [sql(f"CREATE OR REPLACE VIEW IF NOT EXISTS mv WITH METRICS LANGUAGE YAML AS {y}")],
        tables=False,
    )
    case(
        "parse_replace_check_before_missing_clause",
        "parse",
        ref,
        [sql(f"CREATE OR REPLACE VIEW IF NOT EXISTS mv LANGUAGE YAML AS {y}")],
        tables=False,
    )
    case("parse_missing_with_metrics", "parse", ref, [sql(f"CREATE VIEW mv LANGUAGE YAML AS {y}")], tables=False)
    case("parse_missing_language", "parse", ref, [sql(f"CREATE VIEW mv WITH METRICS AS {y}")], tables=False)
    case("parse_missing_both_clauses", "parse", ref, [sql(f"CREATE VIEW mv AS {y}")], tables=False)
    case("parse_language_sql", "parse", ref, [sql(f"CREATE VIEW mv WITH METRICS LANGUAGE SQL AS {y}")], tables=False)
    case(
        "parse_language_python",
        "parse",
        ref,
        [sql(f"CREATE VIEW mv WITH METRICS LANGUAGE PYTHON AS {y}")],
        tables=False,
    )
    case(
        "parse_language_lowercase_yaml",
        "parse",
        ref,
        [sql(f"CREATE VIEW mv WITH METRICS LANGUAGE yaml AS {y}"), sql("SELECT measure(count_sum) FROM mv")],
    )
    case(
        "parse_language_backquoted",
        "parse",
        ref,
        [sql(f"CREATE VIEW mv WITH METRICS LANGUAGE `yaml` AS {y}")],
        tables=False,
    )
    case(
        "parse_as_select",
        "parse",
        ref,
        [sql("CREATE VIEW mv WITH METRICS LANGUAGE YAML AS SELECT * FROM test_table")],
    )
    case(
        "parse_clause_order_free",
        "parse",
        ref,
        [
            sql(f"CREATE VIEW mv COMMENT 'c' LANGUAGE YAML TBLPROPERTIES ('k'='v') WITH METRICS AS {y}"),
            sql("SELECT measure(count_sum) FROM mv"),
        ],
    )
    case(
        "parse_tagged_dollar_quote",
        "parse",
        ref,
        [
            sql("CREATE VIEW mv WITH METRICS LANGUAGE YAML AS $yml$\n" + yaml() + "\n$yml$"),
            sql("SELECT measure(count_sum) FROM mv"),
        ],
    )
    case(
        "parse_temporary",
        "parse",
        "1 no TEMPORARY form",
        [sql(f"CREATE TEMPORARY VIEW mv WITH METRICS LANGUAGE YAML AS {y}")],
        tables=False,
    )
    case(
        "parse_sql_security",
        "parse",
        "1 no SQL SECURITY",
        [sql(f"CREATE VIEW mv WITH METRICS LANGUAGE YAML SQL SECURITY INVOKER AS {y}")],
        tables=False,
    )
    case(
        "parse_schema_binding",
        "parse",
        "1 grammar",
        [sql(f"CREATE VIEW mv WITH METRICS LANGUAGE YAML WITH SCHEMA BINDING AS {y}")],
        tables=False,
    )
    case(
        "parse_alter_view_with_metrics",
        "parse",
        "1 no ALTER VIEW ... WITH METRICS",
        [mv(), sql(f"ALTER VIEW mv WITH METRICS LANGUAGE YAML AS {y}")],
    )
    case(
        "parse_alter_view_as_query",
        "parse",
        "1 ALTER VIEW mv AS <query>",
        [mv(), sql("ALTER VIEW mv AS SELECT 1 AS x"), sql("DESCRIBE mv"), sql("SELECT * FROM mv")],
    )
    case(
        "parse_metrics_measure_identifiers",
        "parse",
        "1 non-reserved METRICS/MEASURE",
        [sql("SELECT 1 AS metrics, 2 AS measure"), sql("CREATE TABLE metrics (measure INT) USING parquet")],
        tables=False,
    )
    case(
        "parse_empty_body",
        "parse",
        "2",
        [sql("CREATE VIEW mv WITH METRICS LANGUAGE YAML AS $$$$")],
        tables=False,
    )


def yaml_cases() -> None:
    ref = "2 YAML matrix"
    rest = yaml()[len("version: 0.1\n") :]

    def ycase(name: str, body: str, read: bool = True, extra: list | None = None) -> None:
        steps = [mv(body)]
        if read:
            steps.append(sql("SELECT region, measure(count_sum) FROM mv GROUP BY region ORDER BY region"))
        case(f"yaml_{name}", "yaml", ref, steps + (extra or []))

    for label, v in [
        ("float", "0.1"),
        ("double_quoted", '"0.1"'),
        ("single_quoted", "'0.1'"),
        ("0_10", "0.10"),
        ("1", "1"),
        ("1_0", "1.0"),
        ("yes", "yes"),
        ("null", "~"),
        ("tag_str", "!!str 0.1"),
    ]:
        ycase(f"version_{label}", f"version: {v}\n{rest}")
    ycase("version_missing", rest)
    ycase("unknown_top_key", yaml() + "\nowner: someone")
    ycase(
        "unknown_column_key",
        yaml(dims=[("region", "region\n    comment: c")], meas=[("count_sum", "sum(count)")]),
    )
    ycase("missing_source", "version: 0.1\n" + "\n".join(rest.split("\n")[1:]))
    ycase("missing_dimension_name", yaml().replace("  - name: region\n    expr: region\n", "  - expr: region\n"))
    ycase("missing_measure_expr", yaml().replace("\n    expr: sum(count)", ""))
    ycase("missing_measure_expr_only", yaml(meas=[]) + "\nmeasures:\n  - name: count_sum")
    ycase("source_null", yaml(source="~"))
    ycase("source_empty", yaml(source='""'))
    ycase("source_invalid", yaml(source="not a ( query"))
    ycase("source_sql_trailing_keyword", yaml(source="SELECT * FROM test_table WHERE"))
    ycase("source_sql_unparsable", yaml(source="SELECT * FROM test_table WHERE count >"))
    ycase("source_statement_not_query", yaml(source="SHOW TABLES"))
    ycase("source_missing_table", yaml(source="no_such_table"))
    ycase("source_sql_missing_table", yaml(source="SELECT * FROM no_such_table"))
    ycase("source_qualified", yaml(source="spark_catalog.mv_yaml_source_qualified.test_table"))
    ycase(
        "dimensions_null",
        "version: 0.1\nsource: test_table\ndimensions: ~\nmeasures:\n  - name: count_sum\n    expr: sum(count)",
        read=False,
        extra=[sql("SELECT measure(count_sum) FROM mv")],
    )
    ycase(
        "measures_null",
        "version: 0.1\nsource: test_table\ndimensions:\n  - name: region\n    expr: region\nmeasures: ~",
        read=False,
        extra=[sql("SELECT region FROM mv GROUP BY region ORDER BY region")],
    )
    ycase("no_dimensions", yaml(dims=[]), read=False, extra=[sql("SELECT measure(count_sum) FROM mv")])
    ycase("no_measures", yaml(meas=[]), read=False, extra=[sql("SELECT region FROM mv GROUP BY region ORDER BY 1")])
    ycase("no_columns", yaml(dims=[], meas=[]), read=False, extra=[sql("DESCRIBE mv")])
    ycase("dimension_expr_unparsable", yaml(dims=[("region", "upper(region")]))
    ycase("measure_expr_unparsable", yaml(meas=[("count_sum", "sum(count")]))
    ycase("filter_unparsable", yaml(filt="region ="))
    ycase("filter_not_boolean", yaml(filt="count"))
    ycase("filter_unknown_column", yaml(filt="nope = 1"))
    ycase("dimension_unknown_column", yaml(dims=[("region", "nope")]))
    ycase("dimension_is_aggregate", yaml(dims=[("region", "sum(count)")]))
    ycase("measure_not_aggregate", yaml(meas=[("count_sum", "count + 1")]))
    ycase("measure_aggregate_plus_column", yaml(meas=[("count_sum", "sum(count) + count")]))
    ycase("measure_nested_aggregate", yaml(meas=[("count_sum", "sum(max(count))")]))
    ycase("measure_window", yaml(meas=[("count_sum", "sum(count) OVER ()")]))
    ycase("dimension_number", yaml(dims=[("region", "1")]), extra=[sql("DESCRIBE mv")])
    ycase(
        "dimension_constant_not_shadowing",
        yaml(dims=[("region", "region"), ("one", "1")]),
        extra=[sql("SELECT one, measure(count_sum) FROM mv GROUP BY one ORDER BY one")],
    )
    ycase("dimension_bool", yaml(dims=[("region", "true")]), extra=[sql("DESCRIBE mv")])
    ycase("dimension_flow_sequence", yaml(dims=[("region", "[a, b]")]))
    ycase("dimension_flow_mapping", yaml(dims=[("region", "{a: b}")]))
    ycase(
        "dimension_block_literal",
        yaml(dims=[("region", "|\n      upper(\n        region)")]),
        extra=[api("table_schema", name="mv")],
    )
    ycase(
        "dimension_block_folded",
        yaml(dims=[("region", ">\n      upper(\n        region)")]),
        extra=[api("table_schema", name="mv")],
    )
    ycase("dimension_tag_str", yaml(dims=[("region", "!!str upper(region)")]))
    ycase("dimension_tag_int", yaml(dims=[("region", "!!int 1")]))
    ycase("dimension_quoted_colon", yaml(dims=[("region", "\"concat(region, ': x')\"")]))
    ycase("name_number", yaml(dims=[("1", "region")]), read=False, extra=[sql("DESCRIBE mv")])
    ycase(
        "name_with_space",
        yaml(dims=[("my region", "region")]),
        read=False,
        extra=[sql("SELECT `my region`, measure(count_sum) FROM mv GROUP BY 1 ORDER BY 1")],
    )
    ycase("duplicate_key_source", "version: 0.1\nsource: no_such_table\n" + rest)
    ycase("duplicate_key_version", "version: 9\n" + yaml())
    ycase("duplicate_dimension_name", yaml(dims=[("region", "region"), ("region", "upper(region)")]), read=False)
    ycase("dimension_measure_same_name", yaml(dims=[("region", "region")], meas=[("region", "sum(count)")]), read=False)
    ycase("duplicate_name_case_insensitive", yaml(dims=[("region", "region"), ("REGION", "upper(region)")]), read=False)
    ycase(
        "anchor_alias",
        "version: 0.1\nsource: test_table\ndimensions:\n  - name: region\n    expr: &e upper(region)\n"
        "  - name: region2\n    expr: *e\nmeasures:\n  - name: count_sum\n    expr: sum(count)",
        extra=[api("table_schema", name="mv")],
    )
    ycase(
        "merge_key",
        "version: 0.1\nsource: test_table\nbase: &b\n  expr: region\ndimensions:\n  - <<: *b\n    name: region\n"
        "measures:\n  - name: count_sum\n    expr: sum(count)",
    )
    ycase(
        "merge_key_with_expr",
        "version: 0.1\nsource: test_table\nbase: &b\n  expr: region\ndimensions:\n  - <<: *b\n    name: region\n"
        "    expr: region\nmeasures:\n  - name: count_sum\n    expr: sum(count)",
    )
    ycase(
        "alias_whole_mapping",
        "version: 0.1\nsource: test_table\ndimensions:\n  - &d\n    name: region\n    expr: region\n  - *d\n"
        "measures:\n  - name: count_sum\n    expr: sum(count)",
    )
    ycase(
        "merge_key_in_list",
        "version: 0.1\nsource: test_table\ndimensions:\n  - &b\n    name: r0\n    expr: region\n  - <<: *b\n"
        "    name: region\nmeasures:\n  - name: count_sum\n    expr: sum(count)",
    )
    ycase("leading_document_marker", "---\n" + yaml())
    ycase("document_end_marker", yaml() + "\n...")
    ycase("second_document", yaml() + "\n---\nversion: 0.1\nsource: other")
    ycase("comment_lines", "# a comment\n" + yaml().replace("dimensions:", "dimensions: # dims"))
    ycase(
        "flow_style_document",
        "{version: 0.1, source: test_table, dimensions: [{name: region, expr: region}], "
        "measures: [{name: count_sum, expr: sum(count)}]}",
    )
    ycase("not_a_mapping", "- a\n- b")
    ycase("scalar_document", "just text")
    ycase("malformed_indent", "version: 0.1\nsource: test_table\ndimensions:\n  - name: region\n   expr: region")
    ycase("malformed_tab", "version: 0.1\nsource: test_table\ndimensions:\n\t- name: region")
    ycase("malformed_unclosed_quote", 'version: 0.1\nsource: "test_table\ndimensions: []')
    ycase("dimensions_not_list", yaml(dims=[]) + "\ndimensions: region")
    ycase("source_sequence", yaml(source="[a, b]"))
    # Double-quoted YAML scalars, so the SQL string literal keeps its single quotes (length counts them).
    long_ok = "\"'" + "a" * 1022 + "'\""
    long_bad = "\"'" + "a" * 1023 + "'\""
    ycase(
        "expr_length_1024",
        yaml(dims=[("region", long_ok)]),
        read=False,
        extra=[sql("SELECT length(region) FROM mv GROUP BY region")],
    )
    ycase("expr_length_1025", yaml(dims=[("region", long_bad)]), read=False)
    ycase("expr_length_1031_measure", yaml(meas=[("count_sum", "sum(count) + " + "0 + " * 254 + "00")]), read=False)
    # Non-constant dimensions (a constant dimension fails at CREATE on its own, see expr_length_1024).
    pre = "concat(region, '"
    for n in (1024, 1025):
        ycase(
            f"expr_length_{n}_concat",
            yaml(dims=[("region", '"' + pre + "a" * (n - len(pre) - 2) + "')" + '"')]),
            read=False,
            extra=[sql("SELECT length(region), measure(count_sum) FROM mv GROUP BY region ORDER BY 1, 2")],
        )
        ycase(
            f"expr_length_{n}_measure",
            yaml(meas=[("count_sum", "sum(count) + " + "0" * (n - 13))]),
            read=False,
            extra=[sql("SELECT measure(count_sum) FROM mv")],
        )
    emoji = "\U0001f600"
    # UTF-16 length 1 + 2*511 + 1 = 1024 (515 code points): accepted if Spark counts UTF-16 units.
    ycase(
        "expr_non_bmp_1024_units",
        yaml(dims=[("region", "\"'" + emoji * 511 + "'\"")]),
        read=False,
        extra=[sql("SELECT length(region) FROM mv GROUP BY region")],
    )
    # UTF-16 length 1 + 2*512 + 1 = 1026 (514 code points).
    ycase("expr_non_bmp_1026_units", yaml(dims=[("region", "\"'" + emoji * 512 + "'\"")]), read=False)
    # concat(region, '<k emoji>'): UTF-16 length 16 + 2k + 2; k = 503 gives 1024 units (521 code points).
    for k in (503, 504):
        ycase(
            f"expr_non_bmp_concat_{16 + 2 * k + 2}_units",
            yaml(dims=[("region", '"' + pre + emoji * k + "')" + '"')]),
            read=False,
            extra=[sql("SELECT length(region), measure(count_sum) FROM mv GROUP BY region ORDER BY 1, 2")],
        )
    # metric_view.where cut at 1024 UTF-16 units splits the surrogate pair at units 1023-1024.
    cut_filter = "\"region <> '" + "x" * 1012 + emoji + "'\""
    ycase(
        "property_cut_splits_surrogate",
        yaml(filt=cut_filter),
        extra=[
            api(
                "via_temp",
                first="SHOW TBLPROPERTIES mv",
                then="SELECT key, length(value) AS len, hex(encode(value, 'UTF-8')) AS utf8_hex_tail FROM _r "
                "WHERE key = 'metric_view.where'",
            ),
        ],
    )
    long_sql = "SELECT * FROM test_table WHERE " + " AND ".join(["count >= 0"] * 100)
    ycase(
        "property_cut_sql_source",
        yaml(source=long_sql),
        extra=[
            api(
                "via_temp",
                first="SHOW TBLPROPERTIES mv",
                then="SELECT key, length(value) AS len FROM _r WHERE key LIKE 'metric_view.%' ORDER BY key",
            ),
        ],
    )


def ddl_cases() -> None:
    ref = "3 DDL"
    case(
        "ddl_describe",
        "ddl",
        ref,
        [
            mv(yaml(filt="product <> 'product_4'")),
            sql("DESCRIBE mv"),
            sql("DESCRIBE EXTENDED mv"),
            sql("DESCRIBE mv region_upper"),
            sql("DESCRIBE EXTENDED mv count_sum"),
            sql("SHOW COLUMNS IN mv"),
            api("table_schema", name="mv"),
        ],
    )
    case(
        "ddl_describe_sql_source",
        "ddl",
        ref,
        [mv(yaml(source="SELECT * FROM test_table WHERE count > 20")), sql("DESCRIBE EXTENDED mv")],
    )
    case(
        "ddl_describe_as_json",
        "ddl",
        ref,
        [mv(), sql("DESCRIBE EXTENDED mv AS JSON")],
    )
    case(
        "ddl_show_tables_views",
        "ddl",
        ref,
        [
            mv(),
            sql("CREATE VIEW plain_v AS SELECT * FROM test_table"),
            sql("SHOW TABLES"),
            sql("SHOW VIEWS"),
            sql("SHOW TABLE EXTENDED LIKE 'mv'"),
            api("list_tables"),
        ],
    )
    case("ddl_show_create_table", "ddl", ref, [mv(), sql("SHOW CREATE TABLE mv"), sql("SHOW CREATE TABLE mv AS SERDE")])
    case("ddl_show_tblproperties", "ddl", ref, [mv(), sql("SHOW TBLPROPERTIES mv")])
    case(
        "ddl_user_metric_view_properties",
        "ddl",
        ref,
        [
            sql(
                create(
                    yaml(),
                    clauses="WITH METRICS LANGUAGE YAML TBLPROPERTIES ('metric_view.from.type'='X', "
                    "'metric_view.from.name'='other', 'metric_view.custom'='y', 'k'='v')",
                )
            ),
            sql("SHOW TBLPROPERTIES mv"),
        ],
    )
    case("ddl_drop_table_on_metric_view", "ddl", ref, [mv(), sql("DROP TABLE mv"), sql("SHOW TABLES")])
    case(
        "ddl_drop_view_on_table",
        "ddl",
        "3 DROP VIEW on a table (I6)",
        [sql("DROP VIEW test_table")],
        gate=True,
    )
    case(
        "ddl_drop_view",
        "ddl",
        ref,
        [mv(), sql("DROP VIEW mv"), sql("SHOW TABLES"), sql("SELECT measure(count_sum) FROM mv")],
    )
    case("ddl_drop_view_if_exists", "ddl", ref, [mv(), sql("DROP VIEW IF EXISTS mv"), sql("DROP VIEW IF EXISTS mv")])
    case(
        "ddl_rename",
        "ddl",
        ref,
        [
            mv(),
            sql("ALTER VIEW mv RENAME TO mv2"),
            sql("SHOW TABLES"),
            sql("SELECT region, measure(count_sum) FROM mv2 GROUP BY region ORDER BY region"),
            sql("DESCRIBE EXTENDED mv2"),
            api("table_schema", name="mv2"),
        ],
    )
    case("ddl_alter_table_rename", "ddl", ref, [mv(), sql("ALTER TABLE mv RENAME TO mv2"), sql("SHOW TABLES")])
    case(
        "ddl_alter_view_set_tblproperties",
        "ddl",
        ref,
        [
            mv(),
            sql("ALTER VIEW mv SET TBLPROPERTIES ('k'='v')"),
            sql("ALTER VIEW mv UNSET TBLPROPERTIES ('k')"),
            sql("SHOW TBLPROPERTIES mv"),
        ],
    )
    case(
        "ddl_alter_view_schema_binding",
        "ddl",
        ref,
        [mv(), sql("ALTER VIEW mv WITH SCHEMA EVOLUTION")],
    )
    case(
        "ddl_comment_on_view",
        "ddl",
        ref,
        [mv(), sql("COMMENT ON VIEW mv IS 'hello'"), sql("DESCRIBE EXTENDED mv")],
    )
    case(
        "ddl_or_replace_existing_metric_view",
        "ddl",
        ref,
        [
            mv(),
            sql(create(yaml(dims=[("region", "region")]), head="CREATE OR REPLACE VIEW")),
            sql("DESCRIBE mv"),
        ],
    )
    case(
        "ddl_or_replace_existing_plain_view",
        "ddl",
        ref,
        [
            sql("CREATE VIEW mv AS SELECT 1 AS x"),
            sql(create(yaml(), head="CREATE OR REPLACE VIEW")),
            sql("DESCRIBE mv"),
        ],
    )
    case(
        "ddl_or_replace_new",
        "ddl",
        ref,
        [sql(create(yaml(), head="CREATE OR REPLACE VIEW")), sql("SELECT measure(count_sum) FROM mv")],
    )
    case("ddl_create_existing", "ddl", ref, [mv(), mv()])
    case(
        "ddl_create_over_table",
        "ddl",
        ref,
        [sql(create(yaml(), name="test_table"))],
    )
    case(
        "ddl_if_not_exists_existing",
        "ddl",
        ref,
        [
            mv(),
            sql(create(yaml(dims=[("region", "region")]), head="CREATE VIEW IF NOT EXISTS")),
            sql("DESCRIBE mv"),
        ],
    )
    case(
        "ddl_plain_view_over_existing_metric_view",
        "ddl",
        ref,
        [mv(), sql("CREATE OR REPLACE VIEW mv AS SELECT 1 AS x"), sql("DESCRIBE EXTENDED mv")],
    )
    case(
        "ddl_column_list_with_comments",
        "ddl",
        ref,
        [
            sql(
                create(
                    yaml(),
                    name="mv (r COMMENT 'the region', p, ru, cs COMMENT 'sum', pa)",
                )
            ),
            sql("DESCRIBE EXTENDED mv"),
            api("table_schema", name="mv"),
            sql("SELECT r, measure(cs) FROM mv GROUP BY r ORDER BY r"),
            sql("SELECT region, measure(count_sum) FROM mv GROUP BY region"),
        ],
    )
    case("ddl_column_list_too_few", "ddl", ref, [sql(create(yaml(), name="mv (a, b)"))])
    case("ddl_column_list_too_many", "ddl", ref, [sql(create(yaml(), name="mv (a, b, c, d, e, f)"))])
    case("ddl_column_list_duplicate", "ddl", ref, [sql(create(yaml(), name="mv (a, a, b, c, d)"))])
    case(
        "ddl_temp_view_source",
        "ddl",
        ref,
        [sql("CREATE TEMPORARY VIEW tv AS SELECT * FROM test_table"), mv(yaml(source="tv"))],
    )
    case(
        "ddl_temp_view_in_sql_source",
        "ddl",
        ref,
        [sql("CREATE TEMPORARY VIEW tv AS SELECT * FROM test_table"), mv(yaml(source="SELECT * FROM tv"))],
    )
    case(
        "ddl_temp_view_in_filter_subquery",
        "ddl",
        ref,
        [
            sql("CREATE TEMPORARY VIEW tv AS SELECT 'product_1' AS p"),
            mv(yaml(filt="product IN (SELECT p FROM tv)")),
        ],
    )
    case(
        "ddl_temp_variable_in_dimension",
        "ddl",
        ref,
        [sql("DECLARE VARIABLE v INT DEFAULT 1"), mv(yaml(dims=[("region", "region"), ("vv", "v")]))],
    )
    case(
        "ddl_metric_view_over_metric_view",
        "ddl",
        "3 a metric view over a metric view",
        [
            mv(),
            mv(
                yaml(source="mv", dims=[("region", "region")], meas=[("total", "measure(count_sum)")]),
                name="mv2",
            ),
            sql("DESCRIBE mv2"),
            sql("SELECT region, measure(total) FROM mv2 GROUP BY region ORDER BY region"),
            sql("SELECT measure(total) FROM mv2"),
        ],
    )
    case(
        "ddl_metric_view_over_metric_view_plain_agg",
        "ddl",
        "3 a metric view over a metric view",
        [
            mv(),
            mv(yaml(source="mv", dims=[("region", "region")], meas=[("total", "sum(count_sum)")]), name="mv2"),
            sql("SELECT region, measure(total) FROM mv2 GROUP BY region ORDER BY region"),
        ],
    )
    case(
        "ddl_metric_view_over_plain_view",
        "ddl",
        ref,
        [
            sql("CREATE VIEW pv AS SELECT * FROM test_table WHERE count > 20"),
            mv(yaml(source="pv")),
            sql("SELECT region, measure(count_sum) FROM mv GROUP BY region ORDER BY region"),
        ],
    )
    case("ddl_insert_into", "ddl", ref, [mv(), sql("INSERT INTO mv VALUES ('a', 'b', 'c', 1, 1.0)")])
    case("ddl_create_table_like", "ddl", ref, [mv(), sql("CREATE TABLE t2 LIKE mv"), sql("DESCRIBE t2")])
    case("ddl_ctas_from_select_star", "ddl", ref, [mv(), sql("CREATE TABLE t2 AS SELECT * FROM mv")])
    case(
        "ddl_ctas_from_aggregate",
        "ddl",
        ref,
        [
            mv(),
            sql("CREATE TABLE t2 AS SELECT region, measure(count_sum) AS c FROM mv GROUP BY region"),
            sql("SELECT * FROM t2 ORDER BY region"),
        ],
    )
    case("ddl_show_partitions", "ddl", ref, [mv(), sql("SHOW PARTITIONS mv")])
    case("ddl_add_partition", "ddl", ref, [mv(), sql("ALTER TABLE mv ADD PARTITION (region='x')")])
    case("ddl_truncate", "ddl", ref, [mv(), sql("TRUNCATE TABLE mv")])
    case("ddl_analyze", "ddl", ref, [mv(), sql("ANALYZE TABLE mv COMPUTE STATISTICS")])
    case("ddl_cache", "ddl", ref, [mv(), sql("CACHE TABLE mv")])
    case("ddl_refresh", "ddl", ref, [mv(), sql("REFRESH TABLE mv")])
    case(
        "ddl_comment_clause",
        "ddl",
        ref,
        [sql(create(yaml(), clauses="WITH METRICS LANGUAGE YAML COMMENT 'hi'")), sql("DESCRIBE EXTENDED mv")],
    )
    case(
        "ddl_qualified_name",
        "ddl",
        ref,
        [
            mv(name="spark_catalog.mv_ddl_qualified_name.mv"),
            sql("SELECT measure(count_sum) FROM spark_catalog.mv_ddl_qualified_name.mv"),
        ],
    )
    case(
        "ddl_create_in_missing_database",
        "ddl",
        ref,
        [mv(name="no_such_db.mv")],
    )


def read_cases() -> None:
    def rcase(name: str, ref: str, queries: list, body: str | None = None, extra_setup: list | None = None) -> None:
        case(f"read_{name}", "read", ref, (extra_setup or []) + [mv(body)] + [sql(q) for q in queries])

    s4 = "4.4 no aggregate"
    rcase("no_aggregate_select_star", s4, ["SELECT * FROM mv"])
    rcase("no_aggregate_select_dimension", s4, ["SELECT region FROM mv"])
    rcase("no_aggregate_select_measure_column", s4, ["SELECT count_sum FROM mv"])
    rcase("no_aggregate_where", s4, ["SELECT region FROM mv WHERE region_upper = 'REGION_1'"])
    rcase("no_aggregate_limit", s4, ["SELECT * FROM mv LIMIT 1"])
    rcase("distinct", s4, ["SELECT DISTINCT region FROM mv"])
    rcase("describe_query", s4, ["DESCRIBE QUERY SELECT * FROM mv"])
    rcase("explain_select_star", s4, ["EXPLAIN SELECT * FROM mv", "EXPLAIN EXTENDED SELECT * FROM mv"])
    rcase("explain_aggregate", s4, ["EXPLAIN SELECT region, measure(count_sum) FROM mv GROUP BY region"])
    case(
        "read_dataframe_no_aggregate",
        "read",
        s4,
        [mv(), api("table_schema", name="mv"), api("table_collect", name="mv")],
    )
    case(
        "read_plain_view_over_select_star",
        "read",
        s4,
        [mv(), sql("CREATE VIEW v AS SELECT * FROM mv"), sql("DESCRIBE v"), sql("SELECT * FROM v")],
    )
    case(
        "read_plain_view_aggregate_over_select_star",
        "read",
        s4,
        [
            mv(),
            sql("CREATE VIEW v AS SELECT * FROM mv"),
            sql("SELECT region, measure(count_sum) FROM v GROUP BY region"),
        ],
    )
    s5 = "4.5 misuse"
    rcase("measure_of_dimension", s5, ["SELECT measure(region) FROM mv"])
    rcase("measure_of_expression", s5, ["SELECT measure(count_sum + 1) FROM mv"])
    rcase("measure_over_table", s5, ["SELECT measure(count) FROM test_table", "SELECT measure(1)"])
    rcase("measure_describe_query_over_table", s5, ["DESCRIBE QUERY SELECT measure(count) FROM test_table"])
    rcase("bare_measure_in_aggregate", s5, ["SELECT region, count_sum FROM mv GROUP BY region"])
    rcase("bare_measure_in_group_by", s5, ["SELECT count_sum, measure(price_avg) FROM mv GROUP BY count_sum"])
    rcase("where_on_measure", s5, ["SELECT region, measure(count_sum) FROM mv WHERE count_sum > 1 GROUP BY region"])
    rcase("measure_in_where", s5, ["SELECT region FROM mv WHERE measure(count_sum) > 1 GROUP BY region"])
    rcase("nested_measure", s5, ["SELECT sum(measure(count_sum)) FROM mv"])
    rcase("measure_no_args", s5, ["SELECT measure() FROM mv", "SELECT measure(count_sum, price_avg) FROM mv"])
    rcase("measure_distinct", s5, ["SELECT measure(DISTINCT count_sum) FROM mv"])
    rcase("measure_filter_clause", s5, ["SELECT measure(count_sum) FILTER (WHERE region = 'region_1') FROM mv"])
    rcase("measure_unknown_column", s5, ["SELECT measure(nope) FROM mv"])
    s6 = "4.6 shapes"
    rcase(
        "having_measure",
        s6,
        ["SELECT region, measure(count_sum) FROM mv GROUP BY region HAVING measure(count_sum) > 100 ORDER BY region"],
    )
    rcase(
        "having_dimension",
        s6,
        ["SELECT region, measure(count_sum) FROM mv GROUP BY region HAVING region <> 'region_1' ORDER BY region"],
    )
    rcase(
        "having_alias",
        s6,
        ["SELECT region, measure(count_sum) AS c FROM mv GROUP BY region HAVING c > 100 ORDER BY region"],
    )
    rcase(
        "order_by_measure",
        s6,
        ["SELECT region, measure(count_sum) FROM mv GROUP BY region ORDER BY measure(count_sum)"],
    )
    rcase(
        "order_by_measure_not_selected", s6, ["SELECT region FROM mv GROUP BY region ORDER BY measure(price_avg) DESC"]
    )
    rcase("order_by_dimension", s6, ["SELECT region, measure(count_sum) FROM mv GROUP BY region ORDER BY region DESC"])
    rcase("order_by_alias", s6, ["SELECT region, measure(count_sum) AS c FROM mv GROUP BY region ORDER BY c"])
    rcase("group_by_all", s6, ["SELECT region, measure(count_sum) FROM mv GROUP BY ALL ORDER BY region"])
    rcase("group_by_ordinal", s6, ["SELECT region, product, measure(count_sum) FROM mv GROUP BY 1, 2 ORDER BY 1, 2"])
    rcase("group_by_unselected", s6, ["SELECT measure(count_sum) FROM mv GROUP BY region_upper ORDER BY 1"])
    rcase(
        "rollup",
        s6,
        [
            "SELECT region_upper, product, measure(count_sum) FROM mv GROUP BY ROLLUP(region_upper, "
            "product) ORDER BY region_upper NULLS FIRST, product NULLS FIRST"
        ],
    )
    rcase(
        "cube",
        s6,
        [
            "SELECT region_upper, measure(count_sum), grouping(region_upper) FROM mv "
            "GROUP BY CUBE(region_upper) ORDER BY region_upper NULLS FIRST"
        ],
    )
    rcase(
        "grouping_sets",
        s6,
        [
            "SELECT region_upper, product, measure(price_avg) FROM mv GROUP BY GROUPING SETS "
            "((region_upper), (product), ()) ORDER BY region_upper NULLS FIRST, product NULLS FIRST"
        ],
    )
    rcase(
        "measure_arithmetic",
        s6,
        [
            "SELECT region, measure(count_sum) * 2, measure(count_sum) / measure(price_avg) "
            "FROM mv GROUP BY region ORDER BY region"
        ],
    )
    rcase("measure_with_plain_aggregate", s6, ["SELECT count(*), measure(count_sum), count(region) FROM mv"])
    rcase("plain_aggregate_only", s6, ["SELECT count(*) FROM mv", "SELECT max(region_upper) FROM mv"])
    rcase("measure_uppercase_name", s6, ["SELECT REGION, MEASURE(COUNT_SUM) FROM mv GROUP BY REGION ORDER BY 1"])
    rcase("measure_alias_name", s6, ["SELECT measure(count_sum) AS total FROM mv"])
    rcase(
        "window_over_measure",
        s6,
        [
            "SELECT region, measure(count_sum), rank() OVER (ORDER BY measure(count_sum) "
            "DESC) AS r FROM mv GROUP BY region ORDER BY region"
        ],
    )
    rcase(
        "subquery_over_aggregate",
        s6,
        [
            "SELECT * FROM (SELECT region, measure(count_sum) AS c FROM mv GROUP BY "
            "region) WHERE c > 100 ORDER BY region"
        ],
    )
    rcase(
        "cte_over_metric_view",
        s6,
        ["WITH t AS (SELECT * FROM mv) SELECT region, measure(count_sum) FROM t GROUP BY region ORDER BY region"],
    )
    rcase(
        "subquery_alias_chain",
        s6,
        [
            "SELECT region, measure(count_sum) FROM (SELECT * FROM mv WHERE "
            "product <> 'product_4') GROUP BY region ORDER BY region"
        ],
    )
    rcase("scalar_subquery", s6, ["SELECT (SELECT measure(count_sum) FROM mv) AS total"])
    rcase(
        "in_subquery",
        s6,
        ["SELECT region FROM test_table WHERE count > (SELECT measure(count_sum) / 10 FROM mv) ORDER BY region, count"],
    )
    rcase(
        "user_alias",
        "4.8 read context",
        ["SELECT m.region, measure(m.count_sum) FROM mv AS m GROUP BY m.region ORDER BY 1"],
    )
    rcase(
        "view_name_qualifier",
        "4.8 read context",
        ["SELECT mv.region, measure(mv.count_sum) FROM mv GROUP BY mv.region ORDER BY 1"],
    )
    rcase(
        "db_qualifier",
        "4.8 read context",
        ["SELECT mv_read_db_qualifier.mv.region FROM mv_read_db_qualifier.mv GROUP BY 1 ORDER BY 1"],
    )
    s7 = "4.7 joins and unions"
    rcase(
        "join_with_table",
        s7,
        ["SELECT a.region, measure(a.count_sum) FROM mv a JOIN test_table b ON a.region = b.region GROUP BY a.region"],
    )
    rcase(
        "self_join",
        s7,
        ["SELECT a.region, measure(a.count_sum) FROM mv a JOIN mv b ON a.region = b.region GROUP BY a.region"],
    )
    rcase(
        "join_of_aggregates",
        s7,
        [
            "SELECT a.region, a.c, b.p FROM (SELECT region, measure(count_sum) AS c FROM mv "
            "GROUP BY region) a JOIN (SELECT region, measure(price_avg) AS p FROM mv GROUP BY "
            "region) b ON a.region = b.region ORDER BY a.region"
        ],
    )
    rcase(
        "sql_union_all",
        s7,
        [
            "SELECT region, measure(count_sum) FROM mv GROUP BY region UNION ALL "
            "SELECT region, measure(count_sum) FROM mv GROUP BY region ORDER BY 1, 2"
        ],
    )
    rcase(
        "union_under_aggregate",
        s7,
        ["SELECT region, measure(count_sum) FROM (SELECT * FROM mv UNION ALL SELECT * FROM mv) GROUP BY region"],
    )
    case(
        "read_dataframe_group_by_measure",
        "read",
        "6 Connect test shape",
        [mv(), api("df_measure", name="mv", group="region", expr="measure(count_sum)")],
    )
    s2 = "4.2 measures see dimensions"
    rcase(
        "shadowed_source_column",
        s2,
        ["SELECT measure(n) FROM mv", "SELECT region, measure(n) FROM mv GROUP BY region ORDER BY region"],
        body=yaml(dims=[("region", "upper(region)")], meas=[("n", "count(DISTINCT region)")]),
    )
    rcase(
        "shadowed_source_column_describe",
        s2,
        ["DESCRIBE mv"],
        body=yaml(dims=[("region", "length(region)")], meas=[("m", "max(region)")]),
    )
    rcase(
        "shadowed_type_mismatch_read",
        s2,
        ["SELECT measure(m) FROM mv", "SELECT region, measure(m) FROM mv GROUP BY region ORDER BY region"],
        body=yaml(dims=[("region", "length(region)")], meas=[("m", "max(region)")]),
    )
    rcase(
        "measure_names_dimension",
        s2,
        ["SELECT measure(n) FROM mv", "SELECT product, measure(n) FROM mv GROUP BY product ORDER BY product"],
        body=yaml(dims=[("product", "product"), ("ru", "upper(region)")], meas=[("n", "count(DISTINCT ru)")]),
    )
    rcase(
        "measure_names_other_measure",
        s2,
        ["SELECT measure(m) FROM mv"],
        body=yaml(dims=[("region", "region")], meas=[("count_sum", "sum(count)"), ("m", "count_sum + 1")]),
    )
    rcase(
        "dimension_names_other_dimension",
        s2,
        ["SELECT r2, measure(c) FROM mv GROUP BY r2 ORDER BY r2"],
        body=yaml(dims=[("r1", "upper(region)"), ("r2", "lower(r1)")], meas=[("c", "sum(count)")]),
    )
    rcase(
        "non_additive_avg",
        s2,
        [
            "SELECT region_upper, measure(price_avg) FROM mv GROUP BY region_upper ORDER BY 1",
            "SELECT measure(price_avg) FROM mv",
        ],
    )
    rcase(
        "filter_before_where",
        "4.3 filter",
        [
            "SELECT measure(count_sum) FROM mv WHERE product = 'product_4'",
            "SELECT measure(count_sum) FROM mv WHERE product = 'product_1'",
        ],
        body=yaml(filt="product <> 'product_4'"),
    )
    rcase(
        "filter_on_dimension_name",
        "4.3 filter",
        ["SELECT measure(count_sum) FROM mv"],
        body=yaml(filt="region_upper = 'REGION_1'"),
    )
    rcase(
        "filter_subquery",
        "4.3 filter",
        ["SELECT measure(count_sum) FROM mv"],
        body=yaml(filt="count > (SELECT avg(count) FROM test_table)"),
    )
    s8 = "4.8 read context"
    case(
        "read_context_reader_namespace",
        "read",
        s8,
        [
            mv(),
            sql("CREATE DATABASE mv_read_context_reader_namespace_b"),
            sql("USE mv_read_context_reader_namespace_b"),
            sql("SELECT measure(count_sum) FROM mv_read_context_reader_namespace.mv"),
            sql("CREATE TABLE test_table (region STRING, product STRING, count INT, price DOUBLE) USING parquet"),
            sql("INSERT INTO test_table VALUES ('z', 'p', 1000, 1.0)"),
            sql("SELECT measure(count_sum), measure(price_avg) FROM mv_read_context_reader_namespace.mv"),
            sql("DESCRIBE mv_read_context_reader_namespace.mv"),
        ],
    )
    case(
        "read_context_temp_view_shadows",
        "read",
        s8,
        [
            mv(),
            sql(
                "CREATE TEMPORARY VIEW test_table AS SELECT 'z' AS region, 'p' AS product, 1000 AS count, 1.0D AS price"
            ),
            sql("SELECT measure(count_sum), measure(price_avg) FROM mv"),
            sql("SELECT region, measure(count_sum) FROM mv GROUP BY region"),
        ],
    )
    case(
        "read_context_temp_view_wrong_type",
        "read",
        s8,
        [
            mv(),
            sql(
                "CREATE TEMPORARY VIEW test_table AS SELECT 'z' AS region, 'p' AS product, 'x' AS count, 1.0D AS price"
            ),
            sql("SELECT measure(count_sum) FROM mv"),
        ],
    )
    case(
        "read_context_source_dropped",
        "read",
        s8,
        [mv(), sql("DROP TABLE test_table"), sql("SELECT measure(count_sum) FROM mv"), sql("DESCRIBE mv")],
    )
    case(
        "read_context_source_column_dropped",
        "read",
        s8,
        [
            mv(),
            sql("DROP TABLE test_table"),
            sql("CREATE TABLE test_table (region STRING, product STRING, price DOUBLE) USING parquet"),
            sql("SELECT measure(price_avg) FROM mv"),
        ],
    )
    case(
        "read_context_source_type_changed",
        "read",
        s8,
        [
            mv(),
            sql("DROP TABLE test_table"),
            sql("CREATE TABLE test_table (region STRING, product STRING, count BIGINT, price DOUBLE) USING parquet"),
            sql("INSERT INTO test_table VALUES ('a', 'b', 5, 1.0)"),
            sql("SELECT measure(count_sum) FROM mv"),
            sql("DESCRIBE mv"),
        ],
    )
    case(
        "read_context_inside_permanent_view",
        "read",
        "4 placement (R4)",
        [
            mv(),
            sql("CREATE VIEW v AS SELECT region, measure(count_sum) AS c FROM mv GROUP BY region"),
            sql("SELECT * FROM v ORDER BY region"),
            sql("CREATE DATABASE mv_read_context_inside_permanent_view_b"),
            sql("USE mv_read_context_inside_permanent_view_b"),
            sql("CREATE TABLE test_table (region STRING, product STRING, count INT, price DOUBLE) USING parquet"),
            sql("INSERT INTO test_table VALUES ('z', 'p', 1000, 1.0)"),
            sql("SELECT * FROM mv_read_context_inside_permanent_view.v ORDER BY region"),
            sql("CREATE TEMPORARY VIEW test_table AS SELECT 'y' AS region, 'p' AS product, 7 AS count, 1.0D AS price"),
            sql("SELECT * FROM mv_read_context_inside_permanent_view.v ORDER BY region"),
            sql("USE mv_read_context_inside_permanent_view"),
            sql("SELECT * FROM v ORDER BY region"),
            sql("SELECT measure(count_sum) FROM mv"),
        ],
    )
    big = [
        sql("CREATE TABLE big (k STRING, b BIGINT, i INT) USING parquet", True),
        sql("INSERT INTO big VALUES ('a', 9223372036854775807, 2147483647), ('a', 1, 1), ('b', 0, 0)", True),
    ]
    big_mv = yaml(source="big", dims=[("k", "k")], meas=[("s", "sum(b)"), ("si", "sum(i)"), ("plus", "max(i) + 1")])
    case(
        "read_ansi_overflow_reader_conf",
        "read",
        "4.8 ANSI is the reader's",
        [
            *big,
            conf("spark.sql.ansi.enabled", "true"),
            mv(big_mv),
            sql("SELECT k, measure(s), measure(si), measure(plus) FROM mv GROUP BY k ORDER BY k"),
            conf("spark.sql.ansi.enabled", "false"),
            sql("SELECT k, measure(s), measure(si), measure(plus) FROM mv GROUP BY k ORDER BY k"),
            sql("SELECT measure(s) FROM mv"),
        ],
        tables=False,
    )
    case(
        "read_ansi_created_off_read_on",
        "read",
        "4.8 ANSI is the reader's",
        [
            *big,
            conf("spark.sql.ansi.enabled", "false"),
            mv(big_mv),
            conf("spark.sql.ansi.enabled", "true"),
            sql("SELECT k, measure(s) FROM mv GROUP BY k ORDER BY k"),
            sql("SELECT k, measure(plus) FROM mv GROUP BY k ORDER BY k"),
        ],
        tables=False,
    )
    zero = [
        sql("CREATE TABLE z (k STRING, d INT) USING parquet", True),
        sql("INSERT INTO z VALUES ('a', 0), ('a', 2), ('b', NULL)", True),
    ]
    case(
        "read_ansi_divide_by_zero_dimension",
        "read",
        "coverage: ANSI, conditional evaluation",
        [
            *zero,
            mv(
                yaml(
                    source="z",
                    dims=[
                        ("k", "k"),
                        ("q", "10 / d"),
                        ("safe", "CASE WHEN d = 0 THEN -1 ELSE 10 / d END"),
                        ("c", "coalesce(10 / nullif(d, 0), -2)"),
                    ],
                    meas=[("n", "count(1)")],
                )
            ),
            sql("SELECT safe, c, measure(n) FROM mv GROUP BY safe, c ORDER BY safe, c"),
            sql("SELECT q, measure(n) FROM mv GROUP BY q ORDER BY q"),
            conf("spark.sql.ansi.enabled", "false"),
            sql("SELECT q, measure(n) FROM mv GROUP BY q ORDER BY q NULLS FIRST"),
        ],
        tables=False,
    )
    case(
        "read_ansi_cast_in_dimension",
        "read",
        "coverage: ANSI, a malformed cast input",
        [
            sql("CREATE TABLE c (s STRING) USING parquet", True),
            sql("INSERT INTO c VALUES ('x'), (NULL)", True),
            mv(yaml(source="c", dims=[("v", "CAST(s AS INT)")], meas=[("n", "count(1)")])),
            conf("spark.sql.ansi.enabled", "false"),
            sql("SELECT v, measure(n) FROM mv GROUP BY v ORDER BY v NULLS FIRST"),
            conf("spark.sql.ansi.enabled", "true"),
            sql("SELECT v, measure(n) FROM mv GROUP BY v ORDER BY v NULLS FIRST"),
        ],
        tables=False,
    )
    case(
        "read_ansi_cast_ascii_space_in_dimension",
        "read",
        "coverage: ANSI, ASCII whitespace in casts",
        [
            sql("CREATE TABLE c (s STRING) USING parquet", True),
            sql("INSERT INTO c VALUES (' 12 '), (NULL)", True),
            mv(yaml(source="c", dims=[("v", "CAST(s AS INT)")], meas=[("n", "count(1)")])),
            conf("spark.sql.ansi.enabled", "false"),
            sql("SELECT v, measure(n) FROM mv GROUP BY v ORDER BY v NULLS FIRST"),
            conf("spark.sql.ansi.enabled", "true"),
            sql("SELECT v, measure(n) FROM mv GROUP BY v ORDER BY v NULLS FIRST"),
        ],
        tables=False,
    )
    case(
        "read_ansi_cast_nbsp_in_dimension",
        "read",
        "coverage: ANSI, a no-break space in casts",
        [
            sql("CREATE TABLE c (s STRING) USING parquet", True),
            sql("INSERT INTO c VALUES ('\\u00a012'), (NULL)", True),
            mv(yaml(source="c", dims=[("v", "CAST(s AS INT)")], meas=[("n", "count(1)")])),
            conf("spark.sql.ansi.enabled", "false"),
            sql("SELECT v, measure(n) FROM mv GROUP BY v ORDER BY v NULLS FIRST"),
            conf("spark.sql.ansi.enabled", "true"),
            sql("SELECT v, measure(n) FROM mv GROUP BY v ORDER BY v NULLS FIRST"),
        ],
        tables=False,
    )
    case(
        "read_nulls_nan_signed_zero",
        "read",
        "coverage: NULL, NaN, -0.0",
        [
            sql("CREATE TABLE f (k STRING, x DOUBLE) USING parquet", True),
            sql(
                "INSERT INTO f VALUES ('a', 1.0), ('a', CAST('NaN' AS DOUBLE)), (NULL, 2.0), (NULL, NULL), "
                "('b', -0.0), ('b', 0.0), ('c', CAST('Infinity' AS DOUBLE)), ('c', CAST('-Infinity' AS DOUBLE))",
                True,
            ),
            mv(
                yaml(
                    source="f",
                    dims=[("k", "k"), ("x", "x")],
                    meas=[("s", "sum(x)"), ("a", "avg(x)"), ("mx", "max(x)"), ("n", "count(x)")],
                )
            ),
            sql("SELECT k, measure(s), measure(a), measure(mx), measure(n) FROM mv GROUP BY k ORDER BY k NULLS FIRST"),
            sql("SELECT x, measure(n) FROM mv GROUP BY x ORDER BY x NULLS FIRST"),
        ],
        tables=False,
    )
    case(
        "read_empty_source",
        "read",
        "coverage: empty input",
        [
            mv(yaml(filt="1 = 0")),
            sql("SELECT measure(count_sum), measure(price_avg) FROM mv"),
            sql("SELECT region, measure(count_sum) FROM mv GROUP BY region"),
        ],
    )
    case(
        "read_decimal_measure",
        "read",
        "coverage: decimal precision",
        [
            sql("CREATE TABLE d (k STRING, v DECIMAL(38, 10)) USING parquet", True),
            sql(
                "INSERT INTO d VALUES ('a', 9999999999999999999999999999.9999999999), "
                "('a', 9999999999999999999999999999.9999999999), ('b', 1.5)",
                True,
            ),
            mv(yaml(source="d", dims=[("k", "k")], meas=[("s", "sum(v)"), ("a", "avg(v)")])),
            sql("DESCRIBE mv"),
            conf("spark.sql.ansi.enabled", "false"),
            sql("SELECT k, measure(s), measure(a) FROM mv GROUP BY k ORDER BY k"),
            conf("spark.sql.ansi.enabled", "true"),
            sql("SELECT k, measure(s) FROM mv GROUP BY k ORDER BY k"),
        ],
        tables=False,
    )
    case(
        "read_session_timezone",
        "read",
        "coverage: session timezone is the reader's",
        [
            sql("CREATE TABLE ts (t TIMESTAMP) USING parquet", True),
            sql("INSERT INTO ts VALUES (TIMESTAMP'2024-03-10 09:30:00'), (TIMESTAMP'2024-03-10 11:30:00')", True),
            conf("spark.sql.session.timeZone", "UTC"),
            mv(yaml(source="ts", dims=[("h", "hour(t)"), ("s", "CAST(t AS STRING)")], meas=[("n", "count(1)")])),
            sql("SELECT h, s, measure(n) FROM mv GROUP BY h, s ORDER BY h, s"),
            conf("spark.sql.session.timeZone", "America/Los_Angeles"),
            sql("SELECT h, s, measure(n) FROM mv GROUP BY h, s ORDER BY h, s"),
        ],
        tables=False,
    )
    case(
        "read_regex_dimension",
        "read",
        "coverage: Java regex dialect",
        [
            mv(yaml(dims=[("r", "regexp_extract(region, '(?i)^(region)_(\\\\d)$', 2)")], meas=[("c", "sum(count)")])),
            sql("SELECT r, measure(c) FROM mv GROUP BY r ORDER BY r"),
        ],
    )
    case(
        "read_binary_unicode",
        "read",
        "coverage: binary, Unicode",
        [
            sql("CREATE TABLE u (s STRING, b BINARY) USING parquet", True),
            sql("INSERT INTO u VALUES ('é', X'FF00'), ('é', X'FF00'), ('\U0001f600', X'')", True),
            mv(yaml(source="u", dims=[("s", "s"), ("len", "length(s)"), ("bh", "hex(b)")], meas=[("n", "count(1)")])),
            sql("SELECT s, len, bh, measure(n) FROM mv GROUP BY s, len, bh ORDER BY len, bh, s"),
        ],
        tables=False,
    )


def gate_cases() -> None:
    ref = "0 versions gate"
    case("gate_create_metric_view", "gate", ref, [mv()], gate=True)
    case("gate_measure_function", "gate", ref, [sql("SELECT measure(count) FROM test_table")], gate=True)


suite_cases()
parse_cases()
yaml_cases()
ddl_cases()
read_cases()
gate_cases()

# ---------------------------------------------------------------------------------------------------------------
# Probe (runs inside the container)
# ---------------------------------------------------------------------------------------------------------------


def _value(v: object) -> object:
    import datetime
    import decimal
    import math

    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return v
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        return {"hex": bytes(v).hex()}
    if isinstance(v, dict):
        return {str(k): _value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_value(x) for x in v]
    return repr(v)


def _schema(st: Any) -> list:
    out = []
    for f in st.fields:
        col = {"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable}
        if f.metadata:
            col["metadata"] = f.metadata
        out.append(col)
    return out


# A JVM's expression ids carry a per-run UUID (`ExprId(6358,a946f8fa-...)`): masked so that two runs agree.
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_EXPR_ID = re.compile(r"ExprId\(\d+,")


def _stable(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return _EXPR_ID.sub("ExprId(N,", _UUID.sub("<uuid>", text))


def _error(spark: Any, e: BaseException, phase: str) -> dict:
    rec = _error_raw(spark, e, phase)
    for k in ("message", "pyspark_conversion_error"):
        if k in rec:
            rec[k] = _stable(rec[k])
    for c in rec.get("causes", []):
        c["message"] = _stable(c["message"])
    return rec


def _error_raw(spark: Any, e: BaseException, phase: str) -> dict:
    jt = getattr(e, "_origin", None) or getattr(e, "java_exception", None)
    if jt is None and e.__context__ is not None and getattr(e.__context__, "java_exception", None) is not None:
        # PySpark failed while converting the JVM error; the Py4J error it was handling holds the JVM throwable.
        rec = _error(spark, e.__context__, phase)
        rec["pyspark_conversion_error"] = f"{type(e).__name__}: {e}"
        return rec
    if jt is None:
        return {"phase": phase, "python_class": type(e).__name__, "message": str(e)}
    jvm = spark._jvm
    rec: dict = {"phase": phase, "java_class": jt.getClass().getName(), "message": jt.getMessage()}
    if jvm.java.lang.Class.forName("org.apache.spark.SparkThrowable").isInstance(jt):
        try:
            cond = jt.getCondition()
        except Exception:
            cond = jt.getErrorClass()
        rec["condition"] = cond
        rec["sqlstate"] = jt.getSqlState()
        params = jt.getMessageParameters()
        rec["parameters"] = {k: params.get(k) for k in sorted(params.keySet())}
        ctxs = []
        for c in jt.getQueryContext():
            if c is None:
                continue
            item: dict = {}
            for attr in ("contextType", "objectType", "objectName", "startIndex", "stopIndex", "fragment"):
                try:
                    val = getattr(c, attr)()
                    item[attr] = val if isinstance(val, (int, str)) or val is None else str(val)
                except Exception:
                    pass
            ctxs.append(item)
        rec["query_context"] = ctxs
    causes = []
    c = jt.getCause()
    while c is not None and len(causes) < 6:
        msg = c.getMessage()
        causes.append({"java_class": c.getClass().getName(), "message": None if msg is None else msg[:4000]})
        c = c.getCause()
    if causes:
        rec["causes"] = causes
    return rec


def _sub_all(rows: list, pattern: str, repl: str) -> list:
    return [[re.sub(pattern, repl, x) if isinstance(x, str) else x for x in r] for r in rows]


def _normalize_rows(text: str, rows: list) -> list:
    up = text.upper()
    if up.startswith("DESCRIBE"):
        rows = [[r[0], "<normalized>", *r[2:]] if r and r[0] in ("Created Time", "Last Access") else r for r in rows]
        rows = _sub_all(rows, r'"(created_time|last_access)":"[^"]*"', r'"\1":"<normalized>"')
    if up.startswith("SHOW TABLE EXTENDED"):
        rows = _sub_all(rows, r"(Created Time|Last Access): [^\n]*", r"\1: <normalized>")
    if up.startswith("EXPLAIN"):
        rows = _sub_all(_sub_all(rows, r"#\d+L?", "#N"), r"plan_id=\d+", "plan_id=N")
        # The local warehouse root: no local path in a fixture; the table path below it stays.
        rows = _sub_all(rows, "file:" + re.escape(SPARK_CONF["spark.sql.warehouse.dir"]), "file:<warehouse>")
    return rows


def _match(outcome: dict) -> str:
    err = outcome.get("error")
    if not err:
        return "exact"
    if err.get("condition") is None or err.get("sqlstate") == "XX000":
        return "class_sqlstate_prefix"
    msg = err.get("message") or ""
    if err.get("java_class", "").endswith("ExtendedAnalysisException") and re.search(r"; line \d+ pos \d+;\n", msg):
        # getMessage() of an analysis error appends the analyzed plan (with expression ids) after the position.
        return "exact_before_plan"
    if err.get("condition") == "INVALID_METRIC_VIEW_YAML" and "Failed to parse YAML:" in msg:
        jackson = ("Unrecognized field", "Missing required creator property", "Cannot deserialize", "Cannot construct")
        if not any(j in msg for j in jackson):
            return "class_sqlstate_prefix"
    return "exact"


def _collect(df: Any, text: str) -> dict:
    rows = [[_value(x) for x in r] for r in df.collect()]
    rows = _normalize_rows(text, rows)
    up = text.lstrip().upper()
    # Query results are a bag unless ordered; command output (DESCRIBE, SHOW, EXPLAIN) keeps Spark's order.
    is_query = text == "" or up.startswith(("SELECT", "WITH", "(", "VALUES", "FROM", "TABLE"))
    ordered = not is_query or "ORDER BY" in up
    if not ordered:
        rows = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))
    return {"rows": rows, "row_order": "spark" if ordered else "sorted"}


def _run_sql(s: Any, text: str) -> dict:
    try:
        df = s.sql(text)
    except Exception as e:
        return {"error": _error(s, e, "sql")}
    out: dict = {"schema": _schema(df.schema)}
    try:
        out.update(_collect(df, text))
    except Exception as e:
        out["error"] = _error(s, e, "collect")
    return out


def _run_api(s: Any, op: str, args: dict) -> dict:
    out: dict = {}
    try:
        if op == "table_schema":
            df = s.table(args["name"])
            out["schema"] = _schema(df.schema)
            out["columns"] = list(df.columns)
        elif op == "table_collect":
            df = s.table(args["name"])
            out["schema"] = _schema(df.schema)
            out.update(_collect(df, ""))
        elif op == "union_self":
            df = s.sql(args["sql"])
            u = df.union(df)
            out["schema"] = _schema(u.schema)
            out.update(_collect(u, ""))
            out["count"] = u.count()
            out["single_count"] = df.count()
            out["distinct_rows"] = _collect(u.distinct(), "")["rows"]
        elif op == "df_measure":
            from pyspark.sql import functions as fn

            df = s.table(args["name"]).groupBy(args["group"]).agg(fn.expr(args["expr"]))
            out["schema"] = _schema(df.schema)
            out.update(_collect(df, ""))
        elif op == "list_tables":
            db = s.catalog.currentDatabase()
            ts = s.catalog.listTables(db)
            out["rows"] = sorted([[t.name, t.tableType, t.isTemporary] for t in ts])
        elif op == "via_temp":
            s.sql(args["first"]).createOrReplaceTempView("_r")
            return _run_sql(s, args["then"])
        else:
            raise ValueError(op)
    except Exception as e:
        out["error"] = _error(s, e, "api")
    return out


def probe(version: str, gate_only: bool) -> None:
    from pyspark.sql import SparkSession

    base = SparkSession.builder.getOrCreate()
    base.sparkContext.setLogLevel("ERROR")
    results = []
    for c in CASES:
        if gate_only and not c["gate"]:
            continue
        s = base.newSession()
        db = "mv_" + c["name"]
        s.sql(f"CREATE DATABASE {db}")
        s.sql(f"USE {db}")
        steps = []
        for step in c["steps"]:
            rec = dict(step)
            if step["kind"] == "sql":
                rec["outcome"] = _run_sql(s, step["sql"])
            elif step["kind"] == "conf":
                s.conf.set(step["key"], step["value"])
                rec["outcome"] = {}
            else:
                rec["outcome"] = _run_api(s, step["op"], step["args"])
            rec["match"] = _match(rec["outcome"])
            if step.get("setup") and rec["outcome"].get("error"):
                # A failed setup step leaves the probes without their data: abort this release's capture.
                raise SystemExit(f"{version}: setup step of {c['name']} failed: {rec['outcome']['error']}")
            steps.append(rec)
        results.append(
            {
                "name": c["name"],
                "group": c["group"],
                "note_ref": c["note_ref"],
                "database": db,
                "target": "compatible",
                "steps": steps,
            }
        )
        try:
            s.sql("USE default")
            for (name,) in s.sql("SHOW DATABASES").collect():
                if name == db or name.startswith(db + "_"):
                    s.sql(f"DROP DATABASE {name} CASCADE")
        except Exception as e:
            print(f"cleanup {db}: {e}", file=sys.stderr)
    print("@@JSON " + json.dumps({"spark_version": base.version, "cases": results}, sort_keys=False))
    base.stop()


# ---------------------------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------------------------


def run_release(version: str) -> dict:
    name, image, _, gate_only = RUNS[version]
    cmd = ["docker", "run", "--rm", "--name", name, "--entrypoint", "", "-e", "HOME=/tmp"]
    cmd += ["-v", f"{SELF}:/probe/gen_metric_view_goldens.py:ro", image, "/opt/spark/bin/spark-submit"]
    cmd += ["--master", "local[2]"]
    for k, v in SPARK_CONF.items():
        cmd += ["--conf", f"{k}={v}"]
    cmd += ["/probe/gen_metric_view_goldens.py", "--probe", version] + (["--gate-only"] if gate_only else [])
    done = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3600)
    for line in done.stdout.splitlines():
        if line.startswith("@@JSON "):
            return json.loads(line[len("@@JSON ") :])
    raise SystemExit(f"Spark {version}: no output (exit {done.returncode}):\n{done.stderr[-3000:]}")


def dump_golden(doc: dict) -> str:
    """The golden as JSON with one case per line, so a review diff shows changed cases, not 20,000 lines."""
    head = {k: v for k, v in doc.items() if k != "cases"}
    lines = [json.dumps(head, ensure_ascii=True)[:-1] + ', "cases": [']
    cases = [json.dumps(c, ensure_ascii=True) for c in doc["cases"]]
    lines += [c + ("," if i + 1 < len(cases) else "") for i, c in enumerate(cases)]
    lines.append("]}")
    return "\n".join(lines) + "\n"


def _fixture(fixture_dir: str, gate_only: bool) -> Path:
    return ROOT / "tests/fixtures" / fixture_dir / "metric-views" / ("gate.json" if gate_only else "cases.json")


def _write(out: Path, doc: dict) -> None:
    for c in doc["cases"]:
        for step in c["steps"]:
            step["match"] = _match(step["outcome"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_golden(doc), encoding="utf-8")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--probe":
        probe(args[1], "--gate-only" in args)
        return
    if args and args[0] == "--rematch":
        # Recompute only the `match` field of existing fixtures (no Spark run).
        for _, _, fixture_dir, gate_only in RUNS.values():
            out = _fixture(fixture_dir, gate_only)
            _write(out, json.loads(out.read_text(encoding="utf-8")))
        return
    only = None
    if args and args[0] == "--only":
        only = set(args[1].split(","))
    for version, (_, image, fixture_dir, gate_only) in RUNS.items():
        if only and version not in only:
            continue
        data = run_release(version)
        if data["spark_version"] != version:
            raise SystemExit(f"image {image} reports Spark {data['spark_version']}, expected {version}")
        doc = {
            "format_version": FORMAT_VERSION,
            "generator": "tests/scripts/gen_metric_view_goldens.py",
            "spark_version": version,
            "image": image,
            "session": "classic PySpark, local[2], in-memory session catalog",
            "spark_conf": SPARK_CONF,
            "gate_only": gate_only,
            "cases": data["cases"],
        }
        out = _fixture(fixture_dir, gate_only)
        _write(out, doc)
        n_steps = sum(len(c["steps"]) for c in data["cases"])
        print(
            f"Spark {version}: {len(data['cases'])} cases, {n_steps} steps -> {out.relative_to(ROOT)}", file=sys.stderr
        )


if __name__ == "__main__":
    main()
