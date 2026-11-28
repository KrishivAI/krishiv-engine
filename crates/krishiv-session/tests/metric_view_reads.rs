// SPDX-License-Identifier: Apache-2.0
//! Reads of a metric view beyond the Spark goldens (`metric_views.rs`): the reader is asked about
//! the source, what a dimension or measure may not call yet, `measure(..)` of a column that is not
//! a measure, one view read twice, and the leftover refusals (design note `P3-metric-views.md` §4,
//! review 2026-11-20).
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::BoxFuture;
use krishiv_session::{
    AccessDecision, AccessPolicy, AllowAll, Principal, PrincipalKind, Privilege, Securable, Session,
};

const VIEW: &str = "CREATE VIEW mv WITH METRICS LANGUAGE YAML AS $$
version: 0.1
source: src
dimensions:
  - name: region
    expr: region
measures:
  - name: total
    expr: sum(n)
$$";

fn session_with(policy: Arc<dyn AccessPolicy>) -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(policy)
        .config("spark.sql.warehouse.dir", dir.path().display().to_string())
        .build()
        .unwrap();
    for sql in [
        "CREATE TABLE src (region STRING, n INT) USING parquet",
        "INSERT INTO src VALUES ('a', 1), ('a', 2), ('b', 5)",
        VIEW,
    ] {
        run(&s, sql).unwrap();
    }
    (s, dir)
}

fn session() -> (Session, tempfile::TempDir) {
    session_with(Arc::new(AllowAll))
}

fn run(s: &Session, sql: &str) -> Result<Vec<RecordBatch>, KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await })
}

fn longs(batches: &[RecordBatch], column: usize) -> Vec<i64> {
    let mut out = longs_unsorted(batches, column);
    out.sort_unstable();
    out
}

/// A column's values in row order.
fn longs_unsorted(batches: &[RecordBatch], column: usize) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(column)
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

fn class(e: &KrishivError) -> String {
    e.error_class().name().to_owned()
}

/// Denies `SELECT` on the table `src` once `armed`; everything else is allowed.
struct DenySource;

impl AccessPolicy for DenySource {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let denied = privilege == Privilege::Select
            && matches!(securable, Securable::Table(t) if t.name == "src");
        Box::pin(std::future::ready(Ok(if denied {
            AccessDecision::Deny {
                reason: "private".to_owned(),
            }
        } else {
            AccessDecision::Allow
        })))
    }
}

#[test]
fn the_reader_is_asked_about_the_source() {
    // The view is created where the source may be read, then read by a reader who may not read it:
    // a metric view's source is the reader's read (no definer rights).
    let (s, _dir) = session();
    let reader = s.new_session_as(
        Principal::new("bo", PrincipalKind::User),
        Arc::new(DenySource),
    );
    let e = run(
        &reader,
        "SELECT region, measure(total) FROM mv GROUP BY region",
    )
    .unwrap_err();
    assert_eq!(class(&e), "KRISHIV_INSUFFICIENT_PERMISSIONS", "{e}");
    let e = run(&reader, "SELECT * FROM mv").unwrap_err();
    assert_eq!(class(&e), "KRISHIV_INSUFFICIENT_PERMISSIONS", "{e}");
}

#[test]
fn a_measure_with_a_subquery_or_a_sql_function_is_refused_at_read() {
    let (s, _dir) = session();
    run(&s, "CREATE TABLE other (m INT) USING parquet").unwrap();
    run(
        &s,
        "CREATE FUNCTION plus_one(x INT) RETURNS INT RETURN x + 1",
    )
    .unwrap();
    for (name, measure) in [
        ("with_subquery", "sum(n) + (SELECT max(m) FROM other)"),
        ("with_function", "sum(plus_one(n))"),
    ] {
        let create = format!(
            "CREATE VIEW {name} WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: src\n\
             dimensions:\n  - name: region\n    expr: region\nmeasures:\n  - name: m\n    expr: {measure}\n$$"
        );
        run(&s, &create).unwrap();
        let e = run(&s, &format!("SELECT measure(m) FROM {name}")).unwrap_err();
        assert_eq!(class(&e), "KRISHIV_UNSUPPORTED_FEATURE", "{name}: {e}");
    }
}

#[test]
fn measure_of_a_column_that_is_not_a_measure_is_sparks_internal_error() {
    let (s, _dir) = session();
    // A projection's alias, even one spelled like a measure, is not looked up by name.
    for sql in [
        "SELECT measure(x) FROM (SELECT region AS x FROM mv)",
        "SELECT measure(total) FROM (SELECT region AS total FROM mv)",
        "SELECT measure(region) FROM mv",
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(class(&e), "INTERNAL_ERROR", "{sql}: {e}");
        assert!(e.to_string().contains("key not found"), "{sql}: {e}");
    }
}

#[test]
fn one_view_read_twice_gives_each_read_its_own_columns() {
    let (s, _dir) = session();
    let rows = run(
        &s,
        "SELECT a.t, b.t FROM (SELECT region, measure(total) AS t FROM mv GROUP BY region) a \
         JOIN (SELECT region, measure(total) AS t FROM mv GROUP BY region) b ON a.region = b.region",
    )
    .unwrap();
    // Pairs, not columns sorted apart: each row joins a region's total to its own.
    let (left, right) = (longs_unsorted(&rows, 0), longs_unsorted(&rows, 1));
    let mut pairs: Vec<(i64, i64)> = left.into_iter().zip(right).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, [(3, 3), (5, 5)]);
    let rows = run(
        &s,
        "SELECT measure(total) FROM mv UNION ALL SELECT measure(total) FROM mv",
    )
    .unwrap();
    assert_eq!(longs(&rows, 0), [8, 8]);
}

#[test]
fn a_read_left_without_an_aggregate_is_analyzable_but_not_plannable() {
    let (s, _dir) = session();
    let df = block_on(s.sql("SELECT * FROM mv")).unwrap();
    // Analysis succeeds (the schema is the stored one) ...
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["region", "total"]);
    // ... and planning refuses it, as Spark's planner does.
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(class(&e), "INTERNAL_ERROR");
    assert!(e.to_string().contains("phase planning failed"), "{e}");
    // A `measure(..)` left over is refused first, with its own text.
    let e = run(&s, "SELECT measure(total) FROM mv JOIN src ON true").unwrap_err();
    assert!(
        e.to_string()
            .contains("UnevaluableAggregateFunc.aggBufferAttributes"),
        "{e}"
    );
}

/// The create aggregate groups by the dimensions' ordinals whatever `spark.sql.groupByOrdinal`
/// says: with it off a create still groups, and a dimension that is an aggregate is still left
/// out of the grouping (Spark groups by the dimensions' attributes).
#[test]
fn a_create_groups_by_its_dimensions_with_ordinals_off() {
    let (s, _dir) = session();
    s.conf().set("spark.sql.groupByOrdinal", "false").unwrap();
    run(
        &s,
        "CREATE VIEW mv2 WITH METRICS LANGUAGE YAML AS $$
version: 0.1
source: src
dimensions:
  - name: region
    expr: upper(region)
  - name: everything
    expr: sum(n)
measures:
  - name: total
    expr: sum(n)
$$",
    )
    .unwrap();
    let err = run(
        &s,
        "SELECT region, measure(total) FROM mv2 GROUP BY region ORDER BY region",
    )
    .unwrap_err();
    // The aggregate dimension makes every read Spark's `MISSING_GROUP_BY`, as the golden
    // `yaml_dimension_is_aggregate` shows; the create itself succeeded above.
    assert_eq!(class(&err), "MISSING_GROUP_BY");
    let rows = run(
        &s,
        "SELECT region, measure(total) FROM mv GROUP BY region ORDER BY region",
    )
    .unwrap();
    assert_eq!(longs(&rows, 1), vec![3, 5]);
}

/// A metric view's operators keep the creator's ANSI mode, the functions under them the reader's
/// (a Spark 4.2.1 run, 2026-11-21: created under ANSI, read without it, `abs(i) + 0` of
/// `INT_MIN` is `INT_MIN`, `-i` overflows).
#[test]
fn an_operator_keeps_the_creators_mode_and_its_operand_the_readers() {
    let (s, _dir) = session();
    run(
        &s,
        "CREATE TABLE t (k STRING, i INT, l BIGINT) USING parquet",
    )
    .unwrap();
    run(
        &s,
        "INSERT INTO t VALUES ('a', -2147483648, -9223372036854775808), ('b', 1, 1)",
    )
    .unwrap();
    s.conf().set("spark.sql.ansi.enabled", "true").unwrap();
    run(
        &s,
        "CREATE VIEW mvo WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: t\ndimensions:\n  - name: a\n    expr: abs(i) + 0\n  - name: n\n    expr: -i\n  - name: k\n    expr: k\nmeasures:\n  - name: c\n    expr: count(1)\n  - name: neg\n    expr: -min(i)\n  - name: negl\n    expr: -min(l)\n$$",
    )
    .unwrap();
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    let got = run(
        &s,
        "SELECT CAST(a AS BIGINT), measure(c) FROM mvo GROUP BY a",
    )
    .unwrap();
    assert_eq!(longs(&got, 0), vec![-2_147_483_648, 1]);
    // The creator's mode also where Spark's rule order adds the measure from the clause
    // (review 60b9e05fb finding 1; each one `ARITHMETIC_OVERFLOW` on Spark 4.2.1).
    for q in [
        "SELECT n, measure(c) FROM mvo GROUP BY n",
        "SELECT n FROM mvo GROUP BY n ORDER BY measure(c)",
        "SELECT k, measure(neg) FROM mvo GROUP BY k",
        "SELECT k FROM mvo GROUP BY k ORDER BY measure(neg)",
        "SELECT k FROM mvo GROUP BY k HAVING measure(negl) > 0L",
        // A sort on a grouping key the output drops rebuilds the aggregate (review cL9v).
        "SELECT measure(negl) FROM mvo GROUP BY k ORDER BY k",
        "SELECT measure(neg) FROM mvo GROUP BY k HAVING k = 'a'",
    ] {
        let e = run(&s, q).unwrap_err();
        assert_eq!(class(&e), "ARITHMETIC_OVERFLOW", "{q}");
    }
}

/// A dimension or measure ending in a `--` comment (each parses alone, as Spark parses it) ends
/// there: the create aggregate keeps its alias and its source (review b047d8901 finding 1).
#[test]
fn an_expression_ending_in_a_comment_keeps_its_name_and_source() {
    let (s, _dir) = session();
    let yaml = |source: &str| {
        format!(
            "CREATE VIEW mv3 WITH METRICS LANGUAGE YAML AS $$
version: 0.1
source: {source}
dimensions:
  - name: r
    expr: upper(region) -- the region
measures:
  - name: rows
    expr: count(1) -- every row
$$"
        )
    };
    let err = run(&s, &yaml("no_such_table")).unwrap_err();
    assert_eq!(class(&err), "TABLE_OR_VIEW_NOT_FOUND");
    run(&s, &yaml("src")).unwrap();
    let rows = run(&s, "SELECT r, measure(rows) FROM mv3 GROUP BY r ORDER BY r").unwrap();
    let schema = rows[0].schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["r", "measure(rows)"]);
    assert_eq!(longs(&rows, 1), vec![1, 2]);
}

/// A release before 4.2 has no metric views: `EXECUTE IMMEDIATE` and a script statement fail to
/// parse them as a top-level statement does (review b047d8901 finding 2).
#[test]
fn a_metric_view_inside_execute_immediate_or_a_script_is_refused_before_4_2() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .profile(krishiv_plan::analyzer::SparkProfile::V4_1)
        .config("spark.sql.warehouse.dir", dir.path().display().to_string())
        .build()
        .unwrap();
    run(&s, "CREATE TABLE src (region STRING, n INT) USING parquet").unwrap();
    let body = VIEW.replacen("CREATE VIEW mv", "CREATE VIEW mv41", 1);
    let quoted = body.replace('\'', "''");
    for sql in [
        body.clone(),
        format!("EXECUTE IMMEDIATE '{quoted}'"),
        format!("BEGIN\n{body};\nEND"),
    ] {
        let err = run(&s, &sql).unwrap_err();
        assert_eq!(class(&err), "PARSE_SYNTAX_ERROR", "{sql}");
    }
    let err = run(&s, "SELECT * FROM mv41").unwrap_err();
    assert_eq!(class(&err), "TABLE_OR_VIEW_NOT_FOUND");
}

/// A dimension is Spark's `UpCast` to its stored type: a source column widened after the create
/// fails the read at analysis, whatever the ANSI mode; one narrowed still reads (a Spark 4.2.1
/// run, 2026-11-21; review b047d8901 finding 3).
#[test]
fn a_source_column_widened_after_the_create_fails_the_read_as_an_up_cast() {
    let (s, _dir) = session();
    run(&s, "CREATE TABLE w (k INT, n INT) USING parquet").unwrap();
    run(
        &s,
        "CREATE VIEW mvw WITH METRICS LANGUAGE YAML AS $$
version: 0.1
source: w
dimensions:
  - name: k
    expr: k
  - name: k1
    expr: k + 1
measures:
  - name: c
    expr: count(1)
$$",
    )
    .unwrap();
    // Spark 4.2.1 gives `CANNOT_UP_CAST_DATATYPE` for every one, in both ANSI modes, also where
    // Krishiv's cast refuses the types outright (review c856eb6ee finding 1).
    for (ty, value) in [
        ("BIGINT", "5000000000"),
        ("STRING", "'x'"),
        ("ARRAY<INT>", "array(1)"),
        ("DATE", "DATE'2020-01-01'"),
        ("BOOLEAN", "true"),
    ] {
        run(&s, "DROP TABLE w").unwrap();
        run(&s, &format!("CREATE TABLE w (k {ty}, n INT) USING parquet")).unwrap();
        run(&s, &format!("INSERT INTO w VALUES ({value}, 1)")).unwrap();
        for ansi in ["true", "false"] {
            s.conf().set("spark.sql.ansi.enabled", ansi).unwrap();
            for q in [
                "SELECT k, measure(c) FROM mvw GROUP BY k",
                "SELECT k1, measure(c) FROM mvw GROUP BY k1",
                "SELECT measure(c) FROM mvw",
            ] {
                let err = run(&s, q).unwrap_err();
                assert_eq!(class(&err), "CANNOT_UP_CAST_DATATYPE", "{ty} {ansi} {q}");
                let params = err.message_parameters();
                let p = |k: &str| {
                    params
                        .iter()
                        .find(|(n, _)| *n == k)
                        .map(|(_, v)| v.as_str())
                };
                assert_eq!(p("sourceType"), Some(format!("\"{ty}\"").as_str()));
                assert_eq!(p("targetType"), Some("\"INT\""));
            }
        }
    }
    run(&s, "DROP TABLE w").unwrap();
    run(&s, "CREATE TABLE w (k SMALLINT, n INT) USING parquet").unwrap();
    run(&s, "INSERT INTO w VALUES (3, 1)").unwrap();
    let rows = run(&s, "SELECT k, measure(c) FROM mvw GROUP BY k").unwrap();
    assert_eq!(longs(&rows, 1), vec![1]);
}

/// Spark's rule never looks through a temporary view (a `View` node) or a CTE reference: the
/// `measure(..)` stays, and the run is Spark's internal error (a Spark 4.2.1 run, 2026-11-21:
/// SQL, DataFrame and global temporary views, and `WITH`; review b047d8901 finding 4).
#[test]
fn a_read_through_a_temporary_view_or_a_cte_is_sparks_internal_error() {
    let (s, _dir) = session();
    run(&s, "CREATE TEMP VIEW tv AS SELECT * FROM mv").unwrap();
    run(&s, "CREATE GLOBAL TEMP VIEW gv AS SELECT * FROM mv").unwrap();
    block_on(s.table("mv").create_or_replace_temp_view("tv2")).unwrap();
    for q in [
        "SELECT region, measure(total) FROM tv GROUP BY region",
        "SELECT measure(total) FROM tv",
        "SELECT region, measure(total) FROM tv2 GROUP BY region",
        "SELECT region, measure(total) FROM global_temp.gv GROUP BY region",
        "WITH w AS (SELECT * FROM mv) SELECT region, measure(total) FROM w GROUP BY region",
    ] {
        let err = run(&s, q).unwrap_err();
        assert_eq!(class(&err), "INTERNAL_ERROR", "{q}");
        assert!(
            err.to_string()
                .contains("UnevaluableAggregateFunc.aggBufferAttributes should not be called"),
            "{q}: {err}"
        );
    }
}

/// `saveAsTable` onto a metric view (a Spark 4.2.1 run, 2026-11-21): `append` is "Saving data
/// into a view is not allowed", `overwrite` replaces it with a managed table, `errorifexists` is
/// `TABLE_OR_VIEW_ALREADY_EXISTS` and `ignore` leaves it (review b047d8901 nit 1). `append` onto
/// a plain permanent view is the same refusal.
#[test]
fn save_as_table_onto_a_metric_view_follows_sparks_modes() {
    let type_of = |s: &Session| -> String {
        let text = |c: &dyn Array, i: usize| match c.as_string_opt::<i32>() {
            Some(a) => a.value(i).to_owned(),
            None => c.as_string_view().value(i).to_owned(),
        };
        let rows = run(s, "DESCRIBE EXTENDED mv").unwrap();
        rows.iter()
            .find_map(|b| {
                (0..b.num_rows())
                    .find(|&i| text(b.column(0), i) == "Type")
                    .map(|i| text(b.column(1), i))
            })
            .unwrap_or_default()
    };
    for (mode, want) in [
        ("append", Err("_LEGACY_ERROR_TEMP_1158")),
        ("overwrite", Ok("MANAGED")),
        ("errorifexists", Err("TABLE_OR_VIEW_ALREADY_EXISTS")),
        ("ignore", Ok("METRIC_VIEW")),
    ] {
        let (s, _dir) = session();
        let got = block_on(s.sql("SELECT 1L AS id"))
            .and_then(|df| block_on(df.write().mode(mode).save_as_table("mv")));
        match want {
            Ok(ty) => {
                got.unwrap_or_else(|e| panic!("{mode}: {e}"));
                assert_eq!(type_of(&s), ty, "{mode}");
            }
            Err(c) => {
                assert_eq!(class(&got.unwrap_err()), c, "{mode}");
                assert_eq!(type_of(&s), "METRIC_VIEW", "{mode}");
            }
        }
    }
}

#[test]
fn save_as_table_append_onto_a_permanent_view_is_refused_as_spark_does() {
    let (s, _dir) = session();
    run(&s, "CREATE VIEW pv AS SELECT 1L AS id").unwrap();
    let got = block_on(s.sql("SELECT 2L AS id"))
        .and_then(|df| block_on(df.write().mode("append").save_as_table("pv")));
    assert_eq!(class(&got.unwrap_err()), "_LEGACY_ERROR_TEMP_1158");
}

/// Spark rewrites a metric-view aggregate only once it holds an aggregate, so an `ORDER BY` or
/// `HAVING` over one whose select list holds none still reads the measures; one needing an
/// implicit cast resolves after the rewrite and misses them. Every case probed on Spark 4.2.1.
#[test]
fn order_by_and_having_over_an_aggregate_with_no_measure_follow_sparks_rule_order() {
    let (s, _dir) = session();
    run(&s, "INSERT INTO src VALUES ('c', 4)").unwrap();
    let regions = |sql: &str| -> Vec<String> {
        run(&s, sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .iter()
            .flat_map(|b| {
                let c = b.column(0);
                (0..c.len())
                    .map(|i| arrow_cast::display::array_value_to_string(c, i).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    // Totals: a 3, b 5, c 4.
    for sql in [
        "SELECT region FROM mv GROUP BY region ORDER BY measure(total) DESC",
        "SELECT region AS r, 'x' AS c FROM mv GROUP BY region ORDER BY -measure(total)",
        "SELECT region FROM mv GROUP BY region ORDER BY CAST(measure(total) AS INT) DESC",
        "SELECT region FROM mv GROUP BY region ORDER BY measure(total) + measure(total) DESC",
        "SELECT region FROM mv GROUP BY ALL ORDER BY measure(total) DESC",
    ] {
        assert_eq!(regions(sql), ["b", "c", "a"], "{sql}");
    }
    assert_eq!(
        regions("SELECT region FROM mv GROUP BY region HAVING measure(total) > 3L ORDER BY region"),
        ["b", "c"]
    );
    // A subquery's own casts are not the clause's: Spark 4.2.1 reads it (review 60b9e05fb nit 1).
    assert_eq!(
        regions(
            "SELECT region FROM mv GROUP BY region \
             HAVING measure(total) > (SELECT max(CAST(n AS BIGINT)) FROM src WHERE n < 5L) \
             ORDER BY region"
        ),
        ["b"]
    );
    // An implicit cast anywhere in the clause: the measure is missing from the rewritten input.
    for (sql, missing) in [
        (
            "SELECT region FROM mv GROUP BY region ORDER BY measure(total) + 1",
            "\"total\"",
        ),
        (
            "SELECT region FROM mv GROUP BY region HAVING measure(total) > 3",
            "\"total\"",
        ),
        (
            "SELECT region FROM mv GROUP BY region ORDER BY concat(region, 1), measure(total)",
            "\"total\"",
        ),
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(
            class(&e),
            "MISSING_ATTRIBUTES.RESOLVED_ATTRIBUTE_MISSING_FROM_INPUT",
            "{sql}: {e}"
        );
        let text = e.to_string();
        assert!(
            text.contains(&format!(
                "{missing} missing from \"n\", \"region\" in operator Aggregate"
            )),
            "{sql}: {text}"
        );
    }
    // Rewritten at once (an aggregate or an expression in the select list, an ordinal or an
    // expression as a key), or by a `HAVING` before the sort: the measure is unresolved.
    for sql in [
        "SELECT region, measure(total) FROM mv GROUP BY region ORDER BY measure(total)",
        "SELECT region, count(1) FROM mv GROUP BY region ORDER BY measure(total)",
        "SELECT upper(region) AS u FROM mv GROUP BY region ORDER BY measure(total)",
        "SELECT region FROM mv GROUP BY 1 ORDER BY measure(total)",
        "SELECT region FROM mv GROUP BY region, upper(region) ORDER BY measure(total)",
        "SELECT region FROM mv GROUP BY region HAVING region <> 'a' ORDER BY measure(total)",
        "SELECT region FROM mv GROUP BY region HAVING measure(total) > 3L ORDER BY measure(total)",
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(class(&e), "UNRESOLVED_COLUMN.WITH_SUGGESTION", "{sql}: {e}");
    }
}

/// A source with two columns of one name reads as Spark 4.2.1 reads it (#941 review thread; a real
/// run, 2026-11-21): the passthrough columns keep their identity, so a measure and a dimension that
/// do not use them resolve (`(1)`, `(3, 1)`). A measure or dimension that names the repeated column
/// is ambiguous when the view is created (review 49995de79): Spark prints
/// `[AMBIGUOUS_REFERENCE] Reference `a` is ambiguous, could be: [`a`, `a`]. SQLSTATE: 42704` for
/// both `sum(a)` and `a + 0` over `SELECT 1 AS a, 2 AS a`, and the view does not exist after.
#[test]
fn a_source_with_a_repeated_column_name_reads_its_measures() {
    let (s, _dir) = session();
    run(
        &s,
        "CREATE VIEW mvdup WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: SELECT 1 AS a, 2 AS a\nmeasures:\n  - name: c\n    expr: count(1)\n$$",
    )
    .unwrap();
    assert_eq!(
        longs(&run(&s, "SELECT measure(c) FROM mvdup").unwrap(), 0),
        vec![1]
    );
    run(
        &s,
        "CREATE VIEW mvdup2 WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: SELECT 1 AS a, 2 AS a, 3 AS b\ndimensions:\n  - name: b\n    expr: b\nmeasures:\n  - name: c\n    expr: count(1)\n$$",
    )
    .unwrap();
    let out = run(&s, "SELECT b, measure(c) FROM mvdup2 GROUP BY b").unwrap();
    assert_eq!(longs(&out, 1), vec![1]);
    for (name, body) in [
        ("mvdupm", "measures:\n  - name: s\n    expr: sum(a)\n"),
        (
            "mvdupd",
            "dimensions:\n  - name: d\n    expr: a + 0\nmeasures:\n  - name: c\n    expr: count(1)\n",
        ),
    ] {
        let create = format!(
            "CREATE VIEW {name} WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: SELECT 1 AS a, 2 AS a\n{body}$$"
        );
        let e = run(&s, &create).unwrap_err();
        assert_eq!(class(&e), "AMBIGUOUS_REFERENCE", "{name}: {e}");
        let e = run(&s, &format!("SELECT * FROM {name}")).unwrap_err();
        assert_eq!(class(&e), "TABLE_OR_VIEW_NOT_FOUND", "{name}: {e}");
    }
}

/// Only the built-in `measure` reads a measure: a qualified SQL function of that name is that
/// function, never the measure's aggregate (#941 review thread). Spark 4.2.1 raises
/// `MISSING_AGGREGATION` for it (a real run, 2026-11-21: `[MISSING_AGGREGATION] The
/// non-aggregating expression "total" is based on columns which are not participating in the GROUP
/// BY clause.`, for both qualified forms), as it does for such a function over a plain table's
/// non-grouped column. Krishiv checks a SQL function's arguments against the grouping before it
/// inlines the body, so a parameter the body never reads raises the same class.
#[test]
fn a_qualified_function_named_measure_is_not_the_built_in() {
    let (s, _dir) = session();
    run(
        &s,
        "CREATE FUNCTION default.measure(x BIGINT) RETURNS BIGINT RETURN 7",
    )
    .unwrap();
    for q in [
        "SELECT region, default.measure(total) FROM mv GROUP BY region",
        "SELECT region, spark_catalog.default.measure(total) FROM mv GROUP BY region",
    ] {
        let e = run(&s, q).unwrap_err();
        assert_eq!(class(&e), "MISSING_AGGREGATION", "{q}: {e}");
    }
    let out = run(&s, "SELECT region, measure(total) FROM mv GROUP BY region").unwrap();
    assert_eq!(longs(&out, 1), vec![3, 5]);
    let out = run(
        &s,
        "SELECT region, builtin.measure(total) FROM mv GROUP BY region",
    )
    .unwrap();
    assert_eq!(longs(&out, 1), vec![3, 5]);
}

/// An asset source followed by a line comment (`source: "src -- note"`) is the table it names:
/// the comment never swallows the clauses built after it (#941 review thread; Spark 4.2.1 reads
/// both views, the filtered one `('b', 5)` here).
#[test]
fn an_asset_source_with_a_line_comment_reads_the_table() {
    let (s, _dir) = session();
    for (name, filter) in [("mvc", ""), ("mvcf", "filter: n > 2\n")] {
        run(
            &s,
            &format!(
                "CREATE VIEW {name} WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: \"src -- note\"\n{filter}dimensions:\n  - name: region\n    expr: region\nmeasures:\n  - name: total\n    expr: sum(n)\n$$"
            ),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    let all = run(&s, "SELECT region, measure(total) FROM mvc GROUP BY region").unwrap();
    assert_eq!(longs(&all, 1), vec![3, 5]);
    let filtered = run(
        &s,
        "SELECT region, measure(total) FROM mvcf GROUP BY region",
    )
    .unwrap();
    assert_eq!(longs(&filtered, 1), vec![5]);
}

/// An expression with an alias of its own (`sum(n) AS subtotal`): Spark's `parseExpression` takes a
/// named expression and the column's name aliases it again, so the view is created and read under
/// the column names (Spark 4.2.1 probe, #941 review thread: `('a', 3), ('b', 5)`; `DESCRIBE` shows
/// `r`, `total`).
#[test]
fn an_expression_with_its_own_alias_is_named_by_its_column() {
    let (s, _dir) = session();
    let create = "CREATE VIEW mva WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: src\ndimensions:\n  - name: r\n    expr: region AS reg\nmeasures:\n  - name: total\n    expr: sum(n) AS subtotal\n$$";
    run(&s, create).unwrap_or_else(|e| panic!("{create}: {e}"));
    let batches = run(
        &s,
        "SELECT r, measure(total) FROM mva GROUP BY r ORDER BY r",
    )
    .unwrap();
    assert_eq!(longs(&batches, 1), vec![3, 5]);
    let schema = batches[0].schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names[0], "r");
    let described = run(&s, "DESCRIBE mva").unwrap();
    let col = described[0].column(0);
    let name = |i| match col.as_string_opt::<i32>() {
        Some(a) => a.value(i).to_owned(),
        None => col.as_string_view().value(i).to_owned(),
    };
    assert_eq!(name(0), "r");
    assert_eq!(name(1), "total");
}

/// A metric view read inside an ordinary view resolves its source's bare names in the enclosing
/// view's namespace, as the view's own names do; read directly, in the reader's (a Spark 4.2.1
/// run, 2026-11-21: `a.v` read from `b` gives `a.t`'s rows, `a.mv` read from `b` gives `b.t`'s).
#[test]
fn a_metric_source_inside_a_view_resolves_in_that_views_namespace() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .config("spark.sql.warehouse.dir", dir.path().display().to_string())
        .build()
        .unwrap();
    for sql in [
        "CREATE NAMESPACE a",
        "CREATE NAMESPACE b",
        "CREATE TABLE a.t (region STRING, n INT) USING parquet",
        "INSERT INTO a.t VALUES ('a', 1)",
        "CREATE TABLE b.t (region STRING, n INT) USING parquet",
        "INSERT INTO b.t VALUES ('b', 100)",
        "USE a",
        "CREATE VIEW a.mv WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: t\ndimensions:\n  - name: region\n    expr: region\nmeasures:\n  - name: total\n    expr: sum(n)\n$$",
        "CREATE VIEW a.v AS SELECT region, measure(total) AS m FROM a.mv GROUP BY region",
        "USE b",
    ] {
        run(&s, sql).unwrap();
    }
    let direct = run(&s, "SELECT measure(total) FROM a.mv").unwrap();
    assert_eq!(longs(&direct, 0), [100]);
    let through_view = run(&s, "SELECT m FROM a.v").unwrap();
    assert_eq!(longs(&through_view, 0), [1]);
}

/// A query source ending in `;` is no source (a Spark 4.2.1 run, 2026-11-21).
#[test]
fn a_query_source_ending_in_a_semicolon_is_invalid() {
    let (s, _dir) = session();
    let e = run(
        &s,
        "CREATE VIEW mv2 WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: SELECT * FROM src;\ndimensions:\n  - name: region\n    expr: region\nmeasures:\n  - name: total\n    expr: sum(n)\n$$",
    )
    .unwrap_err();
    assert_eq!(class(&e), "INVALID_METRIC_VIEW_YAML", "{e}");
    assert!(
        e.to_string().contains("Invalid source: SELECT * FROM src;"),
        "{e}"
    );
}

/// The terminator is decided on tokens, as Spark's `parseQuery` decides (review of 515b0f345): a
/// `;` after the query, before a comment or doubled, is no source; a `;` inside a trailing comment
/// leaves a query source.
#[test]
fn a_query_source_terminator_is_read_from_tokens() {
    let (s, _dir) = session();
    let create = |name: &str, source: &str| {
        run(
            &s,
            &format!(
                "CREATE VIEW {name} WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: {source}\ndimensions:\n  - name: region\n    expr: region\nmeasures:\n  - name: total\n    expr: sum(n)\n$$"
            ),
        )
    };
    for source in ["SELECT * FROM src; -- note", "SELECT * FROM src;;"] {
        let e = create("mv_bad", source).unwrap_err();
        assert_eq!(class(&e), "INVALID_METRIC_VIEW_YAML", "{source}: {e}");
        assert!(
            e.to_string().contains(&format!("Invalid source: {source}")),
            "{source}: {e}"
        );
    }
    create("mv_comment", "SELECT * FROM src -- latest;").unwrap();
    let rows = run(&s, "SELECT measure(total) FROM mv_comment").unwrap();
    assert_eq!(longs(&rows, 0), [8]);
}

/// `HAVING` on a grouping key the output does not show, then `ORDER BY measure(..)`: Spark 4.2.1
/// cannot resolve the measure column in the sort (run 2026-11-21).
#[test]
fn order_by_a_measure_over_having_on_a_hidden_key_is_unresolved() {
    let (s, _dir) = session();
    let e = run(
        &s,
        "SELECT measure(total) FROM mv GROUP BY region HAVING region = 'a' ORDER BY measure(total)",
    )
    .unwrap_err();
    assert_eq!(class(&e), "UNRESOLVED_COLUMN.WITH_SUGGESTION", "{e}");
}

/// Before 4.2 a metric view is a syntax error at the first keyword 4.1 does not know, and the
/// message never quotes the YAML body (Spark 4.1.0 run, 2026-11-21).
#[test]
fn the_pre_4_2_syntax_error_names_the_keyword_not_the_body() {
    let s = Session::builder()
        .profile(krishiv_plan::analyzer::SparkProfile::V4_1)
        .build()
        .unwrap();
    for (sql, token) in [
        (
            "CREATE VIEW v LANGUAGE YAML AS $$\nversion: 0.1\nsecret: hunter2\n$$",
            "'LANGUAGE'",
        ),
        (
            "CREATE VIEW v WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsecret: hunter2\n$$",
            "'METRICS'",
        ),
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(class(&e), "PARSE_SYNTAX_ERROR", "{e}");
        let text = e.to_string();
        assert!(
            text.contains(&format!("Syntax error at or near {token}")),
            "{text}"
        );
        assert!(!text.contains("hunter2"), "{text}");
    }
}
