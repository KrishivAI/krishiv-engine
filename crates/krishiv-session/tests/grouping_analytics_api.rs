// SPDX-License-Identifier: Apache-2.0
//! Grouping analytics through the session: the DataFrame forms (`rollup`,
//! `cube`, `grouping_sets`) against the SQL forms, rolled-up columns' types,
//! `NULL` data against rolled-up `NULL`, and the errors of invalid uses.
//! The Spark goldens are in `grouping_analytics.rs`.
#![forbid(unsafe_code)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int32Type, Int64Type};
use arrow_array::{Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::functions::{col, lit, sum};
use krishiv_session::{DataFrame, Session};

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Tables `t(a, b, v)` (no nulls) and `n(a, v)` (with a real NULL key).
fn session() -> R<Session> {
    let s = Session::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2])),
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    )?;
    s.register_batches("t", schema, vec![batch])?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("v", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, None])),
            Arc::new(Int32Array::from(vec![5, 7, 11])),
        ],
    )?;
    s.register_batches("n", schema, vec![batch])?;
    Ok(s)
}

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int8 => c.as_primitive::<Int8Type>().value(row).to_string(),
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        other => format!("<{other}>"),
    }
}

fn row_strings(batches: &[RecordBatch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(|r| {
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("\t")
            })
        })
        .collect()
}

/// The schema and the rows (tab-separated, sorted) of a DataFrame.
fn run(df: &DataFrame) -> Result<(SchemaRef, Vec<String>), KrishivError> {
    block_on(async {
        let schema = df.schema().await?;
        let mut rows = row_strings(&df.collect().await?);
        rows.sort();
        Ok((schema, rows))
    })
}

fn sql(s: &Session, query: &str) -> Result<(SchemaRef, Vec<String>), KrishivError> {
    run(&block_on(s.sql(query))?)
}

fn rows(s: &Session, query: &str) -> R<Vec<String>> {
    Ok(sql(s, query)?.1)
}

fn sorted(rows: &[&str]) -> Vec<String> {
    let mut rows: Vec<String> = rows.iter().map(|r| (*r).to_owned()).collect();
    rows.sort();
    rows
}

fn error_class(s: &Session, query: &str) -> R<ErrorClass> {
    match sql(s, query) {
        Err(e) => Ok(e.error_class()),
        Ok((_, rows)) => {
            Err(format!("{query} should fail, it returned {} rows", rows.len()).into())
        }
    }
}

#[test]
fn dataframe_rollup_cube_and_grouping_sets_match_their_sql() -> R {
    let s = session()?;
    let sum_v = || [sum(col("v"))];
    let (schema, rollup) = run(&s.table("t").rollup([col("a"), col("b")]).agg(sum_v()))?;
    assert_eq!(
        rollup,
        sorted(&[
            "1\t1\t10",
            "1\t2\t20",
            "2\t1\t30",
            "1\tNULL\t30",
            "2\tNULL\t30",
            "NULL\tNULL\t60",
        ])
    );
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["a", "b", "sum(v)"]);
    assert_eq!(
        rollup,
        rows(&s, "SELECT a, b, sum(v) FROM t GROUP BY ROLLUP(a, b)")?
    );

    let (_, cube) = run(&s.table("t").cube([col("a"), col("b")]).agg(sum_v()))?;
    assert_eq!(
        cube,
        rows(&s, "SELECT a, b, sum(v) FROM t GROUP BY CUBE(a, b)")?
    );
    assert_eq!(cube.len(), 8);

    let sets = s
        .table("t")
        .grouping_sets(
            [vec![col("a")], vec![col("b")], vec![]],
            [col("a"), col("b")],
        )
        .agg(sum_v());
    let (_, sets) = run(&sets)?;
    assert_eq!(
        sets,
        sorted(&[
            "1\tNULL\t30",
            "2\tNULL\t30",
            "NULL\t1\t40",
            "NULL\t2\t20",
            "NULL\tNULL\t60",
        ])
    );
    assert_eq!(
        sets,
        rows(
            &s,
            "SELECT a, b, sum(v) FROM t GROUP BY GROUPING SETS ((a), (b), ())"
        )?
    );
    Ok(())
}

#[test]
fn rolled_up_columns_become_nullable_and_grouping_is_a_tinyint() -> R {
    let s = session()?;
    let (schema, _) = sql(
        &s,
        "SELECT a, b, sum(v), grouping(a), grouping_id(a, b) FROM t GROUP BY ROLLUP(a, b)",
    )?;
    let fields: Vec<(String, DataType, bool)> = schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone(), f.is_nullable()))
        .collect();
    assert_eq!(
        fields,
        [
            ("a".to_owned(), DataType::Int32, true),
            ("b".to_owned(), DataType::Int32, true),
            ("sum(v)".to_owned(), DataType::Int64, true),
            ("grouping(a)".to_owned(), DataType::Int8, false),
            ("grouping_id(a, b)".to_owned(), DataType::Int64, false),
        ]
    );
    Ok(())
}

#[test]
fn grouping_tells_a_null_key_from_a_rolled_up_one() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT a, grouping(a), sum(v) FROM n GROUP BY ROLLUP(a)"
        )?,
        sorted(&["1\t0\t5", "NULL\t0\t18", "NULL\t1\t23"])
    );
    Ok(())
}

#[test]
fn grouping_id_sets_bit_n_minus_1_minus_position() -> R {
    let s = session()?;
    // `grouping_id(a, b)`: `a` is the high bit.
    assert_eq!(
        rows(
            &s,
            "SELECT a, b, grouping_id(a, b) FROM t GROUP BY GROUPING SETS ((a), (b), (a, b))"
        )?,
        sorted(&[
            "1\tNULL\t1",
            "2\tNULL\t1",
            "NULL\t1\t2",
            "NULL\t2\t2",
            "1\t1\t0",
            "1\t2\t0",
            "2\t1\t0",
        ])
    );
    Ok(())
}

#[test]
fn mixed_plain_keys_and_rollup() -> R {
    let s = session()?;
    assert_eq!(
        rows(&s, "SELECT a, b, sum(v) FROM t GROUP BY a, ROLLUP(b)")?,
        sorted(&[
            "1\t1\t10",
            "1\t2\t20",
            "2\t1\t30",
            "1\tNULL\t30",
            "2\tNULL\t30"
        ])
    );
    Ok(())
}

#[test]
fn operands_of_a_commutative_operator_are_one_grouping_expression() -> R {
    let s = session()?;
    // `b + a` is `a + b`: the select item reads the grouping copy, and the
    // two sets are equal, so each group comes twice.
    assert_eq!(
        rows(
            &s,
            "SELECT b + a, sum(v) FROM t GROUP BY GROUPING SETS ((a + b), (b + a))"
        )?,
        sorted(&["2\t10", "2\t10", "3\t50", "3\t50"])
    );
    Ok(())
}

#[test]
fn empty_set_is_a_grand_total_even_over_no_rows() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT count(*), grouping_id() FROM t WHERE v > 100 GROUP BY GROUPING SETS (())"
        )?,
        ["0\t0"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT a, count(*) FROM t WHERE v > 100 GROUP BY ROLLUP(a)"
        )?,
        Vec::<String>::new(),
        "a rollup over no rows has no grand-total row, as in Spark"
    );
    Ok(())
}

#[test]
fn having_and_order_by_read_grouping() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT a, sum(v) FROM t GROUP BY ROLLUP(a) HAVING grouping(a) = 1"
        )?,
        ["NULL\t60"]
    );
    // `ORDER BY` follows the id: the grand total (id 1) comes first, descending.
    let df = block_on(
        s.sql("SELECT a, sum(v) FROM t GROUP BY ROLLUP(a) ORDER BY grouping_id(a) DESC, a"),
    )?;
    assert_eq!(
        row_strings(&block_on(df.collect())?),
        ["NULL\t60", "1\t30", "2\t30"]
    );
    Ok(())
}

#[test]
fn invalid_uses_have_spark_error_classes() -> R {
    let s = session()?;
    let unsupported = ErrorClass::UnsupportedGroupingExpression;
    // Outside a grouping aggregate, in any clause.
    assert_eq!(error_class(&s, "SELECT grouping(a) FROM t")?, unsupported);
    assert_eq!(
        error_class(&s, "SELECT a FROM t GROUP BY a HAVING grouping(a) = 0")?,
        unsupported
    );
    assert_eq!(
        error_class(&s, "SELECT count(*) FROM t ORDER BY grouping(a)")?,
        unsupported
    );
    // An argument that is no grouping expression, and a mismatched id list.
    assert_eq!(
        error_class(&s, "SELECT grouping(v) FROM t GROUP BY ROLLUP(a)")?,
        ErrorClass::GroupingColumnMismatch
    );
    assert_eq!(
        error_class(&s, "SELECT grouping_id(b, a) FROM t GROUP BY ROLLUP(a, b)")?,
        ErrorClass::GroupingIdColumnMismatch
    );
    assert_eq!(
        error_class(&s, "SELECT grouping(a, b) FROM t GROUP BY ROLLUP(a, b)")?,
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    // 65 grouping expressions do not fit the 64-bit id; 64 still do.
    let rollup = |n: usize| {
        let keys: Vec<String> = (0..n).map(|i| format!("a + {i}")).collect();
        format!(
            "SELECT count(*) FROM t GROUP BY ROLLUP({})",
            keys.join(", ")
        )
    };
    block_on(s.sql(&rollup(64)))?; // Analyzer-only: 65 sets would still run.
    assert_eq!(
        error_class(&s, &rollup(65))?,
        ErrorClass::GroupingSizeLimitExceeded
    );
    // A cube has 2^n sets: 17 expressions are too many, 16 still are not.
    let cube = |n: usize| {
        let keys: Vec<String> = (0..n).map(|i| format!("a + {i}")).collect();
        format!(
            "SELECT count(*) FROM t GROUP BY {} WITH CUBE",
            keys.join(", ")
        )
    };
    block_on(s.sql(&cube(16)))?; // Analyzer-only: 65,536 sets would still run.
    assert_eq!(
        error_class(&s, &cube(17))?,
        ErrorClass::GroupingSizeLimitExceeded
    );
    let keys: Vec<_> = (0..17).map(|i| col("a").gt(lit(i))).collect();
    let df = s.table("t").cube(keys).agg([sum(col("v"))]);
    assert_eq!(
        run(&df).err().map(|e| e.error_class()),
        Some(ErrorClass::GroupingSizeLimitExceeded)
    );
    Ok(())
}

#[test]
fn grouping_reads_the_top_bit_of_a_64_expression_id() -> R {
    let s = session()?;
    let keys: Vec<String> = (0..64).map(|i| format!("a + {i}")).collect();
    let query = format!(
        "SELECT grouping(a + 0), grouping(a + 63) FROM t GROUP BY ROLLUP({})",
        keys.join(", ")
    );
    let got = rows(&s, &query)?;
    assert!(
        got.iter().any(|r| r == "1\t1"),
        "the grand total (id -1) reads both bits as 1, got {got:?}"
    );
    assert!(
        !got.iter().any(|r| r.contains('-')),
        "no negative bits, got {got:?}"
    );
    Ok(())
}

#[test]
fn duplicate_sets_keep_their_rows_but_not_late_grouping_calls() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT a, sum(v) FROM t GROUP BY GROUPING SETS ((a), (a))"
        )?,
        sorted(&["1\t30", "1\t30", "2\t30", "2\t30"])
    );
    // Spark reads the set's number where `HAVING` expects the id.
    assert_eq!(
        error_class(
            &s,
            "SELECT a FROM t GROUP BY GROUPING SETS ((a), (a)) HAVING grouping(a) = 0"
        )?,
        ErrorClass::UnsupportedGroupingExpression
    );
    Ok(())
}

#[test]
fn grouping_calls_inside_aggregate_arguments_resolve() -> R {
    let s = session()?;
    assert_eq!(
        rows(&s, "SELECT sum(grouping(a)) FROM t GROUP BY ROLLUP(a)")?,
        sorted(&["0", "0", "3"])
    );
    Ok(())
}
