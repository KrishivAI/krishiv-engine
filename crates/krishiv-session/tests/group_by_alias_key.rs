// SPDX-License-Identifier: Apache-2.0
//! `df.groupBy(expr.alias(name))` groups by `expr` and names the key column `name`, as Spark does
//! (PY-009: the alias used to reach the aggregate as a grouping expression and failed with
//! `INTERNAL_ERROR attribute #0 is not an input column`).
#![forbid(unsafe_code)]

use futures::executor::block_on;
use krishiv_session::Session;
use krishiv_session::functions::{col, count_star, lit};

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[test]
fn an_aliased_expression_key_groups_and_names_the_column() -> R {
    let s = Session::new()?;
    let df = block_on(s.sql("SELECT id FROM range(10)"))?
        .group_by([(col("id") % lit(2)).alias("m")])
        .agg([count_star().alias("n")])
        .order_by([col("m")]);
    let batches = block_on(df.collect())?;
    let names: Vec<_> = batches[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["m", "n"]);
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    assert_eq!(rows, 2);
    Ok(())
}

#[test]
fn aliased_keys_of_explicit_grouping_sets_group_by_the_bare_expression() -> R {
    let s = Session::new()?;
    let m = || (col("id") % lit(2)).alias("m");
    let df = block_on(s.sql("SELECT id FROM range(10)"))?
        .grouping_sets([vec![m()], vec![]], [m()])
        .agg([count_star().alias("n")]);
    let batches = block_on(df.collect())?;
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    assert_eq!(rows, 3);
    Ok(())
}
