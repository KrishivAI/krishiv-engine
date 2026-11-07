// SPDX-License-Identifier: Apache-2.0
//! Lateral column aliases next to a generator in one select list: Spark resolves them before it
//! extracts the generator, so the generated columns are no input columns yet (checked against the
//! Spark 4.2.0 goldens `generators-resolution-edge-cases`).
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(sql: &str) -> Result<Vec<arrow_array::RecordBatch>, krishiv_common::error::KrishivError> {
    block_on(async { Session::new()?.sql(sql).await?.collect().await })
}

#[test]
fn a_lateral_alias_and_the_generators_alias_of_one_name_are_ambiguous() {
    for q in [
        "SELECT 1 as a, explode(array(1, 2, 3)) as a, a * 10",
        "SELECT explode(array(1, 2, 3)) as a, 1 as a, a * 10",
    ] {
        let e = run(q).unwrap_err().to_string();
        assert!(
            e.contains("AMBIGUOUS_LATERAL_COLUMN_ALIAS") && e.contains("2 matches"),
            "{q}: {e}"
        );
    }
}

#[test]
fn a_name_list_of_the_generator_is_no_lateral_alias() {
    let batches =
        run("SELECT 1 as pos, posexplode(array('x', 'y')) as (pos, val), pos * 10").unwrap();
    let names: Vec<String> = batches[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(
        names,
        ["pos", "pos", "val", "(lateralAliasReference(pos) * 10)"]
    );
    // `pos * 10` reads the alias `1 AS pos`, not the generated position.
    let last = batches[0].column(3).as_primitive::<Int32Type>();
    assert!(last.iter().all(|v| v == Some(10)));
    assert_eq!(
        batches
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
}

/// The `INT` column `idx` of every row.
fn ints(sql: &str, idx: usize) -> Vec<i32> {
    run(sql)
        .unwrap()
        .iter()
        .flat_map(|b| {
            b.column(idx)
                .as_primitive::<Int32Type>()
                .iter()
                .map(|v| v.unwrap_or(i32::MIN))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn only_aliases_before_the_reference_count() {
    // Spark 4.2.0 (local run, 2026-11-07): a generator alias after the reading item is no match.
    assert_eq!(
        ints("SELECT 1 AS a, a * 10, explode(array(1, 2, 3)) AS a", 1),
        [10, 10, 10]
    );
    // And one before it, with no lateral alias of the name, is the generated column: 10, 20.
    assert_eq!(
        ints("SELECT explode(array(1, 2)) AS a, a * 10, 1 AS a", 1),
        [10, 20]
    );
    // A generator without an alias is no lateral alias: `col` is the alias `1 AS col`.
    assert_eq!(
        ints("SELECT 1 AS col, explode(array(1, 2)), col * 10", 2),
        [10, 10]
    );
    // An input column of the name wins over both.
    assert_eq!(
        ints(
            "SELECT 1 AS a, explode(array(1, 2, 3)) AS a, t.a FROM VALUES (5) AS t(a)",
            2
        ),
        [5, 5, 5]
    );
    let e = run("SELECT 1 AS a, explode(array(1)) AS a, explode(array(2)) AS a, a * 10");
    assert!(e.is_err(), "several generators are not supported here");
}

#[test]
fn a_generator_reading_a_later_generators_alias_runs_after_it() {
    // Spark 4.2.0 goldens `generators-resolution-edge-cases`: "generator LCA right-to-left".
    assert_eq!(
        ints(
            "SELECT explode(arr) as col, explode(array(array(0), array(1), array(2))) as arr",
            0
        ),
        [0, 1, 2]
    );
    // Left to right still works, and so does a read of an alias defined further right by a
    // generator (the golden lists rows 2, 3, 4 despite its "should fail" comment).
    assert_eq!(
        ints(
            "SELECT explode(array(array(0), array(1), array(2))) as arr, explode(arr) as col",
            1
        ),
        [0, 1, 2]
    );
    assert_eq!(
        ints("SELECT col + 1 as col2, explode(array(1, 2, 3)) as col", 0),
        [2, 3, 4]
    );
}

#[test]
fn generators_read_input_columns_before_later_generator_aliases() {
    // Spark 4.2.0 (local run, 2026-11-07): `arr` is the input column for the first generator,
    // the second one still runs first and its column is only the output `arr`.
    let q = "SELECT explode(arr) AS col, explode(array(array(0), array(1))) AS arr \
             FROM VALUES (array(9, 8)) AS t(arr)";
    assert_eq!(ints(q, 0), [9, 8, 9, 8]);
    // A chain of three runs back to front, the match is case-insensitive.
    assert_eq!(
        ints(
            "SELECT explode(ARR2) AS c, explode(arr1) AS arr2, \
             explode(array(array(array(1)))) AS arr1",
            0
        ),
        [1]
    );
    // A cycle has no order: the first read is unresolved, as in Spark.
    let e = run("SELECT explode(b) AS a, explode(a) AS b")
        .unwrap_err()
        .to_string();
    assert!(e.contains("UNRESOLVED_COLUMN"), "{e}");
}

#[test]
fn the_input_column_match_follows_spark_sql_case_sensitive() {
    // Spark 4.2.0 (local run, 2026-11-07): with `spark.sql.caseSensitive=true`, input `ARR` and
    // the alias `arr` are two names, so the first generator reads the alias (rows 0, 1) and does
    // not read the input; the mirrored case is the same.
    for (input, alias) in [("ARR", "arr"), ("arr", "ARR")] {
        let s = Session::new().unwrap();
        s.conf().set("spark.sql.caseSensitive", "true").unwrap();
        let q = format!(
            "SELECT explode({alias}) AS col, explode(array(array(0), array(1))) AS {alias} \
             FROM VALUES (array(9, 8)) AS t({input})"
        );
        let rows: Vec<i32> = block_on(async { s.sql(&q).await?.collect().await })
            .unwrap()
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<Int32Type>()
                    .iter()
                    .map(|v| v.unwrap_or(i32::MIN))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(rows, [0, 1], "{input}/{alias}");
    }
}
