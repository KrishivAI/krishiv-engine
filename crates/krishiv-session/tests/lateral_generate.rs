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
