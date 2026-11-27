// SPDX-License-Identifier: Apache-2.0
//! `rewrite_data_files` over the row-level table (36e-2a): the planned groups of small files,
//! each one commit of the group's rows through [`RowLevelTable::group_rewriter`], driven as the
//! session's `CALL` path drives them (`scan_files` of the group into the sink).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use krishiv_exec::planner::{
    REWRITE_GROUP_OPTION, REWRITE_WRITE_MAX_OPTION, RewriteGroup, RowLevelOperation, RowLevelWrite,
};

use super::cow_tests::{Dml, append, create};
use super::provider_tests::env;
use super::sink_tests::{Catalog, PerDriver, expected, range, read_back, run};

fn options(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

/// One group's commit: its files' rows, unchanged, into the group sink.
fn run_group(dml: &Dml<'_>, group: &RewriteGroup) {
    let rows: Vec<RecordBatch> = dml.query(
        dml.table.scan_files(&group.files).unwrap(),
        "SELECT id, x, name FROM t",
    );
    let schema = dml.input();
    // One driver writes the group (the planner's single output partition), so the files come
    // from the size ceiling alone.
    let rows: Vec<RecordBatch> = rows
        .into_iter()
        .map(|b| RecordBatch::try_new(Arc::clone(&schema), b.columns().to_vec()).unwrap())
        .collect();
    let mut batches = vec![arrow_select::concat::concat_batches(&schema, &rows).unwrap()];
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(Arc::clone(&schema)));
    }
    let mut request = RowLevelWrite::new(
        RowLevelOperation::Delete,
        "`db`.`t`",
        schema,
        group.files.clone(),
        None,
    );
    request.write_id = Some(format!("rewrite-{}", super::sink::unique_u64()));
    request
        .options
        .insert(REWRITE_GROUP_OPTION.to_owned(), String::new());
    request.options.insert(
        REWRITE_WRITE_MAX_OPTION.to_owned(),
        group.write_max_file_size.to_string(),
    );
    let sink = dml.table.group_rewriter(&request).unwrap();
    let input = Arc::new(PerDriver {
        schema: dml.input(),
        batches,
        stall: false,
    });
    run(dml.env, sink, input, None).0.unwrap();
}

#[test]
fn small_files_are_compacted_into_one_and_no_row_changes() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    for start in [0, 5, 10, 15] {
        append(&env, &catalog, range(start, start + 5));
    }
    let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
    assert_eq!(dml.table.live_data_files().unwrap(), 4);
    let groups = dml
        .table
        .plan_rewrite(&options(&[("min-input-files", "2")]))
        .unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].files.len(), 4);
    run_group(&dml, &groups[0]);

    assert_eq!(read_back(&env, &catalog), expected(0..20));
    let after = Dml::open(&env, &catalog, RowLevelOperation::Delete);
    assert_eq!(after.table.live_data_files().unwrap(), 1);
    // Nothing is left to compact: below `min-input-files` a lone file is a no-op plan.
    assert!(
        after
            .table
            .plan_rewrite(&options(&[("min-input-files", "2")]))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn files_below_min_input_files_plan_no_group() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    append(&env, &catalog, range(5, 10));
    let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
    // The default `min-input-files` is 5.
    assert!(dml.table.plan_rewrite(&options(&[])).unwrap().is_empty());
}

#[test]
fn an_unknown_option_is_refused() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
    assert!(dml.table.plan_rewrite(&options(&[("bogus", "1")])).is_err());
}
