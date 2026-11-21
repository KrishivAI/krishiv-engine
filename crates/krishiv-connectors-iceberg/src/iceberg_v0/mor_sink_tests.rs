// SPDX-License-Identifier: Apache-2.0
//! Tests of the merge-on-read sink (`mor_sink.rs`, PR-36c part 2): the rows a statement's SQL
//! selects, ending with the row id, through the row-level table's rewriter.

use krishiv_exec::planner::RowLevelOperation;

use super::cow_tests::{Dml, append, create, get, live, summary};
use super::provider_tests::env;
use super::sink_tests::{Catalog, expected, range, read_back};

const MOR: &[(&str, &str)] = &[
    ("write.delete.mode", "merge-on-read"),
    ("write.update.mode", "merge-on-read"),
];

#[test]
fn a_merge_on_read_delete_writes_a_delete_file_and_no_data() {
    let env = env();
    let t = create(&env, 2, false, MOR);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    let before = live(&env, &catalog.metadata()).0;
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .mor_delete("id < 3")
        .unwrap();
    let m = catalog.metadata();
    assert_eq!(read_back(&env, &catalog), expected(3..10));
    let (data, deletes) = live(&env, &m);
    assert_eq!(data, before, "no data file is rewritten");
    assert_eq!(deletes.len(), 1);
    let (_, s) = summary(&m);
    assert_eq!(get(&s, "added-position-delete-files"), Some("1"));
}

#[test]
fn a_merge_on_read_update_deletes_the_old_row_and_adds_the_new() {
    let env = env();
    let t = create(&env, 2, false, MOR);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 9));
    Dml::open(&env, &catalog, RowLevelOperation::Update)
        .mor_update("id = 4", "'u'")
        .unwrap();
    let mut want = expected(0..9);
    want.remove(4);
    want.push("4|1|u".to_owned());
    want.sort();
    let mut got = read_back(&env, &catalog);
    got.sort();
    assert_eq!(got, want);
}

#[test]
fn a_merge_on_read_delete_that_hits_nothing_commits_an_overwrite() {
    let env = env();
    let t = create(&env, 2, false, MOR);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    let before = catalog.metadata().current_snapshot_id();
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .mor_delete("id > 100")
        .unwrap();
    let m = catalog.metadata();
    assert_ne!(m.current_snapshot_id(), before);
    let (op, _) = summary(&m);
    assert_eq!(format!("{op:?}").to_lowercase(), "overwrite");
}

#[test]
fn a_v3_merge_on_read_delete_writes_a_dv() {
    let env = env();
    let t = create(&env, 3, false, MOR);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .mor_delete("id % 2 = 0")
        .unwrap();
    let m = catalog.metadata();
    assert_eq!(read_back(&env, &catalog).len(), 5);
    let (_, s) = summary(&m);
    assert_eq!(get(&s, "added-dvs"), Some("1"));
}

#[test]
fn an_unknown_row_level_mode_fails_the_statement() {
    for (op, key) in [
        (RowLevelOperation::Delete, "write.delete.mode"),
        (RowLevelOperation::Update, "write.update.mode"),
    ] {
        let env = env();
        let t = create(&env, 2, false, &[(key, "mor")]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 3));
        assert!(Dml::open(&env, &catalog, op).scan_is_refused(), "{key}");
    }
}
