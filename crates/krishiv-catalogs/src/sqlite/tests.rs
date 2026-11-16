// SPDX-License-Identifier: Apache-2.0
//! The SQLite catalog against the contract every catalog shares (`krishiv_plan::analyzer::contract`) and against what
//! only a file catalog has to answer: a restart, a second connection, a second process, a process killed mid-change.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::contract::{capabilities_agree, write_contract};
use krishiv_plan::analyzer::{
    Catalog, ColumnPosition, CreateMode, NameForm, NamespaceInfo, TableChange, TableColumn,
    TableIdent, TableInfo,
};
use krishiv_types::DataType;

use super::SqliteCatalog;
use crate::location::RemoteLocationPolicy;

fn form() -> NameForm {
    NameForm {
        case_sensitive: false,
        qualified: true,
        temporary_only: false,
    }
}

fn ident(namespace: &str, name: &str) -> TableIdent {
    TableIdent {
        catalog: "lite".to_owned(),
        namespace: vec![namespace.to_owned()],
        name: name.to_owned(),
    }
}

fn info(namespace: &str, name: &str) -> TableInfo {
    TableInfo::new(
        ident(namespace, name),
        vec![
            TableColumn::new("a", DataType::Integer, false),
            TableColumn::new("b", DataType::String, true),
        ],
    )
}

fn class(e: &KrishivError) -> String {
    e.error_class().name().to_owned()
}

fn file() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.db");
    (dir, path)
}

fn open(path: &Path) -> SqliteCatalog {
    SqliteCatalog::open("lite", path, false).unwrap()
}

fn tables(c: &SqliteCatalog, namespace: &str) -> Vec<String> {
    c.list_tables(&[namespace.to_owned()]).unwrap()
}

fn load(c: &SqliteCatalog, name: &str) -> Option<krishiv_plan::analyzer::TableMeta> {
    block_on(c.load_table(&ident("default", name), None, form())).unwrap()
}

#[test]
fn the_sqlite_catalog_meets_the_write_contract_in_memory_and_in_a_file() {
    let memory = SqliteCatalog::open_in_memory("lite", false).unwrap();
    block_on(write_contract(&memory)).unwrap();
    let (_dir, path) = file();
    let on_disk = open(&path);
    block_on(write_contract(&on_disk)).unwrap();
    block_on(capabilities_agree(&on_disk)).unwrap();
    let caps = on_disk.capabilities();
    assert!(caps.writes && caps.alter_columns && caps.namespaces);
    assert!(!caps.time_travel && !caps.staged_replace && !caps.nested_namespaces);
}

#[test]
fn a_catalog_survives_a_restart_and_the_epoch_with_it() {
    let (_dir, path) = file();
    let epoch = {
        let c = open(&path);
        let write = c.as_write().unwrap();
        block_on(write.create_namespace(&["ns".to_owned()], NamespaceInfo::default())).unwrap();
        let mut i = info("ns", "t");
        i.comment = Some("hello".to_owned());
        i.location = Some("s3://bucket/t".to_owned());
        i.properties.insert("k".to_owned(), "v".to_owned());
        i.partition_columns = vec!["a".to_owned()];
        assert!(block_on(write.create_table(i, CreateMode::ErrorIfExists)).unwrap());
        c.epoch()
    };
    let c = open(&path);
    assert_eq!(c.epoch(), epoch);
    assert_eq!(epoch, 2);
    let details = c
        .table_details(&ident("ns", "t"), &RemoteLocationPolicy::default())
        .unwrap()
        .unwrap();
    assert_eq!(details.comment.as_deref(), Some("hello"));
    assert_eq!(details.location.as_deref(), Some("s3://bucket/t"));
    assert_eq!(details.properties.get("k").map(String::as_str), Some("v"));
    assert_eq!(details.partition_columns, ["a"]);
    let meta = block_on(c.load_table(&ident("ns", "t"), None, form()))
        .unwrap()
        .unwrap();
    // The partition column goes last; NOT NULL is kept as declared.
    let columns: Vec<_> = meta
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c.nullable))
        .collect();
    assert_eq!(columns, [("b", true), ("a", false)]);
    assert_eq!(c.list_namespaces().unwrap(), ["default", "ns"]);
}

#[test]
fn types_comments_defaults_and_char_varchar_round_trip() {
    let c = SqliteCatalog::open_in_memory("lite", false).unwrap();
    let write = c.as_write().unwrap();
    let mut i = info("default", "t");
    i.columns.push(TableColumn {
        name: "c".to_owned(),
        data_type: DataType::String,
        nullable: true,
        char_varchar: Some("varchar(5)".to_owned()),
    });
    i.columns.push(TableColumn::new(
        "d",
        krishiv_types::parse_data_type("decimal(10,2)").unwrap(),
        true,
    ));
    i.columns.push(TableColumn::new(
        "e",
        krishiv_types::parse_data_type("map<string,array<struct<x:int,y:double>>>").unwrap(),
        true,
    ));
    block_on(write.create_table(i.clone(), CreateMode::ErrorIfExists)).unwrap();
    let meta = load(&c, "t").unwrap();
    assert_eq!(meta.columns, i.columns);
    let altered = block_on(write.alter_table(
        &ident("default", "t"),
        meta.version.unwrap(),
        &[
            TableChange::AddColumn {
                path: vec!["f".to_owned()],
                data_type: DataType::Long,
                declared: None,
                nullable: true,
                comment: Some("the f".to_owned()),
                position: None::<ColumnPosition>,
                default: None,
            },
            TableChange::SetProperty {
                key: "comment".to_owned(),
                value: "table comment".to_owned(),
            },
            TableChange::SetProperty {
                key: "k".to_owned(),
                value: "v".to_owned(),
            },
        ],
    ))
    .unwrap();
    assert_eq!(
        altered.column_comments.get("f").map(String::as_str),
        Some("the f")
    );
    let details = c
        .table_details(&ident("default", "t"), &RemoteLocationPolicy::default())
        .unwrap()
        .unwrap();
    assert_eq!(details.comment.as_deref(), Some("table comment"));
    assert_eq!(details.properties.get("k").map(String::as_str), Some("v"));
    assert_eq!(load(&c, "t").unwrap(), altered);
}

#[test]
fn a_version_that_is_not_current_is_refused_and_a_recreated_table_has_a_new_one() {
    let c = SqliteCatalog::open_in_memory("lite", false).unwrap();
    let write = c.as_write().unwrap();
    block_on(write.create_table(info("default", "t"), CreateMode::ErrorIfExists)).unwrap();
    let v0 = load(&c, "t").unwrap().version.unwrap();
    // No history: a load at any version but the current one finds nothing.
    assert!(
        block_on(c.load_table(&ident("default", "t"), Some(TableVersion(v0.0 + 1)), form()))
            .unwrap()
            .is_none()
    );
    assert!(
        block_on(c.load_table(&ident("default", "t"), Some(v0), form()))
            .unwrap()
            .is_some()
    );
    block_on(write.drop_table(&ident("default", "t"), false)).unwrap();
    block_on(write.create_table(info("default", "t"), CreateMode::ErrorIfExists)).unwrap();
    let e = block_on(write.alter_table(&ident("default", "t"), v0, &[])).unwrap_err();
    assert_eq!(class(&e), "INTERNAL_ERROR");
    assert_ne!(load(&c, "t").unwrap().version.unwrap(), v0);
}

#[test]
fn errors_carry_their_spark_classes() {
    let c = SqliteCatalog::open_in_memory("lite", false).unwrap();
    let write = c.as_write().unwrap();
    let ns = |n: &str| vec![n.to_owned()];
    let run = |r: Result<(), KrishivError>| class(&r.unwrap_err());
    assert_eq!(
        run(block_on(
            write.create_namespace(&ns("default"), NamespaceInfo::default())
        )),
        "SCHEMA_ALREADY_EXISTS"
    );
    assert_eq!(
        run(block_on(
            write.create_namespace(&ns("a-b"), NamespaceInfo::default())
        )),
        "INVALID_SCHEMA_OR_RELATION_NAME"
    );
    assert_eq!(
        run(block_on(write.create_namespace(
            &["a".to_owned(), "b".to_owned()],
            NamespaceInfo::default()
        ))),
        "UNSUPPORTED_FEATURE.CATALOG_OPERATION"
    );
    assert_eq!(
        run(block_on(write.drop_namespace(&ns("default"), true))),
        "UNSUPPORTED_FEATURE.CATALOG_OPERATION"
    );
    assert_eq!(
        run(block_on(write.drop_namespace(&ns("nope"), false))),
        "SCHEMA_NOT_FOUND"
    );
    let create =
        |i: TableInfo| block_on(write.create_table(i, CreateMode::ErrorIfExists)).map(|_| ());
    assert_eq!(
        class(&create(info("nope", "t")).unwrap_err()),
        "SCHEMA_NOT_FOUND"
    );
    assert_eq!(
        class(&create(info("default", "bad name")).unwrap_err()),
        "INVALID_SCHEMA_OR_RELATION_NAME"
    );
    let mut repeated = info("default", "r");
    repeated
        .columns
        .push(TableColumn::new("A", DataType::Long, true));
    assert_eq!(
        class(&create(repeated).unwrap_err()),
        "COLUMN_ALREADY_EXISTS"
    );
    let mut missing = info("default", "p");
    missing.partition_columns = vec!["zz".to_owned()];
    assert_eq!(
        class(&create(missing).unwrap_err()),
        "PARTITION_COLUMN_NOT_FOUND_IN_SCHEMA"
    );
    create(info("default", "t")).unwrap();
    assert_eq!(
        class(&create(info("default", "T")).unwrap_err()),
        "TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    let other = ident("default", "t");
    let rename = |from: &TableIdent, to: &TableIdent| block_on(write.rename_table(from, to));
    assert_eq!(
        class(&rename(&ident("default", "ghost"), &other).unwrap_err()),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    block_on(write.create_namespace(&ns("two"), NamespaceInfo::default())).unwrap();
    assert!(rename(&other, &ident("two", "t")).is_err());
    assert_eq!(
        class(
            &block_on(write.alter_table(&ident("default", "ghost"), TableVersion(1), &[]))
                .unwrap_err()
        ),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn a_case_sensitive_catalog_keeps_both_spellings() {
    let c = SqliteCatalog::open_in_memory("lite", true).unwrap();
    let write = c.as_write().unwrap();
    for name in ["t", "T"] {
        block_on(write.create_table(info("default", name), CreateMode::ErrorIfExists)).unwrap();
    }
    assert_eq!(tables(&c, "default"), ["T", "t"]);
    let sensitive = NameForm {
        case_sensitive: true,
        ..form()
    };
    assert!(
        block_on(c.load_table(&ident("default", "T"), None, sensitive))
            .unwrap()
            .is_some()
    );
}

#[test]
fn a_file_of_a_newer_format_is_refused_and_a_temporary_only_path_finds_nothing() {
    let (_dir, path) = file();
    drop(open(&path));
    rusqlite::Connection::open(&path)
        .unwrap()
        .pragma_update(None, "user_version", 99)
        .unwrap();
    let e = SqliteCatalog::open("lite", &path, false).unwrap_err();
    assert_eq!(class(&e), "KRISHIV_CATALOG_FAILURE");
    let c = SqliteCatalog::open_in_memory("lite", false).unwrap();
    let temp = NameForm {
        temporary_only: true,
        ..form()
    };
    assert!(
        block_on(c.load_table(&ident("default", "t"), None, temp))
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_second_connection_sees_every_change_and_the_epoch_it_made() {
    let (_dir, path) = file();
    let (a, b) = (open(&path), open(&path));
    let before = b.epoch();
    block_on(
        a.as_write()
            .unwrap()
            .create_table(info("default", "t"), CreateMode::ErrorIfExists),
    )
    .unwrap();
    assert!(b.epoch() > before);
    assert!(load(&b, "t").is_some());
}

#[test]
fn writers_in_threads_take_turns_and_a_name_is_made_once() {
    let (_dir, path) = file();
    let winners: usize = std::thread::scope(|s| {
        let handles: Vec<_> =
            (0..8)
                .map(|i| {
                    let path = path.clone();
                    s.spawn(move || {
                        let c = open(&path);
                        let write = c.as_write().unwrap();
                        block_on(write.create_table(
                            info("default", &format!("own{i}")),
                            CreateMode::ErrorIfExists,
                        ))
                        .unwrap();
                        usize::from(
                            block_on(write.create_table(
                                info("default", "shared"),
                                CreateMode::IgnoreIfExists,
                            ))
                            .unwrap(),
                        )
                    })
                })
                .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    assert_eq!(winners, 1);
    let c = open(&path);
    assert_eq!(tables(&c, "default").len(), 9);
    // One epoch step per change, none lost or repeated.
    assert_eq!(c.epoch(), 9);
}

/// The body of the child processes of the tests below (a test binary run again with the environment set): creates
/// `KRISHIV_SQLITE_CHILD_COUNT` tables named `<prefix><i>` in the catalog file, or until it is killed.
#[test]
fn child_writer() {
    let (Ok(path), Ok(prefix)) = (
        std::env::var("KRISHIV_SQLITE_CHILD_PATH"),
        std::env::var("KRISHIV_SQLITE_CHILD_PREFIX"),
    ) else {
        return;
    };
    let count: usize = std::env::var("KRISHIV_SQLITE_CHILD_COUNT")
        .ok()
        .and_then(|c| c.parse().ok())
        .unwrap_or(usize::MAX);
    let c = open(Path::new(&path));
    let write = c.as_write().unwrap();
    for i in 0..count {
        block_on(write.create_table(
            info("default", &format!("{prefix}{i}")),
            CreateMode::ErrorIfExists,
        ))
        .unwrap();
    }
}

fn child(path: &Path, prefix: &str, count: Option<usize>) -> std::process::Child {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", "sqlite::tests::child_writer", "--nocapture"])
        .env("KRISHIV_SQLITE_CHILD_PATH", path)
        .env("KRISHIV_SQLITE_CHILD_PREFIX", prefix)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(count) = count {
        cmd.env("KRISHIV_SQLITE_CHILD_COUNT", count.to_string());
    }
    cmd.spawn().unwrap()
}

#[test]
fn two_processes_writing_one_file_lose_nothing() {
    let (_dir, path) = file();
    drop(open(&path));
    let mut a = child(&path, "pa", Some(40));
    let mut b = child(&path, "pb", Some(40));
    assert!(a.wait().unwrap().success());
    assert!(b.wait().unwrap().success());
    let c = open(&path);
    assert_eq!(tables(&c, "default").len(), 80);
    assert_eq!(c.epoch(), 80);
}

#[test]
fn a_process_killed_in_the_middle_of_ddl_leaves_a_loadable_catalog() {
    let (_dir, path) = file();
    drop(open(&path));
    // Killed on drop, so a failed assertion below does not leave the unbounded writer running.
    struct Victim(std::process::Child);
    impl Drop for Victim {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let mut victim = Victim(child(&path, "k", None));
    let deadline = Instant::now() + Duration::from_secs(60);
    // Let it make some tables, then kill -9 it while it is making the next.
    while tables(&open(&path), "default").len() < 5 {
        assert!(Instant::now() < deadline, "the child made no tables");
        std::thread::sleep(Duration::from_millis(20));
    }
    victim.0.kill().unwrap();
    victim.0.wait().unwrap();
    let c = open(&path);
    let names = tables(&c, "default");
    // Every table that is there loads whole, and the epoch counts exactly the changes that committed.
    for name in &names {
        assert!(load(&c, name).is_some(), "{name}");
    }
    assert_eq!(c.epoch(), names.len() as u64);
    // The catalog still takes changes.
    block_on(
        c.as_write()
            .unwrap()
            .create_table(info("default", "after"), CreateMode::ErrorIfExists),
    )
    .unwrap();
    assert_eq!(c.epoch(), names.len() as u64 + 1);
}

#[test]
fn one_folding_rule_is_stored_in_the_file_whatever_the_query_says() {
    // Rule: the catalog's own `case_sensitive`, fixed in the file; the query's `NameForm::case_sensitive` is ignored.
    let (_dir, path) = file();
    let c = SqliteCatalog::open("lite", &path, false).unwrap();
    block_on(
        c.as_write()
            .unwrap()
            .create_table(info("default", "T"), CreateMode::ErrorIfExists),
    )
    .unwrap();
    for query in [false, true] {
        let form = NameForm {
            case_sensitive: query,
            ..form()
        };
        for spelling in ["T", "t"] {
            assert!(
                block_on(c.load_table(&ident("default", spelling), None, form))
                    .unwrap()
                    .is_some(),
                "{spelling} under case_sensitive={query}"
            );
        }
        assert!(
            c.table_details(&ident("default", "t"), &RemoteLocationPolicy::default())
                .unwrap()
                .is_some()
        );
    }
    drop(c);
    // The file refuses the other rule.
    assert_eq!(
        class(&SqliteCatalog::open("lite", &path, true).unwrap_err()),
        "KRISHIV_CATALOG_FAILURE"
    );
    // And a case-sensitive file finds `T` under a case-insensitive query.
    let (_dir2, path2) = file();
    let s = SqliteCatalog::open("lite", &path2, true).unwrap();
    block_on(
        s.as_write()
            .unwrap()
            .create_table(info("default", "T"), CreateMode::ErrorIfExists),
    )
    .unwrap();
    assert!(
        block_on(s.load_table(&ident("default", "T"), None, form()))
            .unwrap()
            .is_some()
    );
    assert!(
        block_on(s.load_table(&ident("default", "t"), None, form()))
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_location_another_process_wrote_is_checked_on_the_way_out() {
    let (_dir, path) = file();
    let c = open(&path);
    block_on(
        c.as_write()
            .unwrap()
            .create_table(info("default", "t"), CreateMode::ErrorIfExists),
    )
    .unwrap();
    let raw = rusqlite::Connection::open(&path).unwrap();
    raw.execute(
        "UPDATE tables SET location = 'file:///etc' WHERE name = 't'",
        [],
    )
    .unwrap();
    let e = c
        .table_details(&ident("default", "t"), &RemoteLocationPolicy::default())
        .unwrap_err();
    assert_eq!(class(&e), "KRISHIV_CATALOG_LOCATION_REFUSED");
}

#[test]
fn a_two_part_name_is_requires_single_part_namespace() {
    let c = SqliteCatalog::open_in_memory("lite", false).unwrap();
    assert!(c.single_part_namespace());
}

#[test]
fn the_epoch_never_goes_backwards_or_repeats_after_a_read_fault() {
    let (_dir, path) = file();
    let c = open(&path);
    block_on(
        c.as_write()
            .unwrap()
            .create_table(info("default", "t"), CreateMode::ErrorIfExists),
    )
    .unwrap();
    let before = c.epoch();
    assert_eq!(before, 1);
    // Make the epoch unreadable from outside, as a damaged file would be.
    let raw = rusqlite::Connection::open(&path).unwrap();
    raw.execute_batch("ALTER TABLE epoch RENAME TO epoch_away")
        .unwrap();
    let a = c.epoch();
    let b = c.epoch();
    assert!(a > before && b > a, "{before} {a} {b}");
    raw.execute_batch("ALTER TABLE epoch_away RENAME TO epoch")
        .unwrap();
    let mut seen = vec![before, a, b];
    // The file reads again (at its old value): the answer stays above everything handed out, then follows the file.
    let back = c.epoch();
    assert!(seen.iter().all(|e| back > *e), "{seen:?} {back}");
    seen.push(back);
    // Another process commits changes, which move the file epoch up to and past the fault values.
    for i in 0..(b as usize + 2) {
        raw.execute("UPDATE epoch SET value = value + 1", [])
            .unwrap();
        let e = c.epoch();
        assert!(!seen.contains(&e), "epoch {e} repeated ({i}): {seen:?}");
        assert!(e > *seen.last().unwrap());
        seen.push(e);
    }
}
