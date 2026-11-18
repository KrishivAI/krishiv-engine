// SPDX-License-Identifier: Apache-2.0
//! The read direction of the Iceberg conformance (P3-T028a, PR-53a): tables written by Spark 4.1.0 +
//! iceberg-spark-runtime 1.12.0 (`tests/scripts/gen_iceberg_fixtures.py`, fixtures in
//! `tests/fixtures/iceberg-1.12.0/`) read through [`IcebergTableProvider`] and SQL, against the rows Spark read
//! back from the same snapshots (`EXPECTED.json`), never against a hand-written value.
//!
//! The fixture directory is the root of the file system the test serves: every path in the metadata is
//! `file:/wh/db/<table>/...`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_io::IoConfig;
use serde_json::Value;

use super::provider::IcebergTableProvider;
use super::provider_tests::{Session, provider_session};
use super::table::IcebergTable;
use super::tests::local_on;
use crate::plan::VersionSpec;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/iceberg-1.12.0")
}

fn expected() -> Value {
    let text = std::fs::read_to_string(root().join("wh/EXPECTED.json")).expect("EXPECTED.json");
    serde_json::from_str(&text).expect("EXPECTED.json is JSON")
}

/// A fixture table loaded through a registration whose file system is rooted at the fixture directory.
struct Oracle {
    rt: tokio::runtime::Runtime,
    table: IcebergTable,
    _sched: Arc<krishiv_io::IoScheduler>,
}

fn open_table(name: &str, exp: &Value) -> Oracle {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let store = Arc::new(object_store::local::LocalFileSystem::new_with_prefix(root()).unwrap());
    let (sched, reg) = {
        let _entered = rt.enter();
        local_on(&root(), store, IoConfig::default())
    };
    let metadata = exp[name]["metadata"].as_str().unwrap().to_owned();
    let table = rt
        .block_on(IcebergTable::load(&reg, &metadata, "db", name))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    Oracle {
        rt,
        table,
        _sched: sched,
    }
}

impl Oracle {
    fn session(&self, version: &VersionSpec) -> Session {
        let p = self
            .rt
            .block_on(IcebergTableProvider::open(
                &self.table,
                version,
                &Config::new(),
                4,
            ))
            .unwrap();
        provider_session(p)
    }
}

/// One Spark value as the text `CAST(x AS STRING)` gives in Krishiv (Spark's rendering of it).
fn text(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_owned(),
        Value::String(s) => s.clone(),
        Value::Object(o) if o.len() == 1 && o.contains_key("binary") => {
            o["binary"].as_str().unwrap().to_owned()
        }
        other => other.to_string(),
    }
}

/// `columns` of `table` at `version` through SQL (every column cast to text) against Spark's rows `rows`.
fn assert_rows(o: &Oracle, version: &VersionSpec, columns: &[&str], rows: &[Value], what: &str) {
    let select: Vec<String> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if *c == "bin" {
                format!("lower(hex({c})) AS c{i}")
            } else {
                format!("CAST({c} AS STRING) AS c{i}")
            }
        })
        .collect();
    let sql = format!("SELECT {} FROM t ORDER BY id", select.join(", "));
    let s = o.session(version);
    let got = s.rows(&sql);
    let want: Vec<String> = rows
        .iter()
        .map(|r| {
            columns
                .iter()
                .map(|c| text(&r[c.split('.').next_back().unwrap()]))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    assert_eq!(got, want, "{what}");
}

/// The scalar columns of a table's rows (a struct, list or map value is not compared as text).
fn scalar_columns(exp: &Value, name: &str) -> Vec<String> {
    let rows = exp[name]["rows"].as_array().unwrap();
    exp[name]["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_owned())
        .filter(|c| {
            rows.iter().all(|r| match &r[c] {
                Value::Object(o) => o.len() == 1 && o.contains_key("binary"),
                Value::Array(_) => false,
                _ => true,
            })
        })
        .collect()
}

fn check_current(name: &str) {
    let exp = expected();
    let o = open_table(name, &exp);
    let cols = scalar_columns(&exp, name);
    let cols: Vec<&str> = cols.iter().map(String::as_str).collect();
    assert!(!cols.is_empty());
    assert_rows(
        &o,
        &VersionSpec::Current,
        &cols,
        exp[name]["rows"].as_array().unwrap(),
        name,
    );
}

#[test]
fn spark_wrote_primitive_types_and_two_appends() {
    check_current("basic");
}

#[test]
fn spark_wrote_identity_bucket_and_day_partitions() {
    check_current("partitioned");
}

#[test]
fn spark_wrote_position_delete_files() {
    let exp = expected();
    let deletes = exp["pos_deletes"]["delete_files"].as_array().unwrap();
    assert!(
        deletes.iter().any(|d| d["file_format"] == "PARQUET"),
        "the generator must leave position delete files: {deletes:?}"
    );
    check_current("pos_deletes");
}

#[test]
fn spark_wrote_deletion_vectors() {
    let exp = expected();
    let deletes = exp["dvs"]["delete_files"].as_array().unwrap();
    assert!(
        deletes.iter().any(|d| d["file_format"] == "PUFFIN"),
        "the generator must leave deletion vectors: {deletes:?}"
    );
    check_current("dvs");
}

#[test]
fn spark_evolved_the_schema_and_krishiv_reads_every_snapshot() {
    let exp = expected();
    check_current("evolved");
    let o = open_table("evolved", &exp);
    let as_of = exp["evolved"]["as_of"].as_object().unwrap();
    assert_eq!(
        as_of.len(),
        exp["evolved"]["snapshots"].as_array().unwrap().len(),
        "every snapshot has a time-travel answer"
    );
    for (id, rows) in as_of {
        let rows = rows.as_array().unwrap();
        // The columns of that snapshot's schema, as Spark's `VERSION AS OF` returned them.
        let cols: Vec<&str> = rows[0]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_rows(
            &o,
            &VersionSpec::Id(id.parse().unwrap()),
            &cols,
            rows,
            &format!("snapshot {id}"),
        );
    }
}

#[test]
fn spark_wrote_structs() {
    let exp = expected();
    let o = open_table("nested", &exp);
    let s = o.session(&VersionSpec::Current);
    // The struct's fields, from Spark's own rows (a NULL struct reads as NULL fields).
    let want: Vec<String> = exp["nested"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let st = &r["st"];
            format!("{}|{}|{}", text(&r["id"]), text(&st["a"]), text(&st["b"]))
        })
        .collect();
    assert_eq!(
        s.rows("SELECT CAST(id AS STRING), CAST(st.a AS STRING), CAST(st.b AS STRING) FROM t ORDER BY id"),
        want
    );
}

/// P3-T024 (PR-54a): the Theta sketches Iceberg's `compute_table_stats` wrote (a Puffin statistics file of the
/// current snapshot) against sketches Krishiv builds from the same columns. Below 4,096 distinct values both
/// sketches are exact, so the compact blobs are byte-equal (the hash and serialisation gate for the
/// `datasketches` crate); above that Java's is an Alpha-family sketch and Krishiv's a QuickSelect one, so only
/// the estimates compare (within 3 RSE each, review B1). Java's `ndv` is the blob's estimate truncated.
///
/// Skipped, with a message, while `tests/fixtures/iceberg-1.12.0` lacks the `thetas` tables: the fixtures
/// are written by Spark in Docker (`tests/scripts/gen_iceberg_fixtures.py`) and were not regenerated on a
/// host without it.
#[test]
fn spark_wrote_theta_sketches_and_krishiv_matches_them() {
    use crate::plan::theta_value::{new_sketch, sketched, update_column};
    use arrow_array::Array as _;

    let exp = expected();
    if exp.get("thetas").is_none() || exp.get("thetas_big").is_none() {
        eprintln!(
            "SKIPPED spark_wrote_theta_sketches_and_krishiv_matches_them: EXPECTED.json has no `thetas` \
             tables; regenerate the fixtures on a Docker host (tests/scripts/gen_iceberg_fixtures.py)"
        );
        return;
    }
    // One standard error of a k = 4,096 sketch is about 1/sqrt(4096); the bar is 3 of each side.
    let tolerance = 6.0 / 4096_f64.sqrt();
    let mut exact = 0;
    let mut estimated = 0;
    for name in ["thetas", "thetas_big"] {
        let o = open_table(name, &exp);
        let provider =
            o.rt.block_on(IcebergTableProvider::open(
                &o.table,
                &VersionSpec::Current,
                &Config::new(),
                4,
            ))
            .unwrap();
        let snapshot_id = provider.snapshot().snapshot_id.unwrap();
        let file = o
            .table
            .table
            .metadata()
            .statistics_for_snapshot(snapshot_id)
            .unwrap_or_else(|| panic!("{name}: no statistics file for snapshot {snapshot_id}"))
            .clone();
        let fields = provider.snapshot().fields.clone();
        let s = provider_session(provider);
        for f in fields.iter().filter(|f| sketched(&f.ty)) {
            let col = f.path.root().to_owned();
            let java =
                o.rt.block_on(super::puffin_stats::read_sketch(
                    o.table.table.file_io(),
                    &file,
                    f.id,
                ))
                .unwrap()
                .unwrap_or_else(|| panic!("{name}.{col}: no blob"));
            let mut mine = new_sketch().unwrap();
            for b in s.run(&format!("SELECT {col} FROM t")).unwrap() {
                assert!(
                    update_column(&mut mine, b.column(0).as_ref(), &f.ty),
                    "{name}.{col}: {:?} as {:?}",
                    b.column(0).data_type(),
                    f.ty
                );
            }
            let theirs = java.estimate();
            let ndv: f64 = file
                .blob_metadata
                .iter()
                .find(|b| b.fields == [f.id])
                .and_then(|b| b.properties.get("ndv"))
                .unwrap_or_else(|| panic!("{name}.{col}: no ndv"))
                .parse::<u64>()
                .unwrap() as f64;
            assert_eq!(
                ndv,
                theirs.trunc(),
                "{name}.{col}: the property is the truncated estimate"
            );
            if java.is_estimation_mode() || mine.is_estimation_mode() {
                estimated += 1;
                assert!(
                    (mine.estimate() - theirs).abs() <= tolerance * theirs,
                    "{name}.{col}: Krishiv {} vs Spark {theirs}",
                    mine.estimate()
                );
            } else {
                exact += 1;
                assert_eq!(
                    mine.compact(true).serialize(),
                    java.serialize(),
                    "{name}.{col}: compact blobs differ (estimate {theirs})"
                );
            }
        }
    }
    assert!(
        exact > 0 && estimated > 0,
        "exact {exact}, estimated {estimated}"
    );
}
