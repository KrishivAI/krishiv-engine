// SPDX-License-Identifier: Apache-2.0
//! Checkpoint reads of the oracle-written tables in `tests/fixtures/delta-4.4.1/` (Spark 4.2.0 +
//! delta-spark 4.4.1): V1, V2 with a sidecar, multi-part. Each table is copied to a temporary
//! root without its pre-checkpoint commits, so the state can only come from the checkpoint,
//! then read through the assembled engine; the live files and their record counts must match
//! what Spark read back (`EXPECTED.json`).

use std::path::Path;

use super::*;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/delta-4.4.1")
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// The table `name` read through the engine: (version, live files, records, partitions).
fn read(name: &str) -> (u64, usize, u64, Vec<String>, serde_json::Value) {
    let (version, files, expected) = scan(name, None);
    let records = files.iter().map(|(_, n, _)| n.unwrap()).sum();
    let mut partitions: Vec<String> = files.iter().map(|(_, _, p)| p.clone().unwrap()).collect();
    partitions.sort();
    (version, files.len(), records, partitions, expected)
}

/// The scan files of the table `name` under `predicate`, read from its checkpoint only.
fn scan(
    name: &str,
    predicate: Option<delta_kernel::expressions::Predicate>,
) -> (u64, Vec<ScanEntry>, serde_json::Value) {
    let source = fixtures().join(name);
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("EXPECTED.json")).unwrap()).unwrap();
    let f = Fixture::new();
    copy_dir(&source, &f.dir);
    let last: serde_json::Value =
        serde_json::from_slice(&std::fs::read(f.dir.join("_delta_log/_last_checkpoint")).unwrap())
            .unwrap();
    let checkpoint = last["version"].as_u64().unwrap();
    for v in 0..checkpoint {
        std::fs::remove_file(f.dir.join(format!("_delta_log/{v:020}.json"))).unwrap();
        let _ = std::fs::remove_file(f.dir.join(format!("_delta_log/{v:020}.crc")));
    }
    let root = f.root.clone();
    let (version, files) = f.run(move |e| {
        let snapshot = Snapshot::builder_for(root.as_str()).build(e).unwrap();
        let schema: Vec<String> = snapshot
            .schema()
            .fields()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(schema, ["id", "p", "t"]);
        (snapshot.version(), scan_files(e, &snapshot, predicate))
    });
    (version, files, expected)
}

fn check(name: &str, version: u64) {
    let (got_version, files, records, partitions, expected) = read(name);
    assert_eq!(got_version, version, "{name}");
    let rows = expected["rows"].as_array().unwrap();
    assert_eq!(records, rows.len() as u64, "{name}: records");
    // One file per partition per insert, so one live file per row.
    assert_eq!(files, rows.len(), "{name}: files");
    let mut want: Vec<String> = rows
        .iter()
        .map(|r| r["p"].as_str().unwrap().to_owned())
        .collect();
    want.sort();
    assert_eq!(partitions, want, "{name}: partitions");
}

#[test]
fn reads_a_v1_checkpoint() {
    check("v1-checkpoint", 4);
}

#[test]
fn reads_a_v2_checkpoint_with_sidecars() {
    check("v2-checkpoint", 4);
}

#[test]
fn reads_a_multipart_checkpoint() {
    check("multipart-checkpoint", 3);
}

/// Data skipping on Spark's checkpoint stats: the V1 checkpoint carries `stats_parsed`
/// (`delta.checkpoint.writeStatsAsStruct`), read through the Parquet handler's nested reconcile
/// and the footer, with Spark's timestamp stats conformed to `UTC` microseconds. Every live file
/// holds one row, so the expected files are those of the rows `EXPECTED.json` says match.
#[test]
fn data_skipping_on_a_spark_checkpoint() {
    use delta_kernel::expressions::{Expression as Expr, Predicate as Pred, Scalar};
    let rows = |expected: &serde_json::Value, keep: &dyn Fn(&serde_json::Value) -> bool| {
        expected["rows"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| keep(r))
            .count()
    };
    let (_, files, expected) = scan(
        "v1-checkpoint",
        Some(Pred::lt(Expr::column(["id"]), Expr::literal(2i64))),
    );
    assert_eq!(
        files.len(),
        rows(&expected, &|r| r["id"].as_i64().unwrap() < 2)
    );
    // `t < 2024-01-02 12:00:00 UTC`: the file with t = 2024-01-03 is skipped by its max, and a
    // file whose only `t` is NULL by its null count (`nullCount = numRecords`: no row can match).
    let noon = chrono::DateTime::parse_from_rfc3339("2024-01-02T12:00:00Z")
        .unwrap()
        .timestamp_micros();
    let (_, files, expected) = scan(
        "v1-checkpoint",
        Some(Pred::lt(
            Expr::column(["t"]),
            Expr::literal(Scalar::Timestamp(noon)),
        )),
    );
    let kept = rows(&expected, &|r| {
        r["t"].as_str().is_some_and(|t| t < "2024-01-02 12:00:00")
    });
    assert_eq!(kept, 1);
    assert_eq!(files.len(), kept);
    assert!(
        kept < expected["rows"].as_array().unwrap().len(),
        "something is skipped"
    );
}
