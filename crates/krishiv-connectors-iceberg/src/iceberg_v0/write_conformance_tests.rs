// SPDX-License-Identifier: Apache-2.0
//! The write direction of the Iceberg conformance (P3-T028b, PR-53b): tables Krishiv writes are read by
//! Spark 4.1.0 + iceberg-spark-runtime 1.12.0, which then appends on top, and Krishiv reads that back.
//!
//! Three steps, run in order by `tests/scripts/iceberg_write_conformance.sh` (they need Docker, so both tests are
//! `#[ignore]`d and the default gate never runs them):
//!
//! 1. [`krishiv_writes_the_warehouse`] writes the tables through the append sink into `$KRISHIV_ICEBERG_WRITE_OUT/wh`
//!    (a Hadoop-catalog layout: `db/<table>/metadata/v1.metadata.json` + `version-hint.text`) and records what Krishiv
//!    wrote in `KRISHIV.json`;
//! 2. `tests/scripts/gen_iceberg_write_readback.py` (real Spark) reads each table, appends a row set on top and
//!    writes `READBACK.json`;
//! 3. [`krishiv_agrees_with_spark_readback`] compares `KRISHIV.json` with what Spark read, and reads Spark's
//!    latest metadata through Krishiv.
//!
//! Every expected value in step 3 comes from `READBACK.json`, never from a literal here.
//!
//! Pending (no Krishiv write path on main yet): Iceberg MERGE, position deletes, deletion vectors and equality
//! deletes written by Krishiv (PR-36b units 2-4); Delta MERGE, deletion vectors and CDF (PR-37, #782).

use std::path::PathBuf;
use std::sync::Arc;

use krishiv_exec::planner::TableProvider as _;
use krishiv_exec::testing::Batches;
use krishiv_types::batch::Batch;
use serde_json::{Value, json};

use super::provider_tests::{env_in, provider_session};
use super::sink_tests::{Catalog, PerDriver, input_schema, provider, range, request, rows, run};
use super::table::IcebergTable;
use crate::plan::VersionSpec;

const OUT_VAR: &str = "KRISHIV_ICEBERG_WRITE_OUT";

fn out_dir() -> PathBuf {
    PathBuf::from(
        std::env::var(OUT_VAR).unwrap_or_else(|_| panic!("{OUT_VAR} names the output directory")),
    )
}

/// Table name, format version, identity-partitioned by `x`.
const TABLES: [(&str, u8, bool); 4] = [
    ("unpart", 2, false),
    ("part", 2, true),
    ("v3", 3, false),
    ("v3part", 3, true),
];

fn create(
    env: &super::provider_tests::Env,
    name: &str,
    version: u8,
    partitioned: bool,
) -> iceberg::table::Table {
    use iceberg::spec::{FormatVersion, PartitionSpec, Transform};
    let schema = super::provider_tests::ixn_schema();
    let format = if version == 3 {
        FormatVersion::V3
    } else {
        FormatVersion::V2
    };
    let builder = iceberg::TableCreation::builder()
        .name(name.to_owned())
        .format_version(format);
    let creation = if partitioned {
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_partition_field("x", "x", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        builder
            .schema(schema)
            .partition_spec(spec.into_unbound())
            .build()
    } else {
        builder.schema(schema).build()
    };
    env.rt.block_on(async {
        use iceberg::Catalog as _;
        env.catalog
            .create_table(&iceberg::NamespaceIdent::new("db".to_owned()), creation)
            .await
            .unwrap()
    })
}

/// `id, x, name` of the table at `location` as Krishiv reads it (`CAST` to text, `NULL` for null).
pub(super) fn krishiv_rows(
    env: &super::provider_tests::Env,
    location: &str,
    name: &str,
    version: &VersionSpec,
) -> Vec<String> {
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, location, "db", name))
        .unwrap();
    let p = env.provider(&table, version).unwrap();
    provider_session(p).rows(
        "SELECT CAST(id AS STRING), CAST(x AS STRING), CAST(name AS STRING) FROM t \
         ORDER BY id NULLS FIRST, name",
    )
}

pub(super) fn snapshot_json(m: &iceberg::spec::TableMetadata) -> Vec<Value> {
    let mut snaps: Vec<_> = m.snapshots().collect();
    snaps.sort_by_key(|s| (s.sequence_number(), s.snapshot_id()));
    snaps
        .iter()
        .map(|s| {
            let p = &s.summary().additional_properties;
            json!({
                "snapshot_id": s.snapshot_id(),
                "parent_id": s.parent_snapshot_id(),
                "operation": format!("{:?}", s.summary().operation).to_lowercase(),
                "added_records": p.get("added-records"),
                "total_records": p.get("total-records"),
                "total_data_files": p.get("total-data-files"),
                "krishiv_write_id": p.get(super::append::WRITE_ID_PROP),
            })
        })
        .collect()
}

/// Step 1: Krishiv writes every table in [`TABLES`] with two appends (the first by two drivers, with a row whose
/// id is NULL), into a Hadoop-catalog warehouse Spark can open.
#[test]
#[ignore = "step 1 of tests/scripts/iceberg_write_conformance.sh (needs Docker for the other steps)"]
fn krishiv_writes_the_warehouse() {
    let out = out_dir();
    std::fs::create_dir_all(&out).unwrap();
    let dir = tempfile::Builder::new()
        .prefix("wh")
        .rand_bytes(0)
        .tempdir_in(&out)
        .unwrap();
    let wh = dir.path().to_path_buf();
    let store = Arc::new(object_store::local::LocalFileSystem::new());
    let env = env_in(dir, store, krishiv_io::IoConfig::default());
    let mut record = serde_json::Map::new();
    for (name, version, partitioned) in TABLES {
        let t = create(&env, name, version, partitioned);
        assert_eq!(
            t.metadata().location(),
            format!("{}/db/{name}", krishiv_exec::files::file_url(&wh)),
            "the catalog lays the table out as a Hadoop catalog does"
        );
        let catalog = Catalog::new(&t);
        // First append: two drivers, the second with a NULL id.
        let p = provider(&env, &catalog.location(), &catalog);
        let sink = p.writer(&request(Some("w-1"))).unwrap();
        let input = Arc::new(PerDriver {
            schema: input_schema(),
            batches: vec![
                range(0, 25),
                rows(&[None, Some(25), Some(26), Some(27), Some(28)]),
            ],
            stall: false,
        });
        run(&env, sink, input, None).0.unwrap();
        // Second append on top.
        let p = provider(&env, &catalog.location(), &catalog);
        let sink = p.writer(&request(None)).unwrap();
        let input = Batches::new(input_schema(), vec![Batch::new(range(100, 130))]);
        run(&env, sink, input, None).0.unwrap();

        // The final metadata as the Hadoop catalog names it.
        let meta_dir = wh.join("db").join(name).join("metadata");
        std::fs::copy(
            super::sink_tests::local_path(&catalog.location()),
            meta_dir.join("v1.metadata.json"),
        )
        .unwrap();
        std::fs::write(meta_dir.join("version-hint.text"), "1").unwrap();
        let m = catalog.metadata();
        record.insert(
            name.to_owned(),
            json!({
                "format_version": version,
                "partitioned": partitioned,
                "rows": krishiv_rows(&env, &catalog.location(), name, &VersionSpec::Current),
                "snapshots": snapshot_json(&m),
                "next_row_id": m.next_row_id(),
            }),
        );
    }
    std::fs::write(
        out.join("KRISHIV.json"),
        serde_json::to_vec_pretty(&Value::Object(record)).unwrap(),
    )
    .unwrap();
    // Keep the warehouse: it is the next step's input.
    let _kept = env.dir.keep();
}

fn read_json(name: &str) -> Value {
    let text = std::fs::read_to_string(out_dir().join(name))
        .unwrap_or_else(|e| panic!("{name}: {e} (run the earlier steps first)"));
    serde_json::from_str(&text).unwrap()
}

/// One Spark value as the text `CAST(x AS STRING)` gives.
pub(super) fn text(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_owned(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub(super) fn spark_rows(rows: &Value) -> Vec<String> {
    rows.as_array()
        .unwrap()
        .iter()
        .map(|r| format!("{}|{}|{}", text(&r["id"]), text(&r["x"]), text(&r["name"])))
        .collect()
}

/// Step 3: what Spark read of Krishiv's tables is what Krishiv wrote; Spark's append is on top of Krishiv's last
/// snapshot, and Krishiv reads Spark's result.
#[test]
#[ignore = "step 3 of tests/scripts/iceberg_write_conformance.sh (needs READBACK.json from Spark)"]
fn krishiv_agrees_with_spark_readback() {
    let krishiv = read_json("KRISHIV.json");
    let spark = read_json("READBACK.json");
    assert_eq!(spark["_oracle"]["iceberg"], "1.12.0");
    let store = Arc::new(object_store::local::LocalFileSystem::new());
    let dir = tempfile::tempdir().unwrap();
    let env = env_in(dir, store, krishiv_io::IoConfig::default());
    for (name, _, partitioned) in TABLES {
        let k = &krishiv[name];
        let s = &spark[name];
        // Rows: Krishiv's own read of its table is what Spark read of it (both sorted as text).
        let mut krishiv_sorted: Vec<String> = k["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_str().unwrap().to_owned())
            .collect();
        krishiv_sorted.sort();
        let mut spark_sorted = spark_rows(&s["before"]["rows"]);
        spark_sorted.sort();
        assert_eq!(krishiv_sorted, spark_sorted, "{name}: rows Spark read");
        // Snapshots and history: ids in commit order, operation and counts.
        let ks = k["snapshots"].as_array().unwrap();
        let ss = s["before"]["snapshots"].as_array().unwrap();
        assert_eq!(ks.len(), ss.len(), "{name}: snapshot count");
        for (a, b) in ks.iter().zip(ss) {
            for f in [
                "snapshot_id",
                "parent_id",
                "operation",
                "added_records",
                "total_records",
                "total_data_files",
            ] {
                assert_eq!(text(&a[f]), text(&b[f]), "{name}: snapshot {f}");
            }
        }
        let history: Vec<String> = s["before"]["history"]
            .as_array()
            .unwrap()
            .iter()
            .map(text)
            .collect();
        assert_eq!(
            history,
            ks.iter()
                .map(|x| text(&x["snapshot_id"]))
                .collect::<Vec<_>>(),
            "{name}: history"
        );
        // Schema: Spark's view of the columns.
        let cols: Vec<(String, String)> = s["before"]["schema"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| (text(&c["col_name"]), text(&c["data_type"])))
            .collect();
        assert_eq!(
            cols[..3],
            [
                ("id".to_owned(), "bigint".to_owned()),
                ("x".to_owned(), "int".to_owned()),
                ("name".to_owned(), "string".to_owned())
            ],
            "{name}: schema"
        );
        // Files Spark lists: one set of records per Krishiv data file count.
        let records: i64 = s["before"]["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["record_count"].as_i64().unwrap())
            .sum();
        assert_eq!(
            usize::try_from(records).unwrap(),
            k["rows"].as_array().unwrap().len(),
            "{name}: files"
        );
        assert_eq!(
            s["before"]["files"].as_array().unwrap().len().to_string(),
            text(&ks.last().unwrap()["total_data_files"]),
            "{name}: data files"
        );
        if partitioned {
            let parts = s["before"]["partitions"].as_array().unwrap().len();
            assert_eq!(parts, 3, "{name}: partitions Spark sees (x = 0, 1, 2)");
        }
        // v3 row lineage: Spark's row ids are distinct and below Krishiv's next-row-id.
        if name.starts_with("v3") {
            let ids: Vec<i64> = s["before"]["row_ids"]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: Spark returned no _row_id for a v3 table"))
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let mut uniq = ids.clone();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(
                uniq.len(),
                ids.len(),
                "{name}: Spark's _row_id values are distinct"
            );
            let next = k["next_row_id"].as_i64().unwrap();
            assert!(
                ids.iter().all(|i| (0..next).contains(i)),
                "{name}: row ids below {next}"
            );
            assert_eq!(
                usize::try_from(next).unwrap(),
                ids.len(),
                "{name}: Krishiv's next-row-id counts the rows"
            );
        }

        if name.starts_with("v3") {
            // Each row carries the sequence number of the snapshot that added it: 1 and 2, 30 rows each.
            let seqs: Vec<i64> = s["before"]["last_updated"]
                .as_array()
                .unwrap_or_else(|| {
                    panic!("{name}: Spark returned no sequence numbers for a v3 table")
                })
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            for (n, seq) in [(30, 1), (30, 2)] {
                assert_eq!(
                    seqs.iter().filter(|s| **s == seq).count(),
                    n,
                    "{name}: rows added at sequence {seq}"
                );
            }
        }

        // Spark's append on top of Krishiv's commit.
        let after = &s["after"];
        let last = ks.last().unwrap();
        let after_snaps = after["snapshots"].as_array().unwrap();
        assert_eq!(after_snaps.len(), ks.len() + 1, "{name}");
        assert_eq!(
            text(&after_snaps.last().unwrap()["parent_id"]),
            text(&last["snapshot_id"]),
            "{name}: Spark's snapshot is a child of Krishiv's"
        );
        if name.starts_with("v3") {
            // Spark trusts Krishiv's next-row-id: its two new rows get the next ids, with no overlap.
            let next = k["next_row_id"].as_i64().unwrap();
            let ids: Vec<i64> = after["row_ids"]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: no _row_id after Spark's append"))
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            assert_eq!(
                ids,
                (0..next + 2).collect::<Vec<_>>(),
                "{name}: ids are 0..next-row-id plus Spark's two rows, all distinct"
            );
            let seqs: Vec<i64> = after["last_updated"]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: no sequence numbers after Spark's append"))
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            assert_eq!(
                seqs.iter().filter(|s| **s == 3).count(),
                2,
                "{name}: Spark's two rows carry sequence 3"
            );
        }
        // Krishiv reads Spark's metadata (its file is the one Spark committed).
        let location = after["metadata"].as_str().unwrap().to_owned();
        assert!(
            location.ends_with(".metadata.json"),
            "{name}: Spark names the metadata it committed: {location}"
        );
        let want = {
            let mut w = spark_rows(&after["rows"]);
            w.sort();
            w
        };
        let mut got = krishiv_rows(&env, &location, name, &VersionSpec::Current);
        got.sort();
        assert_eq!(got, want, "{name}: Krishiv reads Spark's table");
        // Time travel to Krishiv's last snapshot still answers Krishiv's rows.
        let mut then = krishiv_rows(
            &env,
            &location,
            name,
            &VersionSpec::Id(last["snapshot_id"].as_i64().unwrap()),
        );
        then.sort();
        assert_eq!(
            then, krishiv_sorted,
            "{name}: time travel to Krishiv's snapshot"
        );
    }
}
