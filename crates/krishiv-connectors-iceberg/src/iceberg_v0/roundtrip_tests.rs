// SPDX-License-Identifier: Apache-2.0
//! Spark and Krishiv on the same Iceberg tables (P3-T028b, PR-53b, round trip): Spark 4.1.0 +
//! iceberg-spark-runtime 1.12.0 makes the tables, Krishiv appends and runs copy-on-write and merge-on-read `DELETE`
//! and `UPDATE` on them, Spark reads that and runs merge-on-read DML on top, Krishiv reads Spark's result.
//!
//! Four steps, run in order by `tests/scripts/iceberg_roundtrip.sh` (Docker, so both Rust steps are `#[ignore]`d):
//!
//! 1. `gen_iceberg_roundtrip.py create` (Spark) writes `mor2` (v2, position deletes), `mor3` (v3, a deletion
//!    vector) and `evolved` (`int` id promoted to `bigint`, a renamed column, unpartitioned then `identity(x)`)
//!    into `$KRISHIV_ICEBERG_RT_OUT/wh` and records its reads in `SPARK-CREATE.json`;
//! 2. [`krishiv_meets_spark_tables`] reads each table (it must read what Spark read), appends, deletes a row of a
//!    Spark file that carries Spark's deletes, and updates one row of Spark's and one of its own (copy-on-write),
//!    then on `mor2`/`mor3` deletes (and on `mor2` updates) through merge-on-read; after each commit it records the
//!    snapshot, its own read and the live delete files in `KRISHIV-RT.json`, and leaves its last metadata as the Hadoop
//!    catalog's next version;
//! 3. `gen_iceberg_roundtrip.py readback` (Spark) reads Krishiv's result, time travels to each Krishiv snapshot,
//!    then runs merge-on-read `DELETE`, `UPDATE` and `MERGE` on top and appends (`SPARK-READBACK.json`);
//! 4. [`krishiv_reads_spark_on_top`] compares: Spark read what Krishiv read at every snapshot, Krishiv's snapshots
//!    are Spark's history, Spark lists Krishiv's delete files, v3 row ids survived Krishiv's rewrite, and Krishiv reads Spark's last metadata (its
//!    deletes against Krishiv's files included) and time travels back to its own snapshots.
//!
//! Every expected value is the other engine's record, never a literal here.

use std::path::PathBuf;
use std::sync::Arc;

use iceberg::spec::{
    DataContentType, DataFileFormat, Manifest, ManifestContentType, ManifestStatus, TableMetadata,
};
use krishiv_exec::planner::{RowLevelOperation, TableProvider as _};
use krishiv_exec::testing::Batches;
use krishiv_exec::{Sink, Source};
use krishiv_types::batch::Batch;
use serde_json::{Value, json};

use super::append_tests::MAX;
use super::cow_tests::Dml;
use super::provider_tests::{Env, env_in};
use super::scan::read_manifest_files;
use super::sink_tests::{Catalog, input_schema, local_path, provider, range, request, rows, run};
use super::write_conformance_tests::{krishiv_rows, snapshot_json, spark_rows, text};
use crate::plan::VersionSpec;
use krishiv_common::error::KrishivError;

const OUT_VAR: &str = "KRISHIV_ICEBERG_RT_OUT";

/// The tables `gen_iceberg_roundtrip.py` makes.
const TABLES: [&str; 3] = ["mor2", "mor3", "evolved"];

fn out_dir() -> PathBuf {
    PathBuf::from(
        std::env::var(OUT_VAR).unwrap_or_else(|_| panic!("{OUT_VAR} names the output directory")),
    )
}

fn read_json(name: &str) -> Value {
    let text = std::fs::read_to_string(out_dir().join(name))
        .unwrap_or_else(|e| panic!("{name}: {e} (run the earlier steps first)"));
    serde_json::from_str(&text).unwrap()
}

fn local_env() -> Env {
    let store = Arc::new(object_store::local::LocalFileSystem::new());
    env_in(
        tempfile::tempdir().unwrap(),
        store,
        krishiv_io::IoConfig::default(),
    )
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

/// The current metadata of the table at `location` (a `file:` path, as Spark records it).
fn metadata_at(location: &str) -> TableMetadata {
    serde_json::from_slice(&std::fs::read(local_path(location)).unwrap()).unwrap()
}

/// The Hadoop catalog's next version of table `name`: `metadata` written as `v<N+1>.metadata.json`, the hint moved.
fn publish_to_hadoop_catalog(name: &str, location: &str) -> String {
    let meta = out_dir().join("wh/db").join(name).join("metadata");
    let hint: u64 = std::fs::read_to_string(meta.join("version-hint.text"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let next = meta.join(format!("v{}.metadata.json", hint + 1));
    std::fs::copy(local_path(location), &next).unwrap();
    std::fs::write(meta.join("version-hint.text"), (hint + 1).to_string()).unwrap();
    format!("file:{}", next.display())
}

/// The live delete files of `m`'s current snapshot as Spark's `delete_files` table lists them (one row per entry, so
/// one per DV of a Puffin file): `content`, `file_format`, `record_count`, sorted.
fn delete_files(env: &Env, m: &TableMetadata) -> Vec<Value> {
    let Some(s) = m.current_snapshot() else {
        return Vec::new();
    };
    let io = env.reg.file_io();
    let list = env
        .rt
        .block_on(read_manifest_files(
            &io,
            s.manifest_list(),
            m.format_version() as u8,
            MAX,
        ))
        .unwrap();
    let mut out = Vec::new();
    for mf in list
        .iter()
        .filter(|mf| mf.content == ManifestContentType::Deletes)
    {
        let bytes = env
            .rt
            .block_on(async { io.new_input(&mf.manifest_path).unwrap().read().await })
            .unwrap();
        let (entries, _) = Manifest::parse_avro(&bytes).unwrap().into_parts();
        for e in entries
            .iter()
            .filter(|e| e.status() != ManifestStatus::Deleted)
        {
            let f = e.data_file();
            let content = match f.content_type() {
                DataContentType::PositionDeletes => 1,
                DataContentType::EqualityDeletes => 2,
                DataContentType::Data => unreachable!("a data file in a delete manifest"),
            };
            let format = match f.file_format() {
                DataFileFormat::Puffin => "PUFFIN",
                DataFileFormat::Parquet => "PARQUET",
                other => panic!("delete file format {other:?}"),
            };
            out.push(json!({"content": content, "file_format": format, "record_count": f.record_count()}));
        }
    }
    sorted_values(out)
}

fn sorted_values(mut v: Vec<Value>) -> Vec<Value> {
    v.sort_by_key(Value::to_string);
    v
}

/// One Krishiv step's record: the snapshot it committed, Krishiv's read after it and the live delete files.
fn step(env: &Env, catalog: &Catalog, name: &str, what: &str) -> Value {
    let m = catalog.metadata();
    json!({
        "what": what,
        "snapshot_id": m.current_snapshot().unwrap().snapshot_id(),
        "rows": krishiv_rows(env, &catalog.location(), name, &VersionSpec::Current),
        "delete_files": delete_files(env, &m),
    })
}

/// Step 2: Krishiv reads Spark's tables, appends, runs copy-on-write `DELETE` and `UPDATE` on them, then
/// merge-on-read `DELETE` (and on v2 `UPDATE`) on the merge-on-read ones.
#[test]
#[ignore = "step 2 of tests/scripts/iceberg_roundtrip.sh (needs SPARK-CREATE.json from Spark)"]
fn krishiv_meets_spark_tables() {
    let spark = read_json("SPARK-CREATE.json");
    assert_eq!(spark["_oracle"]["iceberg"], "1.12.0");
    let env = local_env();
    let mut record = serde_json::Map::new();
    for name in TABLES {
        let created = &spark[name]["spark"];
        let location = created["metadata"].as_str().unwrap();
        // Krishiv reads Spark's table as Spark does (deletes applied, evolved columns by field id).
        let read = krishiv_rows(&env, location, name, &VersionSpec::Current);
        assert_eq!(
            sorted(read.clone()),
            sorted(spark_rows(&created["rows"])),
            "{name}: Krishiv reads Spark's table"
        );
        let catalog = Catalog::with_metadata(metadata_at(location), location.to_owned());
        let mut steps = Vec::new();

        // An append: two drivers, one row with a NULL id.
        let p = provider(&env, &catalog.location(), &catalog);
        let sink = p.writer(&request(Some("rt-append"))).unwrap();
        let input = Arc::new(super::sink_tests::PerDriver {
            schema: input_schema(),
            batches: vec![range(100, 104), rows(&[None, Some(104), Some(105)])],
            stall: false,
        });
        run(&env, sink, input, None).0.unwrap();
        steps.push(step(&env, &catalog, name, "append"));

        // DELETE of a row in a Spark file that has Spark's deletes (ids 1 and 5 went in step 1): the rewrite
        // replaces the file and drops its deletes.
        Dml::open(&env, &catalog, RowLevelOperation::Delete)
            .delete("id = 4")
            .unwrap();
        steps.push(step(&env, &catalog, name, "delete"));

        // UPDATE of one row of Spark's and one of Krishiv's. Id 8 sits in Spark's x = 2 file, the one with Spark's
        // delete of id 5, so the rewrite replaces a second Spark file that has deletes.
        Dml::open(&env, &catalog, RowLevelOperation::Update)
            .update("id IN (8, 101)", "'k'")
            .unwrap();
        steps.push(step(&env, &catalog, name, "update"));

        // Merge-on-read on the tables Spark made merge-on-read (`write.*.mode`): a DELETE of a row in a Spark file
        // without deletes (id 3, x = 0) and of one in a Krishiv file (105), a position delete file per data file on
        // v2 and a DV per data file on v3; then, on v2, an UPDATE of a Spark row and a Krishiv one (a v3 UPDATE
        // through merge-on-read is refused until the session selects the lineage columns). Spark's DML in step 3
        // touches other ids.
        if name != "evolved" {
            Dml::open(&env, &catalog, RowLevelOperation::Delete)
                .mor_delete("id IN (3, 105)")
                .unwrap();
            steps.push(step(&env, &catalog, name, "mor-delete"));
            if name == "mor2" {
                Dml::open(&env, &catalog, RowLevelOperation::Update)
                    .mor_update("id IN (6, 100)", "'r'")
                    .unwrap();
                steps.push(step(&env, &catalog, name, "mor-update"));
            }
        }

        // A second plain append on top of the rewrites.
        let p = provider(&env, &catalog.location(), &catalog);
        let sink = p.writer(&request(None)).unwrap();
        let input = Batches::new(input_schema(), vec![Batch::new(range(110, 112))]);
        run(&env, sink, input, None).0.unwrap();
        steps.push(step(&env, &catalog, name, "append"));

        let published = publish_to_hadoop_catalog(name, &catalog.location());
        record.insert(
            name.to_owned(),
            json!({
                "read": read,
                "steps": steps,
                "snapshots": snapshot_json(&catalog.metadata()),
                "metadata": published,
            }),
        );
    }
    race_tables(&env, &spark, &mut record);
    std::fs::write(
        out_dir().join("KRISHIV-RT.json"),
        serde_json::to_vec_pretty(&Value::Object(record)).unwrap(),
    )
    .unwrap();
}

/// The concurrent-writer tables (PR-53b, added 2026-11-20): v2, copy-on-write, partitioned by `x`.
const RACE_TABLES: [&str; 2] = ["race_k", "race_s"];

/// An append of `ids` through a provider pinned at the metadata file `location`.
fn append_at(
    env: &Env,
    location: &str,
    catalog: &Arc<Catalog>,
    ids: (i64, i64),
) -> (Arc<dyn Sink>, Arc<dyn Source>) {
    let p = provider(env, location, catalog);
    let sink = p.writer(&request(None)).unwrap();
    let input: Arc<dyn Source> =
        Batches::new(input_schema(), vec![Batch::new(range(ids.0, ids.1))]);
    (sink, input)
}

/// One race event: what ran, how it ended (`snapshot_id` when it landed, `class` when it
/// conflicted) and Krishiv's read of the catalog's current metadata after it.
fn race_event(
    env: &Env,
    catalog: &Catalog,
    name: &str,
    what: &str,
    outcome: Result<(), KrishivError>,
) -> Value {
    match outcome {
        Ok(()) => step(env, catalog, name, what),
        Err(e) => json!({
            "what": what,
            "class": e.error_class().name(),
            "message": e.to_string(),
            "rows": krishiv_rows(env, &catalog.location(), name, &VersionSpec::Current),
        }),
    }
}

/// Step 2's concurrent-writer cases. `race_k`: two Krishiv writers pinned at one base: appends
/// (both land, the second rebased), a `DELETE` of `x = 0` (the first lands, the second conflicts:
/// its files are gone), `DELETE`s of `x = 1` and of Spark's rows of `x = 2` (both land). `race_s`:
/// Krishiv pinned at the metadata before Spark's `DELETE` of `x = 0`, committing through a catalog
/// whose current metadata is Spark's: a `DELETE` of `x = 1` (lands), of `x = 0` (conflicts), an
/// append (lands). The `DELETE`s come before the append: Krishiv's conflict filter is every row
/// (the session passes no read filter yet), so a `DELETE` after a concurrent append conflicts
/// under `serializable`, where Spark's filter would not.
fn race_tables(env: &Env, spark: &Value, record: &mut serde_json::Map<String, Value>) {
    for name in RACE_TABLES {
        let created = &spark[name]["spark"];
        let location = created["metadata"].as_str().unwrap().to_owned();
        let catalog = Catalog::with_metadata(metadata_at(&location), location.clone());
        let mut events = Vec::new();
        if name == "race_k" {
            let base = catalog.location();
            let (s1, i1) = append_at(env, &base, &catalog, (203, 205));
            let (s2, i2) = append_at(env, &base, &catalog, (206, 208));
            let r = run(env, s1, i1, None).0;
            events.push(race_event(env, &catalog, name, "append", r));
            let r = run(env, s2, i2, None).0;
            events.push(race_event(env, &catalog, name, "append", r));
            let a = Dml::open(env, &catalog, RowLevelOperation::Delete);
            let b = Dml::open(env, &catalog, RowLevelOperation::Delete);
            events.push(race_event(env, &catalog, name, "delete", a.delete("x = 0")));
            events.push(race_event(env, &catalog, name, "delete", b.delete("x = 0")));
            let c = Dml::open(env, &catalog, RowLevelOperation::Delete);
            let d = Dml::open(env, &catalog, RowLevelOperation::Delete);
            events.push(race_event(env, &catalog, name, "delete", c.delete("x = 1")));
            let r = d.delete("x = 2 AND id < 100");
            events.push(race_event(env, &catalog, name, "delete", r));
        } else {
            let base = spark[name]["base"]["metadata"].as_str().unwrap().to_owned();
            let a = Dml::open_at(env, &base, &catalog, RowLevelOperation::Delete);
            let b = Dml::open_at(env, &base, &catalog, RowLevelOperation::Delete);
            let (s, i) = append_at(env, &base, &catalog, (203, 205));
            events.push(race_event(env, &catalog, name, "delete", a.delete("x = 1")));
            events.push(race_event(env, &catalog, name, "delete", b.delete("x = 0")));
            let r = run(env, s, i, None).0;
            events.push(race_event(env, &catalog, name, "append", r));
        }
        let steps: Vec<Value> = events
            .iter()
            .filter(|e| e.get("snapshot_id").is_some())
            .cloned()
            .collect();
        let published = publish_to_hadoop_catalog(name, &catalog.location());
        record.insert(
            name.to_owned(),
            json!({
                "read": created["rows"],
                "steps": steps,
                "race": events,
                "snapshots": snapshot_json(&catalog.metadata()),
                "metadata": published,
            }),
        );
    }
}

/// Step 4's checks of the races: the outcomes the cases expect, a conflict leaves the table
/// readable and unchanged, Spark reads every landed snapshot as Krishiv read it and lists them in
/// its history with Krishiv's parents, and Krishiv reads Spark's DML on top.
fn check_races(env: &Env, created: &Value, krishiv: &Value, spark: &Value) {
    for name in RACE_TABLES {
        let k = &krishiv[name];
        let s = &spark[name];
        let events = k["race"].as_array().unwrap();
        let outcomes: Vec<&str> = events
            .iter()
            .map(|e| {
                if e.get("snapshot_id").is_some() {
                    "lands"
                } else {
                    "conflicts"
                }
            })
            .collect();
        let expected: &[&str] = if name == "race_k" {
            &["lands", "lands", "lands", "conflicts", "lands", "lands"]
        } else {
            &["lands", "conflicts", "lands"]
        };
        assert_eq!(outcomes, expected, "{name}: outcomes {events:#?}");
        let mut rows = spark_rows(&created[name]["spark"]["rows"]);
        for e in events {
            let now = strings(&e["rows"]);
            if let Some(id) = e.get("snapshot_id") {
                let id = text(id);
                assert_eq!(
                    sorted(spark_rows(&s["at"][&id])),
                    sorted(now.clone()),
                    "{name}: Spark at Krishiv's {} snapshot {id}",
                    e["what"]
                );
                assert_ne!(
                    sorted(now.clone()),
                    sorted(rows.clone()),
                    "{name}: {id} changed rows"
                );
            } else {
                assert_eq!(
                    e["class"], "KRISHIV_ICEBERG_COMMIT_CONFLICT",
                    "{name}: conflict class: {}",
                    e["message"]
                );
                assert_eq!(
                    sorted(now.clone()),
                    sorted(rows.clone()),
                    "{name}: a conflict changes nothing and the table stays readable"
                );
            }
            rows = now;
        }
        // Spark's history holds Krishiv's landed snapshots with Krishiv's parents.
        let spark_snapshots = s["before"]["snapshots"].as_array().unwrap();
        for ks in k["snapshots"].as_array().unwrap() {
            let found = spark_snapshots
                .iter()
                .find(|ss| text(&ss["snapshot_id"]) == text(&ks["snapshot_id"]))
                .unwrap_or_else(|| panic!("{name}: Spark lists snapshot {}", ks["snapshot_id"]));
            assert_eq!(
                text(&found["parent_id"]),
                text(&ks["parent_id"]),
                "{name}: parent"
            );
        }
        assert_eq!(
            sorted(spark_rows(&s["before"]["rows"])),
            sorted(rows),
            "{name}: Spark reads the races' result"
        );
        let location = s["after"]["metadata"].as_str().unwrap();
        assert_eq!(
            sorted(krishiv_rows(env, location, name, &VersionSpec::Current)),
            sorted(spark_rows(&s["after"]["rows"])),
            "{name}: Krishiv reads Spark's DML on top"
        );
    }
}

fn ids_of(rows: &Value) -> Vec<(String, i64)> {
    rows.as_array()
        .unwrap()
        .iter()
        .map(|r| (text(&r["id"]), r["_row_id"].as_i64().unwrap()))
        .collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_str().unwrap().to_owned())
        .collect()
}

/// Step 4: both engines agree on every snapshot, and Krishiv reads Spark's DML on top of its own.
#[test]
#[ignore = "step 4 of tests/scripts/iceberg_roundtrip.sh (needs SPARK-READBACK.json from Spark)"]
fn krishiv_reads_spark_on_top() {
    let created = read_json("SPARK-CREATE.json");
    let krishiv = read_json("KRISHIV-RT.json");
    let spark = read_json("SPARK-READBACK.json");
    let env = local_env();
    check_races(&env, &created, &krishiv, &spark);
    for name in TABLES {
        let k = &krishiv[name];
        let s = &spark[name];
        let steps = k["steps"].as_array().unwrap();
        let last = steps.last().unwrap();

        // Spark reads Krishiv's last snapshot as Krishiv does.
        assert_eq!(
            sorted(spark_rows(&s["before"]["rows"])),
            sorted(strings(&last["rows"])),
            "{name}: Spark reads Krishiv's result"
        );
        // Spark's time travel to every Krishiv snapshot answers Krishiv's read of it.
        for st in steps {
            let id = text(&st["snapshot_id"]);
            assert_eq!(
                sorted(spark_rows(&s["at"][&id])),
                sorted(strings(&st["rows"])),
                "{name}: Spark at Krishiv's {} snapshot {id}",
                st["what"]
            );
        }
        // History: Spark's snapshots are Krishiv's, with the same parents and counts. Both sides are keyed by
        // snapshot id, since Spark orders by commit time (two commits can share a millisecond) and Krishiv by
        // sequence number; the parent ids carry the order.
        let by_id = |v: &Value| -> Vec<Value> {
            let mut v = v.as_array().unwrap().clone();
            v.sort_by_key(|r| text(&r["snapshot_id"]));
            v
        };
        let (ks, ss) = (by_id(&k["snapshots"]), by_id(&s["before"]["snapshots"]));
        assert_eq!(ks.len(), ss.len(), "{name}: snapshot count");
        for (a, b) in ks.iter().zip(&ss) {
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
        let spark_created = created[name]["spark"]["snapshots"].as_array().unwrap();
        assert_eq!(
            ks.len(),
            spark_created.len() + steps.len(),
            "{name}: one snapshot per Krishiv step on top of Spark's"
        );
        // Spark lists exactly the delete files Krishiv's last snapshot holds: Krishiv's own merge-on-read deletes
        // (position delete files on v2, DVs on v3) and the Spark deletes it kept.
        let spark_deletes =
            |v: &Value| sorted_values(v["delete_files"].as_array().unwrap().clone());
        assert_eq!(
            spark_deletes(&s["before"]),
            sorted_values(last["delete_files"].as_array().unwrap().clone()),
            "{name}: Spark's delete files are Krishiv's"
        );
        if name != "evolved" {
            let count = |what: &str| -> usize {
                let i = steps.iter().position(|st| st["what"] == what).unwrap();
                steps[i]["delete_files"].as_array().unwrap().len()
            };
            let was = spark_deletes(&created[name]["spark"]).len();
            let pos = |what: &str| steps.iter().position(|st| st["what"] == what).unwrap();
            // Krishiv's copy-on-write DELETE and UPDATE each replaced a Spark file that had deletes: on v3 each
            // file's DV goes with it (`removeDanglingDeletesFor`); on v2 the partitions' position delete files stay
            // live, as Java keeps them.
            let gone = usize::from(name == "mor3");
            assert_eq!(
                count("append"),
                was,
                "{name}: an append adds no delete file"
            );
            assert_eq!(count("delete"), was - gone, "{name}: copy-on-write DELETE");
            assert_eq!(
                count("update"),
                was - 2 * gone,
                "{name}: copy-on-write UPDATE"
            );
            // The merge-on-read DELETE adds one delete file (v2) or DV (v3) for each of the two data files it hits.
            assert_eq!(
                count("mor-delete"),
                count("update") + 2,
                "{name}: merge-on-read DELETE"
            );
            assert!(
                steps[pos("mor-delete")]["delete_files"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(
                        |d| d["file_format"] == if name == "mor3" { "PUFFIN" } else { "PARQUET" }
                    )
                    .count()
                    >= 2,
                "{name}: Krishiv's deletes are {}",
                if name == "mor3" {
                    "DVs"
                } else {
                    "position delete files"
                }
            );
            assert!(
                spark_deletes(&s["after"]).len() > spark_deletes(&s["before"]).len(),
                "{name}: Spark's merge-on-read DML wrote deletes on top"
            );
        }
        // v3: a row Krishiv only moved keeps the `_row_id` Spark gave it; all ids stay distinct.
        if name == "mor3" {
            let before = ids_of(&created[name]["spark"]["row_ids"]);
            let after = ids_of(&s["before"]["row_ids"]);
            let mut uniq: Vec<i64> = after.iter().map(|(_, r)| *r).collect();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), after.len(), "{name}: row ids distinct");
            for (id, row_id) in &before {
                if let Some((_, now)) = after.iter().find(|(i, _)| i == id) {
                    assert_eq!(now, row_id, "{name}: id {id} keeps its _row_id");
                }
            }
        }

        // Krishiv reads Spark's last metadata: merge-on-read deletes on Krishiv's files, the MERGE, the append.
        let location = s["after"]["metadata"].as_str().unwrap();
        assert_eq!(
            sorted(krishiv_rows(&env, location, name, &VersionSpec::Current)),
            sorted(spark_rows(&s["after"]["rows"])),
            "{name}: Krishiv reads Spark's DML on top of its own"
        );
        // And time travels back to each of its own snapshots through Spark's metadata.
        for st in steps {
            let id = st["snapshot_id"].as_i64().unwrap();
            assert_eq!(
                sorted(krishiv_rows(&env, location, name, &VersionSpec::Id(id))),
                sorted(strings(&st["rows"])),
                "{name}: Krishiv at its own snapshot {id} through Spark's metadata"
            );
        }
    }
}
