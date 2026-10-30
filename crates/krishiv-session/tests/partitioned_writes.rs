// SPDX-License-Identifier: Apache-2.0
//! Partitioned writes (`df.write.partition_by`): Spark's directory layout,
//! the commit modes and the error cases, through a session.
#![expect(clippy::unwrap_used, reason = "test code")]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("part", DataType::Utf8, false),
    ]))
}

/// `(id, part)` rows, deliberately unsorted: the write sorts them.
fn batches() -> Vec<RecordBatch> {
    let ids = [1, 2, 3, 4, 5, 6];
    let parts = ["b", "a", "b", "a", "c", "b"];
    vec![
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(ids[..3].to_vec())),
                Arc::new(StringArray::from(parts[..3].to_vec())),
            ],
        )
        .unwrap(),
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(ids[3..].to_vec())),
                Arc::new(StringArray::from(parts[3..].to_vec())),
            ],
        )
        .unwrap(),
    ]
}

fn session() -> Session {
    let s = Session::new().unwrap();
    s.register_batches("t", schema(), batches()).unwrap();
    s
}

/// Relative file paths under `root`, with the job UUID normalized away.
fn layout(root: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace('\\', "/");
                out.push(normalize_uuid(&rel));
            }
        }
    }
    out.sort();
    out
}

/// `part-<task>-<uuid>-...` and `part-<task>-<uuid>....` both become
/// `part-<task>-UUID-...`, so layouts compare across writes.
fn normalize_uuid(path: &str) -> String {
    let mut out = path.to_owned();
    for marker in ["part-00000-", "part-00001-"] {
        while let Some(i) = out.find(marker) {
            let start = i + marker.len();
            let end = start + 36;
            if out.len() >= end {
                out.replace_range(start..end, "UUID");
            } else {
                break;
            }
        }
    }
    out
}

#[test]
fn partitioned_writes_follow_sparks_layout() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("t");
    let s = session();
    block_on(
        s.table("t")
            .write()
            .partition_by(["part"])
            .parquet(out.to_str().unwrap()),
    )
    .unwrap();
    assert_eq!(
        layout(&out),
        [
            "_SUCCESS",
            "part=a/part-00000-UUID.c000.snappy.parquet",
            "part=b/part-00000-UUID.c001.snappy.parquet",
            "part=c/part-00000-UUID.c002.snappy.parquet",
        ]
    );
    // Row counts per partition match, reading each partition directory.
    let mut counts = BTreeMap::new();
    for part in ["a", "b", "c"] {
        let p = out.join(format!("part={part}"));
        let df = s.read().parquet([p.to_str().unwrap()]);
        let batches = block_on(df.collect()).unwrap();
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        counts.insert(part, rows);
        // Data files hold only the non-partition columns.
        let schema = block_on(s.read().parquet([p.to_str().unwrap()]).schema()).unwrap();
        assert_eq!(
            schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>(),
            ["id"]
        );
    }
    assert_eq!(counts, BTreeMap::from([("a", 2), ("b", 3), ("c", 1)]));
}

#[test]
fn partition_by_order_is_respected() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("t");
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Utf8, false),
        Field::new("v", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["x"])),
            Arc::new(StringArray::from(vec!["y"])),
            Arc::new(Int32Array::from(vec![1])),
        ],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    block_on(
        s.table("t")
            .write()
            .partition_by(["b", "a"])
            .parquet(out.to_str().unwrap()),
    )
    .unwrap();
    assert_eq!(
        layout(&out),
        ["_SUCCESS", "b=y/a=x/part-00000-UUID.c000.snappy.parquet"]
    );
}

#[test]
fn partition_errors_follow_spark() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    // Every column partitioned.
    let e = block_on(
        s.table("t")
            .write()
            .partition_by(["id", "part"])
            .parquet(dir.path().join("a").to_str().unwrap()),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::AllPartitionColumnsNotAllowed);
    assert!(
        e.to_string()
            .contains("Cannot use all columns for partition columns"),
        "{e}"
    );
    // An unknown column.
    let e = block_on(
        s.table("t")
            .write()
            .partition_by(["nope"])
            .parquet(dir.path().join("b").to_str().unwrap()),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::PartitionColumnNotFoundInSchema);
    assert!(e.to_string().contains("nope"), "{e}");
    // A bad save mode still wins over partitioning.
    let e = block_on(
        s.table("t")
            .write()
            .mode("bogus")
            .partition_by(["part"])
            .parquet(dir.path().join("c").to_str().unwrap()),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidSaveMode);
    // Unpartitioned names are unchanged (the `-c000` form, not `.c000`).
    let out = dir.path().join("u");
    block_on(s.table("t").write().parquet(out.to_str().unwrap())).unwrap();
    let names = layout(&out);
    assert_eq!(names.len(), 2);
    assert!(names[1].starts_with("part-00000-UUID-c000"), "{names:?}");
}

#[test]
fn static_overwrite_replaces_everything_dynamic_keeps_other_partitions() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let out = dir.path().join("t").to_str().unwrap().to_owned();
    block_on(s.table("t").write().partition_by(["part"]).parquet(&out)).unwrap();
    // STATIC overwrite with other partitions present removes them: write
    // only `a` rows over the full output.
    let a = Session::new().unwrap();
    a.register_batches(
        "t",
        schema(),
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int32Array::from(vec![7])),
                    Arc::new(StringArray::from(vec!["a"])),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    block_on(
        a.table("t")
            .write()
            .partition_by(["part"])
            .mode("overwrite")
            .parquet(&out),
    )
    .unwrap();
    assert_eq!(
        layout(dir.path().join("t").as_path()),
        ["_SUCCESS", "part=a/part-00000-UUID.c000.snappy.parquet"]
    );
    // DYNAMIC overwrite of `b` keeps `a`.
    s.conf()
        .set("spark.sql.sources.partitionOverwriteMode", "DYNAMIC")
        .unwrap();
    s.register_batches(
        "u",
        schema(),
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int32Array::from(vec![8])),
                    Arc::new(StringArray::from(vec!["b"])),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    block_on(
        s.table("u")
            .write()
            .partition_by(["part"])
            .mode("overwrite")
            .parquet(&out),
    )
    .unwrap();
    assert_eq!(
        layout(dir.path().join("t").as_path()),
        [
            "_SUCCESS",
            "part=a/part-00000-UUID.c000.snappy.parquet",
            "part=b/part-00000-UUID.c000.snappy.parquet",
        ]
    );
}

#[test]
fn max_records_per_file_rolls_files() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("t");
    let s = session();
    block_on(
        s.table("t")
            .write()
            .partition_by(["part"])
            .option("maxRecordsPerFile", "2")
            .parquet(out.to_str().unwrap()),
    )
    .unwrap();
    assert_eq!(
        layout(&out),
        [
            "_SUCCESS",
            "part=a/part-00000-UUID.c000.snappy.parquet",
            "part=b/part-00000-UUID.c001.snappy.parquet",
            "part=b/part-00000-UUID.c002.snappy.parquet",
            "part=c/part-00000-UUID.c003.snappy.parquet",
        ]
    );
    let e = block_on(
        s.table("t")
            .write()
            .option("maxRecordsPerFile", "many")
            .parquet(dir.path().join("bad").to_str().unwrap()),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidFormatOptionValue);
}

#[test]
fn the_partition_sort_spills_and_matches() {
    let spill = tempfile::tempdir().unwrap();
    let spill_root = spill.path().join("spill");
    let plain = Session::new().unwrap();
    let limited = Session::builder()
        .config("krishiv.spill.dir", spill_root.to_str().unwrap())
        .config("krishiv.memory.limit", "6m")
        .build()
        .unwrap();
    // 100,000 rows over 50 partitions, unsorted, under a 6 MB limit:
    // the partition sort spills as needed, and the limited run matches
    // the plain one partition by partition.
    let n = 100_000;
    let ids: Vec<i32> = (0..n).collect();
    let parts: Vec<String> = (0..n).map(|i| format!("p{:02}", (i * 7) % 50)).collect();
    let batch_of = |ids: &[i32], parts: &[String]| {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(parts.to_vec())),
            ],
        )
        .unwrap()
    };
    let rows: Vec<(Vec<i32>, Vec<String>)> = ids
        .chunks(2000)
        .zip(parts.chunks(2000))
        .map(|(a, b)| (a.to_vec(), b.to_vec()))
        .collect();
    for session in [&plain, &limited] {
        session
            .register_batches(
                "t",
                schema(),
                rows.iter().map(|(a, b)| batch_of(a, b)).collect(),
            )
            .unwrap();
    }
    let out_plain = spill.path().join("plain");
    let out_limited = spill.path().join("limited");
    block_on(
        plain
            .table("t")
            .write()
            .partition_by(["part"])
            .parquet(out_plain.to_str().unwrap()),
    )
    .unwrap();
    block_on(
        limited
            .table("t")
            .write()
            .partition_by(["part"])
            .parquet(out_limited.to_str().unwrap()),
    )
    .unwrap();
    // Same partitions with the same row counts (file names differ by UUID).
    let per_partition = |root: &std::path::Path| {
        let mut out = BTreeMap::new();
        // Every directory holding data files, through the session reader.
        let mut dirs = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let mut has_file = false;
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "parquet") {
                    has_file = true;
                }
            }
            if has_file {
                dirs.push(dir);
            }
        }
        for dir in dirs {
            let batches =
                block_on(plain.read().parquet([dir.to_str().unwrap()]).collect()).unwrap();
            let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
            out.insert(
                dir.strip_prefix(root).unwrap().to_str().unwrap().to_owned(),
                rows,
            );
        }
        out
    };
    assert_eq!(per_partition(&out_plain), per_partition(&out_limited));
    assert_eq!(per_partition(&out_plain).len(), 50);
    assert_eq!(
        per_partition(&out_plain).values().sum::<usize>(),
        usize::try_from(n).unwrap()
    );
}

#[test]
fn a_repeated_or_unsupported_partition_column_is_refused_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let out = dir.path().join("dup");
    let e = block_on(
        s.table("t")
            .write()
            .partition_by(["part", "PART"])
            .parquet(out.to_str().unwrap()),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::ColumnAlreadyExists);
    assert!(!out.exists(), "nothing is written");

    // A column type with no partition path text fails at analysis.
    let list = Session::new().unwrap();
    let s2 = block_on(list.sql("SELECT 1 AS id, array(1, 2) AS xs, 'a' AS part")).unwrap();
    let out = dir.path().join("list");
    let e = block_on(
        s2.write()
            .partition_by(["xs"])
            .parquet(out.to_str().unwrap()),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(!out.exists(), "nothing is written");
}

#[test]
fn the_write_option_overrides_the_overwrite_mode_conf() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let out = dir.path().join("t").to_str().unwrap().to_owned();
    block_on(s.table("t").write().partition_by(["part"]).parquet(&out)).unwrap();
    s.register_batches(
        "u",
        schema(),
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int32Array::from(vec![8])),
                    Arc::new(StringArray::from(vec!["b"])),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    // The conf says STATIC (the default); the option makes this DYNAMIC.
    block_on(
        s.table("u")
            .write()
            .partition_by(["part"])
            .option("partitionOverwriteMode", "dynamic")
            .mode("overwrite")
            .parquet(&out),
    )
    .unwrap();
    assert_eq!(
        layout(dir.path().join("t").as_path()),
        [
            "_SUCCESS",
            "part=a/part-00000-UUID.c000.snappy.parquet",
            "part=b/part-00000-UUID.c000.snappy.parquet",
            "part=c/part-00000-UUID.c002.snappy.parquet",
        ]
    );
    // Anything but STATIC or DYNAMIC is refused.
    let e = block_on(
        s.table("u")
            .write()
            .partition_by(["part"])
            .option("partitionOverwriteMode", "bogus")
            .mode("overwrite")
            .parquet(&out),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidFormatOptionValue);
}

#[test]
fn an_empty_partitioned_write_leaves_only_the_success_marker() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let out = dir.path().join("e");
    block_on(
        s.table("t")
            .filter(krishiv_session::functions::lit(false))
            .write()
            .partition_by(["part"])
            .parquet(out.to_str().unwrap()),
    )
    .unwrap();
    assert_eq!(layout(&out), ["_SUCCESS"]);
}
