// SPDX-License-Identifier: Apache-2.0
//! FAILFAST inference over records that are not objects: the type it names
//! is the one Spark 4.2.0 names (each partition's records reduced without a
//! seed, then folded into an empty struct). Expected texts are from a
//! Spark 4.2.0 `spark-shell` run (`local[1]`; `local[*]` gives the same for
//! one file).

use std::sync::Arc;

use krishiv_common::error::{ErrorClass, KrishivError};

use super::JsonFormat;
use super::tests::{classes_of, infer, options};
use crate::format::FileFormat;
use crate::io::{FileRef, IoConfig, IoScheduler, StoreId};

fn failfast(multi_line: bool) -> crate::options::FormatOptions {
    options(&serde_json::json!({
        "mode": "FAILFAST",
        "multiLine": if multi_line { "true" } else { "false" },
    }))
}

/// The type a FAILFAST inference error names, `"STRING"` for
/// `... but found "STRING". ...`.
fn named(e: &KrishivError) -> String {
    assert!(
        classes_of(e).contains(&ErrorClass::InvalidJsonRecordType),
        "{e}"
    );
    let text = e.to_string();
    let found = text.split("but found ").nth(1).unwrap_or_default();
    found.split('"').nth(1).unwrap_or_default().to_owned()
}

/// Several in-memory files, listed in the given order.
fn stage(files: &[(&str, &[u8])]) -> (tokio::runtime::Runtime, Arc<IoScheduler>, Vec<FileRef>) {
    let sched = IoScheduler::new(IoConfig::default());
    let store = Arc::new(object_store::memory::InMemory::new());
    let id = StoreId::new("memory://json");
    sched.register_store(
        id.clone(),
        Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("runtime");
    let refs: Vec<FileRef> = rt.block_on(async {
        let mut refs = Vec::new();
        for (name, bytes) in files {
            let path = object_store::path::Path::from(*name);
            object_store::ObjectStoreExt::put(store.as_ref(), &path, bytes.to_vec().into())
                .await
                .expect("put");
            let meta = object_store::ObjectStoreExt::head(store.as_ref(), &path)
                .await
                .expect("head");
            refs.push(FileRef::from_meta(id.clone(), &meta));
        }
        refs
    });
    (rt, sched, refs)
}

/// Inference over several in-memory files, listed in the given order.
fn infer_files(files: &[(&str, &[u8])], multi_line: bool) -> Result<(), KrishivError> {
    let (rt, sched, refs) = stage(files);
    let io = sched.context();
    rt.block_on(JsonFormat.infer_schema(&io, &refs, &failfast(multi_line)))
        .map(|_| ())
        .map_err(|e| e.into_error("f.json"))
}

#[test]
fn failfast_names_the_scalars_merged_before_the_first_object() {
    // (file, the type Spark 4.2.0 names). The seeded fold this replaced
    // failed the scalar-merge cases (the first, fourth, fifth and sixth);
    // the others pin shapes it already named right.
    let cases: [(&[u8], &str); 9] = [
        // Merged first: BIGINT and STRING are STRING (qa-1's T32-1).
        (b"5\n\"str\"\n{\"a\":1}\n", "STRING"),
        // The object comes second: the scalar before it alone.
        (b"5\n{\"a\":1}\n\"str\"\n", "BIGINT"),
        // The object first: the first scalar after it.
        (b"{\"a\":1}\n5\n\"str\"\n", "BIGINT"),
        // Only scalars: their merged type against the empty root.
        (b"5\ntrue\n", "STRING"),
        (b"5\n\"str\"\n", "STRING"),
        (b"5\n1.5\n{\"a\":1}\n", "DOUBLE"),
        // A top-level array is its element type.
        (b"[1,\"x\"]\n{\"a\":1}\n", "STRING"),
        (b"5\n[{\"b\":2}]\n", "BIGINT"),
        // A null record merges as nothing.
        (b"null\n5\n{\"a\":1}\n", "BIGINT"),
    ];
    for (text, want) in cases {
        let e = infer(text, &failfast(false)).expect_err("FAILFAST refuses");
        assert_eq!(named(&e), want, "{}", String::from_utf8_lossy(text));
    }
}

#[test]
fn failfast_multi_line_reads_the_first_value_of_the_file() {
    // Spark 4.2.0: the record is `5` alone.
    let e = infer(b"5\n\"str\"\n{\"a\":1}\n", &failfast(true)).expect_err("refused");
    assert_eq!(named(&e), "BIGINT");
}

#[test]
fn failfast_reads_one_partition_of_small_files_largest_first() {
    // Pins the split order the reduce sees, not the reduce itself: one
    // partition (small files pack into one), read largest first: the
    // object, then `"str"`, so Spark 4.2.0 (`local[1]`) names STRING where
    // listing order (the object, then `5`) would name BIGINT.
    let files: [(&str, &[u8]); 3] = [
        ("a.json", b"{\"a\":1}\n"),
        ("b.json", b"5\n"),
        ("c.json", b"\"str\"\n"),
    ];
    let e = infer_files(&files, false).expect_err("refused");
    assert_eq!(named(&e), "STRING");
}

#[test]
fn failfast_multi_line_merges_the_files_scalars_first() {
    // Files `5`, `"str"`, `{"a":1}`: Spark 4.2.0 names STRING.
    let files: [(&str, &[u8]); 3] = [
        ("m1.json", b"5"),
        ("m2.json", b"\"str\""),
        ("m3.json", b"{\"a\":1}"),
    ];
    let e = infer_files(&files, true).expect_err("refused");
    assert_eq!(named(&e), "STRING");
    // Listing order: Spark 4.2.0 names these the same in `local[1]` and
    // `local[*]`. Its order within a combined split is Hadoop's, not a
    // contract: files `{"a":1}`, `5`, `"str"` named STRING in one run and
    // BIGINT in the next, so that shape is not pinned here.
    let ml3: [(&str, &[u8]); 3] = [
        ("a.json", b"\"str\""),
        ("b.json", b"{\"a\":1}"),
        ("c.json", b"5"),
    ];
    assert_eq!(
        named(&infer_files(&ml3, true).expect_err("refused")),
        "STRING"
    );
    let ml4: [(&str, &[u8]); 3] = [
        ("a.json", b"5"),
        ("b.json", b"{\"a\":1}"),
        ("c.json", b"\"strstrstr\""),
    ];
    assert_eq!(
        named(&infer_files(&ml4, true).expect_err("refused")),
        "BIGINT"
    );
}

#[test]
fn permissive_and_dropmalformed_schemas_follow_the_reduce() {
    let text = b"5\n\"str\"\n{\"a\":1}\n";
    let permissive = infer(text, &options(&serde_json::json!({}))).expect("schema");
    let names: Vec<&str> = permissive
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(names, ["_corrupt_record", "a"]);
    let drop = infer(
        text,
        &options(&serde_json::json!({ "mode": "DROPMALFORMED" })),
    )
    .expect("schema");
    let names: Vec<&str> = drop.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["a"]);
    // Scalars only: PERMISSIVE is the corrupt column alone.
    let only = infer(b"5\n\"str\"\n", &options(&serde_json::json!({}))).expect("schema");
    assert_eq!(only.fields().len(), 1);
    assert_eq!(only.field(0).name(), "_corrupt_record");
    // A scalar before an object that has a field named like the corrupt
    // column: the scalar meets that object, not an empty root, so the field
    // keeps its own type (Spark 4.2.0: `struct<_corrupt_record:bigint>`;
    // the seeded fold added the column as a string first).
    let named_like = infer(
        b"5\n{\"_corrupt_record\":1}\n",
        &options(&serde_json::json!({})),
    )
    .expect("schema");
    assert_eq!(named_like.fields().len(), 1);
    assert_eq!(
        named_like.field(0).data_type(),
        &arrow_schema::DataType::Int64
    );
}

#[test]
fn each_partition_meets_an_empty_root_before_the_others() {
    // Partition 0 holds `5` then `{"_corrupt_record":1}` (a BIGINT field
    // the scalar keeps), partition 1 only `5`. Spark 4.2.0 folds each
    // partition into `StructType(Nil)` first, so partition 1 adds the
    // corrupt column as a STRING and the merge widens to STRING:
    // `struct<_corrupt_record:string>` (one partition gives BIGINT).
    let (rt, sched, refs) = stage(&[
        ("a.json", b"5\n{\"_corrupt_record\":1}\n"),
        ("b.json", b"5\n"),
    ]);
    let io = sched.context();
    let o = options(&serde_json::json!({
        "inferenceMaxPartitionBytes": "8",
        "inferenceOpenCostInBytes": "8",
    }));
    let schema = rt
        .block_on(JsonFormat.infer_schema(&io, &refs, &o))
        .map_err(|e| e.into_error("f.json"))
        .expect("schema");
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "_corrupt_record");
    assert_eq!(
        schema.field(0).data_type(),
        &arrow_schema::DataType::Utf8View
    );
}

#[test]
fn failfast_names_a_top_level_timestamp_ntz() {
    // Under `spark.sql.timestampType=TIMESTAMP_NTZ` a top-level zone-less
    // timestamp text infers `TIMESTAMP_NTZ`, which FAILFAST names (Spark
    // 4.2.1 and 4.3.0 runs: `... but found "TIMESTAMP_NTZ".`).
    let o = options(&serde_json::json!({
        "mode": "FAILFAST",
        "inferTimestamp": "true",
        "spark.sql.timestampType": "TIMESTAMP_NTZ",
    }));
    let e = infer(b"\"2024-01-02T03:04:05\"\n", &o).expect_err("FAILFAST");
    assert_eq!(named(&e), "TIMESTAMP_NTZ");
}
