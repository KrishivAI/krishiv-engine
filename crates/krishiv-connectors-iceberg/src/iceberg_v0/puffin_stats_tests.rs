// SPDX-License-Identifier: Apache-2.0
//! Puffin NDV statistics (P3-T024, PR-54a) on hand-built table metadata: the provider's `distinct` comes
//! from the pinned snapshot's Theta blob only, and a blob read back from a Puffin file is the sketch that
//! was written.

use std::collections::HashMap;

use iceberg::puffin::{APACHE_DATASKETCHES_THETA_V1, Blob, CompressionCodec, PuffinWriter};
use iceberg::spec::StatisticsFile;
use iceberg::spec::{BlobMetadata, NestedField, PrimitiveType, Struct, Type};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use krishiv_exec::planner::TableProvider as _;
use krishiv_optimizer::StatSource;

use super::provider_tests::{Env, batch, data_file, env, id_field, int64s, strings, write_parquet};
use super::puffin_stats::read_sketch;
use crate::plan::IcebergType;
use crate::plan::VersionSpec;
use crate::plan::theta_value::{new_sketch, update_column};
use arrow_array::Int64Array;
use arrow_schema::DataType as A;

fn blob(ty: &str, snapshot: i64, fields: &[i32], ndv: &str) -> BlobMetadata {
    BlobMetadata {
        r#type: ty.to_owned(),
        snapshot_id: snapshot,
        sequence_number: 1,
        fields: fields.to_vec(),
        properties: HashMap::from([("ndv".to_owned(), ndv.to_owned())]),
    }
}

fn stats_file(snapshot: i64, path: &str, blobs: Vec<BlobMetadata>) -> StatisticsFile {
    StatisticsFile {
        snapshot_id: snapshot,
        statistics_path: path.to_owned(),
        file_size_in_bytes: 1,
        file_footer_size_in_bytes: 1,
        key_metadata: None,
        blob_metadata: blobs,
    }
}

fn set_statistics(
    env: &Env,
    table: &iceberg::table::Table,
    file: StatisticsFile,
) -> iceberg::table::Table {
    env.rt.block_on(async {
        let tx = Transaction::new(table);
        let tx = tx
            .update_statistics()
            .set_statistics(file)
            .apply(tx)
            .unwrap();
        tx.commit(&env.catalog).await.unwrap()
    })
}

/// A table of columns `id` (1) and `name` (2) with two appends (1 row, then 3); the first snapshot's id.
fn two_snapshots(env: &Env) -> (iceberg::table::Table, i64) {
    let schema = iceberg::spec::Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();
    let mut t = env.create("t", schema, None);
    let mut first = 0;
    for (i, ids) in [vec![1_i64], vec![2, 3, 4]].into_iter().enumerate() {
        let names: Vec<&str> = ids.iter().map(|_| "n").collect();
        let b = batch(
            vec![id_field("id", A::Int64, 1), id_field("name", A::Utf8, 2)],
            vec![int64s(&ids), strings(&names)],
        );
        let path = env.at(&format!("f{i}.parquet"));
        let size = write_parquet(&path, &b);
        t = env.append(
            &t,
            vec![data_file(&path, size, ids.len() as u64, Struct::empty())],
        );
        if i == 0 {
            first = t.metadata().current_snapshot().unwrap().snapshot_id();
        }
    }
    (t, first)
}

fn distinct(
    env: &Env,
    t: &iceberg::table::Table,
    version: &VersionSpec,
    col: &str,
) -> Option<(u64, StatSource)> {
    fn stats(p: &super::provider::IcebergTableProvider) -> krishiv_optimizer::TableStats {
        p.statistics().unwrap()
    }
    let table = env.load(t);
    let p = env.provider(&table, version).unwrap();
    stats(&p)
        .columns
        .get(col)
        .and_then(|c| c.distinct.map(|d| (d, c.source)))
}

/// The count is the current snapshot's blob's `ndv`, never above the rows; the older snapshot (whose
/// statistics file is not recorded) has none; a non-Theta blob and a two-field blob give none.
#[test]
fn distinct_comes_from_the_pinned_snapshots_blob() {
    let env = env();
    let (t, first) = two_snapshots(&env);
    let current = t.metadata().current_snapshot().unwrap().snapshot_id();
    let theta = APACHE_DATASKETCHES_THETA_V1;
    let t = set_statistics(
        &env,
        &t,
        stats_file(
            current,
            "file:/nowhere.stats",
            vec![
                blob(theta, current, &[1], "3"),
                // 2 blobs' worth of rows is 4: an over-count is clamped.
                blob(theta, current, &[2], "1000"),
            ],
        ),
    );
    assert_eq!(
        distinct(&env, &t, &VersionSpec::Current, "id"),
        Some((3, StatSource::Sketch))
    );
    assert_eq!(
        distinct(&env, &t, &VersionSpec::Current, "name"),
        Some((4, StatSource::Sketch))
    );
    // The first snapshot has no statistics file of its own: another snapshot's blobs are not used.
    assert_eq!(distinct(&env, &t, &VersionSpec::Id(first), "id"), None);

    // A file for the first snapshot holding blobs that must not count.
    let t = set_statistics(
        &env,
        &t,
        stats_file(
            first,
            "file:/nowhere2.stats",
            vec![
                blob("some-other-blob", first, &[1], "1"),
                blob(theta, first, &[1, 2], "1"),
                // Named for another snapshot than its file.
                blob(theta, current, &[2], "1"),
            ],
        ),
    );
    assert_eq!(distinct(&env, &t, &VersionSpec::Id(first), "id"), None);
    assert_eq!(distinct(&env, &t, &VersionSpec::Id(first), "name"), None);
    // The current snapshot is untouched.
    assert_eq!(
        distinct(&env, &t, &VersionSpec::Current, "id"),
        Some((3, StatSource::Sketch))
    );
}

/// A blob written to a Puffin file (zstd, as Java writes) reads back as the sketch that was written.
#[test]
fn a_theta_blob_reads_back_from_its_puffin_file() {
    let env = env();
    let (t, _) = two_snapshots(&env);
    let current = t.metadata().current_snapshot().unwrap().snapshot_id();
    let mut sketch = new_sketch().unwrap();
    update_column(
        &mut sketch,
        &Int64Array::from_iter_values(0..200),
        &IcebergType::Long,
    );
    let bytes = sketch.compact(true).serialize();
    let path = env.at("stats.puffin");
    let io = t.file_io().clone();
    let size = env.rt.block_on(async {
        let out = io.new_output(&path).unwrap();
        let mut w = PuffinWriter::new(&out, HashMap::new(), false)
            .await
            .unwrap();
        for field in [1, 2] {
            let b = Blob::builder()
                .r#type(APACHE_DATASKETCHES_THETA_V1.to_owned())
                .fields(vec![field])
                .snapshot_id(current)
                .sequence_number(2)
                .data(bytes.clone())
                .properties(HashMap::new())
                .build();
            w.add(b, CompressionCodec::Zstd(Default::default()))
                .await
                .unwrap();
        }
        w.close().await.unwrap()
    });
    let file = stats_file(current, &path, vec![]);
    assert!(size.size > 0);
    let got = env
        .rt
        .block_on(read_sketch(&io, &file, 2))
        .unwrap()
        .expect("a blob for field 2");
    assert_eq!(got.serialize(), bytes);
    assert!((got.estimate() - 200.0).abs() < f64::EPSILON);
    // No blob of field 9.
    assert!(
        env.rt
            .block_on(read_sketch(&io, &file, 9))
            .unwrap()
            .is_none()
    );
}
