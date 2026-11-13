// SPDX-License-Identifier: Apache-2.0
//! The planning baseline of the design note §2.3 (`P3-lakehouse.md` §5.5): a
//! snapshot of 1,000 manifests and 100,000 entries, planned with a predicate,
//! timed as fetch, parse, map and prune. Ignored: run it with
//! `cargo nextest run -p krishiv-connectors-iceberg --run-ignored only -E 'test(planning_baseline)' --no-capture`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{StreamExt, TryStreamExt};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Datum, Literal, Manifest, ManifestListWriter,
    ManifestWriterBuilder, NestedField, PartitionSpec, PrimitiveType, Schema, Struct, Type,
};
use krishiv_io::{CmpOp, ColumnPath, PushdownExpr, ScalarValue};

use super::scan::{parse_manifest, plan, read_manifest_list};
use super::tests::local;
use crate::plan::{
    IcebergType, LiveEntry, ManifestFilter, PartitionFieldInfo, PartitionSpecInfo, PruneOptions,
    Pruner, SchemaField, TableSnapshot, Transform, inherit,
};

const MANIFESTS: usize = 1_000;
const ENTRIES: usize = 100;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "the planning baseline (1,000 manifests, 100,000 entries): run by hand, numbers in the PR"]
async fn planning_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    let io = reg.file_io();
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "x", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap(),
    );
    let spec = PartitionSpec::builder((*schema).clone())
        .with_spec_id(0)
        .add_partition_field("x", "x", iceberg::spec::Transform::Identity)
        .unwrap()
        .build()
        .unwrap();
    let at = |n: &str| format!("file://{}/{n}", dir.path().display());

    // 1,000 manifests of 100 files: manifest m holds partition x = m, file f
    // the ids f*1000 .. f*1000+999.
    let written = Instant::now();
    let manifests: Vec<_> = futures::stream::iter(0..MANIFESTS)
        .map(|m| {
            let (io, schema, spec) = (io.clone(), Arc::clone(&schema), spec.clone());
            let path = at(&format!("m{m}.avro"));
            async move {
                let mut w = ManifestWriterBuilder::new(
                    io.new_output(&path).unwrap(),
                    Some(1),
                    schema,
                    spec,
                )
                .build_v2_data();
                for f in 0..ENTRIES {
                    let lo = i64::try_from(f * 1000).unwrap();
                    let file = DataFileBuilder::default()
                        .content(DataContentType::Data)
                        .file_path(format!("s3://bucket/t/data/x={m}/{f}.parquet"))
                        .file_format(DataFileFormat::Parquet)
                        .file_size_in_bytes(128 << 20)
                        .record_count(1_000_000)
                        .partition(Struct::from_iter([Some(Literal::int(
                            i32::try_from(m).unwrap(),
                        ))]))
                        .partition_spec_id(0)
                        .value_counts(HashMap::from([(1, 1000), (3, 1000)]))
                        .null_value_counts(HashMap::from([(1, 0), (3, 0)]))
                        .lower_bounds(HashMap::from([(1, Datum::long(lo))]))
                        .upper_bounds(HashMap::from([(1, Datum::long(lo + 999))]))
                        .build()
                        .unwrap();
                    w.add_file(file, -1).unwrap();
                }
                w.write_manifest_file().await.unwrap()
            }
        })
        .buffered(16)
        .collect()
        .await;
    let list = at("snap-1.avro");
    let mut lw = ManifestListWriter::v2(
        io.new_output(&list).unwrap().writer().await.unwrap(),
        1,
        None,
        1,
    );
    lw.add_manifests(manifests.into_iter()).unwrap();
    lw.close().await.unwrap();
    eprintln!("written in {:.2?}", written.elapsed());

    // x = 500 AND id < 5000: one manifest, five files.
    let predicate = PushdownExpr::And(vec![
        PushdownExpr::cmp(ColumnPath::new("x"), CmpOp::Eq, ScalarValue::Int32(500)),
        PushdownExpr::cmp(ColumnPath::new("id"), CmpOp::Lt, ScalarValue::Int64(5000)),
    ]);
    let fields = vec![
        SchemaField {
            id: 1,
            path: ColumnPath::new("id"),
            ty: IcebergType::Long,
        },
        SchemaField {
            id: 2,
            path: ColumnPath::new("name"),
            ty: IcebergType::String,
        },
        SchemaField {
            id: 3,
            path: ColumnPath::new("x"),
            ty: IcebergType::Int,
        },
    ];
    let specs = BTreeMap::from([(
        0,
        PartitionSpecInfo {
            spec_id: 0,
            fields: vec![PartitionFieldInfo::new(
                1000,
                3,
                "x".to_owned(),
                Transform::Identity,
                Some(IcebergType::Int),
            )],
        },
    )]);

    // Fetch: the list and every manifest's bytes (8 at once, as the planner).
    let start = Instant::now();
    let entries = read_manifest_list(&io, &list, 2).await.unwrap();
    let bytes: Vec<_> = futures::stream::iter(entries.iter())
        .map(|m| {
            let io = io.clone();
            async move { io.new_input(&m.path).unwrap().read().await }
        })
        .buffered(8)
        .try_collect()
        .await
        .unwrap();
    let fetch = start.elapsed();
    // Parse: iceberg-rust's Avro decode alone.
    let start = Instant::now();
    for b in &bytes {
        Manifest::parse_avro(b).unwrap();
    }
    let parse = start.elapsed();
    // Map: the decode plus Krishiv's values, minus the decode.
    let start = Instant::now();
    let parsed: Vec<_> = bytes
        .iter()
        .map(|b| parse_manifest(b, 0).unwrap())
        .collect();
    let map = start.elapsed().saturating_sub(parse);
    // Prune: inheritance, the manifest summaries and every entry.
    let start = Instant::now();
    let pruner = Pruner::new(&predicate, &fields, &specs, PruneOptions::default());
    let mut kept = 0;
    let mut total = 0;
    for (m, p) in entries.iter().zip(parsed) {
        let live = inherit(m, p).unwrap();
        total += live.len();
        if !pruner.manifest_may_match(m) {
            continue;
        }
        kept += live
            .iter()
            .filter(|e| matches!(e, LiveEntry::Data(d) if pruner.file_may_match(&d.file)))
            .count();
    }
    let prune = start.elapsed();
    assert_eq!((total, kept), (MANIFESTS * ENTRIES, 5));

    // End to end through the planner (manifests pruned before they are read).
    let snapshot = TableSnapshot {
        table_uuid: "baseline".to_owned(),
        format_version: 2,
        snapshot_id: Some(1),
        sequence_number: Some(1),
        timestamp_ms: Some(0),
        manifest_list: Some(list.clone()),
        schema_id: 0,
        schema: krishiv_types::StructType::new([]),
        fields: fields.clone(),
        partition_specs: specs.clone(),
    };
    let start = Instant::now();
    let unfiltered = plan(
        &io,
        &snapshot,
        None,
        ManifestFilter::All,
        PruneOptions::default(),
    )
    .await
    .unwrap();
    let all = start.elapsed();
    let start = Instant::now();
    let filtered = plan(
        &io,
        &snapshot,
        Some(&predicate),
        ManifestFilter::All,
        PruneOptions::default(),
    )
    .await
    .unwrap();
    let pruned = start.elapsed();
    assert_eq!(unfiltered.tasks.len(), MANIFESTS * ENTRIES);
    assert_eq!(filtered.tasks.len(), 5);
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    eprintln!(
        "planning baseline, {MANIFESTS} manifests x {ENTRIES} entries ({} bytes of manifests): \
         fetch {:.1} ms, parse {:.1} ms, map {:.1} ms, prune {:.1} ms; \
         plan() without a predicate {:.1} ms, with x = 500 AND id < 5000 {:.1} ms",
        bytes.iter().map(bytes::Bytes::len).sum::<usize>(),
        ms(fetch),
        ms(parse),
        ms(map),
        ms(prune),
        ms(all),
        ms(pruned),
    );
}
