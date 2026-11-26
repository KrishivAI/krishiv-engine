// SPDX-License-Identifier: Apache-2.0
//! Tests of `INSERT OVERWRITE` into Iceberg tables (`overwrite`, `filter`'s
//! row filter and dropped partitions, PR-36b part 2b) through the
//! provider's `writer`, committed by `sink_tests`' catalog and read back
//! through SQL, with `cow_tests`' table invariants after each case: a
//! truncate, static overwrites on identity, bucket, truncate and day
//! partitions (whole files dropped by the strict projection, a straddling
//! file refused), a metrics-only drop, dynamic overwrites (partitioned,
//! unpartitioned, empty), the delete files and v3 row ids of replaced and
//! untouched partitions, and validation off and on.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Date32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as A, Field, Schema, SchemaRef};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, Literal as IcebergLiteral,
    NestedField, Operation, PartitionSpec, PrimitiveType, Struct, Transform, Type,
};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::planner::{TableProvider, WriteMode, WriteRequest};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, ResolvedExpr,
    ResolvedFunction,
};
use krishiv_types::batch::Batch;

use super::cow_tests::{add_deletes, append, create, get, lineage_rows, live, summary};
use super::provider::IcebergTableProvider;
use super::provider_tests::{Env, env, ixn_schema, provider_session, render};
use super::sink_tests::{Catalog, expected, input_schema, provider, range, read_back, rows, run};
use crate::plan::encode_dv;

/// A column of the table.
fn column(name: &str, id: u64, dt: krishiv_types::DataType) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        BoundColumn {
            id: AttrId(id),
            name: name.to_owned(),
            data_type: dt,
            nullable: true,
        },
    ))))
}

fn call(b: Builtin, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(b),
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: krishiv_types::DataType::Boolean,
            nullable: true,
            deterministic: true,
        },
    ))))
}

fn id() -> Arc<Expr> {
    column("id", 1, krishiv_types::DataType::Long)
}

fn x() -> Arc<Expr> {
    column("x", 2, krishiv_types::DataType::Integer)
}

fn long(v: i64) -> Arc<Expr> {
    Arc::new(Expr::literal(Literal::Long(v)))
}

fn int(v: i32) -> Arc<Expr> {
    Arc::new(Expr::literal(Literal::Integer(v)))
}

/// `INSERT OVERWRITE t PARTITION (x = v)`: Spark's `x <=> v`.
fn static_x(v: i32) -> WriteMode {
    WriteMode::OverwriteByFilter(call(Builtin::EqNullSafe, vec![x(), int(v)]))
}

fn by(b: Builtin, c: Arc<Expr>, v: Arc<Expr>) -> Arc<Expr> {
    call(b, vec![c, v])
}

/// A table `t` of `schema` at format `version`, partitioned by `fields`
/// (source name, transform), with `props`.
fn create_with(
    env: &Env,
    schema: iceberg::spec::Schema,
    version: u8,
    fields: &[(&str, Transform)],
    props: &[(&str, &str)],
) -> iceberg::table::Table {
    let mut spec = PartitionSpec::builder(schema.clone()).with_spec_id(0);
    for (i, (source, t)) in fields.iter().enumerate() {
        spec = spec
            .add_partition_field(*source, format!("p{i}"), *t)
            .unwrap();
    }
    let creation = iceberg::TableCreation::builder()
        .name("t".to_owned())
        .properties(
            props
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect::<std::collections::HashMap<String, String>>(),
        )
        .format_version(if version == 3 {
            FormatVersion::V3
        } else {
            FormatVersion::V2
        })
        .schema(schema)
        .partition_spec(spec.build().unwrap().into_unbound())
        .build();
    env.rt.block_on(async {
        use iceberg::Catalog as _;
        env.catalog
            .create_table(&iceberg::NamespaceIdent::new("db".to_owned()), creation)
            .await
            .unwrap()
    })
}

/// The overwrite `mode` of `batches` (rows of `schema`) through the
/// provider `p`, with write `options`.
fn write_rows(
    env: &Env,
    p: &IcebergTableProvider,
    mode: WriteMode,
    options: &[(&str, &str)],
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
) -> Result<(), KrishivError> {
    let id = format!("ow-{}", super::sink::unique_u64());
    write_rows_as(env, p, mode, options, schema, batches, &id)
}

/// [`write_rows`] under the given write id (a replay carries the id of the first run).
fn write_rows_as(
    env: &Env,
    p: &IcebergTableProvider,
    mode: WriteMode,
    options: &[(&str, &str)],
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    write_id: &str,
) -> Result<(), KrishivError> {
    let mut r = WriteRequest::new(mode, "`db`.`t`", Arc::clone(&schema));
    r.options = options
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect::<BTreeMap<_, _>>();
    r.write_id = Some(write_id.to_owned());
    let sink = p.writer(&r)?;
    let input =
        krishiv_exec::testing::Batches::new(schema, batches.into_iter().map(Batch::new).collect());
    run(env, sink, input, None).0
}

/// [`write_rows`] of [`ixn_schema`] rows through a provider opened now.
fn overwrite(
    env: &Env,
    catalog: &Arc<Catalog>,
    mode: WriteMode,
    options: &[(&str, &str)],
    batches: Vec<RecordBatch>,
) -> Result<(), KrishivError> {
    let p = provider(env, &catalog.location(), catalog);
    write_rows(env, &p, mode, options, input_schema(), batches)
}

fn conflict(e: &KrishivError, text: &str) {
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{e}"
    );
    assert!(e.to_string().contains(text), "{e}");
}

/// The partial-match error: deterministic, so its own class and a SQLSTATE
/// no client retries (review 36b2b M4), Java's text, the files removed.
pub(super) fn partial(e: &KrishivError, text: &str) {
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergOverwritePartialFile,
        "{e}"
    );
    assert_eq!(e.sql_state(), Some("22000"), "{e}");
    assert!(e.to_string().contains(text), "{e}");
    assert!(
        e.to_string().contains("the written files were removed"),
        "{e}"
    );
}

/// `INSERT OVERWRITE` without a partition spec: every file replaced.
#[test]
fn truncate_replaces_every_row() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    overwrite(
        &env,
        &catalog,
        WriteMode::Truncate,
        &[],
        vec![range(50, 53)],
    )
    .unwrap();
    let m = catalog.metadata();
    assert_eq!(read_back(&env, &catalog), expected(50..53));
    let (op, s) = summary(&m);
    assert_eq!(op, Operation::Overwrite);
    assert_eq!(get(&s, "deleted-data-files"), Some("3"));
    assert_eq!(get(&s, "deleted-records"), Some("10"));
    assert_eq!(get(&s, "total-records"), Some("3"));
    assert_eq!(get(&s, "replace-partitions"), None);
    assert_eq!(live(&env, &m).0.len(), 3);
}

/// A static partition on an identity column: the partition's file dropped
/// by the strict projection, the others kept.
#[test]
fn static_overwrite_of_an_identity_partition() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 9));
    overwrite(&env, &catalog, static_x(1), &[], vec![rows(&[Some(100)])]).unwrap();
    // 100 % 3 = 1.
    let mut want = expected([0, 2, 3, 5, 6, 8]);
    want.push("100|1|n100".to_owned());
    want.sort_by_key(|r| r.split('|').next().unwrap().parse::<i64>().unwrap());
    assert_eq!(read_back(&env, &catalog), want);
    let (op, s) = summary(&catalog.metadata());
    assert_eq!(op, Operation::Overwrite);
    assert_eq!(get(&s, "deleted-data-files"), Some("1"));
    assert_eq!(get(&s, "changed-partition-count"), Some("1"));
    live(&env, &catalog.metadata());
}

/// An overwrite whose filter does not translate exactly is refused before
/// anything is written.
#[test]
fn an_untranslatable_filter_is_refused() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    let p = provider(&env, &catalog.location(), &catalog);
    let mut r = WriteRequest::new(
        WriteMode::OverwriteByFilter(by(Builtin::Eq, id(), id())),
        "`db`.`t`",
        input_schema(),
    );
    r.write_id = None;
    let e = p.writer(&r).err().unwrap();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
}

/// Bucket partitions: `id = v` has no strict projection, so a file of its
/// bucket holding other ids is "some, but not all, rows match" (nothing
/// committed); `id >= 0` holds for every file by its metrics.
#[test]
fn bucket_partitions_refuse_a_straddling_file() {
    let env = env();
    let t = create_with(&env, ixn_schema(), 2, &[("id", Transform::Bucket(2))], &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    let before = catalog.metadata().snapshots().count();
    let e = overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteByFilter(by(Builtin::Eq, id(), long(3))),
        &[],
        vec![rows(&[Some(3)])],
    )
    .unwrap_err();
    partial(
        &e,
        "Cannot delete file where some, but not all, rows match filter ref(name=\"id\") == 3:",
    );
    assert_eq!(catalog.metadata().snapshots().count(), before);
    assert_eq!(read_back(&env, &catalog), expected(0..10));
    overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteByFilter(by(Builtin::GtEq, id(), long(0))),
        &[],
        vec![rows(&[Some(7)])],
    )
    .unwrap();
    assert_eq!(read_back(&env, &catalog), expected([7]));
    live(&env, &catalog.metadata());
}

/// Truncate partitions (`truncate(10, id)`): `10 <= id < 20` drops the
/// partition 10 whole and keeps the others; `id < 15` straddles it.
#[test]
fn truncate_partitions_drop_whole_files() {
    let env = env();
    let t = create_with(
        &env,
        ixn_schema(),
        2,
        &[("id", Transform::Truncate(10))],
        &[],
    );
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 30));
    let e = overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteByFilter(by(Builtin::Lt, id(), long(15))),
        &[],
        vec![],
    )
    .unwrap_err();
    partial(&e, "some, but not all, rows match");
    let range_filter = call(
        Builtin::And,
        vec![
            by(Builtin::GtEq, id(), long(10)),
            by(Builtin::Lt, id(), long(20)),
        ],
    );
    overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteByFilter(range_filter),
        &[],
        vec![rows(&[Some(11)])],
    )
    .unwrap();
    let mut want: Vec<i64> = (0..10).collect();
    want.push(11);
    want.extend(20..30);
    assert_eq!(read_back(&env, &catalog), expected(want));
    assert_eq!(
        get(&summary(&catalog.metadata()).1, "deleted-data-files"),
        Some("1")
    );
    live(&env, &catalog.metadata());
}

/// `id long (1), d date (2)`.
fn date_schema() -> iceberg::spec::Schema {
    iceberg::spec::Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "d", Type::Primitive(PrimitiveType::Date)).into(),
        ])
        .build()
        .unwrap()
}

fn date_rows(rows: &[(i64, i32)]) -> RecordBatch {
    RecordBatch::try_new(
        date_input(),
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )),
            Arc::new(Date32Array::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn date_input() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", A::Int64, true),
        Field::new("d", A::Date32, true),
    ]))
}

/// Day partitions: `d >= day 2` drops days 2 and 3 whole, keeps day 1 (the
/// inclusive projection rules it out); a date before 1970 is kept by `d >
/// day -4` (day -3 may hold rows of day -4).
#[test]
fn day_partitions_drop_whole_days() {
    let env = env();
    let t = create_with(&env, date_schema(), 2, &[("d", Transform::Day)], &[]);
    let catalog = Catalog::new(&t);
    let d = || column("d", 2, krishiv_types::DataType::Date);
    let date = |v: i32| Arc::new(Expr::literal(Literal::Date(v)));
    let p = provider(&env, &catalog.location(), &catalog);
    write_rows(
        &env,
        &p,
        WriteMode::Truncate,
        &[],
        date_input(),
        vec![date_rows(&[(1, 1), (2, 2), (3, 2), (4, 3), (5, -3)])],
    )
    .unwrap();
    let ids = |catalog: &Arc<Catalog>| {
        let p = provider(&env, &catalog.location(), catalog);
        render(
            &provider_session(p)
                .run("SELECT id FROM t ORDER BY id")
                .unwrap(),
        )
    };
    let p = provider(&env, &catalog.location(), &catalog);
    write_rows(
        &env,
        &p,
        WriteMode::OverwriteByFilter(by(Builtin::GtEq, d(), date(2))),
        &[],
        date_input(),
        vec![date_rows(&[(9, 5)])],
    )
    .unwrap();
    assert_eq!(ids(&catalog), ["1", "5", "9"]);
    assert_eq!(
        get(&summary(&catalog.metadata()).1, "deleted-data-files"),
        Some("2")
    );
    // Day -3: `d > -4` has no strict projection (old writers' rounding),
    // and the file's metrics (one row, day -3) prove it.
    let p = provider(&env, &catalog.location(), &catalog);
    write_rows(
        &env,
        &p,
        WriteMode::OverwriteByFilter(call(
            Builtin::And,
            vec![
                by(Builtin::Gt, d(), date(-4)),
                by(Builtin::Lt, d(), date(0)),
            ],
        )),
        &[],
        date_input(),
        vec![],
    )
    .unwrap();
    assert_eq!(ids(&catalog), ["1", "9"]);
    assert_eq!(summary(&catalog.metadata()).0, Operation::Delete);
    live(&env, &catalog.metadata());
}

/// An unpartitioned table: a file whose metrics put every row in the filter
/// is dropped, one they rule out kept, one in between refused.
#[test]
fn metrics_alone_drop_files_of_an_unpartitioned_table() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    append(&env, &catalog, range(10, 20));
    let e = overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteByFilter(by(Builtin::GtEq, id(), long(5))),
        &[],
        vec![],
    )
    .unwrap_err();
    partial(&e, "some, but not all, rows match");
    overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteByFilter(by(Builtin::GtEq, id(), long(10))),
        &[],
        vec![range(30, 31)],
    )
    .unwrap();
    let mut want: Vec<i64> = (0..10).collect();
    want.push(30);
    assert_eq!(read_back(&env, &catalog), expected(want));
    assert_eq!(
        get(&summary(&catalog.metadata()).1, "deleted-records"),
        Some("10")
    );
    live(&env, &catalog.metadata());
}

/// A dynamic overwrite replaces the partitions the new rows fall in (x = 0
/// and x = 1) and keeps x = 2; `replace-partitions=true`, an `overwrite`.
#[test]
fn dynamic_overwrite_replaces_two_partitions() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 9));
    overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteDynamic,
        &[],
        vec![rows(&[Some(30), Some(31)])],
    )
    .unwrap();
    let mut want = expected([2, 5, 8, 30, 31]);
    want.sort_by_key(|r| r.split('|').next().unwrap().parse::<i64>().unwrap());
    assert_eq!(read_back(&env, &catalog), want);
    let (op, s) = summary(&catalog.metadata());
    assert_eq!(op, Operation::Overwrite);
    assert_eq!(get(&s, "replace-partitions"), Some("true"));
    assert_eq!(get(&s, "deleted-data-files"), Some("2"));
    assert_eq!(get(&s, "added-data-files"), Some("2"));
    live(&env, &catalog.metadata());
}

/// An unpartitioned table's dynamic overwrite replaces everything; a
/// truncate under `overwrite-mode=dynamic` is one too.
#[test]
fn unpartitioned_dynamic_overwrite_replaces_everything() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    append(&env, &catalog, range(5, 9));
    overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteDynamic,
        &[],
        vec![range(40, 42)],
    )
    .unwrap();
    assert_eq!(read_back(&env, &catalog), expected(40..42));
    let (op, s) = summary(&catalog.metadata());
    assert_eq!(op, Operation::Overwrite);
    assert_eq!(get(&s, "replace-partitions"), Some("true"));
    assert_eq!(get(&s, "deleted-data-files"), Some("2"));
    overwrite(
        &env,
        &catalog,
        WriteMode::Truncate,
        &[("Overwrite-Mode", "DYNAMIC")],
        vec![range(43, 44)],
    )
    .unwrap();
    assert_eq!(read_back(&env, &catalog), expected([43]));
    assert_eq!(
        get(&summary(&catalog.metadata()).1, "replace-partitions"),
        Some("true")
    );
    live(&env, &catalog.metadata());
}

/// An empty dynamic overwrite commits nothing (Spark skips it); an empty
/// static one commits a `delete` snapshot removing the partition.
#[test]
fn an_empty_dynamic_overwrite_skips_the_commit_a_static_one_does_not() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 6));
    let snapshots = catalog.metadata().snapshots().count();
    overwrite(&env, &catalog, WriteMode::OverwriteDynamic, &[], vec![]).unwrap();
    assert_eq!(catalog.metadata().snapshots().count(), snapshots);
    assert_eq!(read_back(&env, &catalog), expected(0..6));
    overwrite(&env, &catalog, static_x(0), &[], vec![]).unwrap();
    let m = catalog.metadata();
    assert_eq!(m.snapshots().count(), snapshots + 1);
    assert_eq!(summary(&m).0, Operation::Delete);
    assert_eq!(read_back(&env, &catalog), expected([1, 2, 4, 5]));
    // An empty truncate of an empty partition set still commits.
    overwrite(&env, &catalog, static_x(7), &[], vec![]).unwrap();
    assert_eq!(catalog.metadata().snapshots().count(), snapshots + 2);
}

/// v3: the DV of a file in a replaced partition goes with it; the rows of
/// the untouched partitions keep their `_row_id`s.
#[test]
fn replaced_partitions_take_their_deletes_and_others_keep_row_ids() {
    let env = env();
    let t = create(&env, 3, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 9));
    add_deletes(&env, &catalog, |files| {
        // The DV of partition x = 1's file: rows 1 and 4.
        let of = files.iter().find(|f| f.contains("x=1")).cloned().unwrap();
        let bitmap: roaring::RoaringTreemap = [0_u64].into_iter().collect();
        let blob = encode_dv(&bitmap).unwrap();
        let mut bytes = b"PFA1".to_vec();
        bytes.extend_from_slice(&blob);
        let path = env.at("x1.puffin");
        std::fs::write(super::sink_tests::local_path(&path), &bytes).unwrap();
        let mut d = DataFileBuilder::default();
        d.content(DataContentType::PositionDeletes)
            .file_path(path)
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(bytes.len() as u64)
            .record_count(1)
            .partition(Struct::from_iter([Some(IcebergLiteral::int(1))]))
            .partition_spec_id(0)
            .referenced_data_file(Some(of))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(blob.len() as i64));
        vec![d.build().unwrap()]
    });
    assert_eq!(live(&env, &catalog.metadata()).1, ["x1.puffin"]);
    let kept_before: Vec<String> = lineage_rows(&env, &catalog)
        .into_iter()
        .filter(|r| r.split('|').next().unwrap().parse::<i64>().unwrap() % 3 != 1)
        .collect();
    overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteDynamic,
        &[],
        vec![rows(&[Some(31)])],
    )
    .unwrap();
    let m = catalog.metadata();
    let (_, deletes) = live(&env, &m);
    assert!(deletes.is_empty(), "{deletes:?}");
    let s = summary(&m).1;
    assert_eq!(get(&s, "removed-dvs"), Some("1"));
    let after: Vec<String> = lineage_rows(&env, &catalog)
        .into_iter()
        .filter(|r| r.split('|').next().unwrap().parse::<i64>().unwrap() % 3 != 1)
        .collect();
    assert_eq!(after, kept_before);
}

/// Validation is off without `isolation-level`: a concurrent append lands
/// between the read and the commit; the rebase re-applies the filter to the
/// new base, so the append into the overwritten partition is removed with
/// it and the one into another partition kept (as Java).
#[test]
fn without_validation_the_rebase_reapplies_the_filter() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 6));
    let p = provider(&env, &catalog.location(), &catalog);
    // Foreign: 10 (x = 1, overwritten) and 11 (x = 2, untouched).
    append(&env, &catalog, range(10, 12));
    write_rows(
        &env,
        &p,
        static_x(1),
        &[],
        input_schema(),
        vec![rows(&[Some(40)])],
    )
    .unwrap();
    let mut want = expected([0, 2, 3, 5, 11, 40]);
    want.sort_by_key(|r| r.split('|').next().unwrap().parse::<i64>().unwrap());
    assert_eq!(read_back(&env, &catalog), want);
    // Two appends and the overwrite.
    assert_eq!(catalog.metadata().snapshots().count(), 3);
    live(&env, &catalog.metadata());
}

/// `isolation-level=serializable` from `validate-from-snapshot-id`: the
/// concurrent append into the overwritten partition conflicts; one into
/// another partition does not.
#[test]
fn serializable_validation_from_the_read_snapshot() {
    for (foreign, fails) in [(10_i64, true), (11, false)] {
        let env = env();
        let t = create(&env, 2, true, &[]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 6));
        let s0 = catalog
            .metadata()
            .current_snapshot_id()
            .unwrap()
            .to_string();
        let p = provider(&env, &catalog.location(), &catalog);
        append(&env, &catalog, rows(&[Some(foreign)]));
        let options = [
            ("isolation-level", "serializable"),
            ("validate-from-snapshot-id", s0.as_str()),
        ];
        let out = write_rows(
            &env,
            &p,
            static_x(1),
            &options,
            input_schema(),
            vec![rows(&[Some(40)])],
        );
        if fails {
            conflict(
                &out.unwrap_err(),
                "Found conflicting files that can contain records matching",
            );
            assert_eq!(catalog.metadata().snapshots().count(), 2);
        } else {
            out.unwrap();
            assert_eq!(catalog.metadata().snapshots().count(), 3);
        }
        live(&env, &catalog.metadata());
    }
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

/// An overwrite with `isolation-level` replayed under its write id (the first run already landed):
/// validation has no exemption for the write's own snapshot, so the replay must be recognised
/// first. It commits nothing, answers success and leaves none of its files behind.
#[test]
fn a_replayed_validated_overwrite_commits_nothing_and_leaves_no_files() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 6));
    let s0 = catalog
        .metadata()
        .current_snapshot_id()
        .unwrap()
        .to_string();
    let p = provider(&env, &catalog.location(), &catalog);
    let options = [
        ("isolation-level", "serializable"),
        ("validate-from-snapshot-id", s0.as_str()),
    ];
    let run = |p: &IcebergTableProvider| {
        write_rows_as(
            &env,
            p,
            static_x(1),
            &options,
            input_schema(),
            vec![rows(&[Some(40)])],
            "replayed-overwrite",
        )
    };
    run(&p).unwrap();
    let snapshots = catalog.metadata().snapshots().count();
    let files = sorted(super::mor_tests::files_ending(&env, ""));
    let replay = provider(&env, &catalog.location(), &catalog);
    run(&replay).unwrap();
    assert_eq!(catalog.metadata().snapshots().count(), snapshots);
    assert_eq!(sorted(super::mor_tests::files_ending(&env, "")), files);
}

/// An unpartitioned dynamic overwrite validates every row: a foreign append
/// conflicts under `serializable`, the message printing Java's `true` for
/// `alwaysTrue` (review 36b2b finding 5).
#[test]
fn an_unpartitioned_dynamic_overwrite_validates_with_the_true_filter() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 3));
    let s0 = catalog
        .metadata()
        .current_snapshot_id()
        .unwrap()
        .to_string();
    let p = provider(&env, &catalog.location(), &catalog);
    append(&env, &catalog, rows(&[Some(7)]));
    let options = [
        ("isolation-level", "serializable"),
        ("validate-from-snapshot-id", s0.as_str()),
    ];
    let e = write_rows(
        &env,
        &p,
        WriteMode::OverwriteDynamic,
        &options,
        input_schema(),
        vec![rows(&[Some(40)])],
    )
    .unwrap_err();
    conflict(
        &e,
        "Found conflicting files that can contain records matching true: [",
    );
}

/// `isolation-level` without `validate-from-snapshot-id` validates the whole
/// history (Java's null starting snapshot): the table's own earlier append
/// into the partition conflicts, and the message names the option. A
/// dynamic overwrite under `snapshot` checks only removed files and
/// deletes, so it commits.
#[test]
fn isolation_without_a_starting_snapshot_names_the_option() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 6));
    let e = overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteDynamic,
        &[("isolation-level", "serializable")],
        vec![rows(&[Some(40)])],
    )
    .unwrap_err();
    conflict(
        &e,
        "Found conflicting files that can contain records matching partitions [x=1]",
    );
    assert!(e.to_string().contains("validate-from-snapshot-id"), "{e}");
    overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteDynamic,
        &[("isolation-level", "snapshot")],
        vec![rows(&[Some(40)])],
    )
    .unwrap();
    let e = overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteDynamic,
        &[("isolation-level", "none")],
        vec![],
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    let e = overwrite(
        &env,
        &catalog,
        WriteMode::Truncate,
        &[
            ("isolation-level", "snapshot"),
            ("validate-from-snapshot-id", "x"),
        ],
        vec![],
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
}

fn double(v: f64) -> Arc<Expr> {
    Arc::new(Expr::literal(Literal::Double(v)))
}

/// A float file whose manifest has bounds and null counts but no NaN counts
/// (older Java writers, other engines) and holds a NaN row: `f < 3.0`
/// cannot be proven for every row (NaN < 3.0 is FALSE in Spark), so the
/// overwrite fails with the partial-match error and the NaN row stays,
/// never removed with the file (review 36b2b B2). `f < NaN` is refused up
/// front (B1).
#[test]
fn nan_rows_of_a_file_without_nan_counts_survive_an_overwrite() {
    use arrow_array::Float64Array;
    use iceberg::spec::Datum;

    use super::evolution_tests::{File, evolved_at, opt, schema};
    use super::provider_tests::id_field;
    let env = env();
    let location = evolved_at(
        &env,
        &[schema(vec![opt(1, "f", PrimitiveType::Double)])],
        vec![
            File::new(
                "old.parquet",
                vec![id_field("f", A::Float64, 1)],
                vec![Arc::new(Float64Array::from(vec![1.0, f64::NAN, 2.0]))],
            )
            .bounds(1, Datum::double(1.0), Datum::double(2.0))
            .counts(1, 3, 0),
        ],
        None,
    );
    let catalog = Catalog::at(&location);
    let input: SchemaRef = Arc::new(Schema::new(vec![Field::new("f", A::Float64, true)]));
    let f = || column("f", 1, krishiv_types::DataType::Double);
    let p = provider(&env, &location, &catalog);
    let e = write_rows(
        &env,
        &p,
        WriteMode::OverwriteByFilter(by(Builtin::Lt, f(), double(3.0))),
        &[],
        Arc::clone(&input),
        vec![],
    )
    .unwrap_err();
    partial(&e, "old.parquet");
    let e = write_rows(
        &env,
        &p,
        WriteMode::OverwriteByFilter(by(Builtin::Lt, f(), double(f64::NAN))),
        &[],
        input,
        vec![],
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    let table = env
        .rt
        .block_on(super::table::IcebergTable::load(
            &env.reg,
            &catalog.location(),
            "db",
            "t",
        ))
        .unwrap();
    let p = env
        .provider(&table, &crate::plan::VersionSpec::Current)
        .unwrap();
    let mut got = provider_session(p).rows("SELECT f FROM t");
    got.sort();
    assert_eq!(got, ["1", "2", "NaN"]);
}

/// A dynamic overwrite after `int -> long` on the identity partition
/// column: the old manifest's 4-byte tuple `v = 5` is the same partition as
/// the new file's 8-byte one, so the old rows are replaced (review 36b2b
/// M3), never kept beside the new ones.
#[test]
fn a_dynamic_overwrite_replaces_a_partition_of_a_promoted_column() {
    use iceberg::spec::Datum;

    use super::evolution_tests::{File, evolved_at, opt, schema};
    use super::provider_tests::id_field;
    let env = env();
    let location = evolved_at(
        &env,
        &[
            schema(vec![
                opt(1, "v", PrimitiveType::Int),
                opt(2, "id", PrimitiveType::Long),
            ]),
            schema(vec![
                opt(1, "v", PrimitiveType::Long),
                opt(2, "id", PrimitiveType::Long),
            ]),
        ],
        vec![
            File::new(
                "old.parquet",
                vec![id_field("v", A::Int32, 1), id_field("id", A::Int64, 2)],
                vec![
                    Arc::new(arrow_array::Int32Array::from(vec![5, 5])),
                    Arc::new(Int64Array::from(vec![1, 2])),
                ],
            )
            .bounds(1, Datum::int(5), Datum::int(5))
            .in_partition(Struct::from_iter([Some(IcebergLiteral::int(5))])),
            File::new(
                "other.parquet",
                vec![id_field("v", A::Int32, 1), id_field("id", A::Int64, 2)],
                vec![
                    Arc::new(arrow_array::Int32Array::from(vec![6])),
                    Arc::new(Int64Array::from(vec![3])),
                ],
            )
            .bounds(1, Datum::int(6), Datum::int(6))
            .in_partition(Struct::from_iter([Some(IcebergLiteral::int(6))])),
        ],
        Some("v"),
    );
    let catalog = Catalog::at(&location);
    let input: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("v", A::Int64, true),
        Field::new("id", A::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input),
        vec![
            Arc::new(Int64Array::from(vec![5])),
            Arc::new(Int64Array::from(vec![10])),
        ],
    )
    .unwrap();
    let p = provider(&env, &location, &catalog);
    write_rows(
        &env,
        &p,
        WriteMode::OverwriteDynamic,
        &[],
        input,
        vec![batch],
    )
    .unwrap();
    let (op, s) = summary(&catalog.metadata());
    assert_eq!(op, Operation::Overwrite);
    assert_eq!(get(&s, "deleted-data-files"), Some("1"));
    let table = env
        .rt
        .block_on(super::table::IcebergTable::load(
            &env.reg,
            &catalog.location(),
            "db",
            "t",
        ))
        .unwrap();
    let p = env
        .provider(&table, &crate::plan::VersionSpec::Current)
        .unwrap();
    let mut got = provider_session(p).rows("SELECT v, id FROM t");
    got.sort();
    assert_eq!(got, ["5|10", "6|3"]);
}

/// A spec of only `void` fields is unpartitioned (`PartitionSpec.isUnpartitioned`):
/// a dynamic overwrite under it replaces the files of every older spec too
/// (`BaseReplacePartitions.apply`, review 36b2b round 3).
#[test]
fn a_dynamic_overwrite_under_a_void_only_spec_replaces_every_spec() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 6));
    let void = iceberg::spec::UnboundPartitionSpec::builder()
        .add_partition_field(2, "x_void", Transform::Void)
        .unwrap()
        .build();
    let m = iceberg::spec::TableMetadataBuilder::new_from_metadata(catalog.metadata(), None)
        .add_partition_spec(void)
        .unwrap()
        .set_default_partition_spec(-1)
        .unwrap()
        .build()
        .unwrap()
        .metadata;
    assert_eq!(m.default_partition_spec_id(), 1);
    let path = env.at("void-spec.metadata.json");
    std::fs::write(
        super::sink_tests::local_path(&path),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    let catalog = Catalog::at(&path);
    overwrite(
        &env,
        &catalog,
        WriteMode::OverwriteDynamic,
        &[],
        vec![range(40, 42)],
    )
    .unwrap();
    assert_eq!(read_back(&env, &catalog), expected(40..42));
    live(&env, &catalog.metadata());
}
