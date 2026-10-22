// SPDX-License-Identifier: Apache-2.0
//! Late materialisation (`row_filter.rs`) through the whole scan: which rows
//! each row group returns shows whether it was decoded with the filter.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::file::properties::WriterProperties;

use super::tests::{Env, rows, write};
use crate::request::{
    CmpOp, Collation, ColumnPath, PushdownExpr, RowFilterPolicy, ScalarValue, ScanRequest,
};
use crate::{FormatOptions, MetadataNeeds};

use super::ParquetFormat;

/// Rows per row group: more than one decoded batch.
const RG: i32 = 5000;

/// Three row groups of `k` (the predicate column) and `v` (a payload):
/// `k = 0` on 0.5% of row group 0, on one row of row group 1 (inside its
/// first batch) and on ten rows of row group 2; `k = 1` everywhere else.
fn file(env: &Env) -> (crate::FileRef, FormatOptions, Arc<Schema>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("v", DataType::Utf8View, true),
    ]));
    let k = (0..3 * RG).map(|i| match i / RG {
        0 => i32::from(i % 200 != 0),
        1 => i32::from(i != RG + 7),
        _ => i32::from(i % RG >= 10),
    });
    let v = (0..3 * RG).map(|i| Some(format!("row {i}")));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from_iter_values(k)) as ArrayRef,
            Arc::new(StringViewArray::from_iter(v)),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(RG as usize))
        .build();
    let f = env.put("late.parquet", write(&[batch], props));
    let o = env.options(&[]);
    (f, o, schema)
}

fn scan(env: &Env, policy: RowFilterPolicy, predicate: PushdownExpr) -> Vec<RecordBatch> {
    let (f, o, _) = file(env);
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &f, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    let mut req = ScanRequest::new(schema, vec![ColumnPath::new("k"), ColumnPath::new("v")]);
    req.predicate = Some(predicate);
    req.row_filter = policy;
    env.scan(&f, &req, &o).unwrap()
}

fn k_is_zero() -> PushdownExpr {
    PushdownExpr::cmp(ColumnPath::new("k"), CmpOp::Eq, ScalarValue::Int32(0))
}

/// Every returned row's payload, to check rows stay aligned with their key.
fn payloads(out: &[RecordBatch]) -> Vec<(i32, String)> {
    out.iter()
        .flat_map(|b| {
            let k = b.column(0).as_primitive::<Int32Type>().clone();
            let v = b.column(1).as_string_view().clone();
            (0..b.num_rows()).map(move |i| (k.value(i), v.value(i).to_owned()))
        })
        .collect()
}

#[test]
fn the_filter_follows_each_row_groups_measured_fraction() {
    let env = Env::new();
    let out = scan(&env, RowFilterPolicy::Auto, k_is_zero());
    // Row group 0: no measurement yet, filtered (25 of 5000). It kept 0.5%,
    // so row group 1 is read whole and its first batch measured (1 of
    // 4096), so row group 2 is filtered again (10 rows).
    assert_eq!(rows(&out), 25 + 5000 + 10);
    let got = payloads(&out);
    assert!(got.iter().all(|(k, v)| {
        let i: i32 = v.trim_start_matches("row ").parse().unwrap();
        // Row i holds k as written.
        *k == match i / RG {
            0 => i32::from(i % 200 != 0),
            1 => i32::from(i != RG + 7),
            _ => i32::from(i % RG >= 10),
        }
    }));
    let zeros: Vec<&String> = got
        .iter()
        .filter(|(k, _)| *k == 0)
        .map(|(_, v)| v)
        .collect();
    assert_eq!(zeros.len(), 25 + 1 + 10, "every matching row survives");
}

#[test]
fn forced_policies() {
    let env = Env::new();
    assert_eq!(rows(&scan(&env, RowFilterPolicy::On, k_is_zero())), 36);
    assert_eq!(rows(&scan(&env, RowFilterPolicy::Off, k_is_zero())), 15_000);
}

/// A batch the filter keeps more than 1% of passes whole: decoding the
/// payload with a fragmented selection would cost more than it drops.
#[test]
fn dense_batches_pass_whole() {
    let env = Env::new();
    let k_is_one = PushdownExpr::cmp(ColumnPath::new("k"), CmpOp::Eq, ScalarValue::Int32(1));
    let out = scan(&env, RowFilterPolicy::On, k_is_one);
    assert_eq!(rows(&out), 15_000);
}

#[test]
fn substring_predicates_filter_while_decoding() {
    let env = Env::new();
    let contains = PushdownExpr::Contains {
        column: ColumnPath::new("v"),
        needle: "row 1000".to_owned(),
        collation: Collation::Utf8Binary,
    };
    // `row 1000` and `row 10000`..`row 10009`; the predicate column `v` is
    // projected but `k` is not part of the predicate, so filtering pays.
    let out = scan(&env, RowFilterPolicy::On, contains);
    let mut got: Vec<String> = payloads(&out).into_iter().map(|(_, v)| v).collect();
    got.sort();
    let mut want = vec!["row 1000".to_owned()];
    want.extend((10_000..10_010).map(|i| format!("row {i}")));
    assert_eq!(got, want);
}

#[test]
fn nothing_outside_the_predicate_reads_everything() {
    let env = Env::new();
    let (f, o, _) = file(&env);
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &f, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    // Only `k` is read: the filter would save nothing and is not used.
    let mut req = ScanRequest::new(Arc::clone(&schema), vec![ColumnPath::new("k")]);
    req.predicate = Some(k_is_zero());
    req.row_filter = RowFilterPolicy::On;
    assert_eq!(rows(&env.scan(&f, &req, &o).unwrap()), 15_000);
    // A row index needs every row of a decoded row group.
    let mut req = ScanRequest::new(schema, vec![ColumnPath::new("k"), ColumnPath::new("v")]);
    req.predicate = Some(k_is_zero());
    req.row_filter = RowFilterPolicy::On;
    req.metadata_cols.row_index = true;
    let out = env.scan(&f, &req, &o).unwrap();
    assert_eq!(rows(&out), 15_000);
    let last = out.last().unwrap();
    assert_eq!(
        last.column(2)
            .as_primitive::<Int64Type>()
            .value(last.num_rows() - 1),
        14_999
    );
}

/// A column arrow-rs reads as `UInt32` is compared as Spark reads it
/// (`BIGINT`), with the literal typed for the file.
#[test]
fn widened_columns_are_compared_as_spark_reads_them() {
    let env = Env::new();
    let schema = Arc::new(Schema::new(vec![
        Field::new("u", DataType::UInt32, false),
        Field::new("p", DataType::Int32, false),
    ]));
    let n = 3000u32;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(UInt32Array::from_iter_values(
                (0..n).map(|i| u32::MAX - n + i),
            )) as ArrayRef,
            Arc::new(Int32Array::from_iter_values(0..3000)),
        ],
    )
    .unwrap();
    let f = env.put("u.parquet", write(&[batch], WriterProperties::default()));
    let o = env.options(&[]);
    let file_schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &f, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    let mut req = ScanRequest::new(
        file_schema,
        vec![ColumnPath::new("u"), ColumnPath::new("p")],
    );
    req.predicate = Some(PushdownExpr::cmp(
        ColumnPath::new("u"),
        CmpOp::GtEq,
        ScalarValue::Int64(i64::from(u32::MAX) - 2),
    ));
    req.row_filter = RowFilterPolicy::On;
    let out = env.scan(&f, &req, &o).unwrap();
    let p: Vec<i32> = out
        .iter()
        .flat_map(|b| b.column(1).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    assert_eq!(p, [2998, 2999]);
}

/// Runtime-filter keys are evaluated while decoding: published before the
/// scan opens, or later, at the next row-group boundary.
#[test]
fn runtime_filter_keys_filter_while_decoding() {
    use crate::{
        ByteRange, FilterId, FilterSnapshot, KeyFilterBuilder, KeyFilterOptions, RuntimeFilter,
        ScanOptions, ScanStream,
    };

    let env = Env::new();
    let n = 20_000i64;
    // Keys spread over every row group, so statistics prune none.
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Utf8View, false),
    ]));
    let k: Vec<i64> = (0..n).map(|i| i * 7919 % n).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::Int64Array::from(k.clone())) as ArrayRef,
            Arc::new(StringViewArray::from_iter_values(
                // Wide enough for the first guess to filter.
                (0..n).map(|i| format!("payload {i:>40}")),
            )),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1000))
        .build();
    let file = env.put("rf.parquet", write(&[batch], props));
    let o = env.options(&[]);
    let mut b = KeyFilterBuilder::new(DataType::Int64, KeyFilterOptions::default()).unwrap();
    b.add(&arrow_array::Int64Array::from(vec![3, 17_001]), None)
        .unwrap();
    let keys = Arc::new(b.finish());
    let request = |filter: &Arc<RuntimeFilter>| {
        let mut req = ScanRequest::all(Arc::clone(&schema));
        req.dynamic
            .push(crate::ScanFilter::new(Arc::clone(filter), &schema).unwrap());
        req
    };
    let keys_of = |out: &[RecordBatch]| {
        let mut ks: Vec<i64> = out
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect();
        ks.sort_unstable();
        ks
    };

    let early = Arc::new(RuntimeFilter::new(
        FilterId(1),
        ColumnPath::new("k"),
        DataType::Int64,
    ));
    early.publish(FilterSnapshot::Final(Arc::clone(&keys)));
    assert_eq!(
        keys_of(&env.scan(&file, &request(&early), &o).unwrap()),
        [3, 17_001]
    );

    let late = Arc::new(RuntimeFilter::new(
        FilterId(2),
        ColumnPath::new("k"),
        DataType::Int64,
    ));
    let req = request(&late);
    let out = env.rt.block_on(async {
        let pool: Arc<dyn krishiv_memory::MemoryPool> =
            krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
        let id = pool.register("scan", None);
        let mut s = ScanStream::open(
            &env.io,
            &ParquetFormat,
            file.clone(),
            ByteRange::whole(file.size),
            &req,
            &o,
            krishiv_memory::Reservation::new(pool, id),
            ScanOptions::default(),
        )
        .await
        .unwrap();
        let first = s.next_batch().await.unwrap().unwrap();
        late.publish(FilterSnapshot::Final(Arc::clone(&keys)));
        let mut out = vec![first];
        while let Some(b) = s.next_batch().await.unwrap() {
            out.push(b);
        }
        out
    });
    // Row group 0 was read before the keys arrived; after it, only the
    // rows holding a key, and every one of them.
    assert_eq!(out[0].num_rows(), 1000);
    let after = keys_of(&out[1..]);
    assert!(after.iter().all(|k| [3, 17_001].contains(k)), "{after:?}");
    let in_first = k[..1000]
        .iter()
        .filter(|k| [3, 17_001].contains(*k))
        .count();
    assert_eq!(after.len() + in_first, 2);
}

/// Other columns smaller than twice the predicate columns do not pay for
/// the predicate columns' second decode: no filter unless forced.
#[test]
fn narrow_payloads_are_not_filtered() {
    let env = Env::new();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("w", DataType::Int32, false),
    ]));
    let k = (0..3 * RG).map(|i| i32::from(i % 1000 != 0));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from_iter_values(k)) as ArrayRef,
            // As compressible as `k`: no larger.
            Arc::new(Int32Array::from_iter_values(
                (0..3 * RG).map(|i| i32::from(i % 1000 != 1)),
            )),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(RG as usize))
        .build();
    let f = env.put("narrow.parquet", write(&[batch], props));
    let o = env.options(&[]);
    let mut req = ScanRequest::new(schema, vec![ColumnPath::new("k"), ColumnPath::new("w")]);
    req.predicate = Some(k_is_zero());
    let out = env.scan(&f, &req, &o).unwrap();
    assert_eq!(rows(&out), 15_000);
    // Forced on, only the matches.
    req.row_filter = RowFilterPolicy::On;
    assert_eq!(rows(&env.scan(&f, &req, &o).unwrap()), 15);
}
