// SPDX-License-Identifier: Apache-2.0
//! Stage 2b through the whole scan (T085): a row group whose all-dictionary
//! chunk lacks every literal of a pushed equality or `IN` is skipped, the
//! rows equal the stage-off run, and chunks that fell back to plain pages
//! or have no dictionary are read as before.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::file::properties::WriterProperties;

use super::super::ParquetFormat;
use super::super::tests::{Env, write};
use crate::request::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue, ScanRequest};
use crate::{FileRef, FormatOptions, MetadataNeeds, ScanUnitMetrics};

/// Two row groups of 10,000 rows each over `v` (strings) and `n` (`BIGINT`):
/// the first holds the even values below 2,000 (each five times), the
/// second the odd ones. Their min/max overlap, so only the dictionary can
/// tell a row group does not hold a value. No Bloom filters.
fn dictionary_file(env: &Env, name: &str, props: WriterProperties) -> FileRef {
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Utf8, false),
        Field::new("n", DataType::Int64, false),
    ]));
    let values: Vec<i64> = (0..10_000_i64)
        .map(|i| 2 * (i % 1_000))
        .chain((0..10_000_i64).map(|i| 2 * (i % 1_000) + 1))
        .collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from_iter_values(
                values.iter().map(|x| format!("{x:08}")),
            )) as ArrayRef,
            Arc::new(Int64Array::from(values)) as ArrayRef,
        ],
    )
    .unwrap();
    env.put(name, write(&[batch], props))
}

fn props() -> parquet::file::properties::WriterPropertiesBuilder {
    WriterProperties::builder()
        .set_max_row_group_row_count(Some(10_000))
        .set_dictionary_enabled(true)
}

fn scan(
    env: &Env,
    f: &FileRef,
    o: &FormatOptions,
    predicate: PushdownExpr,
) -> (Vec<i64>, ScanUnitMetrics) {
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, f, MetadataNeeds::default(), o),
        )
        .unwrap()
        .schema();
    let mut req = ScanRequest::new(schema, vec![ColumnPath::new("n")]);
    req.predicate = Some(predicate);
    let (out, m) = env.scan_metered(f, &req, o).unwrap();
    let mut ns: Vec<i64> = out
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect();
    ns.sort_unstable();
    (ns, m)
}

fn n_eq(x: i64) -> PushdownExpr {
    PushdownExpr::cmp(ColumnPath::new("n"), CmpOp::Eq, ScalarValue::Int64(x))
}

fn v_eq(x: i64) -> PushdownExpr {
    PushdownExpr::cmp(
        ColumnPath::new("v"),
        CmpOp::Eq,
        ScalarValue::Utf8(format!("{x:08}")),
    )
}

#[test]
fn an_integer_point_lookup_skips_the_row_group_whose_dictionary_lacks_it() {
    let env = Env::new();
    let f = dictionary_file(&env, "d.parquet", props().build());
    let (on, m) = scan(&env, &f, &env.options(&[]), n_eq(1_001));
    assert_eq!(m.units_pruned_dictionary, 1, "{m:?}");
    assert_eq!(m.dictionary_rejected, 0);
    let (off, m_off) = scan(
        &env,
        &f,
        &env.options(&[("dictionaryFilter", "false")]),
        n_eq(1_001),
    );
    assert_eq!(m_off.units_pruned_dictionary, 0);
    // The rows the scan returns are only ever a superset of the matches,
    // and the skipped row group held none of them.
    assert!(on.iter().all(|n| n % 2 == 1), "the even row group was read");
    assert!(on.contains(&1_001));
    assert_eq!(
        on.iter().filter(|n| **n == 1_001).count(),
        off.iter().filter(|n| **n == 1_001).count()
    );
}

#[test]
fn a_string_lookup_and_an_in_list_prune_too() {
    let env = Env::new();
    let f = dictionary_file(&env, "d.parquet", props().build());
    let o = env.options(&[]);
    assert_eq!(scan(&env, &f, &o, v_eq(42)).1.units_pruned_dictionary, 1);
    let both = PushdownExpr::In {
        column: ColumnPath::new("n"),
        values: vec![ScalarValue::Int64(3), ScalarValue::Int64(5)],
        collation: Collation::Utf8Binary,
    };
    assert_eq!(scan(&env, &f, &o, both).1.units_pruned_dictionary, 1);
    // One literal in each row group: nothing is skipped.
    let across = PushdownExpr::In {
        column: ColumnPath::new("n"),
        values: vec![ScalarValue::Int64(2), ScalarValue::Int64(3)],
        collation: Collation::Utf8Binary,
    };
    let (ns, m) = scan(&env, &f, &o, across);
    assert_eq!(m.units_pruned_dictionary, 0, "{m:?}");
    assert!(ns.contains(&2) && ns.contains(&3));
}

#[test]
fn a_value_in_no_row_group_skips_both() {
    let env = Env::new();
    let f = dictionary_file(&env, "d.parquet", props().build());
    // Every integer below 2,000 is stored in one of the groups; a string
    // within both min/max ranges but in neither dictionary is in none.
    let missing = PushdownExpr::cmp(
        ColumnPath::new("v"),
        CmpOp::Eq,
        ScalarValue::Utf8("00000042x".into()),
    );
    let (ns, m) = scan(&env, &f, &env.options(&[]), missing);
    assert_eq!(m.units_pruned_dictionary, 2, "{m:?}");
    assert!(ns.is_empty());
}

#[test]
fn a_chunk_that_fell_back_to_plain_pages_is_not_eligible() {
    let env = Env::new();
    // A dictionary page limit this small makes the writer fall back to
    // plain pages part way through each chunk.
    let f = dictionary_file(
        &env,
        "fallback.parquet",
        props().set_dictionary_page_size_limit(64).build(),
    );
    let (ns, m) = scan(&env, &f, &env.options(&[]), n_eq(1_001));
    assert_eq!(
        (m.units_pruned_dictionary, m.dictionary_rejected),
        (0, 0),
        "{m:?}"
    );
    assert!(ns.contains(&1_001));
}

#[test]
fn a_file_without_dictionaries_reads_nothing_extra() {
    let env = Env::new();
    let f = dictionary_file(
        &env,
        "plain.parquet",
        props().set_dictionary_enabled(false).build(),
    );
    let (_, m) = scan(&env, &f, &env.options(&[]), n_eq(1_001));
    assert_eq!((m.units_pruned_dictionary, m.dictionary_rejected), (0, 0));
}

/// `krishiv.io.parquet.dictionaryFilter.enabled` is the session's default
/// for the `dictionaryFilter` read option, and the option beats it.
#[test]
fn the_session_setting_switches_the_dictionary_stage() {
    let env = Env::new();
    let key = "krishiv.io.parquet.dictionaryFilter.enabled";
    let f = dictionary_file(&env, "d.parquet", props().build());
    let pruned = |session: &[(&str, &str)], user: &[(&str, &str)]| {
        let o = env.options_over(session, user);
        scan(&env, &f, &o, n_eq(1_001)).1.units_pruned_dictionary
    };
    assert_eq!(pruned(&[], &[]), 1);
    assert_eq!(pruned(&[(key, "false")], &[]), 0);
    assert_eq!(
        pruned(&[(key, "false")], &[("dictionaryFilter", "true")]),
        1
    );
    assert_eq!(pruned(&[], &[("dictionaryFilter", "false")]), 0);
}

/// Dates (and timestamps) take no dictionary check: their stored bytes are in
/// the writer's calendar, which a Spark < 3.0 file may rebase, so a literal
/// compared to them could prune rows the guarded read would refuse to return.
#[test]
fn dates_take_no_dictionary_check() {
    let env = Env::new();
    let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, false)]));
    let days: Vec<i32> = (0..10_000)
        .map(|i| 2 * (i % 100))
        .chain((0..10_000).map(|i| 2 * (i % 100) + 1))
        .collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Date32Array::from(days)) as ArrayRef],
    )
    .unwrap();
    let f = env.put("dates.parquet", write(&[batch], props().build()));
    let o = env.options(&[]);
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &f, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    let mut req = ScanRequest::new(schema, vec![ColumnPath::new("d")]);
    req.predicate = Some(PushdownExpr::cmp(
        ColumnPath::new("d"),
        CmpOp::Eq,
        ScalarValue::Date32(101),
    ));
    let (_, m) = env.scan_metered(&f, &req, &o).unwrap();
    assert_eq!(
        (m.units_pruned_dictionary, m.dictionary_rejected),
        (0, 0),
        "{m:?}"
    );
}
