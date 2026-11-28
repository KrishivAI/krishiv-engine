// SPDX-License-Identifier: Apache-2.0
//! Strings of a registered table (Python's `createDataFrame`, `Session::register_batches`) arrive
//! as `Utf8`/`LargeUtf8`, at the top level and nested in arrays (an element named `item`), maps
//! and structs; the engine runs them on the views the analyzer types (R1-D-13). Each function gives the rows it gives over the
//! same values written as SQL `VALUES`, which run on the views already. And `hash`/`xxhash64` of a
//! nested string give Spark 4.3.0's values (R1-D-6).
#![cfg(test)]

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, MapBuilder, StringBuilder};
use arrow_array::{
    Array, ArrayRef, Int32Array, LargeStringArray, ListArray, RecordBatch, StructArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

/// The rows of `sql`, each column printed by Arrow's formatter.
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let options = FormatOptions::default().with_null("NULL");
    let mut out = Vec::new();
    for b in &batches {
        let formats: Vec<_> = b
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), &options).unwrap())
            .collect();
        for row in 0..b.num_rows() {
            let cells: Vec<String> = formats.iter().map(|f| f.value(row).to_string()).collect();
            out.push(cells.join(" | "));
        }
    }
    out
}

/// `g`, `s` (`LargeUtf8`), `sa` (`List<LargeUtf8>`), `mp` (`Map<Utf8, Int32>`) and `st`
/// (`Struct<s: LargeUtf8>`): the carriers pyarrow hands over for pandas strings.
fn registered(s: &Session) {
    let strings = ["b", "a", "c"];
    let s_col = LargeStringArray::from(strings.to_vec());
    let items = LargeStringArray::from(vec!["b", "a", "c", "a", "b"]);
    let sa = ListArray::new(
        Arc::new(Field::new("item", DataType::LargeUtf8, true)),
        OffsetBuffer::from_lengths([2, 1, 2]),
        Arc::new(items),
        None,
    );
    let mut mp = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    for (k, v) in [("k", 1), ("j", 2), ("l", 3)] {
        mp.keys().append_value(k);
        mp.values().append_value(v);
        mp.keys().append_value("z");
        mp.values().append_value(v + 10);
        mp.append(true).unwrap();
    }
    let mp = mp.finish();
    let st = StructArray::from(vec![(
        Arc::new(Field::new("s", DataType::LargeUtf8, true)),
        Arc::new(LargeStringArray::from(strings.to_vec())) as ArrayRef,
    )]);
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Arc::new(s_col),
        Arc::new(sa),
        Arc::new(mp),
        Arc::new(st),
    ];
    let schema = Arc::new(Schema::new(
        ["g", "s", "sa", "mp", "st"]
            .iter()
            .zip(&columns)
            .map(|(n, c)| Field::new(*n, c.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
}

const VALUES: &str = "CREATE TEMPORARY VIEW r AS SELECT * FROM VALUES \
    (1, 'b', array('b', 'a'), map('k', 1, 'z', 11), named_struct('s', 'b')), \
    (2, 'a', array('c'), map('j', 2, 'z', 12), named_struct('s', 'a')), \
    (3, 'c', array('a', 'b'), map('l', 3, 'z', 13), named_struct('s', 'c')) \
    AS r(g, s, sa, mp, st)";

#[test]
fn nested_strings_of_a_registered_table_run_as_over_values() {
    let s = Session::new().unwrap();
    registered(&s);
    rows(&s, VALUES);
    for select in [
        "array_sort(sa)",
        "sort_array(sa, false)",
        "array_distinct(sa)",
        "map_keys(mp)",
        "map_values(mp)",
        "map_entries(mp)",
        "filter(sa, x -> x > 'a')",
        "transform(sa, x -> upper(x))",
        "greatest(s, 'a')",
        "least(s, 'b')",
        "struct(g, s)",
        "array(s, s)",
        "st.s",
        "to_json(st)",
        "hash(sa)",
        "xxhash64(st)",
        "hash(s, sa, st)",
    ] {
        let got = rows(&s, &format!("SELECT g, {select} FROM t ORDER BY g"));
        let want = rows(&s, &format!("SELECT g, {select} FROM r ORDER BY g"));
        assert_eq!(got, want, "{select}");
    }
    for agg in ["max(st)", "collect_list(st)", "max(struct(s, g))"] {
        let got = rows(&s, &format!("SELECT g, {agg} FROM t GROUP BY g ORDER BY g"));
        let want = rows(&s, &format!("SELECT g, {agg} FROM r GROUP BY g ORDER BY g"));
        assert_eq!(got, want, "{agg}");
    }
}

/// R1-D-6: Spark 4.3.0 gives `hash(array('a'))` = 1485273170 and `xxhash64(struct('a'))` =
/// -8582455328737087284 (review-1's run, `research/qa/2026-11-26-acceptance-0.3.1-issues.md`).
#[test]
fn hashes_of_nested_strings_are_sparks() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(&s, "SELECT hash(array('a')), xxhash64(struct('a'))"),
        ["1485273170 | -8582455328737087284"]
    );
}
