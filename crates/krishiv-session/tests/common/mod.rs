// SPDX-License-Identifier: Apache-2.0
//! The TPC-H schema for session tests: empty tables (plans do not depend on
//! data) or a small deterministic data set in which every join key matches.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, Date32Array, Decimal128Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType as T, Field, Schema};
use krishiv_session::Session;

pub(crate) const TPCH: [(&str, &[(&str, T)]); 8] = [
    (
        "region",
        &[
            ("r_regionkey", T::Int64),
            ("r_name", T::Utf8),
            ("r_comment", T::Utf8),
        ],
    ),
    (
        "nation",
        &[
            ("n_nationkey", T::Int64),
            ("n_name", T::Utf8),
            ("n_regionkey", T::Int64),
            ("n_comment", T::Utf8),
        ],
    ),
    (
        "supplier",
        &[
            ("s_suppkey", T::Int64),
            ("s_name", T::Utf8),
            ("s_address", T::Utf8),
            ("s_nationkey", T::Int64),
            ("s_phone", T::Utf8),
            ("s_acctbal", T::Decimal128(15, 2)),
            ("s_comment", T::Utf8),
        ],
    ),
    (
        "customer",
        &[
            ("c_custkey", T::Int64),
            ("c_name", T::Utf8),
            ("c_address", T::Utf8),
            ("c_nationkey", T::Int64),
            ("c_phone", T::Utf8),
            ("c_acctbal", T::Decimal128(15, 2)),
            ("c_mktsegment", T::Utf8),
            ("c_comment", T::Utf8),
        ],
    ),
    (
        "part",
        &[
            ("p_partkey", T::Int64),
            ("p_name", T::Utf8),
            ("p_mfgr", T::Utf8),
            ("p_brand", T::Utf8),
            ("p_type", T::Utf8),
            ("p_size", T::Int32),
            ("p_container", T::Utf8),
            ("p_retailprice", T::Decimal128(15, 2)),
            ("p_comment", T::Utf8),
        ],
    ),
    (
        "partsupp",
        &[
            ("ps_partkey", T::Int64),
            ("ps_suppkey", T::Int64),
            ("ps_availqty", T::Int32),
            ("ps_supplycost", T::Decimal128(15, 2)),
            ("ps_comment", T::Utf8),
        ],
    ),
    (
        "orders",
        &[
            ("o_orderkey", T::Int64),
            ("o_custkey", T::Int64),
            ("o_orderstatus", T::Utf8),
            ("o_totalprice", T::Decimal128(15, 2)),
            ("o_orderdate", T::Date32),
            ("o_orderpriority", T::Utf8),
            ("o_clerk", T::Utf8),
            ("o_shippriority", T::Int32),
            ("o_comment", T::Utf8),
        ],
    ),
    (
        "lineitem",
        &[
            ("l_orderkey", T::Int64),
            ("l_partkey", T::Int64),
            ("l_suppkey", T::Int64),
            ("l_linenumber", T::Int32),
            ("l_quantity", T::Decimal128(15, 2)),
            ("l_extendedprice", T::Decimal128(15, 2)),
            ("l_discount", T::Decimal128(15, 2)),
            ("l_tax", T::Decimal128(15, 2)),
            ("l_returnflag", T::Utf8),
            ("l_linestatus", T::Utf8),
            ("l_shipdate", T::Date32),
            ("l_commitdate", T::Date32),
            ("l_receiptdate", T::Date32),
            ("l_shipinstruct", T::Utf8),
            ("l_shipmode", T::Utf8),
            ("l_comment", T::Utf8),
        ],
    ),
];

/// One generated cell.
enum V {
    Int(i64),
    Text(String),
}

/// Row `i` of `table`'s column `col`, for a data set of `n` orders (a multiple
/// of 8): 4 lineitems per order, `n / 2` customers and parts, 2 partsupp rows
/// per part, 10 suppliers, 25 nations, 5 regions. Every foreign key resolves,
/// every (partkey, suppkey) of a lineitem is a partsupp row, even parts are
/// "green", and every 8th order is large enough for Q18's HAVING.
fn cell(table: &str, col: &str, i: usize, n: usize) -> V {
    let int = |v: usize| V::Int(i64::try_from(v).unwrap());
    let half = n / 2;
    match (table, col) {
        ("region", "r_regionkey") | ("nation", "n_nationkey") => int(i),
        ("nation", "n_name") => V::Text(format!("NATION{i:02}")),
        ("nation", "n_regionkey") => int(i % 5),
        ("supplier", "s_suppkey")
        | ("customer", "c_custkey")
        | ("part", "p_partkey")
        | ("orders", "o_orderkey") => int(i + 1),
        ("supplier", "s_nationkey") | ("customer", "c_nationkey") => int(i % 25),
        ("part", "p_name") => V::Text(format!(
            "{} part {i}",
            if i.is_multiple_of(2) { "green" } else { "red" }
        )),
        ("partsupp", "ps_partkey") => int(i / 2 + 1),
        ("partsupp", "ps_suppkey") => int((i / 2 + 1 + i % 2) % 10 + 1),
        ("orders", "o_custkey") => int(i % half + 1),
        ("orders", "o_orderdate") => int(8000 + 10 * i),
        ("lineitem", "l_orderkey") => int(i / 4 + 1),
        ("lineitem", "l_partkey") => int((i / 4 + i % 4) % half + 1),
        ("lineitem", "l_suppkey") => int(((i / 4 + i % 4) % half + 1 + i % 2) % 10 + 1),
        // Scaled by 100 (decimal(15,2)).
        ("lineitem", "l_quantity") => int(100
            * if (i / 4).is_multiple_of(8) {
                80
            } else {
                1 + i % 4
            }),
        ("lineitem", "l_extendedprice") => int(100_000 * (1 + i % 4)),
        ("lineitem", "l_discount") => int(2 * (i % 4)),
        ("lineitem", "l_tax") => int(i % 4),
        ("lineitem", "l_returnflag") => V::Text(["A", "N", "R"][i % 3].to_owned()),
        ("lineitem", "l_linestatus") => V::Text(["O", "F"][i % 2].to_owned()),
        ("lineitem", "l_shipdate") => {
            // The first half of the orders ships before Q1's 1998-09-02
            // cutoff (day 10471), the second half after, so the filter
            // removes rows for every data-set size.
            int(8000 + 10 * (i / 4) + 5 + i % 4 + if (i / 4) >= n / 2 { 2500 } else { 0 })
        }
        _ => V::Text(String::new()),
    }
}

/// The column's values: the generated ones, or a default of its type.
fn column(table: &str, col: &str, ty: &T, rows: usize, n: usize) -> ArrayRef {
    let ints = |f: &dyn Fn(usize) -> i64| -> Vec<i64> {
        (0..rows)
            .map(|i| match cell(table, col, i, n) {
                V::Int(v) => v,
                V::Text(_) => f(i),
            })
            .collect()
    };
    match ty {
        T::Int64 => Arc::new(Int64Array::from(ints(&|i| i64::try_from(i).unwrap()))),
        T::Int32 => Arc::new(Int32Array::from_iter_values(
            ints(&|i| i64::try_from(i % 50).unwrap())
                .into_iter()
                .map(|v| i32::try_from(v).unwrap()),
        )),
        T::Date32 => Arc::new(Date32Array::from_iter_values(
            ints(&|i| 8000 + i64::try_from(i).unwrap())
                .into_iter()
                .map(|v| i32::try_from(v).unwrap()),
        )),
        T::Decimal128(p, s) => Arc::new(
            Decimal128Array::from_iter_values(
                ints(&|i| 100 * (i64::try_from(i % 100).unwrap() + 1))
                    .into_iter()
                    .map(i128::from),
            )
            .with_precision_and_scale(*p, *s)
            .unwrap(),
        ),
        T::Utf8 => Arc::new(StringArray::from_iter_values((0..rows).map(
            |i| match cell(table, col, i, n) {
                V::Text(t) if !t.is_empty() => t,
                _ => format!("{col}{}", i % 7),
            },
        ))),
        other => panic!("no generator for {other}"),
    }
}

/// Registers the eight TPC-H tables in `s`: empty for `orders == 0`, else a
/// data set of `orders` orders (a multiple of 8, at least 8).
pub(crate) fn register(s: &Session, orders: usize) {
    assert!(
        orders.is_multiple_of(8),
        "the data set is built in groups of 8 orders"
    );
    for (name, cols) in TPCH {
        let fields: Vec<_> = cols
            .iter()
            .map(|(n, t)| Field::new(*n, t.clone(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let rows = match name {
            _ if orders == 0 => 0,
            "region" => 5,
            "nation" => 25,
            "supplier" => 10,
            "customer" | "part" => orders / 2,
            "partsupp" | "orders" => orders,
            _ => 4 * orders,
        };
        let batches = if rows == 0 {
            vec![]
        } else {
            let arrays = cols
                .iter()
                .map(|(c, t)| column(name, c, t, rows, orders))
                .collect();
            vec![RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap()]
        };
        s.register_batches(name, schema, batches).unwrap();
    }
}

/// The text of TPC-H query `q`.
pub(crate) fn query(q: usize) -> String {
    let path = format!(
        "{}/../../bench/queries/tpch/q{q:02}.sql",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(path).unwrap()
}
