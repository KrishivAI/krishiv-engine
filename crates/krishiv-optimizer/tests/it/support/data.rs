// SPDX-License-Identifier: Apache-2.0
//! A small, deterministic TPC-H-shaped database whose values are drawn so
//! that every query's predicates select some rows (keys join as in TPC-H;
//! names, types, dates and comments come from the queries' constants).

use super::Rng;
use super::eval::{Db, V, days};

const NATIONS: [(&str, i64); 25] = [
    ("ALGERIA", 0),
    ("ARGENTINA", 1),
    ("BRAZIL", 1),
    ("CANADA", 1),
    ("EGYPT", 4),
    ("ETHIOPIA", 0),
    ("FRANCE", 3),
    ("GERMANY", 3),
    ("INDIA", 2),
    ("INDONESIA", 2),
    ("IRAN", 4),
    ("IRAQ", 4),
    ("JAPAN", 2),
    ("JORDAN", 4),
    ("KENYA", 0),
    ("MOROCCO", 0),
    ("MOZAMBIQUE", 0),
    ("PERU", 1),
    ("CHINA", 2),
    ("ROMANIA", 3),
    ("SAUDI ARABIA", 4),
    ("VIETNAM", 2),
    ("RUSSIA", 3),
    ("UNITED KINGDOM", 3),
    ("UNITED STATES", 1),
];

/// Part sizes, the queries' among them.
const SIZES: [i64; 12] = [1, 3, 5, 9, 10, 14, 15, 19, 23, 36, 45, 49];

/// Nations the queries name, drawn more often.
const HOT_NATIONS: [i64; 8] = [2, 3, 6, 7, 8, 20, 12, 18];

fn s(v: &str) -> V {
    V::Str(v.into())
}

fn table(db: &mut Db, name: &str, cols: &[&str], rows: Vec<Vec<V>>) {
    let cols = cols.iter().map(|c| (*c).to_owned()).collect();
    db.tables.insert(name.to_owned(), (cols, rows));
}

/// `t(a int, b string, c bigint)` and `u(a int, d string)`: a few rows of
/// small values, some NULL, so joins match and predicates split.
pub(crate) fn small(seed: u64) -> Db {
    let mut r = Rng(seed | 1);
    let int = |r: &mut Rng| {
        if r.range(0, 5) == 0 {
            V::Null
        } else {
            V::int(r.range(-2, 3))
        }
    };
    let text = |r: &mut Rng| match r.range(0, 4) {
        0 => V::Null,
        _ => s(r.pick(&["x", "y", "z"])),
    };
    let mut db = Db::default();
    let rows = (0..10)
        .map(|_| vec![int(&mut r), text(&mut r), int(&mut r)])
        .collect();
    table(&mut db, "t", &["a", "b", "c"], rows);
    let rows = (0..8).map(|_| vec![int(&mut r), text(&mut r)]).collect();
    table(&mut db, "u", &["a", "d"], rows);
    db
}

/// The TPC-H database, with `orders` orders.
pub(crate) fn tpch(seed: u64, orders: i64) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let nation = |r: &mut Rng| {
        if r.range(0, 2) == 0 {
            r.range(0, 24)
        } else {
            HOT_NATIONS[usize::try_from(r.range(0, 7)).unwrap()]
        }
    };
    let regions = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"];
    let rows = (0..5).map(|i| vec![V::int(i), s(regions[usize::try_from(i).unwrap()]), s("r")]);
    table(
        &mut db,
        "region",
        &["r_regionkey", "r_name", "r_comment"],
        rows.collect(),
    );
    let rows = NATIONS.iter().enumerate().map(|(i, (n, reg))| {
        vec![
            V::int(i64::try_from(i).unwrap()),
            s(n),
            V::int(*reg),
            s("n"),
        ]
    });
    table(
        &mut db,
        "nation",
        &["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        rows.collect(),
    );

    let suppliers = 12;
    let rows = (1..=suppliers)
        .map(|k| {
            let comment = if k % 5 == 0 {
                "slow Customer x Complaints"
            } else {
                "fine"
            };
            vec![
                V::int(k),
                s(&format!("Supplier#{k:03}")),
                s(&format!("addr{k}")),
                V::int(nation(&mut r)),
                s(&format!("{}-555", 10 + k)),
                V::cents(r.range(-99_999, 999_999)),
                s(comment),
            ]
        })
        .collect();
    let scols = [
        "s_suppkey",
        "s_name",
        "s_address",
        "s_nationkey",
        "s_phone",
        "s_acctbal",
        "s_comment",
    ];
    table(&mut db, "supplier", &scols, rows);

    let parts = 40;
    let mut rows = Vec::new();
    for k in 1..=parts {
        let name = r.pick(&[
            "forest green",
            "forest blue",
            "lime green",
            "navy",
            "green cream",
        ]);
        let kind = format!(
            "{} {} {}",
            r.pick(&["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]),
            r.pick(&["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"]),
            r.pick(&["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"])
        );
        let kind = if k % 7 == 0 {
            "ECONOMY ANODIZED STEEL".to_owned()
        } else {
            kind
        };
        let container = format!(
            "{} {}",
            r.pick(&["SM", "MED", "LG", "JUMBO"]),
            r.pick(&["CASE", "BOX", "BAG", "PKG", "PACK"])
        );
        rows.push(vec![
            V::int(k),
            s(name),
            s(&format!("Manufacturer#{}", r.range(1, 5))),
            s(r.pick(&["Brand#12", "Brand#23", "Brand#34", "Brand#45", "Brand#11"])),
            s(&kind),
            V::int(SIZES[usize::try_from(r.range(0, 11)).unwrap()]),
            s(&container),
            V::cents(r.range(90_000, 200_000)),
            s("p"),
        ]);
    }
    let pcols = [
        "p_partkey",
        "p_name",
        "p_mfgr",
        "p_brand",
        "p_type",
        "p_size",
        "p_container",
        "p_retailprice",
        "p_comment",
    ];
    table(&mut db, "part", &pcols, rows);

    let supp_of = |p: i64, i: i64| (p + i * (suppliers / 4)) % suppliers + 1;
    let mut rows = Vec::new();
    for p in 1..=parts {
        for i in 0..4 {
            rows.push(vec![
                V::int(p),
                V::int(supp_of(p, i)),
                V::int(r.range(1, 9_999)),
                V::cents(r.range(100, 100_000)),
                s("ps"),
            ]);
        }
    }
    let pscols = [
        "ps_partkey",
        "ps_suppkey",
        "ps_availqty",
        "ps_supplycost",
        "ps_comment",
    ];
    table(&mut db, "partsupp", &pscols, rows);

    let customers = 30;
    let rows = (1..=customers)
        .map(|k| {
            let n = nation(&mut r);
            vec![
                V::int(k),
                s(&format!("Customer#{k:03}")),
                s(&format!("caddr{k}")),
                V::int(n),
                s(&format!("{}-{}", 10 + n, 100 + k)),
                V::cents(r.range(-99_999, 999_999)),
                s(r.pick(&[
                    "AUTOMOBILE",
                    "BUILDING",
                    "FURNITURE",
                    "MACHINERY",
                    "HOUSEHOLD",
                ])),
                s("c"),
            ]
        })
        .collect();
    let ccols = [
        "c_custkey",
        "c_name",
        "c_address",
        "c_nationkey",
        "c_phone",
        "c_acctbal",
        "c_mktsegment",
        "c_comment",
    ];
    table(&mut db, "customer", &ccols, rows);

    let (start, end) = (days(1992, 1, 1), days(1998, 8, 2));
    let mut orders_rows = Vec::new();
    let mut lines = Vec::new();
    for k in 1..=orders {
        let date = i32::try_from(r.range(i64::from(start), i64::from(end))).unwrap();
        let custkey = loop {
            let c = r.range(1, customers);
            if c % 3 != 0 {
                break c;
            }
        };
        let comment = if k % 6 == 0 {
            "a special deal requests"
        } else {
            "o"
        };
        orders_rows.push(vec![
            V::int(k),
            V::int(custkey),
            s(r.pick(&["F", "O", "P"])),
            V::cents(r.range(100_000, 50_000_000)),
            V::Date(date),
            s(r.pick(&["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"])),
            s(&format!("Clerk#{}", r.range(1, 5))),
            V::int(0),
            s(comment),
        ]);
        let big = k % 37 == 0;
        for line in 1..=if big { 7 } else { r.range(1, 7) } {
            let part = r.range(1, parts);
            let ship = date + i32::try_from(r.range(1, 121)).unwrap();
            let commit = date + i32::try_from(r.range(30, 90)).unwrap();
            let receipt = ship + i32::try_from(r.range(1, 30)).unwrap();
            let qty = if big { r.range(45, 50) } else { r.range(1, 50) };
            lines.push(vec![
                V::int(k),
                V::int(part),
                V::int(supp_of(part, r.range(0, 3))),
                V::int(line),
                V::int(qty),
                V::cents(qty * r.range(90_000, 200_000) / 100),
                V::cents(r.range(0, 10)),
                V::cents(r.range(0, 8)),
                s(r.pick(&["R", "A", "N"])),
                s(r.pick(&["O", "F"])),
                V::Date(ship),
                V::Date(commit),
                V::Date(receipt),
                s(r.pick(&[
                    "DELIVER IN PERSON",
                    "COLLECT COD",
                    "NONE",
                    "TAKE BACK RETURN",
                ])),
                s(r.pick(&["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"])),
                s("l"),
            ]);
        }
    }
    let ocols = [
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice",
        "o_orderdate",
        "o_orderpriority",
        "o_clerk",
        "o_shippriority",
        "o_comment",
    ];
    table(&mut db, "orders", &ocols, orders_rows);
    let lcols = [
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_returnflag",
        "l_linestatus",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
        "l_shipinstruct",
        "l_shipmode",
        "l_comment",
    ];
    table(&mut db, "lineitem", &lcols, lines);
    db
}
