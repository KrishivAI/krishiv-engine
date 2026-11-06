// SPDX-License-Identifier: Apache-2.0
//! A string (or cast string) `RANGE` frame bound is cast to the key's type: a FLOAT or DOUBLE key reads `'inf'` and
//! `'NaN'` as values, an integer key raises the cast's error. Expected rows are Spark 4.2.0's, from the
//! `postgreSQL/window_part2` golden (`numerics` table).
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;

fn session(dir: &std::path::Path) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .unwrap();
    for sql in [
        "create table numerics (id int, f_float4 float, f_float8 float, f_numeric int) using parquet",
        "insert into numerics values (1,-3,-3,-3),(2,-1,-1,-1),(3,0,0,0),(4,1.1,1.1,1.1),(5,1.12,1.12,1.12),(6,2,2,2),(7,100,100,100)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    s
}

/// `first(id)-last(id)` of every row (ordered by id), or the error text.
fn frames(s: &Session, key: &str, bounds: &str) -> Result<String, String> {
    let sql = format!(
        "select id, first(id) over w, last(id) over w from numerics \
         window w as (order by {key} range between {bounds}) order by id"
    );
    let batches =
        block_on(async { s.sql(&sql).await?.collect().await }).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for b in &batches {
        let (f, l) = (
            b.column(1).as_primitive::<Int32Type>(),
            b.column(2).as_primitive::<Int32Type>(),
        );
        out.extend((0..b.num_rows()).map(|i| format!("{}-{}", f.value(i), l.value(i))));
    }
    Ok(out.join(" "))
}

#[test]
fn infinity_and_nan_bounds_on_float_keys_match_spark() {
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    for key in ["f_float4", "f_float8"] {
        for bounds in [
            "'inf' preceding and 'inf' following",
            "cast('inf' as float) preceding and cast('inf' as float) following",
        ] {
            assert_eq!(
                frames(&s, key, bounds).unwrap(),
                "1-7 1-7 1-7 1-7 1-7 1-7 1-7",
                "{key} {bounds}"
            );
        }
        // NaN sorts last, so `key + NaN` reaches the end of the partition.
        assert_eq!(
            frames(&s, key, "1.1 preceding and 'NaN' following").unwrap(),
            "1-7 2-7 2-7 3-7 4-7 4-7 7-7",
            "{key}"
        );
    }
}

#[test]
fn a_nan_string_bound_on_an_integer_key_is_a_cast_error() {
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    let err = frames(&s, "f_numeric", "1.1 preceding and 'NaN' following").unwrap_err();
    assert!(err.contains("CAST_INVALID_INPUT"), "{err}");
}

#[test]
fn malformed_string_bounds_are_not_read_as_numbers() {
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    for bound in ["'--1'", "'++1'", "'+-1'", "'e5'", "'1x'"] {
        let r = frames(
            &s,
            "f_float8",
            &format!("{bound} preceding and 1 following"),
        );
        assert!(r.is_err(), "{bound}: {r:?}");
    }
}

#[test]
fn a_nan_lower_bound_sorts_above_a_finite_upper_one() {
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    let err = frames(&s, "f_float8", "'NaN' following and 1 following").unwrap_err();
    assert!(
        err.contains("SPECIFIED_WINDOW_FRAME_WRONG_COMPARISON"),
        "{err}"
    );
}

#[test]
fn a_float_cast_bound_is_parsed_at_single_precision() {
    let s = Session::new().unwrap();
    // 1.0000001192092896 is the float after 1.0; the text below is just above half way to it, so reading it as a
    // double first and then a float would give 1.0.
    let sql = "select id, last(id) over (order by k range between current row and \
               cast('1.000000059604644775390625000001' as float) following) \
               from (select 1 id, cast(0.0 as float) k union all select 2, cast(1.0000001192092896 as float)) \
               order by id";
    let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let last = b[0].column(1).as_primitive::<Int32Type>();
    assert_eq!(last.value(0), 2);
}

/// `collect_list(k)` per row of `k` over `RANGE BETWEEN {bounds}`, for the keys 1..=5 cast to `ty` (ordered by `k`).
fn lists(ty: &str, bounds: &str) -> Result<Vec<String>, String> {
    let s = Session::new().map_err(|e| e.to_string())?;
    let sql = format!(
        "select cast(collect_list(k) over (order by k range between {bounds}) as string) \
         from (select cast(c as {ty}) k from values (1),(2),(3),(4),(5) t(c)) order by k"
    );
    let batches =
        block_on(async { s.sql(&sql).await?.collect().await }).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for b in &batches {
        let a = b.column(0).as_string_view();
        out.extend((0..b.num_rows()).map(|i| a.value(i).to_owned()));
    }
    Ok(out)
}

#[test]
fn string_bounds_on_integral_keys_are_read_as_doubles_then_cast() {
    // Spark 4.2.0: '1' PRECEDING AND '2' FOLLOWING on INT and TINYINT keys.
    for ty in ["int", "tinyint", "bigint"] {
        assert_eq!(
            lists(ty, "'1' preceding and '2' following").unwrap(),
            [
                "[1, 2, 3]",
                "[1, 2, 3, 4]",
                "[2, 3, 4, 5]",
                "[3, 4, 5]",
                "[4, 5]"
            ],
            "{ty}"
        );
    }
    assert_eq!(
        lists("int", "'999' preceding and current row").unwrap(),
        [
            "[1]",
            "[1, 2]",
            "[1, 2, 3]",
            "[1, 2, 3, 4]",
            "[1, 2, 3, 4, 5]"
        ]
    );
}

#[test]
fn a_malformed_or_overflowing_string_bound_raises_the_cast_error() {
    // Spark 4.2.0: the string is read as a DOUBLE; the double is then cast to the key's type.
    let err = lists("int", "'x' preceding and current row").unwrap_err();
    assert!(err.contains("CAST_INVALID_INPUT"), "{err}");
    let err = lists("tinyint", "'999' preceding and current row").unwrap_err();
    assert!(err.contains("CAST_OVERFLOW"), "{err}");
}

#[test]
fn interval_month_bounds_on_every_integral_and_decimal_key_count_months() {
    let ints = [
        "[1, 2, 3]",
        "[1, 2, 3, 4]",
        "[2, 3, 4, 5]",
        "[3, 4, 5]",
        "[4, 5]",
    ];
    let decimals = [
        "[1.00, 2.00, 3.00]",
        "[1.00, 2.00, 3.00, 4.00]",
        "[2.00, 3.00, 4.00, 5.00]",
        "[3.00, 4.00, 5.00]",
        "[4.00, 5.00]",
    ];
    for ty in ["tinyint", "smallint", "int", "bigint", "decimal(10,2)"] {
        let want = if ty.starts_with("decimal") {
            decimals
        } else {
            ints
        };
        assert_eq!(
            lists(
                ty,
                "interval '1' month preceding and interval '2' month following"
            )
            .unwrap(),
            want,
            "{ty}"
        );
    }
}

#[test]
fn a_malformed_float_text_bound_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    let r = frames(&s, "f_float8", "'1.2.3' preceding and 1 following");
    assert!(r.is_err(), "{r:?}");
}
