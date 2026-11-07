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
fn a_hopeless_string_bound_raises_the_cast_error_only_under_ansi() {
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    // Spark 4.2.0, ANSI on: 'x' is CAST_INVALID_INPUT (STRING to DOUBLE).
    let err = frames(&s, "f_float8", "'x' preceding and 1 following").unwrap_err();
    assert!(err.contains("CAST_INVALID_INPUT"), "{err}");
    // ANSI off: Spark reads 'x' as a NULL bound (not an error); the engine refuses rather than guess, and does not
    // invent a cast error.
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    let err = frames(&s, "f_float8", "'x' preceding and 1 following").unwrap_err();
    assert!(!err.contains("CAST_INVALID_INPUT"), "{err}");
}

#[test]
fn text_that_spark_reads_is_never_reported_as_a_cast_error() {
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    // Spark 4.2.0 reads '1d' (Java accepts the suffix) and a decimal cast; it rejects '1.2.3'. The engine may refuse
    // any of these, but must not invent a cast error for text it cannot decide.
    for (key, bounds) in [
        ("f_float8", "'1d' preceding and 1 following"),
        (
            "f_numeric",
            "cast('1.5' as decimal(3,1)) preceding and 1 following",
        ),
        ("f_float8", "'1.2.3' preceding and 1 following"),
    ] {
        if let Err(e) = frames(&s, key, bounds) {
            assert!(!e.contains("CAST_INVALID_INPUT"), "{key} {bounds}: {e}");
        }
    }
}

#[test]
fn mixed_bounds_on_a_float_key_differ_in_type_and_the_message_names_the_key_type() {
    // Spark 4.2.0 (local run), DOUBLE and FLOAT keys: the number is cast to the key's type in the message.
    for ty in ["double", "float"] {
        let err = lists(ty, "interval '1' month preceding and 1 following").unwrap_err();
        assert!(
            err.contains("SPECIFIED_WINDOW_FRAME_DIFF_TYPES"),
            "{ty}: {err}"
        );
        let want = format!("\"INTERVAL MONTH\" <> \"{}\"", ty.to_uppercase());
        assert!(err.contains(&want), "{ty}: {err}");
        // Two different interval types differ too; the same interval type twice is RANGE_FRAME_INVALID_TYPE.
        for bounds in [
            "interval '1' year preceding and interval '1' month following",
            "interval '1' month preceding and interval '1' day following",
        ] {
            let err = lists(ty, bounds).unwrap_err();
            assert!(
                err.contains("SPECIFIED_WINDOW_FRAME_DIFF_TYPES"),
                "{ty} {bounds}: {err}"
            );
        }
        let err = lists(
            ty,
            "interval '1' month preceding and interval '2' month following",
        )
        .unwrap_err();
        assert!(err.contains("RANGE_FRAME_INVALID_TYPE"), "{ty}: {err}");
    }
    // An INT key casts the interval to its own type, and the frame runs (Spark 4.2.0 gave [1, 2], [1, 2, 3], [2, 3]
    // for the keys 1 to 3).
    assert_eq!(
        lists("int", "interval '1' month preceding and 1 following").unwrap(),
        ["[1, 2]", "[1, 2, 3]", "[2, 3, 4]", "[3, 4, 5]", "[4, 5]"]
    );
}

#[test]
fn a_boolean_key_reports_the_numeric_bound_beside_an_interval() {
    // Spark 4.2.0 (local run): the numeric bound, cast to BOOLEAN, is the unacceptable one; its location is named.
    let err = lists("boolean", "interval '1' month preceding and 1 following").unwrap_err();
    assert!(
        err.contains("SPECIFIED_WINDOW_FRAME_UNACCEPTED_TYPE"),
        "{err}"
    );
    assert!(err.contains("upper bound"), "{err}");
    let err = lists("boolean", "1 preceding and interval '1' month following").unwrap_err();
    assert!(
        err.contains("SPECIFIED_WINDOW_FRAME_UNACCEPTED_TYPE"),
        "{err}"
    );
    assert!(err.contains("lower bound"), "{err}");
}

#[test]
fn a_decimal_key_rounds_a_finer_bound_half_up_as_the_cast_does() {
    // Spark casts the bound to the key's DECIMAL(3, 0): 1.5 is 2, 1.4 is 1, and 2.5 PRECEDING is -2.5 cast to -3.
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    for sql in [
        "create table dk (id int, k decimal(3,0)) using parquet",
        "insert into dk values (1,1),(2,2),(3,3),(4,4),(5,5)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let frames_of = |bounds: &str| {
        let sql = format!(
            "select id, first(id) over w, last(id) over w from dk \
             window w as (order by k range between {bounds}) order by id"
        );
        let b = block_on(async { s.sql(&sql).await?.collect().await }).unwrap();
        let mut out = Vec::new();
        for b in &b {
            let (f, l) = (
                b.column(1).as_primitive::<Int32Type>(),
                b.column(2).as_primitive::<Int32Type>(),
            );
            out.extend((0..b.num_rows()).map(|i| (f.value(i), l.value(i))));
        }
        out
    };
    assert_eq!(
        frames_of("1.5 preceding and 1.5 following"),
        frames_of("2 preceding and 2 following")
    );
    assert_eq!(
        frames_of("1.4 preceding and 1.4 following"),
        frames_of("1 preceding and 1 following")
    );
    // HALF_UP, not HALF_EVEN: 2.5 rounds to 3 (HALF_EVEN would give 2).
    assert_eq!(
        frames_of("2.5 preceding and 2.5 following"),
        frames_of("3 preceding and 3 following")
    );
}

#[test]
fn an_integral_or_decimal_bound_the_key_cannot_hold_raises_under_ansi() {
    // Spark 4.2.0 (local run, ANSI on): CAST_OVERFLOW for an integral key, NUMERIC_VALUE_OUT_OF_RANGE for DECIMAL.
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    s.conf().set("spark.sql.ansi.enabled", "true").unwrap();
    for sql in [
        "create table ek (id int, t tinyint, k decimal(3,0)) using parquet",
        "insert into ek values (1,1,1),(2,2,2)",
        "create table ei (i int) using parquet",
        "insert into ei values (1),(2)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    for (key, bounds, class) in [
        ("t", "129 preceding and current row", "CAST_OVERFLOW"),
        (
            "k",
            "1000 preceding and current row",
            "NUMERIC_VALUE_OUT_OF_RANGE",
        ),
        ("i", "current row and 2147483648 following", "CAST_OVERFLOW"),
    ] {
        let table = if key == "i" { "ei" } else { "ek" };
        let sql =
            format!("select count(*) over (order by {key} range between {bounds}) from {table}");
        let err = block_on(async { s.sql(&sql).await?.collect().await })
            .unwrap_err()
            .to_string();
        assert!(err.contains(class), "{key} {bounds}: {err}");
    }
}

#[test]
fn an_out_of_range_bound_stays_refused_with_ansi_off() {
    // Spark gives a NULL bound here (empty frames); Krishiv does not match that yet and refuses.
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    for sql in [
        "create table eo (t tinyint) using parquet",
        "insert into eo values (1),(2)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let err = block_on(async {
        s.sql(
            "select count(*) over (order by t range between 129 preceding and current row) from eo",
        )
        .await?
        .collect()
        .await
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("UNSUPPORTED_FEATURE"), "{err}");
}
