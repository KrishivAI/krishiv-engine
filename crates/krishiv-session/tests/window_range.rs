// SPDX-License-Identifier: Apache-2.0
//! `RANGE` window frames with value bounds. Expected rows and errors are from a Spark 4.2.0 run.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(async {
        s.sql(
            "CREATE TEMPORARY VIEW t AS SELECT * FROM VALUES (1,'a'),(2,'a'),(2,'a'),(4,'a'),\
             (NULL,'a'),(7,'a'),(3,'b'),(NULL,'b'),(NULL,'b') AS t(k,g)",
        )
        .await?
        .collect()
        .await
    })
    .unwrap();
    s
}

/// Each row of `sql` (one string column) as text.
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_string_view();
            (0..c.len())
                .map(|i| c.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn class(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql}: no error"),
        Err(e) => e.error_class().name().to_owned(),
    }
}

/// Offsets on both sides, descending keys, NULL keys (their frame is the NULLs) and `NULLS LAST`.
#[test]
fn integer_keys_with_offsets_nulls_and_direction() {
    let s = session();
    let got = rows(
        &s,
        "SELECT concat_ws(',', g, coalesce(cast(k AS string), 'N'), c1, coalesce(cast(c2 AS string), 'N'), c3, c4, c5) FROM (\
         SELECT g, k, count(*) OVER (PARTITION BY g ORDER BY k RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) c1, \
         sum(k) OVER (PARTITION BY g ORDER BY k RANGE BETWEEN CURRENT ROW AND 2 FOLLOWING) c2, \
         count(*) OVER (PARTITION BY g ORDER BY k DESC RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) c3, \
         count(*) OVER (PARTITION BY g ORDER BY k DESC NULLS LAST RANGE BETWEEN UNBOUNDED PRECEDING AND 2 FOLLOWING) c4, \
         count(*) OVER (PARTITION BY g ORDER BY k NULLS LAST RANGE BETWEEN 2 PRECEDING AND 1 PRECEDING) c5 \
         FROM t) ORDER BY g, k, c1",
    );
    assert_eq!(
        got,
        [
            "a,N,1,N,1,6,1",
            "a,1,3,5,3,5,0",
            "a,2,3,8,2,5,1",
            "a,2,3,8,2,5,1",
            "a,4,1,4,1,4,2",
            "a,7,1,7,1,1,0",
            "b,N,2,N,2,3,2",
            "b,N,2,N,2,3,2",
            "b,3,1,3,1,1,0",
        ]
    );
}

fn pairs(s: &Session, sql: &str) -> Vec<String> {
    rows(
        s,
        &format!(
            "SELECT concat_ws(',', coalesce(cast(x AS string), 'N'), c, e) FROM ({sql}) ORDER BY x"
        ),
    )
}

/// Doubles: `NaN` is greatest and equal to itself, infinities and signed zeros.
#[test]
fn double_keys_order_nan_and_infinities_as_spark() {
    let s = session();
    let got = rows(
        &s,
        "SELECT concat_ws(',', coalesce(cast(x AS string), 'N'), c, e) FROM (\
         SELECT x, count(*) OVER (ORDER BY x RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) c, \
         count(*) OVER (ORDER BY x DESC RANGE BETWEEN 0.5 PRECEDING AND CURRENT ROW) e \
         FROM VALUES (1.0D),(1.5D),(2.5D),(CAST('NaN' AS DOUBLE)),(CAST('Infinity' AS DOUBLE)),\
         (CAST('-Infinity' AS DOUBLE)),(-0.0D),(0.0D),(CAST(NULL AS DOUBLE)),(CAST('NaN' AS DOUBLE)) AS d(x)) \
         ORDER BY x, c",
    );
    assert_eq!(
        got,
        [
            "N,1,1",
            "-Infinity,1,1",
            "-0.0,3,2",
            "0.0,3,2",
            "1.0,4,2",
            "1.5,3,1",
            "2.5,2,1",
            "Infinity,1,1",
            "NaN,2,2",
            "NaN,2,2",
        ]
    );
}

/// Decimal keys take decimal bounds at their scale; an integer key truncates a fractional bound.
#[test]
fn decimal_and_integer_keys_cast_the_bounds_to_the_key_type() {
    let s = session();
    let got = pairs(
        &s,
        "SELECT x, count(*) OVER (ORDER BY x RANGE BETWEEN 0.5 PRECEDING AND 0.5 FOLLOWING) c, \
         count(*) OVER (ORDER BY x RANGE BETWEEN 1 PRECEDING AND 2 FOLLOWING) e \
         FROM VALUES (CAST(1.5 AS DECIMAL(5,1))),(CAST(2.0 AS DECIMAL(5,1))),(CAST(2.4 AS DECIMAL(5,1))),\
         (CAST(NULL AS DECIMAL(5,1))) AS m(x)",
    );
    assert_eq!(got, ["N,1,1", "1.5,2,3", "2.0,3,3", "2.4,2,3"]);
    let got = rows(
        &s,
        "SELECT concat_ws(',', x, c) FROM (SELECT x, count(*) OVER (ORDER BY x RANGE BETWEEN 1.5 PRECEDING AND 1.9 FOLLOWING) c \
         FROM VALUES (1),(2),(4) AS i(x)) ORDER BY x",
    );
    assert_eq!(got, ["1,2", "2,2", "4,1"]);
}

/// A date key takes an `int` bound in days.
#[test]
fn date_keys_take_day_bounds() {
    let s = session();
    let got = rows(
        &s,
        "SELECT concat_ws(',', cast(x AS string), c) FROM (SELECT x, count(*) OVER (ORDER BY x RANGE BETWEEN 2 PRECEDING AND CURRENT ROW) c \
         FROM VALUES (DATE'2020-01-01'),(DATE'2020-01-03'),(DATE'2020-01-04') AS dt(x)) ORDER BY x",
    );
    assert_eq!(got, ["2020-01-01,1", "2020-01-03,2", "2020-01-04,2"]);
}

/// A key moved past its type raises under ANSI and wraps otherwise (Spark's sliding frame keeps its edges).
#[test]
fn a_key_moved_past_its_type_raises_or_wraps() {
    let s = session();
    let sql = "SELECT concat_ws(',', cast(x AS string), c) FROM (SELECT x, count(*) OVER (ORDER BY x RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) c \
               FROM VALUES (CAST(126 AS TINYINT)),(CAST(127 AS TINYINT)),(CAST(-128 AS TINYINT)) AS b(x)) ORDER BY x";
    assert_eq!(class(&s, sql), "BINARY_ARITHMETIC_OVERFLOW");
    block_on(async {
        s.sql("SET spark.sql.ansi.enabled=false")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(rows(&s, sql), ["-128,0", "126,2", "127,2"]);
}

/// Spark's checks of a `RANGE` frame's key and bounds.
#[test]
fn range_frame_checks() {
    let s = session();
    let q = |over: &str| format!("SELECT count(*) OVER ({over}) FROM VALUES (1, 'a') AS t(v, w)");
    for (over, expected) in [
        (
            "ORDER BY v, w RANGE BETWEEN CURRENT ROW AND 1 FOLLOWING",
            "DATATYPE_MISMATCH.RANGE_FRAME_MULTI_ORDER",
        ),
        (
            "RANGE BETWEEN CURRENT ROW AND 1 FOLLOWING",
            "DATATYPE_MISMATCH.RANGE_FRAME_WITHOUT_ORDER",
        ),
        (
            "ORDER BY v RANGE BETWEEN 1 FOLLOWING AND 1 PRECEDING",
            "DATATYPE_MISMATCH.SPECIFIED_WINDOW_FRAME_WRONG_COMPARISON",
        ),
        (
            "ORDER BY w RANGE BETWEEN CURRENT ROW AND 1 FOLLOWING",
            "DATATYPE_MISMATCH.SPECIFIED_WINDOW_FRAME_UNACCEPTED_TYPE",
        ),
        (
            "ORDER BY current_timestamp() RANGE BETWEEN CURRENT ROW AND 1 FOLLOWING",
            "DATATYPE_MISMATCH.RANGE_FRAME_INVALID_TYPE",
        ),
        (
            "ORDER BY cast(w AS binary) RANGE BETWEEN CURRENT ROW AND 1 FOLLOWING",
            "DATATYPE_MISMATCH.RANGE_FRAME_INVALID_TYPE",
        ),
    ] {
        assert_eq!(class(&s, &q(over)), expected, "{over}");
    }
}

/// The bounds are computed only when a row is compared, and `CURRENT ROW` is a bound of the sliding frame too.
#[test]
fn sliding_frames_match_spark_at_the_edges_of_the_key_type() {
    let s = session();
    let q = |keys: &str, frame: &str| {
        format!(
            "SELECT concat_ws(',', cast(x AS string), c) FROM (SELECT x, count(*) OVER (ORDER BY x RANGE BETWEEN {frame}) c \
             FROM VALUES {keys} AS b(x)) ORDER BY x"
        )
    };
    // No row is compared with 28 + 100, so nothing overflows.
    assert_eq!(
        rows(
            &s,
            &q(
                "(CAST(27 AS TINYINT)),(CAST(28 AS TINYINT))",
                "100 PRECEDING AND 100 FOLLOWING"
            )
        ),
        ["27,2", "28,2"]
    );
    block_on(async {
        s.sql("SET spark.sql.ansi.enabled=false")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(
        rows(
            &s,
            &q(
                "(CAST(126 AS TINYINT)),(CAST(127 AS TINYINT)),(CAST(-128 AS TINYINT))",
                "1 PRECEDING AND CURRENT ROW"
            )
        ),
        ["-128,0", "126,1", "127,2"]
    );
}

/// A FLOAT key adds its bound in single precision.
#[test]
fn float_keys_add_in_single_precision() {
    let s = session();
    let got = rows(
        &s,
        "SELECT concat_ws(',', cast(x AS string), c) FROM (SELECT x, count(*) OVER (ORDER BY x RANGE BETWEEN CURRENT ROW AND 0.2 FOLLOWING) c \
         FROM VALUES (CAST(0.1 AS FLOAT)),(CAST(0.3 AS FLOAT)) AS b(x)) ORDER BY x",
    );
    assert_eq!(got, ["0.1,2", "0.3,1"]);
}

/// FLOAT bounds are single precision before they are ordered: 16777217 and 16777216 are the same FLOAT.
#[test]
fn float_bounds_are_ordered_after_narrowing() {
    let s = session();
    let got = rows(
        &s,
        "SELECT concat_ws(',', cast(x AS string), c) FROM (SELECT x, count(*) OVER (ORDER BY x RANGE BETWEEN 16777217 FOLLOWING AND 16777216 FOLLOWING) c \
         FROM VALUES (CAST(1 AS FLOAT)) AS b(x)) ORDER BY x",
    );
    assert_eq!(got, ["1.0,0"]);
}

/// A decimal target past the key's precision still orders (Spark adds without an overflow check).
#[test]
fn a_decimal_target_past_the_precision_is_kept() {
    let s = session();
    let got = rows(
        &s,
        "SELECT cast(count(*) OVER (ORDER BY x RANGE BETWEEN CURRENT ROW AND 1 FOLLOWING) AS string) \
         FROM VALUES (CAST(99 AS DECIMAL(2,0))) AS b(x)",
    );
    assert_eq!(got, ["1"]);
}

/// A DATE target past the day range wraps without ANSI.
#[test]
fn a_date_target_past_the_day_range_wraps_without_ansi() {
    let s = session();
    block_on(async {
        s.sql("SET spark.sql.ansi.enabled=false")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let got = rows(
        &s,
        "SELECT cast(count(*) OVER (ORDER BY x RANGE BETWEEN CURRENT ROW AND 2147483647 FOLLOWING) AS string) \
         FROM VALUES (DATE'2020-01-01') AS b(x)",
    );
    assert_eq!(got, ["0"]);
}

/// A bound that does not fit the key's decimal type is refused instead of evaluated (Spark raises the cast overflow under ANSI; not reproduced yet).
#[test]
fn a_decimal_bound_past_the_key_precision_is_not_accepted() {
    let s = session();
    let sql = "SELECT count(*) OVER (ORDER BY x RANGE BETWEEN CURRENT ROW AND 100 FOLLOWING) \
               FROM VALUES (CAST(1 AS DECIMAL(2,0))) AS b(x)";
    assert_eq!(class(&s, sql), "KRISHIV_UNSUPPORTED_FEATURE");
}

/// `128 PRECEDING` is `-128` once cast, which a TINYINT key holds.
#[test]
fn a_tinyint_bound_of_128_preceding_fits() {
    let s = session();
    let got = rows(
        &s,
        "SELECT cast(count(*) OVER (ORDER BY x RANGE BETWEEN 128 PRECEDING AND CURRENT ROW) AS string) \
         FROM VALUES (CAST(0 AS TINYINT)) AS t(x)",
    );
    assert_eq!(got, ["1"]);
}

/// A descending order negates the bound in the key's type (Spark's `UnaryMinus`): under ANSI the negation of
/// the type's minimum raises even though adding it to the key would fit.
#[test]
fn a_descending_bound_is_negated_in_the_key_type() {
    let s = session();
    let sql = "SELECT count(*) OVER (ORDER BY x DESC RANGE BETWEEN 128 PRECEDING AND CURRENT ROW) \
               FROM VALUES (CAST(-1 AS TINYINT)) AS t(x)";
    assert_eq!(class(&s, sql), "ARITHMETIC_OVERFLOW");
}

/// A wide decimal bound on a DOUBLE key is rounded once, as `BigDecimal.doubleValue` does
/// (dividing the unscaled value as a double would round twice and land one ulp high).
#[test]
fn a_wide_decimal_bound_on_a_double_key_is_correctly_rounded() {
    let s = session();
    let got = pairs(
        &s,
        "SELECT x, count(*) OVER (ORDER BY x RANGE BETWEEN CURRENT ROW AND 0.49741166850405868 FOLLOWING) c, 0 e \
         FROM VALUES (0.0D),(0.49741166850405866D) AS m(x)",
    );
    assert_eq!(got, ["0.0,2,0", "0.49741166850405866,1,0"]);
}
