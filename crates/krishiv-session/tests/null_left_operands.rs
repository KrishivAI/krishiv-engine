// SPDX-License-Identifier: Apache-2.0
//! A null-intolerant binary operator returns NULL on a NULL left value
//! without evaluating its right operand (Spark's `BinaryExpression.eval`), so
//! an error in the right operand on such a row never surfaces. Found by the
//! nightly expression differential (2026-11-04); the right operand still
//! raises on the rows where the left one is set.

use arrow_array::Array;
use arrow_array::cast::AsArray;
use krishiv_session::Session;

/// The single string column of `sql` (ANSI on, the default), or the error text.
fn strings(sql: &str) -> Result<Vec<Option<String>>, String> {
    let s = Session::new().map_err(|e| e.to_string())?;
    let batches = futures::executor::block_on(async { s.sql(sql).await?.collect().await })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for b in &batches {
        let a = b.column(0).as_string_view();
        out.extend((0..a.len()).map(|i| a.is_valid(i).then(|| a.value(i).to_string())));
    }
    Ok(out)
}

fn some(v: &[Option<&str>]) -> Vec<Option<String>> {
    v.iter().map(|x| x.map(str::to_string)).collect()
}

#[test]
fn arithmetic_does_not_evaluate_the_right_side_on_null_left_rows() {
    // Row 0: left NULL, right 1/0 would raise DIVIDE_BY_ZERO.
    let got = strings("select cast(nullif(id, 0) + 1 / id as string) from range(3) order by id");
    assert_eq!(got.unwrap(), some(&[None, Some("2.0"), Some("2.5")]));
}

#[test]
fn comparisons_do_not_evaluate_the_right_side_on_null_left_rows() {
    let got = strings("select cast(nullif(id, 0) < 1 / id as string) from range(3) order by id");
    assert_eq!(got.unwrap(), some(&[None, Some("false"), Some("false")]));
}

#[test]
fn date_add_and_nullif_do_not_evaluate_the_right_side_on_null_left_rows() {
    let got = strings(
        "select cast(date_add(if(id = 0, cast(null as date), date'2000-01-01'), cast(1 / id as int)) as string) \
         from range(3) order by id",
    );
    assert_eq!(
        got.unwrap(),
        some(&[None, Some("2000-01-02"), Some("2000-01-01")])
    );
    let got =
        strings("select cast(nullif(nullif(id, 0), 1 / id) as string) from range(3) order by id");
    assert_eq!(got.unwrap(), some(&[None, None, Some("2")]));
}

#[test]
fn a_failing_right_side_still_raises_where_the_left_is_set() {
    let err =
        strings("select cast(nullif(id, 2) + 1 / (id - 1) as string) from range(3)").unwrap_err();
    assert!(err.contains("DIVIDE_BY_ZERO"), "{err}");
    // The left side fails first.
    let err = strings("select cast(1 / (id - id) + 1 / id as string) from range(3)").unwrap_err();
    assert!(err.contains("DIVIDE_BY_ZERO"), "{err}");
}
