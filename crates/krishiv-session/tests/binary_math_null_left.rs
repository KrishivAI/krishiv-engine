// SPDX-License-Identifier: Apache-2.0
//! Spark's `BinaryMathExpression`s (`power`, `atan2`, `hypot`) return NULL on a NULL left operand without
//! evaluating the right one, so an ANSI error on the right never surfaces (found by the nightly differential).

use futures::executor::block_on;
use krishiv_session::Session;

#[test]
fn a_null_left_operand_hides_a_right_operand_error() {
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.ansi.enabled", "true").unwrap();
    for f in ["power", "pow", "atan2", "hypot"] {
        let sql = format!(
            "SELECT {f}(a, CAST(z % 0.0D AS DOUBLE)) IS NULL FROM (SELECT CASE WHEN id > 5 THEN 1.0D END AS a, id AS z FROM range(1)) AS t"
        );
        let b = block_on(block_on(s.sql(&sql)).unwrap().collect()).unwrap();
        assert_eq!(b[0].num_rows(), 1, "{sql}");
    }
    // A set left operand still evaluates the right one.
    let r = block_on(s.sql("SELECT power(a, CAST(z % 0.0D AS DOUBLE)) FROM (SELECT CASE WHEN id < 5 THEN 2.0D END AS a, id AS z FROM range(1)) AS t"));
    let err = match r {
        Err(e) => e.to_string(),
        Ok(df) => block_on(df.collect()).unwrap_err().to_string(),
    };
    assert!(err.contains("REMAINDER_BY_ZERO"), "{err}");
}
