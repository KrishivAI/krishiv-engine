// SPDX-License-Identifier: Apache-2.0
//! `SHOW FUNCTIONS` and `DESCRIBE FUNCTION` (SQL-13), against a Spark 4.2.0 run (`tests/scripts/gen_function_descriptions.py`
//! for the built-ins' text; the shapes below from a throw-away container probe).
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    let batches = block_on(block_on(s.sql(sql))?.collect())?;
    let mut out = Vec::new();
    for b in &batches {
        assert_eq!(b.num_columns(), 1);
        let c = b.column(0).as_string_view();
        out.extend((0..b.num_rows()).map(|i| c.value(i).to_owned()));
    }
    Ok(out)
}

fn err(s: &Session, sql: &str) -> String {
    match block_on(s.sql(sql)).and_then(|d| block_on(d.collect())) {
        Ok(_) => panic!("{sql} should fail"),
        Err(e) => e.error_class().name().to_owned(),
    }
}

#[test]
fn show_functions_lists_builtins_sorted_and_filters_like_spark() -> R {
    let s = Session::new()?;
    assert_eq!(rows(&s, "SHOW SYSTEM FUNCTIONS LIKE 'abs'")?, ["abs"]);
    assert_eq!(
        rows(&s, "SHOW FUNCTIONS LIKE 'ABS|bit_le*'")?,
        ["abs", "bit_length"]
    );
    let all = rows(&s, "SHOW FUNCTIONS")?;
    assert!(all.len() > 300);
    assert!(all.windows(2).all(|w| w[0] <= w[1]), "sorted");
    assert!(all.iter().any(|f| f == "abs"));
    // The legacy form names the pattern as a bare word.
    assert_eq!(rows(&s, "SHOW FUNCTIONS abs")?, ["abs"]);
    assert!(rows(&s, "SHOW USER FUNCTIONS")?.is_empty());
    Ok(())
}

#[test]
fn show_functions_qualifies_a_stored_function_and_user_scope_keeps_it() -> R {
    let s = Session::new()?;
    block_on(block_on(s.sql("CREATE FUNCTION f(x INT) RETURNS INT RETURN x + 1"))?.collect())?;
    block_on(
        block_on(s.sql("CREATE TEMPORARY FUNCTION g(x INT) RETURNS INT RETURN x + 2"))?.collect(),
    )?;
    assert_eq!(
        rows(&s, "SHOW USER FUNCTIONS")?,
        ["g", "spark_catalog.default.f"]
    );
    assert_eq!(
        rows(&s, "SHOW FUNCTIONS LIKE 'f'")?,
        ["spark_catalog.default.f"]
    );
    assert_eq!(
        rows(&s, "SHOW FUNCTIONS IN default LIKE 'f'")?,
        ["spark_catalog.default.f"]
    );
    assert_eq!(
        rows(&s, "SHOW FUNCTIONS default.f")?,
        ["spark_catalog.default.f"]
    );
    assert_eq!(
        err(&s, "SHOW FUNCTIONS IN nosuchns LIKE 'a'"),
        "SCHEMA_NOT_FOUND"
    );
    Ok(())
}

#[test]
fn describe_function_prints_spark_s_rows() -> R {
    let s = Session::new()?;
    let short = rows(&s, "DESCRIBE FUNCTION ABS")?;
    assert_eq!(
        short,
        [
            "Function: abs",
            "Class: org.apache.spark.sql.catalyst.expressions.Abs",
            "Usage: abs(expr) - Returns the absolute value of the numeric or interval value.",
        ]
    );
    let long = rows(&s, "DESCRIBE FUNCTION EXTENDED `abs`")?;
    assert_eq!(long[..3], short[..]);
    assert!(long[3].starts_with("Extended Usage:\n"));
    assert!(long[3].contains("      > SELECT abs(-1);\n       1\n"));
    // An operator has no class row, and no extended text without `EXTENDED`.
    assert_eq!(
        rows(&s, "DESCRIBE FUNCTION `!=`")?,
        [
            "Function: !=",
            "Usage: expr1 != expr2 - Returns true if `expr1` is not equal to `expr2`."
        ]
    );
    assert_eq!(err(&s, "DESCRIBE FUNCTION nosuch"), "UNRESOLVED_ROUTINE");
    assert_eq!(err(&s, "DESCRIBE FUNCTION nosuch.f"), "UNRESOLVED_ROUTINE");
    Ok(())
}
