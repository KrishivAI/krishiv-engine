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

/// A session function that shares the name of the built-in `abs`.
#[derive(Debug)]
struct AbsShadow;

impl krishiv_session::ScalarUdf for AbsShadow {
    fn name(&self) -> &str {
        "abs"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &krishiv_session::Signature {
        static SIG: std::sync::OnceLock<krishiv_session::Signature> = std::sync::OnceLock::new();
        SIG.get_or_init(|| {
            krishiv_session::Signature::Exact(vec![krishiv_session::UdfInputType::Exact(
                krishiv_types::DataType::String,
            )])
        })
    }

    fn return_field(
        &self,
        _: &[krishiv_session::ArgField],
    ) -> Result<krishiv_session::ReturnField, krishiv_session::UdfError> {
        Ok(krishiv_session::ReturnField {
            data_type: krishiv_types::DataType::String,
            nullable: true,
        })
    }

    fn invoke(
        &self,
        args: &[krishiv_session::UdfArg<'_>],
        _: usize,
        _: &krishiv_session::InvokeContext,
    ) -> Result<arrow_array::ArrayRef, krishiv_session::UdfError> {
        Ok(args[0].array())
    }
}

/// The error class a statement fails with (`"OK"` if it does not fail).
fn err(s: &Session, sql: &str) -> String {
    match block_on(s.sql(sql)).and_then(|d| block_on(d.collect())) {
        Ok(_) => "OK".to_owned(),
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
    // Every row is a name in Spark's own catalog (the vendored Spark 4.2.0 list): no internal helper.
    let catalog = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../krishiv-functions/vendored/spark-functions.tsv"
    ))?;
    let spark: std::collections::HashSet<&str> = catalog
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split('\t').next())
        .collect();
    let unknown: Vec<&String> = all.iter().filter(|f| !spark.contains(f.as_str())).collect();
    assert!(unknown.is_empty(), "not Spark functions: {unknown:?}");
    // The legacy form names the pattern as a bare word.
    assert_eq!(rows(&s, "SHOW FUNCTIONS abs")?, ["abs"]);
    assert!(rows(&s, "SHOW USER FUNCTIONS")?.is_empty());
    // A session function that shares a built-in's name is a user function.
    s.register_udf(AbsShadow)?;
    assert_eq!(rows(&s, "SHOW USER FUNCTIONS LIKE 'abs'")?, ["abs"]);
    // A pattern is a regular expression after `*` is expanded, as in Spark.
    assert_eq!(rows(&s, "SHOW SYSTEM FUNCTIONS LIKE 'a.s'")?, ["abs"]);
    assert_eq!(
        err(&s, "SHOW bogus FUNCTIONS LIKE 'abs'"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
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
    let mut expected = short;
    expected.push(
        "Extended Usage:\n    Examples:\n      > SELECT abs(-1);\n       1\n      > SELECT abs(INTERVAL -'1-1' YEAR TO MONTH);\n       1-1\n  \n    Since: 1.2.0\n"
            .to_owned(),
    );
    assert_eq!(long, expected);
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
