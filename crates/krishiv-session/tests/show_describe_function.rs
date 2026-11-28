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

/// The rows of `DESCRIBE FUNCTION EXTENDED`, with the creation time (a clock reading) checked for its
/// shape and replaced by `<time>`.
fn extended(s: &Session, sql: &str) -> R<Vec<String>> {
    let mut out = rows(s, sql)?;
    for r in &mut out {
        if let Some(t) = r.strip_prefix("Create Time:   ") {
            // `Fri Nov 27 21:03:05 UTC 2026`, as Java's `Date.toString` in UTC. Spark prints it in the JVM's
            // default zone; Krishiv prints UTC by design, as its other creation times do.
            assert_eq!(t.len(), 28, "{t}");
            assert!(t.contains(" UTC "), "{t}");
            *r = "Create Time:   <time>".to_owned();
        }
    }
    Ok(out)
}

/// A SQL function prints Spark 4.3.0's rows (`DescribeFunctionCommand.describeSQLFunction`; the
/// expected text is from a Spark 4.3.0 Connect run of the same statements, R1-D-9).
#[test]
fn describe_function_of_a_sql_function_prints_spark_s_rows() -> R {
    let s = Session::new()?;
    s.conf().set("spark.sql.session.timeZone", "Etc/UTC")?;
    s.conf().set("spark.sql.ansi.enabled", "true")?;
    block_on(
        block_on(s.sql("create temporary function tf(x int default 7) returns int return x+1"))?
            .collect(),
    )?;
    assert_eq!(
        rows(&s, "describe function tf")?,
        [
            "Function: tf",
            "Type:     SCALAR",
            "Input:    x INT",
            "Returns:  INT"
        ]
    );
    let configs = [
        "Configs:       spark.sql.ansi.enabled=true",
        "               spark.sql.session.timeZone=Etc/UTC",
    ];
    let mut expected = vec![
        "Function:      tf",
        "Type:          SCALAR",
        "Input:         x INT DEFAULT 7",
        "Returns:       INT",
        "Deterministic: true",
        "Data Access:   CONTAINS SQL",
    ];
    expected.extend(configs);
    expected.extend(["Create Time:   <time>", "Body:          x+1"]);
    assert_eq!(extended(&s, "describe function extended tf")?, expected);
    // The session qualifiers name the same temporary function.
    assert_eq!(rows(&s, "describe function session.tf")?[0], "Function: tf");
    assert_eq!(
        rows(&s, "describe function system.session.tf")?[0],
        "Function: tf"
    );

    // Names and types are padded to the longest of each; comments and defaults only with EXTENDED.
    block_on(block_on(s.sql(
        "create temporary function c1(alpha int comment 'the a', b decimal(10,2) default  1 + 2, \
         c array<string> default array('x'), d struct<p:int,q:string> default null) comment 'my fn' return alpha",
    ))?.collect())?;
    assert_eq!(
        rows(&s, "describe function c1")?,
        [
            "Function: c1",
            "Type:     SCALAR",
            "Input:    alpha INT                      ",
            "          b     DECIMAL(10,2)            ",
            "          c     ARRAY<STRING>            ",
            "          d     STRUCT<p: INT, q: STRING>",
            "Returns:  INT",
        ]
    );
    let mut expected = vec![
        "Function:      c1",
        "Type:          SCALAR",
        "Input:         alpha INT                       'the a'",
        "               b     DECIMAL(10,2)             DEFAULT 1 + 2",
        "               c     ARRAY<STRING>             DEFAULT array('x')",
        "               d     STRUCT<p: INT, q: STRING> DEFAULT null",
        "Returns:       INT",
        "Comment:       my fn",
        "Deterministic: true",
        "Data Access:   CONTAINS SQL",
    ];
    expected.extend(configs);
    expected.extend(["Create Time:   <time>", "Body:          alpha"]);
    assert_eq!(extended(&s, "describe function extended c1")?, expected);

    // Determinism is the body's unless declared; a body that reads a relation reads SQL data; a
    // body of type NULL has no `Returns:` row.
    for (sql, deterministic, access) in [
        (
            "create temporary function r1() returns double return rand()",
            "false",
            "CONTAINS SQL",
        ),
        (
            "create temporary function r2() returns double not deterministic return 1.0",
            "false",
            "CONTAINS SQL",
        ),
        (
            "create temporary function rd() return (select max(id) from range(3))",
            "true",
            "CONTAINS SQL",
        ),
    ] {
        block_on(block_on(s.sql(sql))?.collect())?;
        let name = &sql[26..28];
        let got = extended(&s, &format!("describe function extended {name}"))?;
        assert_eq!(got[4], format!("Deterministic: {deterministic}"), "{sql}");
        assert_eq!(got[5], format!("Data Access:   {access}"), "{sql}");
    }
    // A call of a SQL function whose body is nondeterministic is nondeterministic, even when the callee
    // is declared DETERMINISTIC (Spark's `SQLScalarFunction.deterministic` includes the inlined body);
    // the same through a SQL table function read in a subquery.
    for sql in [
        "create temporary function g1() returns double return rand()",
        "create temporary function f1() returns double return g1()",
        "create temporary function g2() returns double deterministic return rand()",
        "create temporary function f2() returns double return g2()",
        "create temporary function t1() returns table (x double) return select rand() as x",
        "create temporary function f3() returns double return (select max(x) from t1())",
    ] {
        block_on(block_on(s.sql(sql))?.collect())?;
    }
    for (name, deterministic) in [
        ("g1", "false"),
        ("f1", "false"),
        ("g2", "true"),
        ("f2", "false"),
        ("t1", "false"),
        ("f3", "false"),
    ] {
        let got = extended(&s, &format!("describe function extended {name}"))?;
        let row = got.iter().find(|r| r.starts_with("Deterministic:"));
        assert_eq!(
            row.map(String::as_str),
            Some(format!("Deterministic: {deterministic}").as_str()),
            "{name}"
        );
    }
    block_on(block_on(s.sql("create temporary view v9 as select 1 as k"))?.collect())?;
    block_on(
        block_on(s.sql("create temporary function rv() return (select max(k) from v9)"))?.collect(),
    )?;
    assert_eq!(
        extended(&s, "describe function extended rv")?[5],
        "Data Access:   READS SQL DATA"
    );
    block_on(block_on(s.sql("create temporary function nn() return null"))?.collect())?;
    assert_eq!(
        rows(&s, "describe function nn")?,
        ["Function: nn", "Type:     SCALAR", "Input:    ()"]
    );

    // A set SQL configuration is captured, an execution one (`spark.sql.shuffle.`) is not.
    s.conf().set("spark.sql.caseSensitive", "true")?;
    s.conf().set("spark.sql.shuffle.partitions", "7")?;
    block_on(block_on(s.sql("create temporary function k1() returns int return 1"))?.collect())?;
    let got = extended(&s, "describe function extended k1")?;
    assert_eq!(
        got[6..9],
        [
            "Configs:       spark.sql.ansi.enabled=true",
            "               spark.sql.caseSensitive=true",
            "               spark.sql.session.timeZone=Etc/UTC",
        ]
    );
    assert_eq!(got[9], "Create Time:   <time>");
    // Spark's deny list names two single-pass resolver keys exactly; the others are captured.
    s.conf().set(
        "spark.sql.analyzer.singlePassResolver.dualRunWithLegacy",
        "false",
    )?;
    s.conf().set(
        "spark.sql.analyzer.singlePassResolver.returnSinglePassResultInDualRun",
        "false",
    )?;
    block_on(block_on(s.sql("create temporary function k2() returns int return 1"))?.collect())?;
    let got = extended(&s, "describe function extended k2")?;
    assert_eq!(
        got[6..10],
        [
            "Configs:       spark.sql.analyzer.singlePassResolver.returnSinglePassResultInDualRun=false",
            "               spark.sql.ansi.enabled=true",
            "               spark.sql.caseSensitive=true",
            "               spark.sql.session.timeZone=Etc/UTC",
        ]
    );
    assert_eq!(got[10], "Create Time:   <time>");
    Ok(())
}

/// A stored table function is named by catalog, namespace and name; its columns are padded like
/// parameters, and a `RETURNS TABLE` without columns shows the query's.
#[test]
fn describe_function_of_a_stored_table_function() -> R {
    let s = Session::new()?;
    s.conf().set("spark.sql.session.timeZone", "Etc/UTC")?;
    s.conf().set("spark.sql.ansi.enabled", "true")?;
    block_on(block_on(s.sql("create database d9db"))?.collect())?;
    block_on(
        block_on(s.sql(
            "create function d9db.tt(a int) returns table(b int, c string) return select a, 'x'",
        ))?
        .collect(),
    )?;
    assert_eq!(
        rows(&s, "describe function d9db.tt")?,
        [
            "Function: spark_catalog.d9db.tt",
            "Type:     TABLE",
            "Input:    a INT",
            "Returns:  b INT   ",
            "          c STRING",
        ]
    );
    assert_eq!(
        extended(&s, "describe function extended spark_catalog.d9db.tt")?,
        [
            "Function:      spark_catalog.d9db.tt",
            "Type:          TABLE",
            "Input:         a INT",
            "Returns:       b INT   ",
            "               c STRING",
            "Deterministic: true",
            "Data Access:   CONTAINS SQL",
            "Configs:       spark.sql.ansi.enabled=true",
            "               spark.sql.session.timeZone=Etc/UTC",
            "Create Time:   <time>",
            "Body:          select a, 'x'",
        ]
    );
    block_on(block_on(s.sql(
        "create temporary function t2(n int) returns table return select n as m, 'z' as longername",
    ))?.collect())?;
    assert_eq!(
        rows(&s, "describe function t2")?[3..],
        ["Returns:  m          INT   ", "          longername STRING"]
    );
    Ok(())
}
