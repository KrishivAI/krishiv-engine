// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2 `SQLQueryTestSuite` goldens for `mode`, the `regr_*`
//! aggregates, the aggregate forms of `bit_and`/`bit_or`/`bit_xor`,
//! `listagg`/`string_agg` and `try_sum`/`try_avg`: each claimed statement
//! of the vendored `mode`, `linear-regression`, `listagg`, `bitwise`,
//! `try_aggregates` (and `nonansi/try_aggregates`) and
//! `postgreSQL/aggregates_part2` result files runs through a session over
//! the tables the file creates (registered through the Rust API), and its
//! schema, rows (in order when the statement orders them) or error class,
//! SQLSTATE and message parameters must be Spark's.
//!
//! Claimed: the `mode.sql` statements over `basic_pays` (global, grouped,
//! the `DISTINCT` and deterministic errors, the ordered forms and the
//! window forms and the `FILTER` ones; not the interval ones, which need
//! interval literals) and the
//! `VALUES`-based ones but the map inputs (no `map` constructor); every
//! `linear-regression.sql` statement but the view setup; every `listagg.sql` statement but the interval one (no
//! interval literals; the `string_agg` ones resolve as `listagg`); the
//! `bitwise.sql` aggregate statements over `bitwise_test`; the `try_aggregates.sql` statements over integers,
//! decimals and doubles, in both ANSI modes (not the interval ones, which
//! need interval literals); the `postgreSQL/aggregates_part2.sql` bit
//! aggregate statements.
//!
//! Not claimed (they stay in `known-failures-cases`): the interval statements
//! (`INTERVAL` literals are unsupported); the map inputs (no `map`
//! constructor); the `try_avg` statements over
//! `(9223372036854775807L, 1L)` (the file renders doubles with Rust
//! `Debug`, its shortest round-trip `4.611686018427388E18`, while
//! Spark's Java `Double.toString` prints 17 significant digits,
//! `4.6116860184273879E18`; the golden harness renders Java-style and
//! covers those statements exactly); the `listagg` timestamp-literal
//! statement (timestamps in a non-fixed-offset session zone are
//! unsupported); the `histogram_numeric` statements of `group-by.sql`.
//!
//! The floating `regr_*` values compare within `1e-9` relative: Spark
//! reassociates the pairwise sums when it merges partial states, so the
//! last ulp depends on partitioning (as the engine's does).
#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// One golden statement: the query and its expected schema and output.
struct Case {
    query: String,
    schema: String,
    output: String,
}

fn load(file: &str) -> R<Vec<Case>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results")
        .join(file);
    let text = std::fs::read_to_string(path)?;
    let blocks: Vec<&str> = text.split("-- !query\n").skip(1).collect();
    let cases: Vec<Case> = blocks
        .iter()
        .filter_map(|block| {
            let (query, rest) = block.split_once("\n-- !query schema\n")?;
            let (schema, output) = rest.split_once("\n-- !query output\n")?;
            Some(Case {
                query: query.trim().to_owned(),
                schema: schema.trim().to_owned(),
                output: output.trim_end().to_owned(),
            })
        })
        .collect();
    if cases.len() != blocks.len() {
        return Err(format!("dropped statements of {file}").into());
    }
    Ok(cases)
}

fn spark_type(t: &DataType) -> String {
    match t {
        DataType::Int8 => "tinyint".into(),
        DataType::Int16 => "smallint".into(),
        DataType::Int32 => "int".into(),
        DataType::Int64 => "bigint".into(),
        DataType::Float32 => "float".into(),
        DataType::Float64 => "double".into(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "string".into(),
        DataType::Binary | DataType::BinaryView => "binary".into(),
        DataType::Boolean => "boolean".into(),
        DataType::Null => "void".into(),
        DataType::Decimal128(p, s) | DataType::Decimal64(p, s) => {
            format!("decimal({p},{s})")
        }
        DataType::Date32 => "date".into(),
        DataType::Timestamp(_, _) => "timestamp".into(),
        DataType::List(f) => format!("array<{}>", spark_type(f.data_type())),
        DataType::Map(f, _) => match f.data_type() {
            DataType::Struct(fs) if fs.len() == 2 => format!(
                "map<{},{}>",
                spark_type(fs[0].data_type()),
                spark_type(fs[1].data_type())
            ),
            _ => "map<?,?>".to_owned(),
        },
        other => format!("{other}"),
    }
}

/// Spark's rendering of a decimal: plain fixed-point digits.
fn decimal_str(unscaled: i128, scale: i8) -> String {
    let neg = unscaled < 0;
    let mut digits = unscaled.unsigned_abs().to_string();
    let scale = usize::try_from(scale).unwrap_or(0);
    if scale > 0 {
        while digits.len() <= scale {
            digits.insert(0, '0');
        }
        digits.insert(digits.len() - scale, '.');
    }
    if neg { format!("-{digits}") } else { digits }
}

fn nested(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Null => "NULL".into(),
        DataType::Int8 => c.as_primitive::<Int8Type>().value(row).to_string(),
        DataType::Int16 => c.as_primitive::<Int16Type>().value(row).to_string(),
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Float32 => format!("{:?}", c.as_primitive::<Float32Type>().value(row)),
        DataType::Float64 => format!("{:?}", c.as_primitive::<Float64Type>().value(row)),
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        DataType::Decimal128(_, s) => {
            decimal_str(c.as_primitive::<Decimal128Type>().value(row), *s)
        }
        DataType::Decimal64(_, s) => {
            decimal_str(i128::from(c.as_primitive::<Decimal64Type>().value(row)), *s)
        }
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        DataType::Map(_, _) => {
            let m = c.as_map();
            let keys = m.keys();
            let values = m.values();
            let off = m.value_offsets();
            let (start, end) = (
                usize::try_from(off[row]).unwrap_or(0),
                usize::try_from(off[row + 1]).unwrap_or(0),
            );
            let mut entries: Vec<String> = (start..end)
                .map(|i| {
                    format!(
                        "{}:\"{}\"",
                        nested(keys.as_ref(), i),
                        nested(values.as_ref(), i)
                    )
                })
                .collect();
            entries.sort();
            format!("{{{}}}", entries.join(","))
        }
        other => format!("<{other}>"),
    }
}

fn cell(c: &dyn Array, row: usize) -> String {
    nested(c, row)
}

/// Whether `query` has an `ORDER BY` at parenthesis depth 0, outside
/// string literals, quoted identifiers and comments.
fn outer_order_by(query: &str) -> bool {
    let mut top = String::with_capacity(query.len());
    let mut depth = 0_usize;
    let mut chars = query.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' | '`' => {
                while let Some(n) = chars.next() {
                    if n == '\\' {
                        chars.next();
                    } else if n == c {
                        break;
                    }
                }
                top.push(' ');
            }
            '-' if chars.peek() == Some(&'-') => {
                for n in chars.by_ref() {
                    if n == '\n' {
                        break;
                    }
                }
                top.push(' ');
            }
            '/' if chars.peek() == Some(&'*') => {
                let mut prev = ' ';
                for n in chars.by_ref() {
                    if prev == '*' && n == '/' {
                        break;
                    }
                    prev = n;
                }
                top.push(' ');
            }
            '(' => {
                depth += 1;
                top.push(' ');
            }
            ')' => {
                depth = depth.saturating_sub(1);
                top.push(' ');
            }
            _ if depth == 0 => top.push(c.to_ascii_lowercase()),
            _ => {}
        }
    }
    let mut words = top
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .peekable();
    while let Some(w) = words.next() {
        if w == "order" && words.peek() == Some(&"by") {
            return true;
        }
    }
    false
}

/// The JSON body of a golden error: class, SQLSTATE, message parameters.
fn expected_error(output: &str) -> R<(String, String, Value)> {
    let body = output.split_once('\n').ok_or("an error without a body")?.1;
    let json: Value = serde_json::Deserializer::from_str(body)
        .into_iter::<Value>()
        .next()
        .ok_or("no error JSON")??;
    let text = |k: &str| json[k].as_str().unwrap_or_default().to_owned();
    Ok((
        text("errorClass"),
        text("sqlState"),
        json["messageParameters"].clone(),
    ))
}

/// Runs `query` over `s`: the schema and tab-separated rows.
fn run(s: &Session, query: &str) -> R<(String, Vec<String>)> {
    let (schema, batches) = block_on(async {
        let df = s.sql(query).await?;
        let schema = df.schema().await?;
        Ok::<_, KrishivError>((schema, df.collect().await?))
    })?;
    let columns: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| format!("{}:{}", f.name(), spark_type(f.data_type())))
        .collect();
    let rows = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(|r| {
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("\t")
            })
        })
        .collect();
    Ok((format!("struct<{}>", columns.join(",")), rows))
}

/// Runs `case` over `s`; the disagreement with Spark, if any. Float cells
/// compare within `1e-9` relative when `tolerant` (the `regr_*` family).
fn check(s: &Session, case: &Case, tolerant: bool) -> R<Option<String>> {
    let result = block_on(async {
        let df = s.sql(&case.query).await?;
        df.collect().await.map(|_| ())
    });
    if case.schema == "struct<>" {
        let (class, state, params) = expected_error(&case.output)?;
        let Err(e) = result else {
            return Ok(Some(format!("{}\n  expected error {class}", case.query)));
        };
        let got: serde_json::Map<String, Value> = e
            .message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), Value::String(v.clone())))
            .collect();
        let same = e.error_class().name() == class
            && e.sql_state() == Some(state.as_str())
            && Value::Object(got.clone()) == params;
        return Ok((!same).then(|| {
            format!(
                "{}\n  error {} {:?} {got:?}\n  want {class} {state} {params}",
                case.query,
                e.error_class().name(),
                e.sql_state()
            )
        }));
    }
    if let Err(e) = result {
        return Ok(Some(format!("{}\n  error {e}", case.query)));
    }
    let (schema, mut rows) = run(s, &case.query)?;
    let mut want: Vec<String> = case.output.lines().map(str::to_owned).collect();
    if !outer_order_by(&case.query) {
        rows.sort();
        want.sort();
    }
    if schema != case.schema {
        return Ok(Some(format!(
            "{}\n  schema {schema} want {}",
            case.query, case.schema
        )));
    }
    if tolerant {
        if rows.len() != want.len() {
            return Ok(Some(format!(
                "{}\n  rows {rows:?}\n  want {want:?}",
                case.query
            )));
        }
        for (g, w) in rows.iter().zip(&want) {
            let (gc, wc): (Vec<&str>, Vec<&str>) =
                (g.split('\t').collect(), w.split('\t').collect());
            if gc.len() != wc.len() {
                return Ok(Some(format!(
                    "{}\n  rows {rows:?}\n  want {want:?}",
                    case.query
                )));
            }
            for (g, w) in gc.iter().zip(wc) {
                match (g.parse::<f64>(), w.parse::<f64>()) {
                    _ if **g == *w => {}
                    (Ok(g), Ok(w)) => {
                        let tol = 1e-9 * w.abs().max(1.0);
                        if (g - w).abs() > tol {
                            return Ok(Some(format!(
                                "{}\n  rows {rows:?}\n  want {want:?}",
                                case.query
                            )));
                        }
                    }
                    _ => {
                        return Ok(Some(format!(
                            "{}\n  rows {rows:?}\n  want {want:?}",
                            case.query
                        )));
                    }
                }
            }
        }
        return Ok(None);
    }
    Ok((rows != want).then(|| format!("{}\n  rows {rows:?}\n  want {want:?}", case.query)))
}

fn check_count(bad: &[String], claimed: usize, want: usize) -> R {
    if claimed != want {
        return Err(format!("{claimed} statements claimed, expected {want}").into());
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!("{} mismatches:\n{}", bad.len(), bad.join("\n")).into())
    }
}

/// Registers `name` with explicit `fields` and `arrays`.
fn register(s: &Session, name: &str, fields: Vec<Field>, arrays: Vec<ArrayRef>) -> R {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    s.register_batches(name, schema, vec![batch])?;
    Ok(())
}

/// The `basic_pays` view of `mode.sql`.
fn basic_pays(s: &Session) -> R {
    register(
        s,
        "basic_pays",
        vec![
            Field::new("employee_name", DataType::Utf8, false),
            Field::new("department", DataType::Utf8, false),
            Field::new("salary", DataType::Int32, false),
        ],
        vec![
            Arc::new(StringArray::from(vec![
                "Diane Murphy",
                "Mary Patterson",
                "Jeff Firrelli",
                "William Patterson",
                "Gerard Bondur",
                "Anthony Bow",
                "Leslie Jennings",
                "Leslie Thompson",
                "Julie Firrelli",
                "Steve Patterson",
                "Foon Yue Tseng",
                "George Vanauf",
                "Loui Bondur",
                "Gerard Hernandez",
                "Pamela Castillo",
                "Larry Bott",
                "Barry Jones",
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "Accounting",
                "Accounting",
                "Accounting",
                "Accounting",
                "Accounting",
                "Accounting",
                "IT",
                "IT",
                "Sales",
                "Sales",
                "Sales",
                "Sales",
                "SCM",
                "SCM",
                "SCM",
                "SCM",
                "SCM",
            ])) as ArrayRef,
            Arc::new(Int32Array::from(vec![
                8435, 9998, 8992, 8870, 11472, 6627, 8113, 5186, 9181, 9441, 6660, 10563, 10449,
                6949, 11303, 11798, 10586,
            ])) as ArrayRef,
        ],
    )
}

/// The `testRegression` view of `linear-regression.sql`.
fn test_regression(s: &Session) -> R {
    register(
        s,
        "testRegression",
        vec![
            Field::new("k", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
            Field::new("x", DataType::Int32, true),
        ],
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
            ])) as ArrayRef,
            Arc::new(Int32Array::from(vec![
                Some(10),
                Some(10),
                Some(20),
                Some(25),
                Some(30),
                None,
            ])) as ArrayRef,
            Arc::new(Int32Array::from(vec![
                None,
                Some(11),
                Some(22),
                None,
                Some(35),
                Some(40),
            ])) as ArrayRef,
        ],
    )
}

/// The `df` view of `listagg.sql`.
fn listagg_df(s: &Session) -> R {
    register(
        s,
        "df",
        vec![
            Field::new("col1", DataType::Utf8, true),
            Field::new("col2", DataType::Utf8, true),
        ],
        vec![
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("a"),
                Some("b"),
                Some("b"),
                None,
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("b"),
                Some("c"),
                Some("c"),
                Some("d"),
                None,
            ])) as ArrayRef,
        ],
    )
}

/// The `df2` view of `listagg.sql`.
fn listagg_df2(s: &Session) -> R {
    register(
        s,
        "df2",
        vec![
            Field::new("col1", DataType::Int32, true),
            Field::new("col2", DataType::Boolean, true),
        ],
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(false),
            ])) as ArrayRef,
        ],
    )
}

/// The `bitwise_test` view of `bitwise.sql`.
fn bitwise_test(s: &Session) -> R {
    register(
        s,
        "bitwise_test",
        vec![
            Field::new("b1", DataType::Int32, true),
            Field::new("b2", DataType::Int32, true),
            Field::new("b3", DataType::Int32, true),
            Field::new("b4", DataType::Int64, true),
        ],
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(7)])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(1), Some(3), Some(7)])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(1), Some(4), Some(7)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef,
        ],
    )
}

/// The `bitwise_test` view of `postgreSQL/aggregates_part2.sql`.
fn postgres_bitwise_test(s: &Session) -> R {
    register(
        s,
        "bitwise_test",
        vec![
            Field::new("b1", DataType::Int32, true),
            Field::new("b2", DataType::Int32, true),
            Field::new("b3", DataType::Int32, true),
            Field::new("b4", DataType::Int64, true),
        ],
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(3), Some(7)])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(1), Some(3), Some(7)])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(1), Some(3), Some(7)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef,
        ],
    )
}

/// `FILTER (WHERE ...)` drops rows before every new aggregate folds them.
#[test]
fn filter_restricts_the_new_aggregates() -> R {
    let s = Session::builder().build()?;
    bitwise_test(&s)?;
    let (_, rows) = run(
        &s,
        "SELECT bit_and(b1) FILTER (WHERE b1 > 1), bit_or(b1) FILTER (WHERE b1 < 7), \
         bit_xor(b1) FILTER (WHERE b1 >= 2), regr_count(b1, b2) FILTER (WHERE b1 > 1), \
         listagg(CAST(b1 AS STRING), ',') WITHIN GROUP (ORDER BY b1) FILTER (WHERE b1 > 1), \
         mode(b1) FILTER (WHERE b1 = 2) FROM bitwise_test",
    )?;
    assert_eq!(rows, ["2\t3\t5\t2\t2,7\t2"]);
    // Grouped: each group folds only its own passing rows.
    let (_, rows) = run(
        &s,
        "SELECT b1 > 1 AS big, bit_or(b1) FILTER (WHERE b1 < 7), \
         listagg(CAST(b1 AS STRING), ',') WITHIN GROUP (ORDER BY b1) FILTER (WHERE b1 > 1), \
         size(collect_list(b1) FILTER (WHERE b1 > 1)), \
         mode() WITHIN GROUP (ORDER BY b1) FILTER (WHERE b1 > 1) \
         FROM bitwise_test GROUP BY b1 > 1 ORDER BY big",
    )?;
    assert_eq!(rows, ["false\t1\tNULL\t0\tNULL", "true\t2\t2,7\t2\t2"]);
    Ok(())
}

/// A table of four rows for the ordered-set rewrites below.
fn ordered_t(s: &Session) -> R {
    let strs = |v: Vec<&str>| Arc::new(StringArray::from(v)) as ArrayRef;
    let ints = |v: Vec<i32>| Arc::new(Int32Array::from(v)) as ArrayRef;
    register(
        s,
        "t",
        vec![
            Field::new("v", DataType::Utf8, true),
            Field::new("o", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
            Field::new("z", DataType::Int32, true),
            Field::new("p", DataType::Utf8, true),
        ],
        vec![
            strs(vec!["a", "b", "c", "d"]),
            ints(vec![3, 1, 2, 4]),
            ints(vec![1, 1, 2, 2]),
            ints(vec![1, 2, 2, 1]),
            strs(vec!["x", "x", "y", "y"]),
        ],
    )
}

/// A `WITHIN GROUP` ordering survives the rewrites that rebuild a call: the
/// distinct-aggregate expansion, `PIVOT` and the `ROLLUP` re-typing. The
/// expected values follow from the ordering (not a Spark run).
#[test]
fn within_group_survives_rewrites() -> R {
    let s = Session::builder().build()?;
    ordered_t(&s)?;
    // Mixed with several distinct sets: `v` ordered by `o` is b, c, a, d.
    let (_, rows) = run(
        &s,
        "SELECT listagg(v, ',') WITHIN GROUP (ORDER BY o), count(DISTINCT y), \
         count(DISTINCT z) FROM t",
    )?;
    assert_eq!(rows, ["b,c,a,d\t2\t2"]);
    let (_, rows) = run(
        &s,
        "SELECT mode() WITHIN GROUP (ORDER BY y), count(DISTINCT y), count(DISTINCT z) FROM t",
    )?;
    assert_eq!(rows, ["1\t2\t2"]);
    // `PIVOT`: each pivot value orders its own rows, descending by `o`.
    let (_, rows) = run(
        &s,
        "SELECT * FROM (SELECT v, o, p FROM t) \
         PIVOT (listagg(v, ',') WITHIN GROUP (ORDER BY o DESC) FOR p IN ('x', 'y'))",
    )?;
    assert_eq!(rows, ["a,b\td,c"]);
    // A grouping call inside the ordering, over `ROLLUP`.
    let (_, rows) = run(
        &s,
        "SELECT p, mode() WITHIN GROUP (ORDER BY grouping(p)) FROM t \
         GROUP BY ROLLUP(p) ORDER BY p NULLS FIRST",
    )?;
    assert_eq!(rows, ["NULL\t1", "x\t0", "y\t0"]);
    Ok(())
}

/// `listagg(x)` built outside SQL has no delimiter: it is NULL, as the SQL
/// layer writes it (the values join with nothing between them).
#[test]
fn listagg_without_a_delimiter_from_the_column_api() -> R {
    use krishiv_session::functions::{call_function, col};
    let s = Session::builder().build()?;
    ordered_t(&s)?;
    let df = s.table("t").select([call_function("listagg", [col("v")])]);
    let batches = block_on(df.collect())?;
    assert_eq!(cell(batches[0].column(0).as_ref(), 0), "abcd");
    Ok(())
}

/// `mode.sql`: the `VALUES`-based statements (no view needed), except
/// the `map` inputs (the `map` constructor is unsupported).
#[test]
fn mode_forms_match_spark() -> R {
    let cases = load("mode.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for (i, case) in cases.iter().enumerate() {
        if (4..=13).contains(&i) && ![10, 11, 12].contains(&i) {
            claimed += 1;
            let s = Session::builder().build()?;
            bad.extend(check(&s, case, false)?);
        }
    }
    check_count(&bad, claimed, 7)
}

/// `mode.sql` over `basic_pays`: global, grouped, the `DISTINCT` and
/// deterministic errors, the ordered forms and the window forms.
/// (`map` inputs need the `map` constructor, which is unsupported; the
/// interval ones need interval literals.)
#[test]
fn mode_goldens_match_spark() -> R {
    let cases = load("mode.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for (i, case) in cases.iter().enumerate() {
        // 25 is the `intervals` setup and 26-27 need interval literals.
        if [1, 2, 3, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24].contains(&i) {
            claimed += 1;
            let s = Session::builder().build()?;
            basic_pays(&s)?;
            bad.extend(check(&s, case, false)?);
        }
    }
    check_count(&bad, claimed, 14)
}

/// `linear-regression.sql` over `testRegression`: every statement but the
/// view setup. The
/// floating values compare within `1e-9` relative.
#[test]
fn linear_regression_matches_spark() -> R {
    let cases = load("linear-regression.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for (i, case) in cases.iter().enumerate() {
        if i != 0 {
            claimed += 1;
            let s = Session::builder().build()?;
            test_regression(&s)?;
            bad.extend(check(&s, case, true)?);
        }
    }
    check_count(&bad, claimed, 34)
}

/// `listagg.sql`: every statement but the interval one (no interval
/// literals). `df` and `df2` are registered; the rest is inline `VALUES`.
/// `string_agg` resolves as `listagg`. The timestamp statement passes
/// here (its Spark error is reproduced) but stays in the golden
/// baseline: the harness runs a non-fixed-offset session zone, where
/// timestamp literals are unsupported.
#[test]
fn listagg_matches_spark() -> R {
    let cases = load("listagg.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for (i, case) in cases.iter().enumerate() {
        if i < 2 || i == 30 {
            continue;
        }
        claimed += 1;
        let s = Session::builder().build()?;
        listagg_df(&s)?;
        listagg_df2(&s)?;
        bad.extend(check(&s, case, false)?);
    }
    check_count(&bad, claimed, 48)
}

/// `bitwise.sql` aggregates over `bitwise_test`.
#[test]
fn bitwise_aggregates_match_spark() -> R {
    let cases = load("bitwise.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for (i, case) in cases.iter().enumerate() {
        if (21..=26).contains(&i) {
            claimed += 1;
            let s = Session::builder().build()?;
            bitwise_test(&s)?;
            bad.extend(check(&s, case, false)?);
        }
    }
    check_count(&bad, claimed, 6)
}

/// `try_aggregates.sql` in ANSI mode: every statement but the interval
/// ones (no interval literals) and the 2^62 `try_avg` (Spark prints 17
/// significant digits).
#[test]
fn try_aggregates_match_spark() -> R {
    let cases = load("try_aggregates.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for (i, case) in cases.iter().enumerate() {
        if [6, 7, 8, 9, 14, 15, 20, 22, 23, 24, 25, 30, 31].contains(&i) {
            continue;
        }
        claimed += 1;
        let s = Session::builder().build()?;
        bad.extend(check(&s, case, false)?);
    }
    check_count(&bad, claimed, 19)
}

/// `nonansi/try_aggregates.sql` with ANSI off.
#[test]
fn nonansi_try_aggregates_match_spark() -> R {
    let cases = load("nonansi/try_aggregates.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for (i, case) in cases.iter().enumerate() {
        if [6, 7, 8, 9, 14, 15, 20, 22, 23, 24, 25, 29, 30, 31].contains(&i) {
            continue;
        }
        claimed += 1;
        let s = Session::builder().build()?;
        s.conf().set("spark.sql.ansi.enabled", "false")?;
        bad.extend(check(&s, case, false)?);
    }
    check_count(&bad, claimed, 18)
}

/// The ordered-aggregate refusals: Spark's error classes for the shapes
/// the engine does not implement (`mode` without `DISTINCT`, a value
/// with no ordering and an ordering with no value, `listagg(DISTINCT)`
/// ordered by anything but its value, and an ordered `listagg` window
/// over anything but the whole partition).
#[test]
fn within_group_refusals_match_spark() -> R {
    let cases = [
        (
            "SELECT mode(DISTINCT x) FROM (VALUES (1), (1), (2)) AS t(x)",
            "INVALID_WITHIN_GROUP_EXPRESSION.DISTINCT_UNSUPPORTED",
        ),
        (
            "SELECT mode() FROM (VALUES (1)) AS t(x)",
            "INVALID_WITHIN_GROUP_EXPRESSION.WITHIN_GROUP_MISSING",
        ),
        (
            "SELECT mode(x) WITHIN GROUP (ORDER BY x) FROM (VALUES (1)) AS t(x)",
            "INVALID_WITHIN_GROUP_EXPRESSION.WRONG_NUM_ORDERINGS",
        ),
        (
            "SELECT listagg(DISTINCT CAST(a AS STRING)) WITHIN GROUP (ORDER BY b) FROM (VALUES ('x', 'y')) AS t(a, b)",
            "INVALID_WITHIN_GROUP_EXPRESSION.MISMATCH_WITH_DISTINCT_INPUT",
        ),
        (
            "SELECT listagg(DISTINCT col, ',') WITHIN GROUP (ORDER BY col) FROM (VALUES (TIMESTAMP '2024-01-01 10:00:00')) AS t(col)",
            "INVALID_WITHIN_GROUP_EXPRESSION.MISMATCH_WITH_DISTINCT_INPUT_UNSAFE_CAST",
        ),
        (
            "SELECT listagg(x) WITHIN GROUP (ORDER BY x) OVER (ORDER BY y) FROM (VALUES ('a', 1)) AS t(x, y)",
            "INVALID_WINDOW_SPEC_FOR_AGGREGATION_FUNC",
        ),
    ];
    let mut bad = Vec::new();
    for (query, class) in cases {
        let s = Session::builder().build()?;
        let err = block_on(async {
            let df = s.sql(query).await?;
            df.collect().await.map(|_| ())
        })
        .err();
        match err {
            Some(e) if e.error_class().name() == class => {}
            got => bad.push(format!(
                "{query}\n  got {:?}\n  want error {class}",
                got.map(|e| e.to_string())
            )),
        }
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!("{} mismatches:\n{}", bad.len(), bad.join("\n")).into())
    }
}

/// Window use for the non-moment aggregates, and the moment refusal:
/// `bit_or` runs cumulatively through the accumulator interface and
/// unordered `listagg` sees the whole partition, while `regr_slope`
/// stays refused (the moment aggregates have no window form).
#[test]
fn window_forms_match_spark() -> R {
    let s = Session::builder().build()?;
    bitwise_test(&s)?;
    let (schema, rows) = run(
        &s,
        "SELECT bit_or(b1) OVER (ORDER BY b2 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM bitwise_test ORDER BY b2",
    )?;
    assert_eq!(
        schema,
        "struct<bit_or(b1) OVER (ORDER BY b2 ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW):int>"
    );
    assert_eq!(rows, ["1", "3", "7"]);
    let s = Session::builder().build()?;
    listagg_df(&s)?;
    listagg_df2(&s)?;
    let (schema, rows) = run(&s, "SELECT listagg(col2) OVER () FROM df")?;
    assert_eq!(
        schema,
        "struct<listagg(col2, NULL) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING):string>"
    );
    assert_eq!(rows, ["bccd", "bccd", "bccd", "bccd", "bccd"]);
    let s = Session::builder().build()?;
    test_regression(&s)?;
    let err = block_on(async {
        let df = s
            .sql("SELECT regr_slope(y, x) OVER () FROM testRegression")
            .await?;
        df.collect().await.map(|_| ())
    })
    .expect_err("regr_slope has no window form");
    assert_eq!(err.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    Ok(())
}

/// `postgreSQL/aggregates_part2.sql`: the bit aggregate statements,
/// including the empty-input one (aggregates over no rows are NULL).
#[test]
fn postgres_bit_aggregates_match_spark() -> R {
    let cases = load("postgreSQL/aggregates_part2.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for (i, case) in cases.iter().enumerate() {
        if (2..=8).contains(&i) {
            claimed += 1;
            let s = Session::builder().build()?;
            postgres_bitwise_test(&s)?;
            bad.extend(check(&s, case, false)?);
        }
    }
    check_count(&bad, claimed, 7)
}
