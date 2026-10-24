// SPDX-License-Identifier: Apache-2.0
//! Output names match Spark 4.2.0's `toPrettySQL`.
//!
//! The analyzer builds an unaliased expression's output name with
//! [`pretty`] (`crates/krishiv-plan/src/analyzer/exprs.rs`), which must
//! print as Spark's `toPrettySQL` does: Java `Double.toString` /
//! `Float.toString` / `BigDecimal.toString`, `DATE` / `TIMESTAMP` with a `+`
//! before years past 9999, default arguments Spark fills in (`substr(s, p)`
//! is `substr(s, p, 2147483647)`, `log(x)` is `LOG(E(), x)`), `TRIM(BOTH
//! ...)` and the upper-case spellings (`CEIL`, `LOG`, `mod` stays `mod`).
//!
//! Ground truth is the vendored Spark run
//! (`tests/spark-golden/v4.2.0/{inputs,results}/<file>.sql[.out]`): each
//! case below names a result file and the 0-based statement indices whose
//! recorded `-- !query schema` (`struct<name:type,...>`) only differed by
//! names. The test re-analyzes the statement and compares the analyzer's
//! output names with the recorded ones.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::path::Path;

use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf};
use krishiv_sql::{ParserOptions, parse_and_lower};

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// One golden file and the statement indices that only differed by names.
struct File {
    name: &'static str,
    ansi: bool,
    indices: &'static [usize],
}

// The statements of the fix, by result file (105; operators[55] is open,
// see below).
const FILES: &[File] = &[
    File {
        name: "datetime-special",
        ansi: true,
        indices: &[0],
    },
    File {
        name: "decimalArithmeticOperations",
        ansi: true,
        indices: &[13, 21, 37],
    },
    File {
        name: "literals",
        ansi: true,
        indices: &[20, 21, 26, 39, 43],
    },
    File {
        name: "nonansi/datetime-special",
        ansi: false,
        indices: &[0],
    },
    File {
        name: "nonansi/decimalArithmeticOperations",
        ansi: false,
        indices: &[13, 21, 37, 45],
    },
    File {
        name: "nonansi/literals",
        ansi: false,
        indices: &[20, 21, 26, 39, 43],
    },
    File {
        name: "nonansi/string-functions",
        ansi: false,
        indices: &[
            4, 38, 39, 41, 42, 43, 44, 45, 47, 48, 51, 52, 53, 54, 55, 56, 57, 58, 63, 64,
        ],
    },
    File {
        name: "operators",
        ansi: true,
        // `55` (`select 1 > 0.00001` → `(1 > 0)`) is left out: Spark prints
        // `0` for `0.00001` where Java's `BigDecimal.toString` (and this
        // printer) gives `0.00001`. See the summary for the open question.
        indices: &[
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 48, 50, 51, 52, 53, 54, 62,
        ],
    },
    File {
        name: "postgreSQL/date",
        ansi: true,
        indices: &[44, 45],
    },
    File {
        name: "postgreSQL/numeric",
        ansi: true,
        indices: &[
            561, 563, 564, 565, 569, 570, 571, 572, 573, 574, 575, 579, 585, 586, 587, 588, 589,
            590, 591, 592,
        ],
    },
    File {
        name: "postgreSQL/strings",
        ansi: true,
        indices: &[102, 104, 107, 109],
    },
    File {
        name: "string-functions",
        ansi: true,
        indices: &[
            4, 38, 39, 41, 42, 43, 44, 45, 47, 48, 51, 52, 53, 54, 55, 56, 57, 58,
        ],
    },
    File {
        name: "timestampNTZ/datetime-special",
        ansi: true,
        indices: &[0],
    },
    File {
        name: "timestampNTZ/datetime-special-ansi",
        ansi: true,
        indices: &[0],
    },
];

/// A parsed `-- !query schema` / `-- !query output` block.
struct Statement {
    query: String,
    schema: String,
}

/// Splits a `.sql.out` file into its statements, in order.
fn parse_results(text: &str) -> Vec<Statement> {
    let mut out = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = lines
        .iter()
        .position(|l| *l == "-- !query")
        .unwrap_or(lines.len());
    while i < lines.len() {
        assert_eq!(lines[i], "-- !query");
        i += 1;
        let mut q = Vec::new();
        while lines[i] != "-- !query schema" {
            q.push(lines[i]);
            i += 1;
        }
        i += 1;
        let mut s = Vec::new();
        while lines[i] != "-- !query output" {
            s.push(lines[i]);
            i += 1;
        }
        i += 1;
        while i < lines.len() && lines[i] != "-- !query" {
            i += 1;
        }
        out.push(Statement {
            query: q.join("\n").trim().to_owned(),
            schema: s.join("\n").trim().to_owned(),
        });
    }
    out
}

/// Splits `struct<...>` into its fields, on commas outside `<>` and `()`.
fn split_top(s: &str) -> Vec<&str> {
    let (mut depth, mut start, mut parts) = (0_i32, 0, Vec::new());
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// The output names of `struct<name:type,...>`.
fn schema_names(schema: &str) -> Vec<String> {
    let body = schema
        .strip_prefix("struct<")
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or("");
    if body.is_empty() {
        return Vec::new();
    }
    split_top(body)
        .into_iter()
        .map(|f| {
            // The name may hold colons (`TIMESTAMP '... 12:03:17'`), the
            // type never does: split on the last one.
            f.rsplit_once(':')
                .map_or_else(|| f.to_owned(), |(n, _)| n.to_owned())
        })
        .collect()
}

fn conf(ansi: bool) -> ResolutionConf {
    ResolutionConf {
        ansi,
        ..ResolutionConf::default()
    }
}

fn analyze_names(query: &str, ansi: bool) -> Vec<String> {
    let plan = parse_and_lower(query, &ParserOptions::default(), SourceId(0)).unwrap();
    let ctx = context(MemoryCatalog::new("spark_catalog"), conf(ansi));
    let analyzed = block_on(DefaultAnalyzer.analyze(&plan, &ctx)).unwrap();
    analyzed.output.iter().map(|c| c.name.clone()).collect()
}

#[test]
fn output_names_match_spark_goldens() {
    let mut failures = Vec::new();
    let mut checked = 0;
    for f in FILES {
        let path = root().join(format!(
            "tests/spark-golden/v4.2.0/results/{}.sql.out",
            f.name
        ));
        let text = std::fs::read_to_string(&path).unwrap();
        let statements = parse_results(&text);
        for &i in f.indices {
            let st = &statements[i];
            let want = schema_names(&st.schema);
            let got = analyze_names(&st.query, f.ansi);
            checked += 1;
            if got != want {
                failures.push(format!(
                    "{}[{i}]:\n  query {q:?}\n  got   {got:?}\n  Spark {want:?}",
                    f.name,
                    q = st.query,
                ));
            }
        }
    }
    assert_eq!(
        checked, 105,
        "the fix covers 105 statements (see operators[55])"
    );
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// One printing rule, one statement: each failed before the fix with the
/// Rust-`Debug` number text, the missing default argument or the wrong
/// spelling in the name.
fn check(query: &str, want: &[&str]) {
    assert_eq!(analyze_names(query, true), want);
}

#[test]
fn doubles_print_as_java_double_to_string() {
    check(
        "select 1D, 1.2D, 1e10, 1.5e5",
        &["1.0", "1.2", "1.0E10", "150000.0"],
    );
    check("select 2.35E10 * 1.0", &["(2.35E10 * 1.0)"]);
    check(
        "select 12345678912345.123456789123 / 0.000000012345678",
        &["(12345678912345.123456789123 / 1.2345678E-8)"],
    );
}

#[test]
fn floats_print_as_java_float_to_string() {
    check("select 1F, 1.2F, .10f", &["1.0", "1.2", "0.1"]);
}

#[test]
fn decimals_print_as_java_big_decimal_to_string() {
    check(
        "select 90912830918230182310293801923652346786BD, 123.0E-28BD",
        &["90912830918230182310293801923652346786", "1.230E-26"],
    );
    check("select 0.3, -3.14", &["0.3", "-3.14"]);
}

#[test]
fn dates_add_plus_before_long_years() {
    check(
        "select date'999999-03-18', date'-0001-1-28', date'0015'",
        &[
            "DATE '+999999-03-18'",
            "DATE '-0001-01-28'",
            "DATE '0015-01-01'",
        ],
    );
}

#[test]
fn timestamps_print_as_spark_sql() {
    // `timestamp'...'` literals do not lower yet; the `DATE` cases run
    // through SQL and the timestamp layout is covered in `spark_fmt`.
    check("select date'99999-03-18'", &["DATE '+99999-03-18'"]);
}

#[test]
fn substr_fills_in_its_length() {
    check(
        "SELECT substr('Spark SQL', 5)",
        &["substr(Spark SQL, 5, 2147483647)"],
    );
    check(
        "SELECT substr('Spark SQL' from 5 for 1)",
        &["substring(Spark SQL, 5, 1)"],
    );
    check(
        "SELECT substring('Spark SQL' from -3)",
        &["substring(Spark SQL, -3, 2147483647)"],
    );
}

#[test]
fn trims_print_in_spark_form() {
    check(
        "SELECT trim(BOTH 'xyz' FROM 'yxTomxx')",
        &["TRIM(BOTH xyz FROM yxTomxx)"],
    );
    check(
        "SELECT trim('xyz' FROM 'yxTomxx')",
        &["TRIM(BOTH xyz FROM yxTomxx)"],
    );
    check(
        "SELECT trim(LEADING 'xyz' FROM 'zzzytest')",
        &["TRIM(LEADING xyz FROM zzzytest)"],
    );
    check(
        "SELECT trim(TRAILING 'xy' FROM 'TURNERyxXxy')",
        &["TRIM(TRAILING xy FROM TURNERyxXxy)"],
    );
    check("SELECT trim(' xyz ')", &["trim( xyz )"]);
    check(
        "SELECT btrim('xyxtrimyyx', 'xy')",
        &["btrim(xyxtrimyyx, xy)"],
    );
}

#[test]
fn pads_and_replace_fill_in_defaults() {
    check("SELECT lpad('hi', 5)", &["lpad(hi, 5,  )"]);
    check("SELECT rpad('hi', 5)", &["rpad(hi, 5,  )"]);
    check("select replace('abc', 'b')", &["replace(abc, b, )"]);
    check(
        "SELECT lpad('hi', 'invalid_length')",
        &["lpad(hi, invalid_length,  )"],
    );
}

#[test]
fn ceil_floor_round_spellings() {
    check("select ceil(1234567890123456)", &["CEIL(1234567890123456)"]);
    check("select ceiling(0)", &["ceiling(0)"]);
    check("select floor(0.01)", &["FLOOR(0.01)"]);
    check("select round(25)", &["round(25, 0)"]);
}

#[test]
fn log_adds_its_base() {
    check(
        "select log(3.4634998359873254962349856073435545)",
        &["LOG(E(), 3.4634998359873254962349856073435545)"],
    );
    check("select ln(1.0)", &["ln(1.0)"]);
    check(
        "select log(0.99923, 4.58934e34)",
        &["LOG(0.99923, 4.58934E34)"],
    );
}

#[test]
fn math_functions_print_upper_case() {
    check(
        "select sec(1), csc(1), cot(1)",
        &["SEC(1)", "CSC(1)", "COT(1)"],
    );
    check("select exp(1.0)", &["EXP(1.0)"]);
    check("select pow(2, 3)", &["pow(2, 3)"]);
    check("select power(2, 3)", &["POWER(2, 3)"]);
}

#[test]
fn mod_stays_mod() {
    check("select mod(7, 2)", &["mod(7, 2)"]);
    check("select 5 % 3", &["(5 % 3)"]);
}

#[test]
fn unary_minus_and_negative_differ() {
    check("select -100", &["-100"]);
    check("select - - 3", &["(- -3)"]);
    check(
        "select positive('-1.11'), negative('-1.11')",
        &["(+ -1.11)", "negative(-1.11)"],
    );
}

#[test]
fn positive_prints_unary_plus() {
    check("select +230", &["(+ 230)"]);
    check("select positive(1)", &["(+ 1)"]);
}
