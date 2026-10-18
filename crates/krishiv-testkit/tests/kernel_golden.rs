// SPDX-License-Identifier: Apache-2.0
//! The literal-only statements of Spark's `comparator.sql`,
//! `decimalArithmeticOperations.sql` (ANSI and non-ANSI) and `operators.sql`
//! (Spark 4.2 has no `arithmetic.sql`; `operators.sql` holds its arithmetic)
//! evaluated on `krishiv-kernels` against their recorded outputs (P2-T021a).
//! Statement-level runs (tables, DDL, other operators) wait for the SQL
//! engine; they are listed, not failed.
#![forbid(unsafe_code)]

use std::fmt::Write as _;

use krishiv_testkit::kernel_eval::run_golden;

type R = Result<(), Box<dyn std::error::Error>>;

#[test]
fn literal_statements_match_spark() -> R {
    let dir = krishiv_testkit::tests_dir().join("spark-golden/v4.2.0/results");
    let mut report = String::new();
    let mut failures = String::new();
    for (file, ansi) in [
        ("comparator.sql.out", true),
        ("decimalArithmeticOperations.sql.out", true),
        ("nonansi/decimalArithmeticOperations.sql.out", false),
        ("operators.sql.out", true),
        ("try_arithmetic.sql.out", true),
        ("nonansi/try_arithmetic.sql.out.java21", false),
        ("string-functions.sql.out", true),
        ("nonansi/string-functions.sql.out", false),
        ("date.sql.out", true),
        ("nonansi/date.sql.out", false),
        ("timestamp.sql.out", true),
        ("nonansi/timestamp.sql.out", false),
        ("datetime-special.sql.out", true),
        ("datetime-legacy.sql.out", true),
    ] {
        let t = run_golden(&std::fs::read_to_string(dir.join(file))?, ansi)?;
        writeln!(
            report,
            "{file}: {} statements, {} evaluated on the kernels, {} passed, {} failed, {} need the engine",
            t.statements,
            t.evaluated,
            t.passed,
            t.failed.len(),
            t.needs_engine.len()
        )?;
        for (q, why) in &t.needs_engine {
            writeln!(report, "    engine: {} -- {why}", q.replace('\n', " "))?;
        }
        for (q, why) in &t.failed {
            writeln!(failures, "{file}: {} -- {why}", q.replace('\n', " "))?;
        }
    }
    std::fs::write(
        std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("kernel-golden-report.txt"),
        &report,
    )?;
    println!("{report}");
    assert!(failures.is_empty(), "{failures}");
    Ok(())
}
