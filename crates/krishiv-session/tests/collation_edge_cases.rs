// SPDX-License-Identifier: Apache-2.0
//! Collation edge cases (issue #621): the capital sharp s against `ss`, a final capital sigma,
//! a tab and a no-break space at the end of an `_RTRIM` string, and `<=>` over a collated column.
//!
//! Every expected value is the answer of a real Spark 4.2.0 run: `lowerCaseCodePoints` is the
//! `tests/fixtures/spark-4.2.0/collation-lowercase.json` fixture (made by `gen_case_mapping.py`
//! against a real Spark), which maps `U+1E9E` -> `U+00DF`, `U+03A3` and `U+03C2` -> `U+03C3` and
//! leaves `U+0009` / `U+00A0` alone; `_RTRIM` drops trailing `U+0020`s only.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

/// The first column of the single row of `sql`, as text.
fn one(sql: &str) -> String {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let c = &batches[0].columns()[0];
    assert_eq!(batches[0].num_rows(), 1, "{sql}");
    if c.is_null(0) {
        "NULL".to_owned()
    } else if let Some(v) = c.as_string_view_opt() {
        v.value(0).to_owned()
    } else if let Some(v) = c.as_string_opt::<i32>() {
        v.value(0).to_owned()
    } else {
        format!("{c:?}")
    }
}

/// The value of the boolean expression `sql`, cast to text the way the Spark run prints it.
fn truth(sql: &str) -> String {
    one(&format!("SELECT CAST(({sql}) AS STRING)"))
}

/// The value of the row count `sql`, as text.
fn count(sql: &str) -> String {
    one(&format!("SELECT CAST(({sql}) AS STRING)"))
}

/// `U+1E9E` (capital sharp s) against `U+00DF` under `UTF8_LCASE`: the Spark fixture maps
/// `U+1E9E` to the single code point `U+00DF`, with no `ss` expansion, so the two spellings are
/// one key.
#[test]
fn capital_sharp_s_and_sharp_s_are_one_lcase_key() {
    for (sql, want) in [
        ("'ß' COLLATE utf8_lcase = '\u{1e9e}'", "true"),
        ("'\u{1e9e}' COLLATE utf8_lcase = 'ß'", "true"),
        ("'\u{1e9e}' COLLATE utf8_lcase = '\u{1e9e}'", "true"),
        // The simple mapping is a single code point, not the `ss` expansion of `upper`.
        ("'\u{1e9e}' COLLATE utf8_lcase = 'ss'", "false"),
        ("'ß' COLLATE utf8_lcase = 'ss'", "false"),
        ("'\u{1e9e}' COLLATE utf8_lcase = 'b'", "false"),
        // `UTF8_BINARY` sees the bytes: the two spellings are different strings there.
        ("'\u{1e9e}' COLLATE utf8_binary = 'ß'", "false"),
    ] {
        assert_eq!(truth(sql), want, "{sql}");
    }
    // One key: `U+1E9E` and `U+00DF` fold together, `ss` stays its own.
    assert_eq!(
        count(
            "SELECT count(*) FROM (SELECT DISTINCT c COLLATE utf8_lcase AS c \
             FROM VALUES ('\u{1e9e}'),('ß'),('ß'),('ss') t(c))"
        ),
        "2"
    );
}

/// A final capital sigma under `UTF8_LCASE`: `lowerCaseCodePoints` is context-free, so `U+03A3`
/// maps to `U+03C3` even word-finally (the JDK final-sigma rule does not apply), and the final
/// sigma `U+03C2` maps to `U+03C3` as well.
#[test]
fn final_capital_sigma_lowers_to_sigma() {
    for (sql, want) in [
        // Word-final capital sigma equals plain sigma, not the final sigma code point.
        ("'ΟΔΟΣ' COLLATE utf8_lcase = 'οδοσ'", "true"),
        ("'Σ' COLLATE utf8_lcase = 'σ'", "true"),
        // The final sigma maps to sigma too, so it compares equal as well.
        ("'ΟΔΟΣ' COLLATE utf8_lcase = 'οδος'", "true"),
        ("'ς' COLLATE utf8_lcase = 'σ'", "true"),
        // Different letters stay different.
        ("'ΟΔΟΣ' COLLATE utf8_lcase = 'οδολ'", "false"),
        // `UTF8_BINARY` sees the written code points: capital sigma is not sigma.
        ("'ΟΔΟΣ' COLLATE utf8_binary = 'οδοσ'", "false"),
        ("'Σ' COLLATE utf8_binary = 'ς'", "false"),
    ] {
        assert_eq!(truth(sql), want, "{sql}");
    }
}

/// `_RTRIM` drops trailing spaces (`U+0020`) only: a tab and a no-break space at the end survive,
/// and the trailing space behind a tab is still dropped.
#[test]
fn rtrim_trims_spaces_and_keeps_tab_and_nbsp() {
    for (sql, want) in [
        // The control: trailing spaces go.
        ("'a  ' COLLATE utf8_binary_rtrim = 'a'", "true"),
        ("'a ' COLLATE utf8_binary_rtrim = 'a'", "true"),
        // A trailing tab and a trailing no-break space are not spaces: they stay.
        ("'a\t' COLLATE utf8_binary_rtrim = 'a'", "false"),
        ("'a\u{a0}' COLLATE utf8_binary_rtrim = 'a'", "false"),
        ("'a\t' COLLATE utf8_binary_rtrim = 'a\t'", "true"),
        ("'a\u{a0}' COLLATE utf8_binary_rtrim = 'a\u{a0}'", "true"),
        ("'a\t' COLLATE utf8_binary_rtrim = 'a\u{a0}'", "false"),
        // The space behind the tab is trimmed; the tab itself is not.
        ("'a\t ' COLLATE utf8_binary_rtrim = 'a\t'", "true"),
        ("'a \t' COLLATE utf8_binary_rtrim = 'a \t'", "true"),
        // `UTF8_LCASE_RTRIM` folds case first, then trims spaces only.
        ("'A\t' COLLATE utf8_lcase_rtrim = 'a\t'", "true"),
        ("'A\t' COLLATE utf8_lcase_rtrim = 'a'", "false"),
        ("'A\u{a0}' COLLATE utf8_lcase_rtrim = 'a\u{a0}'", "true"),
        // Under `UTF8_LCASE` (no rtrim) the tab is part of the key either way.
        ("'a\t' COLLATE utf8_lcase = 'A\t'", "true"),
    ] {
        assert_eq!(truth(sql), want, "{sql}");
    }
}

/// `NULL <=> NULL` over a collated column: the null-safe comparison keeps the null row that `=`
/// drops, and a null column value is null-safe equal to itself.
#[test]
fn null_safe_equal_over_a_collated_column() {
    let s = Session::new().unwrap();
    // A view of one non-null and one null row over a `UTF8_LCASE` column (no warehouse file
    // system: the assertions are about the comparison, not the storage).
    let view = "create temporary view t621 as select * from (values                  (cast('A' as string collate utf8_lcase)),                  (cast(null as string collate utf8_lcase))) v(c)";
    block_on(block_on(s.sql(view)).unwrap().collect()).unwrap_or_else(|e| panic!("{view}: {e}"));
    let text = |sql: &str| -> String {
        // Booleans and counts print the way the Spark run prints them.
        let sql = format!("SELECT CAST(({sql}) AS STRING)");
        let sql = sql.as_str();
        let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
        let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert_eq!(batches[0].num_rows(), 1, "{sql}");
        let c = &batches[0].columns()[0];
        let cell = if c.is_null(0) {
            "NULL".to_owned()
        } else if let Some(v) = c.as_string_view_opt() {
            v.value(0).to_owned()
        } else if let Some(v) = c.as_string_opt::<i32>() {
            v.value(0).to_owned()
        } else {
            format!("{c:?}")
        };
        // Booleans and counts print the way the Spark run prints them.
        if cell == "true" || cell == "false" || cell.chars().all(|c| c.is_ascii_digit()) {
            cell
        } else {
            panic!("{sql}: unexpected cell {cell:?}");
        }
    };
    // Both rows pass `c <=> c`: 'A' against 'A' and NULL against NULL.
    assert_eq!(text("SELECT count(*) FROM t621 WHERE c <=> c"), "2");
    // Ordinary `=` is null-propagating: the null row fails it.
    assert_eq!(text("SELECT count(*) FROM t621 WHERE c = c"), "1");
    // The null row is the one a null-safe comparison against NULL picks.
    assert_eq!(
        text("SELECT count(*) FROM t621 WHERE c <=> CAST(NULL AS STRING)"),
        "1"
    );
    // `NULL <=> NULL` itself, answered from the collated column.
    assert_eq!(
        text("SELECT c <=> CAST(NULL AS STRING) FROM t621 WHERE c IS NULL"),
        "true"
    );
    // A non-null value is not null-safe equal to NULL.
    assert_eq!(
        text("SELECT c <=> CAST(NULL AS STRING) FROM t621 WHERE c IS NOT NULL"),
        "false"
    );
}
