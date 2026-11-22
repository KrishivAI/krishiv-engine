// SPDX-License-Identifier: Apache-2.0
//! The kernels behind `JSON_TABLE` (Spark 4.3, the flat form): the row texts of a document, and the value or the
//! existence of a path in one row text.
//!
//! `JSON_TABLE(input, 'row path' COLUMNS (...))` becomes a `posexplode` of [`json_table_rows`] with each column read
//! from the row text by [`json_table_value`] / [`json_table_exists`] and cast to the column type. Paths are the
//! simple wildcard-free ones (`$`, `.key`, `['key']`, `[n]`), plus one trailing `[*]` on the row path, which makes a
//! row of each element of the array it follows (a value that is no array has no rows, or an error under `ERROR ON ERROR`).
//!
//! Spark's behaviour, from a 4.3.0 run: the input must be one well-formed JSON value and nothing else (single
//! quotes allowed); a NULL, blank or malformed input has no rows unless `ERROR ON ERROR`
//! (`MALFORMED_RECORD_IN_PARSING`, as does a value a trailing `[*]` follows that is no array); a missing row path has no rows; the first of duplicate keys wins. A column's
//! value is NULL for a missing path or a JSON `null`, a string bare, anything else as compact JSON whose numbers
//! print as Java's `BigDecimal.toString` does (`1.50` stays, `1e2` is `1E+2`, `-0` is `0`). A path exists even
//! when its value is `null`.

use std::sync::Arc;

use arrow_array::builder::{ArrayBuilder, BooleanBuilder, StringViewBuilder};
use arrow_array::{Array, ArrayRef, StringArray};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, Field};
use krishiv_common::error::{
    InternalError, KrishivError, MalformedRecordInParsingWithoutSuggestion,
};

use super::parse::{Lexer, LexerFlags};
use super::path::{Owned, Step, capture, parse_path};
use super::write::push_quoted;

/// The document text as a value: one JSON value and nothing after it.
fn parse_document(text: &str) -> Option<Owned> {
    let flags = LexerFlags {
        allow_single_quotes: true,
        ..LexerFlags::strict()
    };
    let mut lexer = Lexer::new(text, flags);
    let value = capture(&mut lexer, 0).ok()?;
    lexer.skip_trivia().ok()?;
    lexer.peek().ok()?.is_none().then_some(value)
}

/// The value `steps` (fields and indexes only) reach in `v`; the first of duplicate keys.
fn navigate<'a>(mut v: &'a Owned, steps: &[Step]) -> Option<&'a Owned> {
    for step in steps {
        v = match (step, v) {
            (Step::Field(name), Owned::Object(fields)) => {
                &fields.iter().find(|(k, _)| k == name)?.1
            }
            (Step::Index(i), Owned::Array(items)) => items.get(*i)?,
            _ => return None,
        };
    }
    Some(v)
}

/// Java's `BigDecimal.toString` of the JSON number written `raw`.
fn big_decimal_string(raw: &str) -> String {
    let (negative, body) = match raw.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, raw.strip_prefix('+').unwrap_or(raw)),
    };
    let (mantissa, exponent) = match body.find(['e', 'E']) {
        Some(at) => (
            &body[..at],
            body[at.saturating_add(1)..].parse::<i64>().unwrap_or(0),
        ),
        None => (body, 0),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{int}{frac}");
    let digits = digits.trim_start_matches('0');
    let zero = digits.is_empty();
    let digits = if zero { "0" } else { digits };
    let scale = i64::try_from(frac.len())
        .unwrap_or(0)
        .saturating_sub(exponent);
    let len = i64::try_from(digits.len()).unwrap_or(1);
    let adjusted = len.saturating_sub(1).saturating_sub(scale);
    let mut out = String::new();
    if negative && !zero {
        out.push('-');
    }
    if scale == 0 {
        out.push_str(digits);
    } else if scale > 0 && adjusted >= -6 {
        let scale = usize::try_from(scale).unwrap_or(0);
        if digits.len() > scale {
            let (head, tail) = digits.split_at(digits.len().saturating_sub(scale));
            out.push_str(head);
            out.push('.');
            out.push_str(tail);
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat(scale.saturating_sub(digits.len())));
            out.push_str(digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('E');
        if adjusted >= 0 {
            out.push('+');
        }
        out.push_str(&adjusted.to_string());
    }
    out
}

/// Compact JSON of `v`, numbers as [`big_decimal_string`] prints them.
fn render(v: &Owned, out: &mut String) {
    match v {
        Owned::Null => out.push_str("null"),
        Owned::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Owned::Num(raw) => out.push_str(&big_decimal_string(raw)),
        Owned::Str(s) => push_quoted(s, out),
        Owned::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                render(item, out);
            }
            out.push(']');
        }
        Owned::Object(fields) => {
            out.push('{');
            for (i, (name, value)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_quoted(name, out);
                out.push(':');
                render(value, out);
            }
            out.push('}');
        }
    }
}

fn render_string(v: &Owned) -> String {
    let mut s = String::new();
    render(v, &mut s);
    s
}

/// The row texts of every input, as `ARRAY<STRING>` (never NULL).
///
/// `path` is the row path (validated by the caller); with `error_on_error` a malformed input raises
/// `MALFORMED_RECORD_IN_PARSING`, otherwise it has no rows.
///
/// # Errors
/// `MALFORMED_RECORD_IN_PARSING` as above; `INTERNAL_ERROR` past the 32-bit offset limit.
pub fn json_table_rows(
    json: &StringArray,
    path: &str,
    error_on_error: bool,
) -> Result<ArrayRef, KrishivError> {
    let mut steps = parse_path(path).ok_or_else(|| InternalError {
        message: format!("JSON_TABLE row path '{path}' was not validated"),
    })?;
    let wildcard = matches!(steps.last(), Some(Step::Wildcard));
    if wildcard {
        steps.pop();
    }
    let mut items = StringViewBuilder::new();
    let mut offsets = vec![0_i32];
    for i in 0..json.len() {
        let malformed = |text: &str| -> KrishivError {
            MalformedRecordInParsingWithoutSuggestion {
                bad_record: text.to_owned(),
                fail_fast_mode: "FAILFAST".to_owned(),
            }
            .into()
        };
        if json.is_null(i) {
            if error_on_error {
                return Err(malformed("null"));
            }
        } else if json.value(i).trim().is_empty() {
            if error_on_error {
                return Err(malformed(json.value(i)));
            }
        } else {
            let text = json.value(i);
            match parse_document(text) {
                None if error_on_error => return Err(malformed(text)),
                None => {}
                Some(doc) => match (navigate(&doc, &steps), wildcard) {
                    (Some(Owned::Array(elements)), true) => {
                        for e in elements {
                            items.append_value(render_string(e));
                        }
                    }
                    // A value the `[*]` follows that is no array is malformed for Spark; a missing one has no rows.
                    (Some(_), true) if error_on_error => return Err(malformed(text)),
                    (Some(node), false) => items.append_value(render_string(node)),
                    _ => {}
                },
            }
        }
        offsets.push(i32::try_from(items.len()).map_err(|_| InternalError {
            message: "JSON_TABLE rows past the 32-bit offset limit".to_owned(),
        })?);
    }
    let field = Arc::new(Field::new("element", ArrowType::Utf8View, false));
    arrow_array::ListArray::try_new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(items.finish()),
        None,
    )
    .map(|a| Arc::new(a) as ArrayRef)
    .map_err(|e| {
        InternalError {
            message: e.to_string(),
        }
        .into()
    })
}

/// The value at `path` in each row text: NULL for a missing path, a JSON `null` or a NULL row.
#[must_use]
pub fn json_table_value(row: &StringArray, path: &StringArray) -> ArrayRef {
    let mut out = Vec::with_capacity(row.len());
    for i in 0..row.len() {
        let value = (row.is_valid(i) && path.is_valid(i))
            .then(|| {
                let steps = parse_path(path.value(i))?;
                let doc = parse_document(row.value(i))?;
                match navigate(&doc, &steps)? {
                    Owned::Null => None,
                    Owned::Str(s) => Some(s.clone()),
                    other => Some(render_string(other)),
                }
            })
            .flatten();
        out.push(value);
    }
    Arc::new(StringArray::from(out))
}

/// Whether `path` exists in each row text (a JSON `null` exists); NULL for a NULL row or path.
#[must_use]
pub fn json_table_exists(row: &StringArray, path: &StringArray) -> ArrayRef {
    let mut out = BooleanBuilder::with_capacity(row.len());
    for i in 0..row.len() {
        if row.is_null(i) || path.is_null(i) {
            out.append_null();
            continue;
        }
        let found = parse_path(path.value(i))
            .zip(parse_document(row.value(i)))
            .is_some_and(|(steps, doc)| navigate(&doc, &steps).is_some());
        out.append_value(found);
    }
    Arc::new(out.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_print_as_big_decimal() {
        for (raw, want) in [
            ("1.50", "1.50"),
            ("1e2", "1E+2"),
            ("1.0E2", "1.0E+2"),
            ("-0", "0"),
            ("-0.0", "0.0"),
            ("0.10", "0.10"),
            ("12345678901234567890", "12345678901234567890"),
            ("1E+2", "1E+2"),
            ("100000000000000000000.5", "100000000000000000000.5"),
            ("1.0", "1.0"),
            ("-12.5", "-12.5"),
            ("0.0000001", "1E-7"),
            ("0.000001", "0.000001"),
            ("5e-1", "0.5"),
        ] {
            assert_eq!(big_decimal_string(raw), want, "{raw}");
        }
    }

    fn rows(json: &str, path: &str) -> Vec<String> {
        let out = json_table_rows(&StringArray::from(vec![json]), path, false).unwrap();
        let list = out
            .as_any()
            .downcast_ref::<arrow_array::ListArray>()
            .unwrap();
        let values = list.value(0);
        let strings = arrow_array::cast::AsArray::as_string_view(&values);
        strings.iter().map(|s| s.unwrap().to_owned()).collect()
    }

    /// Row counts and values from a Spark 4.3.0 run.
    #[test]
    fn rows_follow_the_row_path() {
        assert_eq!(
            rows(r#"[{"a":1},2,"x",null,[3]]"#, "$[*]"),
            [r#"{"a":1}"#, "2", r#""x""#, "null", "[3]"]
        );
        assert_eq!(rows(r#"{"a":1}"#, "$"), [r#"{"a":1}"#]);
        assert!(rows(r#"{"a":1}"#, "$.q").is_empty());
        assert!(rows(r#"{"a":1}"#, "$.q[*]").is_empty());
        assert!(rows(r#"{"a":1}"#, "$.a[*]").is_empty());
        assert_eq!(rows(r#"{"a":{"b":[1,2]}}"#, "$.a.b[*]"), ["1", "2"]);
        assert_eq!(rows(r#"{"a":{"b":[1,2]}}"#, "$.a.b"), ["[1,2]"]);
        assert!(rows("not json", "$").is_empty());
        assert!(rows(r#"{"a":1} trailing"#, "$").is_empty());
        assert_eq!(rows("{'a':1}", "$"), [r#"{"a":1}"#]);
        assert!(rows("", "$").is_empty());
        assert_eq!(rows("[1,2]", "$[1]"), ["2"]);
    }

    #[test]
    fn a_malformed_input_raises_with_error_on_error() {
        let e = json_table_rows(&StringArray::from(vec!["not json"]), "$", true).unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::MalformedRecordInParsingWithoutSuggestion
        );
    }

    #[test]
    fn error_on_error_raises_for_null_blank_and_non_array_row_values() {
        let class = krishiv_common::error::ErrorClass::MalformedRecordInParsingWithoutSuggestion;
        for (input, path) in [
            (None, "$[*]"),
            (Some(""), "$[*]"),
            (Some("  "), "$[*]"),
            (Some(r#"{"a":1}"#), "$[*]"),
            (Some(r#"{"a":1}"#), "$.a[*]"),
        ] {
            let e = json_table_rows(&StringArray::from(vec![input]), path, true).unwrap_err();
            assert_eq!(e.error_class(), class, "{input:?} {path}");
            // Without the clause the same inputs have no rows: one valid, empty list.
            let out = json_table_rows(&StringArray::from(vec![input]), path, false)
                .unwrap_or_else(|e| panic!("{input:?} {path}: {e}"));
            let lists = arrow_array::cast::AsArray::as_list::<i32>(&out);
            assert_eq!(lists.len(), 1, "{input:?} {path}");
            assert!(lists.is_valid(0), "{input:?} {path}");
            assert_eq!(lists.value(0).len(), 0, "{input:?} {path}");
        }
        // A missing container has no rows, even with ERROR ON ERROR.
        for path in ["$.b[*]", "$.b"] {
            let out = json_table_rows(&StringArray::from(vec![r#"{"a":1}"#]), path, true).unwrap();
            assert_eq!(out.len(), 1);
            assert_eq!(
                arrow_array::cast::AsArray::as_list::<i32>(&out)
                    .value(0)
                    .len(),
                0
            );
        }
    }

    fn value(row: &str, path: &str) -> Option<String> {
        let out = json_table_value(
            &StringArray::from(vec![row]),
            &StringArray::from(vec![path]),
        );
        arrow_array::cast::AsArray::as_string::<i32>(&out)
            .iter()
            .next()
            .flatten()
            .map(str::to_owned)
    }

    /// The unusual key spellings the SQL lowering accepts select the key written, and `EXISTS` agrees.
    #[test]
    fn unusual_key_spellings_select_their_keys() {
        for (key, path) in [
            ("c'd", "$.c'd"),
            ("a b", "$.a b"),
            ("a]b", "$.a]b"),
            ("a b", "$['a b']"),
        ] {
            let doc = format!("{{{key:?}:7,\"other\":8}}");
            assert_eq!(value(&doc, path).as_deref(), Some("7"), "{path}");
            let exists = json_table_exists(
                &StringArray::from(vec![doc.as_str()]),
                &StringArray::from(vec![path]),
            );
            assert!(
                arrow_array::cast::AsArray::as_boolean(&exists).value(0),
                "{path}"
            );
            assert_eq!(value("{\"other\":8}", path), None, "{path}");
        }
        let nested = r#"{"a b":{"c":9}}"#;
        assert_eq!(value(nested, "$['a b'].c").as_deref(), Some("9"));
    }

    #[test]
    fn values_print_as_spark_does() {
        let doc = r#"{"a": [ 1.50 , 1e2, "x\ny" ], "b": { "k" : 1.50 , "z" : "é\"q" }, "c":1.50, "d":1e2, "e":-0.0, "f":"é", "n":null}"#;
        assert_eq!(value(doc, "$.a").as_deref(), Some(r#"[1.50,1E+2,"x\ny"]"#));
        assert_eq!(
            value(doc, "$.b").as_deref(),
            Some(r#"{"k":1.50,"z":"é\"q"}"#)
        );
        assert_eq!(value(doc, "$.c").as_deref(), Some("1.50"));
        assert_eq!(value(doc, "$.d").as_deref(), Some("1E+2"));
        assert_eq!(value(doc, "$.e").as_deref(), Some("0.0"));
        assert_eq!(value(doc, "$.f").as_deref(), Some("é"));
        assert_eq!(value(doc, "$.n"), None);
        assert_eq!(value(doc, "$.missing"), None);
        assert_eq!(
            value(r#"[1,[2,3],{"a":4}]"#, "$[1][0]").as_deref(),
            Some("2")
        );
        assert_eq!(value(r#"{"a":1,"a":2}"#, "$.a").as_deref(), Some("1"));
        assert_eq!(
            value(r#"{"a":{"b":{"c":5}}}"#, "$.a.b.c").as_deref(),
            Some("5")
        );
        assert_eq!(value(r#"{"a":1}"#, "$['a']").as_deref(), Some("1"));
        assert_eq!(value("2", "$.a"), None);
        assert_eq!(value(r#""x""#, "$").as_deref(), Some("x"));
        assert_eq!(value("null", "$"), None);
        assert_eq!(value("true", "$").as_deref(), Some("true"));
        assert_eq!(value("false", "$").as_deref(), Some("false"));
        assert_eq!(
            value(r#"{"a":[true,false,null]}"#, "$.a").as_deref(),
            Some("[true,false,null]")
        );
    }

    #[test]
    fn a_null_value_exists() {
        let ex = |row: &str, path: &str| {
            let out = json_table_exists(
                &StringArray::from(vec![row]),
                &StringArray::from(vec![path]),
            );
            out.as_any()
                .downcast_ref::<arrow_array::BooleanArray>()
                .unwrap()
                .value(0)
        };
        assert!(ex(r#"{"a":true,"b":null}"#, "$.a"));
        assert!(ex(r#"{"a":true,"b":null}"#, "$.b"));
        assert!(!ex(r#"{"a":true,"b":null}"#, "$.c"));
        assert!(ex(r#"{"a":true}"#, "$"));
        let nulls = json_table_exists(
            &StringArray::from(vec![None, Some("{}")]),
            &StringArray::from(vec![Some("$"), None]),
        );
        assert_eq!(nulls.null_count(), 2);
    }
}
