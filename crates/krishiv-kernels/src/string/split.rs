// SPDX-License-Identifier: Apache-2.0
//! `split(str, delimiter[, limit])` for delimiters that are plain text.
//!
//! Spark's delimiter is a Java regular expression. This kernel handles the
//! delimiters that are literal strings: no regex metacharacter, or a single
//! backslash-escaped one (`\.`, `\|`). [`literal_delimiter`] decides; a
//! regular expression delimiter is `KRISHIV_UNSUPPORTED_FEATURE` until the
//! regex engine lands.

use std::sync::Arc;

use arrow_array::builder::{GenericStringBuilder, ListBuilder};
use arrow_array::{ListArray, OffsetSizeTrait};
use arrow_schema::{DataType, Field};

use super::{Slot, StrOperand, all_valid, row_count};
use crate::error::KernelError;

/// Java regex metacharacters (`Pattern` syntax characters).
const META: &str = "\\^$.|?*+()[]{}";

/// The text `regex` matches, if it matches nothing else: the empty pattern,
/// or characters without regex meaning (a backslash may escape one
/// metacharacter). `None` for a real regular expression.
#[must_use]
pub fn literal_delimiter(regex: &str) -> Option<String> {
    let mut out = String::with_capacity(regex.len());
    let mut chars = regex.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            // Java: a backslash before a non-alphabetic character quotes it.
            let n = chars.next().filter(|n| !n.is_alphanumeric())?;
            out.push(n);
        } else if META.contains(c) {
            return None;
        } else {
            out.push(c);
        }
    }
    Some(out)
}

/// Splits `s` at `delimiter`: at most `limit` pieces when `limit > 0` (the
/// last holds the rest), as many as there are otherwise. An empty delimiter
/// splits into characters.
fn pieces<'a>(s: &'a str, delimiter: &str, limit: i32, out: &mut Vec<&'a str>) {
    out.clear();
    let max = usize::try_from(limit).ok().filter(|n| *n > 0);
    let full = |out: &Vec<&str>| max.is_some_and(|m| out.len().saturating_add(1) >= m);
    if delimiter.is_empty() {
        // One piece per character; with a limit, the last piece is the rest.
        let mut rest = s;
        while let Some(c) = rest.chars().next() {
            if full(out) {
                break;
            }
            let (head, tail) = rest.split_at(c.len_utf8());
            out.push(head);
            rest = tail;
        }
        if !rest.is_empty() || out.is_empty() {
            out.push(rest);
        }
        return;
    }
    let mut rest = s;
    while !full(out) {
        match rest.split_once(delimiter) {
            Some((head, tail)) => {
                out.push(head);
                rest = tail;
            }
            None => break,
        }
    }
    out.push(rest);
}

/// `split(input, delimiter, limit)` with a literal `delimiter` (see
/// [`literal_delimiter`]); `limit <= 0` means no limit. The result is a
/// `List<Utf8>` whose items are never NULL.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths or no column;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the pieces do not fit the offset type.
pub fn split_literal<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    delimiter: &str,
    limit: crate::ops::Operand<'_, arrow_array::types::Int32Type>,
) -> Result<ListArray, KernelError> {
    let slots: [&dyn Slot; 2] = [&input, &limit];
    let rows = row_count(&slots)?;
    let values = GenericStringBuilder::<i32>::new();
    let item = Field::new("item", DataType::Utf8, false);
    let mut builder = ListBuilder::new(values).with_field(Arc::new(item));
    let mut parts = Vec::new();
    let mut total = 0_usize;
    for i in 0..rows {
        if !all_valid(&slots, i) {
            builder.append(false);
            continue;
        }
        let s = input.at(i);
        pieces(s, delimiter, super::int_at(limit, i), &mut parts);
        // Pieces cover at most the input; each list element adds offsets.
        total = total.saturating_add(s.len());
        super::check_offset::<i32>(total.saturating_add(parts.len()))?;
        for p in &parts {
            builder.values().append_value(p);
        }
        builder.append(true);
    }
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(s: &str, d: &str, limit: i32) -> Vec<String> {
        let mut out = Vec::new();
        pieces(s, d, limit, &mut out);
        out.into_iter().map(str::to_owned).collect()
    }

    #[test]
    fn literal_delimiters() {
        assert_eq!(literal_delimiter(",").as_deref(), Some(","));
        assert_eq!(literal_delimiter("").as_deref(), Some(""));
        assert_eq!(literal_delimiter("\\|").as_deref(), Some("|"));
        assert_eq!(literal_delimiter("a\\.b").as_deref(), Some("a.b"));
        assert_eq!(literal_delimiter("😀").as_deref(), Some("😀"));
        for regex in [".", "a|b", "[ABC]", "\\d", "a+", "(x)", "\\", "^", "a{2}"] {
            assert_eq!(literal_delimiter(regex), None, "{regex}");
        }
    }

    /// Values recorded from Spark 4.2.
    #[test]
    fn pieces_match_spark() {
        assert_eq!(split("a,b,,", ",", -1), ["a", "b", "", ""]);
        assert_eq!(split("a,b,,", ",", 0), ["a", "b", "", ""]);
        assert_eq!(split("a,b,,", ",", 2), ["a", "b,,"]);
        assert_eq!(split("a,b,,", ",", 1), ["a,b,,"]);
        assert_eq!(split("", ",", -1), [""]);
        assert_eq!(split(",a", ",", -1), ["", "a"]);
        assert_eq!(split("11", "1", -1), ["", "", ""]);
        assert_eq!(split("1", "1", 0), ["", ""]);
        assert_eq!(split("a1b1c", "1", 2), ["a", "b1c"]);
        assert_eq!(split("a1b1c", "1", 3), ["a", "b", "c"]);
        assert_eq!(split("a1b1c", "1", 4), ["a", "b", "c"]);
        assert_eq!(split("aXXb", "XX", -1), ["a", "b"]);
        assert_eq!(split("a😀b😀c", "😀", -1), ["a", "b", "c"]);
        assert_eq!(split("abc", "", -1), ["a", "b", "c"]);
        assert_eq!(split("abc", "", 2), ["a", "bc"]);
        assert_eq!(split("abc", "", 1), ["abc"]);
        assert_eq!(split("abc", "", 10), ["a", "b", "c"]);
        assert_eq!(split("", "", -1), [""]);
        assert_eq!(split("", "", 2), [""]);
    }
}
