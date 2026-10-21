// SPDX-License-Identifier: Apache-2.0
//! Regular-expression functions: `regexp_replace`.
//!
//! Spark evaluates patterns with `java.util.regex`. The patterns are
//! translated to the syntax of [`fancy_regex`] (the `regex` crate plus
//! back-references and look-around) by [`java::translate`], which keeps
//! Java's meaning where the two differ: `\d`, `\w`, `\s` are ASCII, `.`
//! stops at every line terminator, `$` also matches before a final line
//! terminator, `\Q..\E` quotes, and `\<`/`\>` are literals. Constructs with
//! no translation (possessive quantifiers, `\G`, the `U`/`d` flags, Unicode
//! blocks) are rejected as an invalid pattern.
//!
//! The replacement follows `Matcher.appendReplacement`: `$n` / `${name}` are
//! groups, `\x` is the literal `x`. Replacement errors are raised on the
//! first match, as Java does.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StringArray};
use arrow_schema::DataType;
use fancy_regex::Regex;
use krishiv_common::error::{
    DatatypeMismatchValueOutOfRange, InvalidParameterValuePattern, KrishivError,
};

use super::args::{Inputs, Prim, Str, internal, is_string};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

mod java;

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["regexp_replace"];

/// Binds the regular-expression function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let well_typed = matches!(types.len(), 3 | 4)
        && types[..3].iter().all(is_string)
        && types
            .get(3)
            .is_none_or(|t| matches!(t, DataType::Int8 | DataType::Int16 | DataType::Int32));
    if !well_typed {
        return Some((listed, Err(bad_types(listed, args))));
    }
    let with_position = types.len() == 4;
    let binding = Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        regexp_replace(x, with_position)
    });
    Some((listed, Ok(binding)))
}

/// A pattern compiled from its Java source.
struct Compiled {
    /// The pattern as written, for the error message.
    source: String,
    /// Matches `$` at the very end of the input only (or the only variant
    /// when the pattern has no non-multiline `$`).
    plain: Regex,
    /// Matches `$` before a final line terminator as well; only used for
    /// inputs that end with one.
    terminated: Option<Regex>,
}

impl Compiled {
    fn new(pattern: &str) -> Result<Self, KrishivError> {
        let invalid = || -> KrishivError {
            InvalidParameterValuePattern {
                function_name: "`regexp_replace`".to_owned(),
                parameter: "`regexp`".to_owned(),
                value: format!("'{pattern}'"),
            }
            .into()
        };
        let plain = java::translate(pattern, false).map_err(|_| invalid())?;
        let build = |p: &str| Regex::new(p).map_err(|_| invalid());
        Ok(Self {
            source: pattern.to_owned(),
            terminated: plain
                .has_dollar
                .then(|| java::translate(pattern, true).map_err(|_| invalid()))
                .transpose()?
                .map(|t| build(&t.text))
                .transpose()?,
            plain: build(&plain.text)?,
        })
    }

    /// The regex for an input ending (or not) in a line terminator.
    fn for_input(&self, text: &str) -> &Regex {
        match &self.terminated {
            Some(t) if text.ends_with(['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}']) => t,
            _ => &self.plain,
        }
    }
}

/// One piece of a replacement, resolved against a pattern.
enum Part {
    Text(String),
    Group(usize),
}

/// `appendReplacement`'s reading of `rep` for a pattern with `groups`
/// groups (group 0 included) named `names`.
fn parse_replacement(
    rep: &str,
    groups: usize,
    names: &[(String, usize)],
) -> Result<Vec<Part>, KrishivError> {
    let err = |m: &str| -> KrishivError { internal(m.to_owned()) };
    let chars: Vec<char> = rep.chars().collect();
    let mut parts = Vec::new();
    let mut text = String::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                i += 1;
                let c = chars
                    .get(i)
                    .ok_or_else(|| err("character to be escaped is missing"))?;
                text.push(*c);
                i += 1;
            }
            '$' => {
                i += 1;
                let first = *chars
                    .get(i)
                    .ok_or_else(|| err("Illegal group reference: group index is missing"))?;
                if first == '{' {
                    i += 1;
                    let start = i;
                    while chars.get(i).is_some_and(char::is_ascii_alphanumeric) {
                        i += 1;
                    }
                    if i == start {
                        return Err(err("named capturing group has 0 length name"));
                    }
                    if chars.get(i) != Some(&'}') {
                        return Err(err("named capturing group is missing trailing '}'"));
                    }
                    let name: String = chars[start..i].iter().collect();
                    i += 1;
                    let index = names
                        .iter()
                        .find(|(n, _)| *n == name)
                        .map(|(_, g)| *g)
                        .ok_or_else(|| internal(format!("No group with name {{{name}}}")))?;
                    flush(&mut text, &mut parts);
                    parts.push(Part::Group(index));
                } else {
                    let digit = first
                        .to_digit(10)
                        .ok_or_else(|| err("Illegal group reference"))?
                        as usize;
                    i += 1;
                    if digit >= groups {
                        return Err(internal(format!("No group {digit}")));
                    }
                    let mut index = digit;
                    while let Some(d) = chars.get(i).and_then(|c| c.to_digit(10)) {
                        let wider = index * 10 + d as usize;
                        if wider >= groups {
                            break;
                        }
                        index = wider;
                        i += 1;
                    }
                    flush(&mut text, &mut parts);
                    parts.push(Part::Group(index));
                }
            }
            c => {
                text.push(c);
                i += 1;
            }
        }
    }
    flush(&mut text, &mut parts);
    Ok(parts)
}

fn flush(text: &mut String, parts: &mut Vec<Part>) {
    if !text.is_empty() {
        parts.push(Part::Text(std::mem::take(text)));
    }
}

/// The byte offset of UTF-16 offset `units` in `text`, `None` when it is at
/// or past the end (Java's `position < length` test, in UTF-16 units).
fn byte_offset(text: &str, units: usize) -> Option<usize> {
    let mut seen = 0;
    for (at, c) in text.char_indices() {
        if seen >= units {
            return Some(at);
        }
        seen += c.len_utf16();
    }
    None
}

/// The memo of the compiled pattern and replacement of the last row.
#[derive(Default)]
struct Cache {
    pattern: Option<(String, Compiled)>,
    replacement: Option<(String, Vec<Part>)>,
}

impl Cache {
    fn compiled(&mut self, pattern: &str) -> Result<&Compiled, KrishivError> {
        if self.pattern.as_ref().is_none_or(|(p, _)| p != pattern) {
            self.pattern = Some((pattern.to_owned(), Compiled::new(pattern)?));
            self.replacement = None;
        }
        self.pattern
            .as_ref()
            .map(|(_, c)| c)
            .ok_or_else(|| internal("regexp_replace: no compiled pattern".to_owned()))
    }
}

/// `regexp_replace(str, regexp, rep[, position])` over a batch.
fn regexp_replace(x: &Inputs<'_>, with_position: bool) -> Result<ArrayRef, KrishivError> {
    let strings = x.strings(0)?;
    let patterns = x.str(1)?;
    let replacements = x.str(2)?;
    let positions = with_position.then(|| x.int(3)).transpose()?;
    let mut cache = Cache::default();
    let mut out = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        let source = strings.is_valid(row).then(|| strings.value(row));
        let pattern = operand(&patterns, row);
        let rep = operand(&replacements, row);
        let position = match &positions {
            Some(Prim::Array(a)) => a.is_valid(row).then(|| a.value(row)),
            Some(Prim::Scalar(v)) => *v,
            None => Some(1),
        };
        let (Some(source), Some(pattern), Some(rep), Some(position)) =
            (source, pattern, rep, position)
        else {
            out.push(None);
            continue;
        };
        if position < 1 {
            return Err(DatatypeMismatchValueOutOfRange {
                sql_expr: "\"regexp_replace\"".to_owned(),
                expr_name: "position".to_owned(),
                value_range: "(0, 2147483647]".to_owned(),
                current_value: position.to_string(),
            }
            .into());
        }
        out.push(Some(replace_row(
            &mut cache, source, pattern, rep, position,
        )?));
    }
    Ok(Arc::new(StringArray::from(out)) as ArrayRef)
}

fn operand(s: &Str, row: usize) -> Option<&str> {
    match s {
        Str::Array(a) => a.is_valid(row).then(|| a.value(row)),
        Str::Scalar(v) => v.as_deref(),
    }
}

/// One row: every match of `pattern` at or after the 1-based `position`
/// replaced, the text before it kept.
fn replace_row(
    cache: &mut Cache,
    source: &str,
    pattern: &str,
    rep: &str,
    position: i32,
) -> Result<String, KrishivError> {
    let start = if position == 1 {
        0
    } else {
        // `usize::try_from` cannot fail for `position >= 1`.
        match byte_offset(source, usize::try_from(position - 1).unwrap_or(usize::MAX)) {
            Some(at) => at,
            None => return Ok(source.to_owned()),
        }
    };
    cache.compiled(pattern)?;
    let Some((_, compiled)) = &cache.pattern else {
        return Err(internal("regexp_replace: no compiled pattern".to_owned()));
    };
    let (head, tail) = source.split_at(start);
    let re = compiled.for_input(tail);
    let at = |e: fancy_regex::Error| internal(format!("{}: {e}", compiled.source));
    let mut out = String::with_capacity(source.len());
    out.push_str(head);
    let (mut last, mut from) = (0, 0);
    let mut parts: Option<&[Part]> = None;
    while from <= tail.len() {
        let Some(caps) = re.captures_from_pos(tail, from).map_err(at)? else {
            break;
        };
        let Some(whole) = caps.get(0) else { break };
        let parts = match parts {
            Some(p) => p,
            None => {
                if cache.replacement.as_ref().is_none_or(|(r, _)| r != rep) {
                    let names: Vec<(String, usize)> = re
                        .capture_names()
                        .enumerate()
                        .filter_map(|(i, n)| n.map(|n| (n.to_owned(), i)))
                        .collect();
                    let parsed = parse_replacement(rep, re.captures_len(), &names)?;
                    cache.replacement = Some((rep.to_owned(), parsed));
                }
                let p = cache
                    .replacement
                    .as_ref()
                    .map(|(_, p)| p.as_slice())
                    .ok_or_else(|| internal("regexp_replace: no replacement".to_owned()))?;
                parts = Some(p);
                p
            }
        };
        out.push_str(&tail[last..whole.start()]);
        for part in parts {
            match part {
                Part::Text(t) => out.push_str(t),
                Part::Group(g) => {
                    if let Some(m) = caps.get(*g) {
                        out.push_str(m.as_str());
                    }
                }
            }
        }
        last = whole.end();
        from = if whole.end() == whole.start() {
            match tail[whole.end()..].chars().next() {
                Some(c) => whole.end() + c.len_utf8(),
                None => break,
            }
        } else {
            whole.end()
        };
    }
    out.push_str(&tail[last..]);
    Ok(out)
}

#[cfg(test)]
mod tests;
