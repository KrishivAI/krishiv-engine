// SPDX-License-Identifier: Apache-2.0
//! Regular-expression functions: `regexp`, `regexp_like`, `regexp_extract`,
//! `regexp_count`, `regexp_instr`, `regexp_substr` and `regexp_replace`.
//!
//! Spark evaluates patterns with `java.util.regex`. The patterns are
//! translated to the syntax of [`fancy_regex`] (the `regex` crate plus
//! back-references and look-around) by [`java::translate`], which keeps
//! Java's meaning where the two differ: `\d`, `\w`, `\s` are ASCII, `.`
//! stops at every line terminator, `$` also matches before a final line
//! terminator, `\Q..\E` quotes, POSIX `\p{..}` classes, possessive
//! quantifiers (as atomic groups) and Java's inline flags. `\<`/`\>` are
//! literals. Constructs with no translation (stacked quantifiers, `\G`, the
//! `U` flag, Unicode blocks) are rejected as an invalid pattern, as are
//! patterns past [`java::MAX_PATTERN_LEN`] characters or with a counted
//! repetition past [`java::MAX_REPETITION`].
//!
//! The replacement follows `Matcher.appendReplacement`: `$n` / `${name}` are
//! groups, `\x` is the literal `x`. Replacement errors are raised on the
//! first match, as Java does.
//!
//! `regexp_extract` takes the group index `idx` (default 1).
//! `regexp_instr`'s `idx` (default 0) only selects which index is
//! validated: the reported position is always the whole match's. An index
//! outside `0..=groupCount` is `INVALID_PARAMETER_VALUE.REGEX_GROUP_INDEX`,
//! checked lazily: only once the pattern matched, as in Spark, so a
//! non-matching input never raises it.
//!
//! Divergence from Spark: patterns with back-references or look-around are
//! matched by backtracking, which can take exponential time on a crafted
//! input (ReDoS). The JVM has no limit and runs until the query is killed;
//! here one match attempt stops after [`BACKTRACK_LIMIT`] steps with
//! `KRISHIV_REGEXP_BACKTRACK_LIMIT_EXCEEDED`. Patterns without those
//! constructs run in linear time and never reach the limit. The look-around
//! the translation adds for `$` and the multiline anchors does not change
//! that: it only follows the pattern (`(a+)+$` stays linear on a near miss
//! ending in a line terminator; pinned in the tests).
//!
//! Positions (`regexp_replace`'s `position`, `regexp_instr`'s result) count
//! UTF-16 units, as Java's `Matcher` does; the goldens only pin ASCII, so
//! supplementary characters stay unverified against Spark 4.2.0.

use std::sync::Arc;

use arrow_array::builder::{GenericStringBuilder, ListBuilder};
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, ListArray, StringArray};
use arrow_schema::{DataType, Field};
use fancy_regex::{Regex, RegexBuilder, RuntimeError};
use krishiv_common::error::{
    DatatypeMismatchValueOutOfRange, InvalidParameterValuePattern,
    InvalidParameterValueRegexGroupIndex, InvalidRegexpReplace, KrishivError,
    KrishivRegexpBacktrackLimitExceeded, UnableToAcquireMemory,
};

use super::args::{Inputs, Prim, Str, internal, is_string};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

mod java;

/// The most backtracking steps one match attempt may take (the
/// `fancy_regex` default, set explicitly so an upgrade cannot change it).
pub(crate) const BACKTRACK_LIMIT: usize = 1_000_000;

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "regexp",
    "regexp_like",
    "regexp_extract",
    "regexp_count",
    "regexp_instr",
    "regexp_substr",
    "regexp_replace",
];

/// Binds the regular-expression function `name`.
///
/// `rlike` is Spark's `RLike` operator (the infix `RLIKE` and the `rlike`
/// function form): the analyzer types it as `Builtin::RLike`, so it is not
/// a registry function and stays out of [`NAMES`]. The executor binds it
/// here to the `regexp_like` kernel, reporting Spark's `rlike` name in
/// pattern errors.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = if name == "rlike" {
        "rlike"
    } else {
        listed(NAMES, name)?
    };
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let strings = |n: usize| types.len() == n && types.iter().all(is_string);
    let indexed = (types.len() == 2 && types.iter().all(is_string))
        || (types.len() == 3
            && types[..2].iter().all(is_string)
            && matches!(types[2], DataType::Int8 | DataType::Int16 | DataType::Int32));
    let binding = match (listed, types.len()) {
        ("regexp" | "regexp_like" | "rlike", 2) if strings(2) => {
            let function: &'static str = match listed {
                "regexp" => "`regexp`",
                "rlike" => "`rlike`",
                _ => "`regexp_like`",
            };
            Binding::new(DataType::Boolean, move |x: &Inputs<'_>| find(x, function))
        }
        ("regexp_extract", 2 | 3) if indexed => {
            let with_idx = types.len() == 3;
            Binding::new(DataType::Utf8, move |x: &Inputs<'_>| extract(x, with_idx))
        }
        ("regexp_count", 2) if strings(2) => {
            Binding::new(DataType::Int32, |x: &Inputs<'_>| count(x))
        }
        ("regexp_instr", 2 | 3) if indexed => {
            let with_idx = types.len() == 3;
            Binding::new(DataType::Int32, move |x: &Inputs<'_>| instr(x, with_idx))
        }
        ("regexp_substr", 2) if strings(2) => {
            Binding::new(DataType::Utf8, |x: &Inputs<'_>| substr(x))
        }
        ("regexp_replace", 3 | 4) => {
            let well_typed = types[..3].iter().all(is_string)
                && types.get(3).is_none_or(|t| {
                    matches!(t, DataType::Int8 | DataType::Int16 | DataType::Int32)
                });
            if !well_typed {
                return Some((listed, Err(bad_types(listed, args))));
            }
            let with_position = types.len() == 4;
            Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                regexp_replace(x, with_position)
            })
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}

/// A pattern compiled from its Java source.
#[derive(Clone)]
struct Compiled {
    /// The pattern as written, for the error message.
    source: String,
    /// The backticked function name, for the error message.
    function: &'static str,
    /// Matches `$` at the very end of the input only (or the only variant
    /// when the pattern has no non-multiline `$`).
    plain: Regex,
    /// Matches `$` before a final line terminator as well; only used for
    /// inputs that end with one.
    terminated: Option<Regex>,
}

impl Compiled {
    fn new(pattern: &str, function: &'static str) -> Result<Self, KrishivError> {
        let invalid = || -> KrishivError {
            InvalidParameterValuePattern {
                function_name: function.to_owned(),
                parameter: "`regexp`".to_owned(),
                value: format!("'{pattern}'"),
            }
            .into()
        };
        let plain = java::translate(pattern, false).map_err(|_| invalid())?;
        let build = |p: &str| {
            RegexBuilder::new(p)
                .backtrack_limit(BACKTRACK_LIMIT)
                .build()
                .map_err(|_| invalid())
        };
        Ok(Self {
            source: pattern.to_owned(),
            function,
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

    /// [`BACKTRACK_LIMIT`] exceeded while matching `self`.
    fn backtrack(&self, e: fancy_regex::Error) -> KrishivError {
        match e {
            fancy_regex::Error::RuntimeError(RuntimeError::BacktrackLimitExceeded) => {
                KrishivRegexpBacktrackLimitExceeded {
                    pattern: format!("'{}'", self.source),
                    function_name: self.function.to_owned(),
                    limit: BACKTRACK_LIMIT.to_string(),
                }
                .into()
            }
            e => internal(format!("{}: {e}", self.source)),
        }
    }

    /// `idx` against the pattern's group count, `INVALID_PARAMETER_VALUE
    /// .REGEX_GROUP_INDEX` when it names no group. Checked lazily, only
    /// once the pattern matched (as in Spark).
    fn group(&self, idx: i32) -> Result<usize, KrishivError> {
        let groups = self.plain.captures_len().saturating_sub(1);
        match usize::try_from(idx).ok().filter(|n| *n <= groups) {
            Some(n) => Ok(n),
            None => Err(InvalidParameterValueRegexGroupIndex {
                parameter: "`idx`".to_owned(),
                function_name: self.function.to_owned(),
                group_count: groups.to_string(),
                group_index: idx.to_string(),
            }
            .into()),
        }
    }
}

/// Java `Matcher.find` from byte `from`: the whole-match span and where the
/// next search resumes (`None` resumes nowhere: stop after this match).
/// An empty match advances past one character, as Java does.
type FindStep = Option<((usize, usize), Option<usize>)>;

fn find_at(compiled: &Compiled, text: &str, from: usize) -> Result<FindStep, KrishivError> {
    let re = compiled.for_input(text);
    let Some(caps) = re
        .captures_from_pos(text, from)
        .map_err(|e| compiled.backtrack(e))?
    else {
        return Ok(None);
    };
    let Some(whole) = caps.get(0) else {
        return Ok(None);
    };
    let next = if whole.end() == whole.start() {
        text[whole.end()..]
            .chars()
            .next()
            .map(|c| whole.end() + c.len_utf8())
    } else {
        Some(whole.end())
    };
    Ok(Some(((whole.start(), whole.end()), next)))
}

/// Iterates Java `find` over `text`: every whole-match span.
fn find_all(compiled: &Compiled, text: &str) -> Result<Vec<(usize, usize)>, KrishivError> {
    let mut out = Vec::new();
    let mut from = Some(0);
    while let Some(at) = from {
        let Some(((start, end), next)) = find_at(compiled, text, at)? else {
            break;
        };
        out.push((start, end));
        from = next;
    }
    Ok(out)
}

/// The memo of the compiled pattern of the last row: a constant pattern
/// compiles once per expression, a column pattern once per distinct value.
#[derive(Default)]
struct Cache {
    pattern: Option<(String, Compiled)>,
}

impl Cache {
    fn compiled(
        &mut self,
        pattern: &str,
        function: &'static str,
    ) -> Result<&Compiled, KrishivError> {
        if self.pattern.as_ref().is_none_or(|(p, _)| p != pattern) {
            self.pattern = Some((pattern.to_owned(), Compiled::new(pattern, function)?));
        }
        self.pattern
            .as_ref()
            .map(|(_, c)| c)
            .ok_or_else(|| internal("no compiled pattern".to_owned()))
    }
}

fn operand(s: &Str, row: usize) -> Option<&str> {
    match s {
        Str::Array(a) => a.is_valid(row).then(|| a.value(row)),
        Str::Scalar(v) => v.as_deref(),
    }
}

fn idx_at(p: &Option<Prim<Int32Type>>, row: usize) -> Option<i32> {
    match p {
        Some(Prim::Array(a)) => a.is_valid(row).then(|| a.value(row)),
        Some(Prim::Scalar(v)) => *v,
        None => None,
    }
}

/// `regexp` / `regexp_like`: Java `find`, true on any match.
fn find(x: &Inputs<'_>, function: &'static str) -> Result<ArrayRef, KrishivError> {
    let strings = x.strings(0)?;
    let patterns = x.str(1)?;
    let mut cache = Cache::default();
    let mut out = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        let (Some(source), Some(pattern)) = (
            strings.is_valid(row).then(|| strings.value(row)),
            operand(&patterns, row),
        ) else {
            out.push(None);
            continue;
        };
        let compiled = cache.compiled(pattern, function)?;
        out.push(Some(find_at(compiled, source, 0)?.is_some()));
    }
    Ok(Arc::new(BooleanArray::from(out)) as ArrayRef)
}

/// `regexp_extract(str, regexp[, idx])`: group `idx` (default 1) of the
/// first match; `""` when the pattern or the group did not match.
fn extract(x: &Inputs<'_>, with_idx: bool) -> Result<ArrayRef, KrishivError> {
    const FUNCTION: &str = "`regexp_extract`";
    let strings = x.strings(0)?;
    let patterns = x.str(1)?;
    let idxs = with_idx.then(|| x.int(2)).transpose()?;
    let mut cache = Cache::default();
    let mut out = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        let (Some(source), Some(pattern)) = (
            strings.is_valid(row).then(|| strings.value(row)),
            operand(&patterns, row),
        ) else {
            out.push(None);
            continue;
        };
        let Some(idx) = idx_at(&idxs, row).or(if with_idx { None } else { Some(1) }) else {
            out.push(None);
            continue;
        };
        let compiled = cache.compiled(pattern, FUNCTION)?;
        let re = compiled.for_input(source);
        let found = re
            .captures_from_pos(source, 0)
            .map_err(|e| compiled.backtrack(e))?;
        // The index is validated only once the pattern matched (as in
        // Spark): a non-matching input never raises.
        let text = match found.as_ref() {
            Some(caps) => {
                let group = compiled.group(idx)?;
                caps.get(group)
                    .map_or_else(String::new, |m| m.as_str().to_owned())
            }
            None => String::new(),
        };
        out.push(Some(text));
    }
    Ok(Arc::new(StringArray::from(out)) as ArrayRef)
}

/// `regexp_count(str, regexp)`: the number of `find` iterations.
fn count(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    const FUNCTION: &str = "`regexp_count`";
    let strings = x.strings(0)?;
    let patterns = x.str(1)?;
    let mut cache = Cache::default();
    let mut out = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        let (Some(source), Some(pattern)) = (
            strings.is_valid(row).then(|| strings.value(row)),
            operand(&patterns, row),
        ) else {
            out.push(None);
            continue;
        };
        let compiled = cache.compiled(pattern, FUNCTION)?;
        let n = find_all(compiled, source)?.len();
        out.push(Some(i32::try_from(n).unwrap_or(i32::MAX)));
    }
    Ok(Arc::new(Int32Array::from(out)) as ArrayRef)
}

/// The 1-based position of byte `at` in `text`, in UTF-16 units (as Java's
/// `Matcher.start` reports it).
fn utf16_position(text: &str, at: usize) -> i32 {
    let units = text[..at].encode_utf16().count();
    i32::try_from(units + 1).unwrap_or(i32::MAX)
}

/// `regexp_instr(str, regexp[, idx])`: the 1-based position of the start
/// of the whole match in the first match (the group index `idx`, default 0,
/// is only validated); 0 when the pattern did not match.
fn instr(x: &Inputs<'_>, with_idx: bool) -> Result<ArrayRef, KrishivError> {
    const FUNCTION: &str = "`regexp_instr`";
    let strings = x.strings(0)?;
    let patterns = x.str(1)?;
    let idxs = with_idx.then(|| x.int(2)).transpose()?;
    let mut cache = Cache::default();
    let mut out = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        let (Some(source), Some(pattern)) = (
            strings.is_valid(row).then(|| strings.value(row)),
            operand(&patterns, row),
        ) else {
            out.push(None);
            continue;
        };
        let Some(idx) = idx_at(&idxs, row).or(if with_idx { None } else { Some(0) }) else {
            out.push(None);
            continue;
        };
        let compiled = cache.compiled(pattern, FUNCTION)?;
        let re = compiled.for_input(source);
        let found = re
            .captures_from_pos(source, 0)
            .map_err(|e| compiled.backtrack(e))?;
        // The index is validated only once the pattern matched (as in
        // Spark): a non-matching input never raises.
        let at = match found.as_ref() {
            Some(caps) => {
                compiled.group(idx)?;
                caps.get(0).map(|m| utf16_position(source, m.start()))
            }
            None => None,
        };
        out.push(Some(at.unwrap_or(0)));
    }
    Ok(Arc::new(Int32Array::from(out)) as ArrayRef)
}

/// `regexp_substr(str, regexp)`: the first match, or NULL when the pattern
/// did not match (an empty match is NULL too, as in Spark).
fn substr(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    const FUNCTION: &str = "`regexp_substr`";
    let strings = x.strings(0)?;
    let patterns = x.str(1)?;
    let mut cache = Cache::default();
    let mut out: Vec<Option<String>> = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        let (Some(source), Some(pattern)) = (
            strings.is_valid(row).then(|| strings.value(row)),
            operand(&patterns, row),
        ) else {
            out.push(None);
            continue;
        };
        let compiled = cache.compiled(pattern, FUNCTION)?;
        let found = find_at(compiled, source, 0)?;
        out.push(match found {
            Some(((start, end), _)) if start != end => Some(source[start..end].to_owned()),
            _ => None,
        });
    }
    Ok(Arc::new(StringArray::from(out)) as ArrayRef)
}

/// Spark `split` around the matches of `compiled` in `s` (Java
/// `Pattern.split` semantics, with Spark's `limit == 0` mapped to `-1`):
/// at most `limit` pieces when `limit > 0` (the last holds the rest),
/// otherwise as many as there are, keeping trailing empty pieces.
fn split_pattern(compiled: &Compiled, s: &str, limit: i32) -> Result<Vec<String>, KrishivError> {
    let spans = find_all(compiled, s)?;
    let limited = limit > 0;
    let cap = usize::try_from(limit).unwrap_or(usize::MAX).max(1);
    let mut parts = Vec::new();
    let mut index = 0;
    for (start, end) in spans {
        if !limited || parts.len() < cap - 1 {
            if index == 0 && start == 0 && start == end {
                // A leading empty match splits nothing (as in Java).
                continue;
            }
            parts.push(s[index..start].to_owned());
            index = end;
        } else if parts.len() == cap - 1 {
            parts.push(s[index..].to_owned());
            index = s.len();
        }
    }
    if index == 0 {
        // No match found (a skipped leading empty does not count).
        return Ok(vec![s.to_owned()]);
    }
    if !limited || parts.len() < cap {
        parts.push(s[index..].to_owned());
    }
    Ok(parts)
}

/// Spark 4.2 `split` of a non-empty `s` by the empty pattern: one piece per
/// character; with `0 < limit < chars` the last holds the rest.
fn split_empty(s: &str, limit: i32) -> Vec<String> {
    let chars: Vec<&str> = {
        let mut out = Vec::new();
        let mut rest = s;
        while let Some(c) = rest.chars().next() {
            let (head, tail) = rest.split_at(c.len_utf8());
            out.push(head);
            rest = tail;
        }
        out
    };
    let count = match usize::try_from(limit) {
        Ok(n) if n > 0 && n < chars.len() => n,
        _ => chars.len(),
    };
    // `s` is non-empty, so `count >= 1`.
    let mut parts: Vec<String> = chars[..count.saturating_sub(1)]
        .iter()
        .map(ToString::to_string)
        .collect();
    parts.push(chars[count.saturating_sub(1)..].concat());
    parts
}

/// The pieces of Spark `split(s, pattern, limit)`: the empty-pattern form
/// for a non-empty `s` and an empty Java `pattern`, else [`split_pattern`].
fn split_pieces(
    s: &str,
    pattern: &str,
    compiled: &Compiled,
    limit: i32,
) -> Result<Vec<String>, KrishivError> {
    if !s.is_empty() && pattern.is_empty() {
        return Ok(split_empty(s, limit));
    }
    split_pattern(compiled, s, limit)
}

/// A Java regex delimiter split for `str_to_map`: the pattern compiles
/// once, then splits every row's text with Spark's limit semantics (as
/// [`split_pieces`]). An invalid pattern is
/// `INVALID_PARAMETER_VALUE.PATTERN`, as for `split`.
#[derive(Clone)]
pub(super) struct Splitter {
    pattern: String,
    compiled: Compiled,
}

impl Splitter {
    /// Compiles the Java regex `pattern` for [`Self::split`].
    pub(super) fn new(pattern: &str) -> Result<Self, KrishivError> {
        Ok(Self {
            pattern: pattern.to_owned(),
            compiled: Compiled::new(pattern, "`str_to_map`")?,
        })
    }

    /// Splits `s` with Spark's limit semantics.
    pub(super) fn split(&self, s: &str, limit: i32) -> Result<Vec<String>, KrishivError> {
        split_pieces(s, &self.pattern, &self.compiled, limit)
    }
}

/// Spark `split(str, pattern[, limit])` over a column: one row per input,
/// NULL in for NULL out. `limit` holds one entry per row (`None` is NULL).
/// The pattern compiles once per call.
///
/// The list holds non-nullable `Utf8` items under the field `item`, as
/// Spark's `array<string>` reads here.
pub fn split_regex(
    input: &StringArray,
    pattern: &str,
    limit: &[Option<i32>],
) -> Result<ListArray, KrishivError> {
    const FUNCTION: &str = "`split`";
    if limit.len() != input.len() {
        return Err(internal(format!(
            "split_regex: {} limits for {} rows",
            limit.len(),
            input.len()
        )));
    }
    let compiled = Compiled::new(pattern, FUNCTION)?;
    let values = GenericStringBuilder::<i32>::new();
    let item = Field::new("item", DataType::Utf8, false);
    let mut builder = ListBuilder::new(values).with_field(Arc::new(item));
    let mut total = 0_usize;
    for (row, lim) in limit.iter().enumerate() {
        let (Some(s), Some(lim)) = (input.is_valid(row).then(|| input.value(row)), *lim) else {
            builder.append(false);
            continue;
        };
        let pieces = split_pieces(s, pattern, &compiled, lim)?;
        // The pieces cover at most the input; each list element adds offsets.
        total = total.saturating_add(s.len());
        if total.saturating_add(pieces.len()) > i32::MAX as usize {
            return Err(UnableToAcquireMemory {
                requested_bytes: total.to_string(),
                received_bytes: "0".to_owned(),
            }
            .into());
        }
        for p in &pieces {
            builder.values().append_value(p);
        }
        builder.append(true);
    }
    Ok(builder.finish())
}

/// `regexp_extract_all(str, regexp[, idx])` over one row: group `idx`
/// (default 1) of every match, `""` for groups that took no part (as in
/// Spark). The index is validated once the pattern matched at least once.
pub fn extract_all_pieces(
    pattern: &str,
    source: &str,
    idx: Option<i32>,
    function: &'static str,
) -> Result<Vec<String>, KrishivError> {
    let compiled = Compiled::new(pattern, function)?;
    extract_all(&compiled, source, idx.unwrap_or(1))
}

/// Group `idx` of every match of `compiled` in `source`.
fn extract_all(compiled: &Compiled, source: &str, idx: i32) -> Result<Vec<String>, KrishivError> {
    let re = compiled.for_input(source);
    let mut out = Vec::new();
    let mut from = Some(0);
    let mut group = None;
    while let Some(at) = from {
        let Some(caps) = re
            .captures_from_pos(source, at)
            .map_err(|e| compiled.backtrack(e))?
        else {
            break;
        };
        let Some(whole) = caps.get(0) else { break };
        let checked = match group {
            Some(g) => g,
            None => {
                let g = compiled.group(idx)?;
                group = Some(g);
                g
            }
        };
        out.push(
            caps.get(checked)
                .map_or_else(String::new, |m| m.as_str().to_owned()),
        );
        from = if whole.end() == whole.start() {
            source[whole.end()..]
                .chars()
                .next()
                .map(|c| whole.end() + c.len_utf8())
        } else {
            Some(whole.end())
        };
    }
    Ok(out)
}

/// One piece of a replacement, resolved against a pattern.
enum Part {
    Text(String),
    Group(usize),
}

/// Spark's `INVALID_REGEXP_REPLACE`, raised for any replacement Java's
/// `appendReplacement` rejects.
fn invalid_replacement(source: &str, pattern: &str, rep: &str, position: i32) -> KrishivError {
    InvalidRegexpReplace {
        source: source.to_owned(),
        pattern: pattern.to_owned(),
        replacement: rep.to_owned(),
        position: position.to_string(),
    }
    .into()
}

/// `appendReplacement`'s reading of `rep` for a pattern with `groups`
/// groups (group 0 included) named `names`; `None` where Java throws.
fn parse_replacement(rep: &str, groups: usize, names: &[(String, usize)]) -> Option<Vec<Part>> {
    let chars: Vec<char> = rep.chars().collect();
    let mut parts = Vec::new();
    let mut text = String::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                i += 1;
                // A trailing backslash has no character to escape.
                let c = chars.get(i)?;
                text.push(*c);
                i += 1;
            }
            '$' => {
                i += 1;
                let first = *chars.get(i)?;
                if first == '{' {
                    i += 1;
                    let start = i;
                    while chars.get(i).is_some_and(char::is_ascii_alphanumeric) {
                        i += 1;
                    }
                    // An empty name, a missing `}` or an unknown name.
                    if i == start || chars.get(i) != Some(&'}') {
                        return None;
                    }
                    let name: String = chars[start..i].iter().collect();
                    i += 1;
                    let index = names.iter().find(|(n, _)| *n == name).map(|(_, g)| *g)?;
                    flush(&mut text, &mut parts);
                    parts.push(Part::Group(index));
                } else {
                    let digit = first.to_digit(10)? as usize;
                    i += 1;
                    if digit >= groups {
                        return None;
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
    Some(parts)
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
struct ReplaceCache {
    pattern: Option<(String, Compiled)>,
    replacement: Option<(String, Vec<Part>)>,
}

impl ReplaceCache {
    fn compiled(&mut self, pattern: &str) -> Result<&Compiled, KrishivError> {
        const FUNCTION: &str = "`regexp_replace`";
        if self.pattern.as_ref().is_none_or(|(p, _)| p != pattern) {
            self.pattern = Some((pattern.to_owned(), Compiled::new(pattern, FUNCTION)?));
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
    let mut cache = ReplaceCache::default();
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

/// One row: every match of `pattern` at or after the 1-based `position`
/// replaced, the text before it kept.
fn replace_row(
    cache: &mut ReplaceCache,
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
    let at = |e: fancy_regex::Error| compiled.backtrack(e);
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
                    let parsed = parse_replacement(rep, re.captures_len(), &names)
                        .ok_or_else(|| invalid_replacement(source, pattern, rep, position))?;
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
