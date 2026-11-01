// SPDX-License-Identifier: Apache-2.0
//! Spark `LIKE`: `%` matches any run of characters, `_` any single
//! character (a code point, newlines included), and `escape` makes the next
//! character literal (only `%`, `_` and the escape itself may follow it).

use arrow_array::{
    Array, BooleanArray, GenericStringArray, OffsetSizeTrait, Scalar, StringViewArray,
};
use arrow_buffer::BooleanBuffer;
use memchr::memmem::Finder;

use super::{Slot, StrOperand, bool_result, row_count};
use crate::error::KernelError;

/// One element of a compiled pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Literal(char),
    /// `_`.
    One,
    /// `%` (consecutive `%` collapse to one).
    Many,
}

/// A compiled `LIKE` pattern. Patterns made only of literals and `%` work on
/// bytes: comparisons at the ends and substring searches between them. UTF-8
/// finds a valid needle only at a character boundary, so these agree with a
/// match over code points. Patterns with `_` run a linear backtracking matcher
/// over the code points.
#[derive(Debug, Clone)]
pub struct LikePattern {
    shape: Shape,
}

#[derive(Debug, Clone)]
enum Shape {
    /// `%`: everything.
    All,
    /// `abc`.
    Equals(String),
    /// `abc%`.
    Prefix(String),
    /// `%abc`.
    Suffix(String),
    /// `%abc%`.
    Contains(Box<Finder<'static>>),
    /// `ab%cd%ef`, either end possibly empty.
    Segments(Segments),
    General(Vec<Token>),
}

/// The literal runs of a pattern with at least two `%` and no `_`.
#[derive(Debug, Clone)]
struct Segments {
    prefix: Vec<u8>,
    /// Between the `%`s, in order; none is empty.
    middle: Vec<Finder<'static>>,
    suffix: Vec<u8>,
}

impl Segments {
    /// The earliest place for each run leaves the most room for the next, so
    /// a greedy search needs no backtracking.
    fn matches(&self, s: &[u8]) -> bool {
        let Some(rest) = s.strip_prefix(self.prefix.as_slice()) else {
            return false;
        };
        let Some(mut rest) = rest.strip_suffix(self.suffix.as_slice()) else {
            return false;
        };
        for finder in &self.middle {
            let Some(at) = finder.find(rest) else {
                return false;
            };
            rest = rest
                .get(at.saturating_add(finder.needle().len())..)
                .unwrap_or_default();
        }
        true
    }
}

impl LikePattern {
    /// Compiles `pattern`; `row` is reported in the error.
    ///
    /// # Errors
    /// `INVALID_FORMAT.ESC_AT_THE_END` for a trailing escape and
    /// `INVALID_FORMAT.ESC_IN_THE_MIDDLE` for an escape before any other
    /// character.
    pub fn compile(pattern: &str, escape: char, row: usize) -> Result<Self, KernelError> {
        Ok(Self {
            shape: Shape::of(&tokenize(pattern, escape, row)?),
        })
    }

    /// Whether `s` matches.
    #[must_use]
    pub fn matches(&self, s: &str) -> bool {
        match &self.shape {
            Shape::All => true,
            Shape::Equals(p) => s == p,
            Shape::Prefix(p) => s.starts_with(p.as_str()),
            Shape::Suffix(p) => s.ends_with(p.as_str()),
            Shape::Contains(f) => f.find(s.as_bytes()).is_some(),
            Shape::Segments(g) => g.matches(s.as_bytes()),
            Shape::General(tokens) => general_match(tokens, s),
        }
    }

    /// Whether each row of `column` matches; a NULL row's bit means nothing.
    /// The shape is chosen once, not per row.
    #[must_use]
    pub fn matches_views(&self, column: &StringViewArray) -> BooleanBuffer {
        fn run(column: &StringViewArray, f: impl Fn(&str) -> bool) -> BooleanBuffer {
            BooleanBuffer::collect_bool(column.len(), |i| f(column.value(i)))
        }
        match &self.shape {
            Shape::All => BooleanBuffer::new_set(column.len()),
            Shape::Equals(p) => run(column, |s| s == p),
            Shape::Prefix(p) => run(column, |s| s.starts_with(p.as_str())),
            Shape::Suffix(p) => run(column, |s| s.ends_with(p.as_str())),
            Shape::Contains(f) => run(column, |s| f.find(s.as_bytes()).is_some()),
            Shape::Segments(g) => run(column, |s| g.matches(s.as_bytes())),
            Shape::General(tokens) => run(column, |s| general_match(tokens, s)),
        }
    }
}

fn tokenize(pattern: &str, escape: char, row: usize) -> Result<Vec<Token>, KernelError> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        let token = if c == escape {
            match chars.next() {
                None => return Err(KernelError::like_escape_at_end(row, pattern)),
                Some(n) if n == '_' || n == '%' || n == escape => Token::Literal(n),
                Some(n) => return Err(KernelError::like_escape_before(row, pattern, n)),
            }
        } else {
            match c {
                '%' if tokens.last() == Some(&Token::Many) => continue,
                '%' => Token::Many,
                '_' => Token::One,
                other => Token::Literal(other),
            }
        };
        tokens.push(token);
    }
    Ok(tokens)
}

impl Shape {
    fn of(tokens: &[Token]) -> Self {
        let literals = |ts: &[Token]| -> Option<String> {
            ts.iter()
                .map(|t| match t {
                    Token::Literal(c) => Some(*c),
                    _ => None,
                })
                .collect()
        };
        match tokens {
            [] => Self::Equals(String::new()),
            [Token::Many] => Self::All,
            _ if tokens.iter().filter(|t| **t == Token::Many).count() >= 2 => {
                Self::segments(tokens).unwrap_or_else(|| Self::General(tokens.to_vec()))
            }
            [Token::Many, rest @ ..] => {
                literals(rest).map_or_else(|| Self::General(tokens.to_vec()), Self::Suffix)
            }
            [rest @ .., Token::Many] => {
                literals(rest).map_or_else(|| Self::General(tokens.to_vec()), Self::Prefix)
            }
            all => literals(all).map_or_else(|| Self::General(tokens.to_vec()), Self::Equals),
        }
    }

    /// A pattern of literal runs between at least two `%`; `None` with a `_`.
    fn segments(tokens: &[Token]) -> Option<Self> {
        let mut runs = tokens
            .split(|t| *t == Token::Many)
            .map(|run| {
                run.iter()
                    .map(|t| match t {
                        Token::Literal(c) => Some(*c),
                        _ => None,
                    })
                    .collect::<Option<String>>()
            })
            .collect::<Option<Vec<_>>>()?;
        let suffix = runs.pop()?.into_bytes();
        let mut runs = runs.into_iter();
        let prefix = runs.next()?.into_bytes();
        let middle: Vec<Finder<'static>> = runs
            .map(|run| Finder::new(run.as_bytes()).into_owned())
            .collect();
        Some(match middle.as_slice() {
            [only] if prefix.is_empty() && suffix.is_empty() => {
                Self::Contains(Box::new(only.clone()))
            }
            _ => Self::Segments(Segments {
                prefix,
                middle,
                suffix,
            }),
        })
    }
}

/// Wildcard matching with one backtrack point (the last `%`): linear for
/// patterns with a single `%`, quadratic in the worst case, never
/// exponential.
fn general_match(tokens: &[Token], s: &str) -> bool {
    let next_char = |at: usize| s.get(at..).and_then(|r| r.chars().next());
    let (mut t, mut at) = (0_usize, 0_usize);
    // Token after the last `%`, and where in `s` it resumes.
    let mut retry: Option<(usize, usize)> = None;
    loop {
        let c = next_char(at);
        let advanced = match (tokens.get(t), c) {
            (Some(Token::Many), _) => {
                t = t.saturating_add(1);
                retry = Some((t, at));
                continue;
            }
            (Some(Token::One), Some(c)) => Some(c),
            (Some(Token::Literal(l)), Some(c)) if *l == c => Some(c),
            (None, None) => return true,
            _ => None,
        };
        if let Some(c) = advanced {
            t = t.saturating_add(1);
            at = at.saturating_add(c.len_utf8());
            continue;
        }
        // Mismatch: let the last `%` swallow one more character.
        let Some((after, from)) = retry else {
            return false;
        };
        let Some(skipped) = next_char(from) else {
            return false;
        };
        let resume = from.saturating_add(skipped.len_utf8());
        retry = Some((after, resume));
        t = after;
        at = resume;
    }
}

/// Rewrites a validated `pattern` (escape `escape`) into arrow-string's
/// dialect: `%` and `_` are wildcards and a backslash is the only escape.
fn to_arrow_pattern(pattern: &str, escape: char) -> String {
    let mut out = String::with_capacity(pattern.len());
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        let (literal, c) = if c == escape {
            // A trailing escape cannot reach here: `compile` rejected it.
            (true, chars.next().unwrap_or(c))
        } else {
            (false, c)
        };
        if (literal && matches!(c, '%' | '_')) || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// `str LIKE pattern ESCAPE escape`.
///
/// # Errors
/// `INVALID_FORMAT.ESC_AT_THE_END` / `ESC_IN_THE_MIDDLE` for a malformed
/// pattern in a row where both operands are non-NULL;
/// `INTERNAL_ERROR` for operands of different lengths or two literals.
pub fn like<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    pattern: StrOperand<'_, O>,
    escape: char,
) -> Result<BooleanArray, KernelError> {
    let slots: [&dyn Slot; 2] = [&input, &pattern];
    let rows = row_count(&slots)?;
    // A column against one literal pattern, the usual shape, runs on
    // arrow-string's matchers once the pattern is validated here (Spark's
    // escape errors are ours; an all-NULL column never reaches them).
    if let (StrOperand::Array(col), StrOperand::Scalar(Some(p))) = (input, pattern) {
        if col.null_count() == col.len() {
            return Ok(BooleanArray::new_null(rows));
        }
        let first = (0..col.len()).find(|&i| col.is_valid(i)).unwrap_or(0);
        LikePattern::compile(p, escape, first)?;
        let arrow_pattern = Scalar::new(GenericStringArray::<O>::from(vec![to_arrow_pattern(
            p, escape,
        )]));
        // arrow-string compiles a regex for general patterns and refuses one
        // over the regex size limit (a very long pattern); the matcher below
        // has no such limit, so that case runs on it.
        if let Ok(result) = arrow_string::like::like(col, &arrow_pattern) {
            return Ok(result);
        }
    }
    // The pattern of the last row, compiled once while it repeats (always,
    // for a literal pattern).
    let mut compiled: Option<(String, LikePattern)> = None;
    let mut error = None;
    let result = bool_result(rows, &slots, |i| {
        if error.is_some() {
            return false;
        }
        let p = pattern.at(i);
        let cached = compiled.as_ref().filter(|(src, _)| src == p);
        let m = match cached {
            Some((_, m)) => m,
            None => match LikePattern::compile(p, escape, i) {
                Ok(m) => &compiled.insert((p.to_owned(), m)).1,
                Err(e) => {
                    error = Some(e);
                    return false;
                }
            },
        };
        m.matches(input.at(i))
    });
    match error {
        Some(e) => Err(e),
        None => Ok(result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(pattern: &str, s: &str) -> bool {
        LikePattern::compile(pattern, '\\', 0).unwrap().matches(s)
    }

    #[test]
    fn shapes_and_matching() {
        assert!(m("", "") && !m("", "a"));
        assert!(m("%", "") && m("%%", "anything"));
        assert!(m("abc", "abc") && !m("abc", "abcd"));
        assert!(m("ab%", "abc") && !m("ab%", "a"));
        assert!(m("%bc", "abc") && !m("%bc", "bcd"));
        assert!(m("%b%", "abc") && !m("%b%", "xyz"));
        assert!(m("a_c", "abc") && m("a_c", "a😀c") && !m("a_c", "ac"));
        assert!(m("_", "\n") && m("%", "\n") && m("a%b", "a\n\nb"));
        assert!(m("a%b_c%", "axxbyc") && m("a%b_c%", "abyczzz"));
        assert!(!m("a%b_c", "abc") && !m("%_%_%", "x"));
        // Backtracking over a repeated prefix.
        assert!(m("%ab%ab_", "ababab_") && !m("%ab%abc", "abababab"));
        assert!(m("é%", "éa") && !m("É", "é"));
        assert!(m("\\%", "%") && m("\\_", "_") && m("\\\\", "\\"));
        assert!(!m("\\%", "a"));
    }

    #[test]
    fn segments_match_in_order_without_overlap() {
        assert!(m("%special%requests%", "xx special yy requests zz"));
        assert!(!m("%special%requests%", "requests special"));
        assert!(m("a%b%c", "a b c") && m("a%b%c", "abc"));
        assert!(!m("ab%b%bc", "abbc") && m("ab%b%bc", "abbbc"));
        assert!(!m("a%a", "a") && m("a%a", "aa"));
        assert!(m("%é%😀", "xéy😀") && !m("%é%😀", "😀é"));
        assert!(m("%aa%aa%", "aaaa") && !m("%aa%aa%", "aaa"));
    }

    proptest::proptest! {
        /// Every shape answers as the code-point matcher does, for one row
        /// and over a view column.
        #[test]
        fn shapes_agree_with_the_matcher(
            pattern in "[ab%_é😀]{0,8}",
            rows in proptest::collection::vec("[abé😀]{0,16}", 1..8),
        ) {
            let tokens = tokenize(&pattern, '\\', 0).unwrap();
            let compiled = LikePattern::compile(&pattern, '\\', 0).unwrap();
            let column = StringViewArray::from_iter_values(&rows);
            let bits = compiled.matches_views(&column);
            for (i, row) in rows.iter().enumerate() {
                let want = general_match(&tokens, row);
                proptest::prop_assert_eq!(compiled.matches(row), want, "{} on {}", pattern, row);
                proptest::prop_assert_eq!(bits.value(i), want, "{} on {}", pattern, row);
            }
        }
    }

    #[test]
    fn escape_errors_and_other_escapes() {
        let e = LikePattern::compile("a\\", '\\', 3).unwrap_err();
        assert_eq!(e.row(), Some(3));
        assert_eq!(
            e.to_string(),
            "[INVALID_FORMAT.ESC_AT_THE_END] The format is invalid: 'a\\\\'. The escape \
             character is not allowed to end with. SQLSTATE: 42601"
        );
        let e = LikePattern::compile("a\\b", '\\', 0).unwrap_err();
        assert_eq!(
            e.to_string(),
            "[INVALID_FORMAT.ESC_IN_THE_MIDDLE] The format is invalid: 'a\\\\b'. The escape \
             character is not allowed to precede 'b'. SQLSTATE: 42601"
        );
        let p = LikePattern::compile("a$%b$$", '$', 0).unwrap();
        assert!(p.matches("a%b$") && !p.matches("axb$"));
        // A `\` is an ordinary character under another escape.
        assert!(
            LikePattern::compile("a\\b", '$', 0)
                .unwrap()
                .matches("a\\b")
        );
    }
}
