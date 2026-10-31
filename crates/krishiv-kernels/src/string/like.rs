// SPDX-License-Identifier: Apache-2.0
//! Spark `LIKE`: `%` matches any run of characters, `_` any single
//! character (a code point, newlines included), and `escape` makes the next
//! character literal (only `%`, `_` and the escape itself may follow it).

use arrow_array::{Array, BooleanArray, GenericStringArray, OffsetSizeTrait, Scalar};

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

/// A compiled `LIKE` pattern. Patterns made only of literals and `%` at the
/// ends take plain string comparisons; the rest run a linear backtracking
/// matcher over the code points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LikePattern {
    shape: Shape,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    Contains(String),
    General(Vec<Token>),
}

impl LikePattern {
    /// Compiles `pattern`; `row` is reported in the error.
    ///
    /// # Errors
    /// `INVALID_FORMAT.ESC_AT_THE_END` for a trailing escape and
    /// `INVALID_FORMAT.ESC_IN_THE_MIDDLE` for an escape before any other
    /// character.
    pub fn compile(pattern: &str, escape: char, row: usize) -> Result<Self, KernelError> {
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
        Ok(Self {
            shape: Shape::of(&tokens),
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
            Shape::Contains(p) => s.contains(p.as_str()),
            Shape::General(tokens) => general_match(tokens, s),
        }
    }
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
            [Token::Many, mid @ .., Token::Many] if !mid.is_empty() => {
                literals(mid).map_or_else(|| Self::General(tokens.to_vec()), Self::Contains)
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
