// SPDX-License-Identifier: Apache-2.0
//! Oracle-style number formats (`to_number` / `try_to_number`, `to_char` /
//! `to_varchar`): one parser module shared by both directions.
//!
//! This is a port of Spark 4.2's `ToNumberParser`
//! (`sql/catalyst/util/ToNumberParser.scala`): the format characters `9 0
//! . , D G $ S MI PR` (case-insensitive; the format is upper-cased before
//! parsing), the `INVALID_FORMAT.*` analysis errors, the token rules of
//! `parse` (whitespace skipped inside digit runs, `,` the only grouping
//! separator, `.` the only decimal point, `S`/`MI` toggling the sign, `PR`
//! as `<...>`) and the padding rules of `format` (spaces for `9`, zeros for
//! leading `0`, `#` on overflow).
//!
//! Java `DecimalFormat` patterns (`format_number` with a string pattern)
//! are a different language; they live in [`decimal_format`].

#![expect(
    clippy::arithmetic_side_effects,
    reason = "indexes advance over short pattern/input texts and are bounded by their lengths; \
              digit counts are bounded by the pattern"
)]

use krishiv_common::error::{
    InvalidFormatContThousandsSeps, InvalidFormatCurMustBeforeDec, InvalidFormatCurMustBeforeDigit,
    InvalidFormatEmpty, InvalidFormatMismatchInput, InvalidFormatThousandsSepsMustBeforeDec,
    InvalidFormatUnexpectedToken, InvalidFormatWrongNumDigit, InvalidFormatWrongNumToken,
    KrishivError,
};

use crate::error::KernelError;

pub mod convert;
pub mod decimal_format;

/// A parsed number format: the token sequence plus the decimal type the
/// pattern describes.
#[derive(Debug, Clone)]
pub struct NumberFormat {
    tokens: Vec<Token>,
    precision: u32,
    scale: u32,
}

/// One token of a parsed format.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    /// A run of digit slots (and grouping separators) with the slots in
    /// reverse order, so leading optional slots are skipped first.
    DigitGroups {
        tokens: Vec<GroupToken>,
        digits: Vec<Digits>,
    },
    DecimalPoint,
    DollarSign,
    /// `S`: an optional `+`/`-`.
    PlusOrMinus,
    /// `MI`: an optional `-`.
    Minus,
    OpeningAngleBracket,
    ClosingAngleBracket,
    /// Anything else (after upper-casing).
    Invalid(char),
}

/// A slot of a [`Token::DigitGroups`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupToken {
    Digits(Digits),
    Separator,
}

/// A run of `0`/`9` slots: exactly, or at most, this many input digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Digits {
    Exactly(u32),
    AtMost(u32),
}

impl Digits {
    fn len(&self) -> u32 {
        match self {
            Self::Exactly(n) | Self::AtMost(n) => *n,
        }
    }
}

impl Token {
    /// As `INVALID_FORMAT.WRONG_NUM_TOKEN` and `UNEXPECTED_TOKEN` print it.
    fn name(&self) -> String {
        match self {
            Self::DigitGroups { .. } => "digit sequence".to_owned(),
            Self::DecimalPoint => ". or D".to_owned(),
            Self::DollarSign => "$".to_owned(),
            Self::PlusOrMinus => "S".to_owned(),
            Self::Minus => "MI".to_owned(),
            Self::OpeningAngleBracket | Self::ClosingAngleBracket => "PR".to_owned(),
            Self::Invalid(c) => format!("character '{c}''"),
        }
    }

    /// Whether `other` may stand in for `self` in the token-order check
    /// (any digit group stands in for any other).
    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::DigitGroups { .. }, Self::DigitGroups { .. }) => true,
            (a, b) => a == b,
        }
    }
}

/// The format as an `INVALID_FORMAT` error prints it: single-quoted, with
/// `\` and `'` escaped (Spark's `toSQLValue`).
fn quoted(format: &str) -> String {
    let mut out = String::with_capacity(format.len() + 2);
    out.push('\'');
    for c in format.chars() {
        if c == '\'' || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('\'');
    out
}

/// A pre-grouping token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Raw {
    Digits(Digits),
    Separator,
    DecimalPoint,
    DollarSign,
    PlusOrMinus,
    Minus,
    OpeningAngleBracket,
    ClosingAngleBracket,
    Invalid(char),
}

fn flush(tokens: &mut Vec<Token>, group: &mut Vec<GroupToken>, digits: &mut Vec<Digits>) {
    if !group.is_empty() {
        group.reverse();
        digits.reverse();
        tokens.push(Token::DigitGroups {
            tokens: std::mem::take(group),
            digits: std::mem::take(digits),
        });
    }
}

/// Parses `format` (as written, any case) into a [`NumberFormat`].
#[must_use]
pub fn parse_format(format: &str) -> NumberFormat {
    let upper = format.to_uppercase();
    let chars: Vec<char> = upper.chars().collect();
    let mut raw: Vec<Raw> = Vec::new();
    let mut i = 0;
    let mut reached_point = false;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '0' => {
                let start = i;
                while i < chars.len() && (chars[i] == '0' || chars[i] == '9') {
                    i += 1;
                }
                let n = u32::try_from(i - start).unwrap_or(u32::MAX);
                raw.push(if reached_point {
                    Raw::Digits(Digits::AtMost(n))
                } else {
                    Raw::Digits(Digits::Exactly(n))
                });
            }
            '9' => {
                let start = i;
                while i < chars.len() && (chars[i] == '0' || chars[i] == '9') {
                    i += 1;
                }
                raw.push(Raw::Digits(Digits::AtMost(
                    u32::try_from(i - start).unwrap_or(u32::MAX),
                )));
            }
            '.' | 'D' => {
                raw.push(Raw::DecimalPoint);
                reached_point = true;
                i += 1;
            }
            ',' | 'G' => {
                raw.push(Raw::Separator);
                i += 1;
            }
            '$' => {
                raw.push(Raw::DollarSign);
                i += 1;
            }
            'S' => {
                raw.push(Raw::PlusOrMinus);
                i += 1;
            }
            'M' if i + 1 < chars.len() && chars[i + 1] == 'I' => {
                raw.push(Raw::Minus);
                i += 2;
            }
            'P' if i + 1 < chars.len() && chars[i + 1] == 'R' => {
                // `PR` wraps the whole format in angle brackets.
                raw.push(Raw::OpeningAngleBracket);
                raw.push(Raw::ClosingAngleBracket);
                i += 2;
            }
            _ => {
                raw.push(Raw::Invalid(c));
                i += 1;
            }
        }
    }
    // Combine consecutive digit and separator tokens into groups.
    let mut tokens: Vec<Token> = Vec::new();
    let mut group: Vec<GroupToken> = Vec::new();
    let mut digits: Vec<Digits> = Vec::new();
    let mut prepend_open = false;
    for token in raw {
        match token {
            Raw::Digits(d) => {
                group.push(GroupToken::Digits(d));
                digits.push(d);
            }
            Raw::Separator => group.push(GroupToken::Separator),
            Raw::OpeningAngleBracket => prepend_open = true,
            Raw::DecimalPoint => {
                flush(&mut tokens, &mut group, &mut digits);
                tokens.push(Token::DecimalPoint);
            }
            Raw::DollarSign => {
                flush(&mut tokens, &mut group, &mut digits);
                tokens.push(Token::DollarSign);
            }
            Raw::PlusOrMinus => {
                flush(&mut tokens, &mut group, &mut digits);
                tokens.push(Token::PlusOrMinus);
            }
            Raw::Minus => {
                flush(&mut tokens, &mut group, &mut digits);
                tokens.push(Token::Minus);
            }
            Raw::ClosingAngleBracket => {
                flush(&mut tokens, &mut group, &mut digits);
                tokens.push(Token::ClosingAngleBracket);
            }
            Raw::Invalid(c) => {
                flush(&mut tokens, &mut group, &mut digits);
                tokens.push(Token::Invalid(c));
            }
        }
    }
    flush(&mut tokens, &mut group, &mut digits);
    if prepend_open {
        tokens.insert(0, Token::OpeningAngleBracket);
    }
    let precision = tokens
        .iter()
        .map(|t| match t {
            Token::DigitGroups { digits, .. } => digits.iter().map(Digits::len).sum(),
            _ => 0,
        })
        .sum();
    let scale = match tokens.iter().position(|t| *t == Token::DecimalPoint) {
        Some(point) => tokens[point..]
            .iter()
            .map(|t| match t {
                Token::DigitGroups { digits, .. } => digits.iter().map(Digits::len).sum(),
                _ => 0,
            })
            .sum(),
        None => 0,
    };
    NumberFormat {
        tokens,
        precision,
        scale,
    }
}

impl NumberFormat {
    /// The decimal type the pattern describes: `(precision, scale)`.
    #[must_use]
    pub fn decimal_type(&self) -> (u32, u32) {
        (self.precision, self.scale)
    }

    /// Spark's `checkInputDataTypes`: the `INVALID_FORMAT.*` analysis
    /// errors for an invalid pattern.
    ///
    /// # Errors
    /// The pattern's `INVALID_FORMAT` subclass, quoting the format as
    /// written.
    pub fn validate(&self, format: &str) -> Result<(), KrishivError> {
        let quoted = quoted(format);
        let first_dollar = self.tokens.iter().position(|t| *t == Token::DollarSign);
        let first_digits = self
            .tokens
            .iter()
            .position(|t| matches!(t, Token::DigitGroups { .. }));
        let first_point = self.tokens.iter().position(|t| *t == Token::DecimalPoint);
        if format.is_empty() {
            return Err(InvalidFormatEmpty { format: quoted }.into());
        }
        if first_digits.is_none() {
            return Err(InvalidFormatWrongNumDigit { format: quoted }.into());
        }
        if let (Some(digits), Some(dollar)) = (first_digits, first_dollar)
            && digits < dollar
        {
            return Err(InvalidFormatCurMustBeforeDigit { format: quoted }.into());
        }
        if let (Some(point), Some(dollar)) = (first_point, first_dollar)
            && point < dollar
        {
            return Err(InvalidFormatCurMustBeforeDec { format: quoted }.into());
        }
        // Thousands separators need digits around them. (The grouped slots
        // are reversed, so the ends are the format's ends.)
        let bad_groups = self.tokens.iter().enumerate().any(|(i, t)| {
            let Token::DigitGroups { tokens, .. } = t else {
                return false;
            };
            if first_point.is_some_and(|p| i > p) {
                return false;
            }
            tokens.iter().enumerate().any(|(j, slot)| {
                if *slot != GroupToken::Separator {
                    return false;
                }
                if j == 0 || j + 1 == tokens.len() {
                    return true;
                }
                matches!(tokens[j - 1], GroupToken::Separator)
                    || matches!(tokens[j + 1], GroupToken::Separator)
            })
        });
        if bad_groups {
            return Err(InvalidFormatContThousandsSeps { format: quoted }.into());
        }
        // No separators after the decimal point.
        let seps_after = match first_point {
            Some(p) => self.tokens[p + 1..].iter().any(|t| match t {
                Token::DigitGroups { tokens, digits } => tokens.len() > digits.len(),
                _ => false,
            }),
            None => false,
        };
        if seps_after {
            return Err(InvalidFormatThousandsSepsMustBeforeDec { format: quoted }.into());
        }
        for token in [
            Token::DecimalPoint,
            Token::PlusOrMinus,
            Token::Minus,
            Token::DollarSign,
            Token::ClosingAngleBracket,
        ] {
            if self.tokens.iter().filter(|t| *t == &token).count() > 1 {
                return Err(InvalidFormatWrongNumToken {
                    token: token.name(),
                    format: quoted,
                }
                .into());
            }
        }
        // `[MI|S] [$] [0|9|G|,]* [.|D] [0|9]* [$] [PR|MI|S]`.
        let digit_group = Token::DigitGroups {
            tokens: Vec::new(),
            digits: Vec::new(),
        };
        let slots: &[&[Token]] = &[
            &[Token::OpeningAngleBracket],
            &[Token::Minus, Token::PlusOrMinus],
            &[Token::DollarSign],
            std::slice::from_ref(&digit_group),
            &[Token::DecimalPoint],
            std::slice::from_ref(&digit_group),
            &[Token::DollarSign],
            &[Token::Minus, Token::PlusOrMinus, Token::ClosingAngleBracket],
        ];
        let mut at = 0;
        for slot in slots {
            if at < self.tokens.len() && slot.iter().any(|want| want.matches(&self.tokens[at])) {
                at += 1;
            }
        }
        if at < self.tokens.len() {
            return Err(InvalidFormatUnexpectedToken {
                token: self.tokens[at].name(),
                format: quoted,
            }
            .into());
        }
        Ok(())
    }

    /// Parses `input` against the pattern: the sign and the digit strings
    /// before and after the decimal point.
    ///
    /// # Errors
    /// [`ParseFailure`] when the input does not match; the caller renders
    /// it as `INVALID_FORMAT.MISMATCH_INPUT` (`to_number`) or NULL
    /// (`try_to_number`).
    pub fn parse_input(&self, input: &str) -> Result<ParsedNumber, ParseFailure> {
        let chars: Vec<char> = input.chars().collect();
        let mut before = String::new();
        let mut after = String::new();
        let mut reached_point = false;
        let mut reached_open = false;
        let mut reached_close = false;
        let mut negate = false;
        let mut i = 0;
        for token in &self.tokens {
            let current = chars.get(i).copied();
            match token {
                Token::DigitGroups { digits, .. } => {
                    let mut sizes: Vec<usize> = Vec::new();
                    let mut group_len = 0_usize;
                    while i < chars.len() {
                        let c = chars[i];
                        if is_java_whitespace(c) {
                            i += 1;
                        } else if c.is_ascii_digit() {
                            group_len += 1;
                            if reached_point {
                                after.push(c);
                            } else {
                                before.push(c);
                            }
                            i += 1;
                        } else if c == ',' {
                            sizes.push(group_len);
                            group_len = 0;
                            i += 1;
                        } else {
                            sizes.push(group_len);
                            break;
                        }
                    }
                    if i == chars.len() {
                        sizes.push(group_len);
                    }
                    // Innermost first, like the reversed expected slots.
                    sizes.reverse();
                    if sizes.len() > digits.len() {
                        return Err(ParseFailure);
                    }
                    for (k, want) in digits.iter().enumerate() {
                        let actual =
                            u32::try_from(sizes.get(k).copied().unwrap_or(0)).unwrap_or(u32::MAX);
                        match want {
                            Digits::Exactly(n) if actual != *n => return Err(ParseFailure),
                            Digits::AtMost(n) if actual > *n => return Err(ParseFailure),
                            _ => {}
                        }
                    }
                }
                Token::DecimalPoint => {
                    if current == Some('.') {
                        reached_point = true;
                        i += 1;
                    }
                }
                Token::DollarSign => {
                    if current != Some('$') {
                        return Err(ParseFailure);
                    }
                    i += 1;
                }
                Token::PlusOrMinus => match current {
                    Some('+') => i += 1,
                    Some('-') => {
                        negate = !negate;
                        i += 1;
                    }
                    _ => {}
                },
                Token::Minus => {
                    if current == Some('-') {
                        negate = !negate;
                        i += 1;
                    }
                }
                Token::OpeningAngleBracket => {
                    if current == Some('<') {
                        if reached_open {
                            return Err(ParseFailure);
                        }
                        reached_open = true;
                        i += 1;
                    }
                }
                Token::ClosingAngleBracket => {
                    if current == Some('>') {
                        if !reached_open {
                            return Err(ParseFailure);
                        }
                        reached_close = true;
                        negate = !negate;
                        i += 1;
                    }
                }
                // Unreachable: invalid patterns fail validation, and
                // validation runs before any parse.
                Token::Invalid(_) => return Err(ParseFailure),
            }
        }
        if i < chars.len() || reached_open != reached_close {
            return Err(ParseFailure);
        }
        if before.is_empty() && after.is_empty() {
            return Err(ParseFailure);
        }
        Ok(ParsedNumber {
            negate,
            before,
            after,
        })
    }

    /// Formats a decimal value (as plain text, e.g. `"-0012.3400"`) with
    /// the pattern.
    ///
    /// # Errors
    /// [`ParseFailure`] when the value does not fit; the caller renders it
    /// as `INVALID_FORMAT.MISMATCH_INPUT`.
    pub fn format_decimal(&self, plain: &str) -> Result<String, ParseFailure> {
        let (before, after) = match plain.split_once('.') {
            Some((b, a)) => (b.to_owned(), a.to_owned()),
            None => (plain.to_owned(), String::new()),
        };
        let before = before.trim_start_matches(['-', '0']);
        let after: String = after
            .chars()
            .rev()
            .skip_while(|c| *c == '0')
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        // The pattern's digit counts before and after the point.
        let mut reached_point = false;
        let (mut want_before, mut want_after) = (0_usize, 0_usize);
        for token in &self.tokens {
            match token {
                Token::DigitGroups { digits, .. } => {
                    let n: usize = digits.iter().map(|d| d.len() as usize).sum();
                    if reached_point {
                        want_after += n;
                    } else {
                        want_before += n;
                    }
                }
                Token::DecimalPoint => reached_point = true,
                _ => {}
            }
        }
        let (before, after) = if want_before < before.len() || want_after < after.len() {
            ("#".repeat(want_before), "#".repeat(want_after))
        } else {
            (
                format!("{}{before}", " ".repeat(want_before - before.len())),
                format!("{after}{}", "0".repeat(want_after - after.len())),
            )
        };
        let before_chars: Vec<char> = before.chars().collect();
        let after_chars: Vec<char> = after.chars().collect();
        let mut out = String::new();
        let (mut bi, mut ai) = (0_usize, 0_usize);
        let mut reached_point = false;
        let negative = plain.starts_with('-');
        for token in &self.tokens {
            match token {
                Token::DigitGroups { tokens, .. } => {
                    for slot in tokens.iter().rev() {
                        match slot {
                            GroupToken::Digits(d) => {
                                for _ in 0..d.len() {
                                    if reached_point {
                                        out.push(after_chars[ai]);
                                        ai += 1;
                                    } else {
                                        match before_chars[bi] {
                                            ' ' if matches!(d, Digits::Exactly(_)) => {
                                                out.push('0');
                                            }
                                            ' ' => add_space_checking_signs(&mut out),
                                            c => out.push(c),
                                        }
                                        bi += 1;
                                    }
                                }
                            }
                            GroupToken::Separator => {
                                if out.chars().last().is_some_and(|c| c.is_ascii_digit()) {
                                    out.push(',');
                                } else {
                                    add_space_checking_signs(&mut out);
                                }
                            }
                        }
                    }
                }
                Token::DecimalPoint => {
                    if out.ends_with(' ') {
                        out.pop();
                        out.push('0');
                    }
                    out.push('.');
                    reached_point = true;
                }
                Token::DollarSign => out.push('$'),
                Token::PlusOrMinus => {
                    strip_trailing_point(&mut out);
                    if negative {
                        add_char_checking_spaces(&mut out, '-');
                    } else {
                        add_char_checking_spaces(&mut out, '+');
                    }
                }
                Token::Minus => {
                    if negative {
                        strip_trailing_point(&mut out);
                        add_char_checking_spaces(&mut out, '-');
                    } else {
                        out.push(' ');
                    }
                }
                Token::OpeningAngleBracket => {
                    if negative {
                        out.push('<');
                    }
                }
                Token::ClosingAngleBracket => {
                    strip_trailing_point(&mut out);
                    if negative {
                        add_char_checking_spaces(&mut out, '>');
                    } else {
                        out.push(' ');
                        out.push(' ');
                    }
                }
                Token::Invalid(_) => return Err(ParseFailure),
            }
        }
        if bi < before_chars.len() || ai < after_chars.len() {
            return Err(ParseFailure);
        }
        strip_trailing_point(&mut out);
        if out.is_empty() || out == "+" || out == "-" {
            Ok("0".to_owned())
        } else {
            Ok(out)
        }
    }
}

/// Java's `Character.isWhitespace`: ASCII controls, `Cc` separators and
/// Unicode spaces except the non-breaking ones.
///
/// Non-ASCII digits are not consumed as digits (Java would append them and
/// fail later building the decimal; here the run simply ends).
fn is_java_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{B}' | '\u{C}' | '\r' | '\u{1C}'..='\u{1F}'
    ) || (c.is_whitespace() && !matches!(c, '\u{A0}' | '\u{2007}' | '\u{202F}'))
}

/// A parsed input: the sign and the digit strings around the point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedNumber {
    /// Whether the input negates the result.
    pub negate: bool,
    /// The digits before the decimal point.
    pub before: String,
    /// The digits after the decimal point.
    pub after: String,
}

impl ParsedNumber {
    /// The unscaled value at `scale`: `before` with `after` padded to
    /// `scale` digits.
    ///
    /// # Errors
    /// [`ParseFailure`] when the digits do not fit an `i128` (patterns
    /// above 38 digits fail earlier, at the decimal type).
    pub fn unscaled(&self, scale: u32) -> Result<i128, ParseFailure> {
        let pad = (scale as usize).saturating_sub(self.after.len());
        let digits = format!("{}{}{}", self.before, self.after, "0".repeat(pad));
        let digits = digits.trim_start_matches('0');
        let value: i128 = if digits.is_empty() {
            0
        } else {
            digits.parse().map_err(|_| ParseFailure)?
        };
        Ok(if self.negate { -value } else { value })
    }
}

/// The input (or value) does not match the pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseFailure;

impl ParseFailure {
    /// `INVALID_FORMAT.MISMATCH_INPUT` for `input` of `input_type`
    /// against `format` (all as written).
    #[must_use]
    pub fn mismatch(
        self,
        row: Option<usize>,
        input_type: &str,
        input: &str,
        format: &str,
    ) -> KernelError {
        KernelError::at(
            row,
            InvalidFormatMismatchInput {
                format: format.to_owned(),
                input_type: input_type.to_owned(),
                input: input.to_owned(),
            }
            .into(),
        )
    }
}

/// Adds `c` after swapping it left past trailing spaces.
fn add_char_checking_spaces(out: &mut String, c: char) {
    out.push(c);
    while out.len() >= 2 {
        let chars: Vec<char> = out.chars().collect();
        let n = chars.len();
        if chars[n - 2] == ' ' && chars[n - 1] == c {
            let mut chars = chars;
            chars[n - 1] = ' ';
            chars[n - 2] = c;
            *out = chars.into_iter().collect();
        } else {
            break;
        }
    }
}

/// Appends a space, swapping it left past a trailing sign or `<`.
fn add_space_checking_signs(out: &mut String) {
    out.push(' ');
    while out.len() >= 2 {
        let chars: Vec<char> = out.chars().collect();
        let n = chars.len();
        if matches!(chars[n - 2], '+' | '-' | '<') && chars[n - 1] == ' ' {
            let mut chars = chars;
            chars[n - 1] = chars[n - 2];
            chars[n - 2] = ' ';
            *out = chars.into_iter().collect();
        } else {
            break;
        }
    }
}

/// Turns a trailing lone decimal point into a space.
fn strip_trailing_point(out: &mut String) {
    if let Some(i) = out.find('.') {
        let next = out[i + 1..].chars().next();
        if next.is_none() || next == Some(' ') {
            out.replace_range(i..i + 1, " ");
        }
    }
}

/// Upper-case hex of bytes.
#[must_use]
pub fn hex_bytes(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0xF) as usize] as char);
    }
    out
}

/// A long in minimal upper-case hex (`0` for zero), as
/// `Long.toHexString`.
#[must_use]
pub fn hex_long(v: i64) -> String {
    if v == 0 {
        return "0".to_owned();
    }
    let zeros = v.leading_zeros();
    let len = usize::try_from((u64::BITS - zeros).div_ceil(4)).unwrap_or(0);
    let mut digits = vec![0_u8; len];
    let mut rest = v.cast_unsigned();
    for d in digits.iter_mut().rev() {
        *d = b"0123456789ABCDEF"[(rest & 0xF) as usize];
        rest >>= 4;
    }
    String::from_utf8(digits).unwrap_or_default()
}

/// A long in binary (`0` for zero, 64 digits for negatives), as
/// `Long.toBinaryString`.
#[must_use]
pub fn bin_long(v: i64) -> String {
    format!("{v:b}")
}

/// One hex digit, or `None` (Java's `HexFormat.fromHexDigit` throws on
/// anything else, which `unhex` turns into NULL).
fn hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Hex text to bytes (an odd length pads a leading zero nibble);
/// `None` for any other character.
#[must_use]
pub fn unhex_text(text: &str) -> Option<Vec<u8>> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::with_capacity(bytes.len() / 2 + 1);
    let mut i = 0;
    if bytes.len() % 2 == 1 {
        out.push(hex_digit(bytes[0])?);
        i = 1;
    }
    while i < bytes.len() {
        out.push(hex_digit(bytes[i])? << 4 | hex_digit(bytes[i + 1])?);
        i += 2;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(pattern: &str) -> NumberFormat {
        let f = parse_format(pattern);
        f.validate(pattern).unwrap();
        f
    }

    #[test]
    fn decimal_types_follow_the_digit_slots() {
        for (pattern, precision, scale) in [
            ("000", 3, 0),
            ("000.0", 4, 1),
            ("00,000", 5, 0),
            ("$00.00", 4, 2),
            ("S000", 3, 0),
            ("00,000.9MI", 6, 1),
            ("00,000.9PR", 6, 1),
            ("999", 3, 0),
            ("99G999D9S", 6, 1),
            ("$99.99", 4, 2),
        ] {
            let f = parsed(pattern);
            assert_eq!(f.decimal_type(), (precision, scale), "{pattern}");
        }
    }

    #[test]
    fn golden_inputs_parse() {
        // `tests/spark-golden/v4.2.0/`: `string-functions.sql` (`to_number`).
        for (input, pattern, negate, before, after) in [
            ("454", "000", false, "454", ""),
            ("454.2", "000.0", false, "454", "2"),
            ("12,454", "00,000", false, "12454", ""),
            ("$78.12", "$00.00", false, "78", "12"),
            ("+454", "S000", false, "454", ""),
            ("-454", "S000", true, "454", ""),
            ("12,454.8-", "00,000.9MI", true, "12454", "8"),
            ("00,454.8-", "00,000.9MI", true, "00454", "8"),
            ("<00,454.8>", "00,000.9PR", true, "00454", "8"),
        ] {
            let f = parsed(pattern);
            let got = f.parse_input(input).unwrap();
            assert_eq!(
                (got.negate, got.before.as_str(), got.after.as_str()),
                (negate, before, after),
                "{input} {pattern}"
            );
            let (p, s) = f.decimal_type();
            let unscaled = got.unscaled(s).unwrap();
            let mut digits = format!("{before}{after}");
            digits.push_str(&"0".repeat(s as usize - after.len()));
            let want: i128 = digits.parse().unwrap_or(0) * if negate { -1 } else { 1 };
            assert_eq!(unscaled, want, "{input} {pattern}");
            let _ = p;
        }
    }

    #[test]
    fn golden_values_format() {
        // `charvarchar.sql` (`to_varchar`).
        for (plain, pattern, want) in [
            ("78.12", "$99.99", "$78.12"),
            ("111.11", "99.9", "##.#"),
            ("12454.8", "99,999.9S", "12,454.8+"),
            ("-12454.8", "99G999D9S", "12,454.8-"),
            ("12454", "99G999", "12,454"),
            ("454", "999", "454"),
            ("454.00", "000D00", "454.00"),
        ] {
            let f = parsed(pattern);
            assert_eq!(f.format_decimal(plain).unwrap(), want, "{plain} {pattern}");
        }
    }

    #[test]
    fn invalid_patterns_raise_invalid_format() {
        use krishiv_common::error::ErrorClass;
        for (pattern, class) in [
            ("", ErrorClass::InvalidFormatEmpty),
            ("1", ErrorClass::InvalidFormatWrongNumDigit),
            ("1.5", ErrorClass::InvalidFormatWrongNumDigit),
            ("TRUE", ErrorClass::InvalidFormatWrongNumDigit),
            ("abc", ErrorClass::InvalidFormatWrongNumDigit),
            ("000$", ErrorClass::InvalidFormatCurMustBeforeDigit),
            ("00.00$", ErrorClass::InvalidFormatCurMustBeforeDigit),
            ("$.00$", ErrorClass::InvalidFormatWrongNumToken),
            ("00..00", ErrorClass::InvalidFormatWrongNumToken),
            ("00,00,", ErrorClass::InvalidFormatContThousandsSeps),
            ("00,,000", ErrorClass::InvalidFormatContThousandsSeps),
            (
                "000.0,0",
                ErrorClass::InvalidFormatThousandsSepsMustBeforeDec,
            ),
            ("S00S", ErrorClass::InvalidFormatWrongNumToken),
            ("00MIMI", ErrorClass::InvalidFormatWrongNumToken),
            ("00PRPR", ErrorClass::InvalidFormatWrongNumToken),
            ("$00X00", ErrorClass::InvalidFormatUnexpectedToken),
            ("00$00", ErrorClass::InvalidFormatCurMustBeforeDigit),
        ] {
            let f = parse_format(pattern);
            let err = f.validate(pattern).unwrap_err();
            assert_eq!(err.error_class(), class, "{pattern}");
        }
        // `PR` must wrap: `<` without `>` (and vice versa) fails at parse.
        let f = parsed("00,000.9PR");
        assert!(f.parse_input("<00,454.8").is_err());
        assert!(f.parse_input("00,454.8>").is_err());
        // Case-insensitive: `mi`, `d`, `g`, `s` work.
        let f = parsed("00,000.9mi");
        assert!(f.parse_input("12,454.8-").is_ok());
    }

    #[test]
    fn decimal_format_subset_matches_spark() {
        use super::decimal_format::{Value, format};
        // Classic `format_number(x, d)` shapes.
        assert_eq!(
            format("#,###,###,###,###,###,##0.0000", Value::Float(12332.123456)).unwrap(),
            "12,332.1235"
        );
        assert_eq!(
            format("#,###,###,###,###,###,##0.0", Value::Float(12332.123456)).unwrap(),
            "12,332.1"
        );
        assert_eq!(
            format("#,###,###,###,###,###,##0", Value::Float(12332.123456)).unwrap(),
            "12,332"
        );
        assert_eq!(
            format("##################.###", Value::Float(12332.123456)).unwrap(),
            "12332.123"
        );
        assert_eq!(
            format("abc", Value::Float(12332.123456)).unwrap(),
            "abc12332"
        );
        assert_eq!(format("###.###", Value::Float(66.421)).unwrap(), "66.421");
        // Negative `d` is the caller's NULL.
        assert_eq!(format("#,##0", Value::Int(-5)).unwrap(), "-5");
        assert_eq!(format("#,##0", Value::Int(1234567)).unwrap(), "1,234,567");
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;

    #[test]
    fn unexpected_token_names_digit_groups() {
        // A third digit group has nowhere to go in the token order.
        let f = parse_format("00PR00");
        let err = f.validate("00PR00").unwrap_err();
        assert_eq!(
            err.error_class(),
            krishiv_common::error::ErrorClass::InvalidFormatUnexpectedToken
        );
        assert!(err.to_string().contains("digit sequence"), "{err}");
    }

    #[test]
    fn backslashes_are_invalid_pattern_characters() {
        let f = parse_format("00\\00");
        let err = f.validate("00\\00").unwrap_err();
        assert_eq!(
            err.error_class(),
            krishiv_common::error::ErrorClass::InvalidFormatUnexpectedToken
        );
        assert!(err.to_string().contains("character '\\''"), "{err}");
    }

    #[test]
    fn currency_after_the_point_is_refused() {
        let f = parse_format(".MI$00");
        let err = f.validate(".MI$00").unwrap_err();
        assert_eq!(
            err.error_class(),
            krishiv_common::error::ErrorClass::InvalidFormatCurMustBeforeDec
        );
    }

    #[test]
    fn angle_brackets_must_pair() {
        let f = parse_format("00,000.9PR");
        assert!(f.parse_input("<<00,454.8>").is_err());
        assert!(f.parse_input("<00,454.8").is_err());
        assert!(f.parse_input("00,454.8>").is_err());
        assert!(f.parse_input("<00,454.8>").is_ok());
    }

    #[test]
    fn sign_arms_format() {
        // `MI`: a trailing `-` for negatives, a space otherwise.
        assert_eq!(parsed("99MI").format_decimal("-8").unwrap(), " 8-");
        assert_eq!(parsed("99MI").format_decimal("8").unwrap(), " 8 ");
        // `PR`: angle brackets for negatives, spaces otherwise (the
        // opening bracket leads the pattern).
        assert_eq!(parsed("99PR").format_decimal("-8").unwrap(), " <8>");
        assert_eq!(parsed("99PR").format_decimal("8").unwrap(), " 8  ");
        // Leading zeros fill `0` runs.
        assert_eq!(parsed("000").format_decimal("5").unwrap(), "005");
        // Grouping without digits is spaces, not commas.
        assert_eq!(parsed("99,999").format_decimal("5").unwrap(), "     5");
        // A lone decimal point becomes a space after the digits.
        assert_eq!(parsed("99.MI").format_decimal("5").unwrap(), " 5  ");
    }

    #[test]
    fn zero_with_a_sign_is_zero() {
        assert_eq!(parsed("S99").format_decimal("0").unwrap(), "  +");
    }

    #[test]
    fn format_without_validation_refuses_garbage() {
        // Without validation, an invalid token fails at format time.
        let f = parse_format("00X00");
        assert!(f.format_decimal("1").is_err());
    }

    fn parsed(pattern: &str) -> NumberFormat {
        let f = parse_format(pattern);
        f.validate(pattern).unwrap();
        f
    }
}

#[cfg(test)]
mod reach_tests {
    use super::parse_format;

    fn parsed(pattern: &str) -> super::NumberFormat {
        let f = parse_format(pattern);
        f.validate(pattern).unwrap();
        f
    }

    #[test]
    fn dollar_must_be_present_to_match() {
        assert!(parsed("$00").parse_input("78").is_err());
    }

    #[test]
    fn invalid_tokens_fail_without_validation() {
        // Without validation, an invalid token fails at parse time.
        assert!(parse_format("00X00").parse_input("1").is_err());
    }
}
