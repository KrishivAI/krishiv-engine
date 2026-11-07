// SPDX-License-Identifier: Apache-2.0
//! The two name syntaxes of Connect: attribute names (`UnresolvedAttribute`,
//! the target of a star) and multipart identifiers (table names).

use krishiv_common::error::{KrishivError, ParseSyntaxError};
pub(crate) use krishiv_common::names::parse_attribute_name;

/// Splits a table name such as ``db.`my table` `` into its parts, as Spark's
/// `parseMultipartIdentifier` does: parts are unquoted words of letters,
/// digits and `_` that do not read as a numeric literal (`123`, `12L`,
/// `1e3`), or backtick-quoted text with ` `` ` for a backtick, separated by
/// dots; whitespace around the dots is ignored.
///
/// # Errors
/// `PARSE_SYNTAX_ERROR` naming the offending text when the name is empty or
/// does not follow that grammar.
pub(crate) fn parse_multipart_identifier(text: &str) -> Result<Vec<String>, KrishivError> {
    let syntax = |at: &str| {
        KrishivError::from(ParseSyntaxError {
            error: if at.is_empty() {
                "end of input".to_owned()
            } else {
                format!("'{at}'")
            },
            hint: String::new(),
        })
    };
    let mut chars = text.char_indices().peekable();
    let mut parts = Vec::new();
    loop {
        while chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        let Some(&(start, first)) = chars.peek() else {
            return Err(syntax(""));
        };
        let mut part = String::new();
        if first == '`' {
            chars.next();
            loop {
                match chars.next() {
                    None => return Err(syntax("")),
                    Some((_, '`')) => {
                        if chars.next_if(|(_, c)| *c == '`').is_some() {
                            part.push('`');
                        } else {
                            break;
                        }
                    }
                    Some((_, c)) => part.push(c),
                }
            }
        } else {
            while let Some((_, c)) = chars.next_if(|(_, c)| c.is_ascii_alphanumeric() || *c == '_')
            {
                part.push(c);
            }
            if part.is_empty() {
                return Err(syntax(&text[start..start + first.len_utf8()]));
            }
            if is_numeric_literal(&part) {
                return Err(syntax(&part));
            }
        }
        parts.push(part);
        while chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        match chars.next() {
            None => return Ok(parts),
            Some((_, '.')) => {}
            Some((at, c)) => return Err(syntax(&text[at..at + c.len_utf8()])),
        }
    }
}

/// Whether an unquoted word is a numeric literal to Spark's SQL lexer, which
/// prefers a literal to an identifier of the same length: digits, with an
/// optional exponent (`E` and digits; a sign would end the word), and then
/// nothing, or `L`, `S` or `Y` (without an exponent), or `F`, `D` or `BD`,
/// in any case. `123abc` and `1e3x` are longer as identifiers, so they are.
fn is_numeric_literal(word: &str) -> bool {
    let upper = word.to_ascii_uppercase();
    let digits = upper.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return false;
    }
    let mut rest = &upper[digits..];
    let mut exponent = false;
    if let Some(after) = rest.strip_prefix('E') {
        let n = after.bytes().take_while(u8::is_ascii_digit).count();
        if n == 0 {
            return false;
        }
        rest = &after[n..];
        exponent = true;
    }
    match rest {
        "" | "F" | "D" | "BD" => true,
        "L" | "S" | "Y" => !exponent,
        _ => false,
    }
}

#[cfg(test)]
mod tests;
