// SPDX-License-Identifier: Apache-2.0
//! Spark's attribute-name syntax (`UnresolvedAttribute.parseAttributeName`) and the name quoting of its
//! `toSQLId`, shared by the Connect translation and the checks that report a repeated column.

use crate::error::{ColumnAlreadyExists, InvalidAttributeNameSyntax, KrishivError};

/// Splits an attribute name at its dots, as Spark's
/// `AttributeNameParser.parseAttributeName` does. A name part may be quoted
/// with backticks, inside which a dot is part of the name and ` `` ` is one
/// backtick: ``a.`b.c` `` is `["a", "b.c"]`. A quoted part must be a whole
/// part, and a backtick may only appear inside one.
///
/// # Errors
/// `INVALID_ATTRIBUTE_NAME_SYNTAX` for unpaired backticks, a quoted part
/// followed or preceded by other text, an empty part between dots, or a
/// leading or trailing dot.
pub fn parse_attribute_name(name: &str) -> Result<Vec<String>, KrishivError> {
    let bad = || {
        KrishivError::from(InvalidAttributeNameSyntax {
            name: name.to_owned(),
        })
    };
    let chars: Vec<char> = name.chars().collect();
    let mut parts = Vec::new();
    let mut part = String::new();
    let mut in_backtick = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if in_backtick {
            if c == '`' {
                if next == Some('`') {
                    part.push('`');
                    i += 1;
                } else {
                    in_backtick = false;
                    if next.is_some_and(|n| n != '.') {
                        return Err(bad());
                    }
                }
            } else {
                part.push(c);
            }
        } else if c == '`' {
            if !part.is_empty() {
                return Err(bad());
            }
            in_backtick = true;
        } else if c == '.' {
            let after_dot = i > 0 && chars[i - 1] == '.';
            if i == 0 || after_dot || next.is_none() {
                return Err(bad());
            }
            parts.push(std::mem::take(&mut part));
        } else {
            part.push(c);
        }
        i += 1;
    }
    if in_backtick {
        return Err(bad());
    }
    parts.push(part);
    Ok(parts)
}

/// `parts` quoted as Spark's `toSQLId(Seq[String])` does: each part in backticks, a backtick doubled,
/// joined with dots (`` `a`.`b` ``).
/// A leading `__auto_generated_subquery_name` part (the alias Spark gives an unnamed subquery) is left out
/// when more parts follow, as Spark does.
#[must_use]
pub fn quote_parts(parts: &[String]) -> String {
    let parts = match parts {
        [first, rest @ ..] if first == AUTO_GENERATED_SUBQUERY_NAME && !rest.is_empty() => rest,
        all => all,
    };
    let quote = |p: &String| format!("`{}`", p.replace('`', "``"));
    parts.iter().map(quote).collect::<Vec<_>>().join(".")
}

/// The alias Spark gives a subquery written without one (`SubqueryAlias.generateSubqueryName`).
const AUTO_GENERATED_SUBQUERY_NAME: &str = "__auto_generated_subquery_name";

/// `name` quoted as Spark's `toSQLId(String)` does: split as an attribute name, then each part quoted
/// (`a.b` is `` `a`.`b` ``, `a b` is `` `a b` ``).
///
/// # Errors
/// `INVALID_ATTRIBUTE_NAME_SYNTAX` when `name` is not an attribute name (see [`parse_attribute_name`]).
pub fn to_sql_id(name: &str) -> Result<String, KrishivError> {
    parse_attribute_name(name).map(|parts| quote_parts(&parts))
}

/// The error Spark 4.2.0 raises for a column name that repeats (`SchemaUtils.checkColumnNameDuplication`):
/// `COLUMN_ALREADY_EXISTS` with the name quoted by [`to_sql_id`], or, because Spark quotes it that way while
/// building the message, `INVALID_ATTRIBUTE_NAME_SYNTAX` when the name is not an attribute name
/// (`` a`b ``, `.a`, `a..b`). `name` is the repeated name as Spark shows it (lower case unless
/// `spark.sql.caseSensitive`).
#[must_use]
pub fn column_already_exists(name: &str) -> KrishivError {
    match to_sql_id(name) {
        Ok(column_name) => ColumnAlreadyExists { column_name }.into(),
        Err(e) => e,
    }
}

/// Spark's `SchemaUtils.checkColumnNameDuplication`: when a name of `names` repeats (compared in lower case
/// unless `case_sensitive`), the [`column_already_exists`] error for the smallest repeated name in Scala's
/// string order (UTF-16 code units), whatever order the names come in.
///
/// # Errors
/// `COLUMN_ALREADY_EXISTS`, or `INVALID_ATTRIBUTE_NAME_SYNTAX` when that name is not an attribute name.
pub fn check_column_name_duplication<'a>(
    names: impl IntoIterator<Item = &'a str>,
    case_sensitive: bool,
) -> Result<(), KrishivError> {
    let mut folded: Vec<String> = names
        .into_iter()
        .map(|n| {
            if case_sensitive {
                n.to_owned()
            } else {
                n.to_lowercase()
            }
        })
        .collect();
    folded.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    match folded.windows(2).find(|w| w[0] == w[1]) {
        Some(w) => Err(column_already_exists(&w[0])),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
