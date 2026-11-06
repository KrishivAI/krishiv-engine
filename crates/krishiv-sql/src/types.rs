// SPDX-License-Identifier: Apache-2.0
//! Data types written in SQL, resolved through `krishiv-types`' DDL type
//! parser (`CatalystSqlParser.parseDataType`), so `CAST(x AS t)` and the
//! type string `t` mean the same type by construction.

use std::fmt;

use krishiv_types::{DataType, parse_data_type};

use crate::ast::{DataTypeKind, DataTypeSpec, StructField};

impl fmt::Display for DataTypeSpec {
    /// The type as a canonical DDL string (`decimal(10,2)`,
    /// `array<struct<`a`:int not null comment 'c'>>`), with any
    /// `COLLATE ...` suffix.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            DataTypeKind::Primitive { name, args } if args.is_empty() => f.write_str(name),
            DataTypeKind::Primitive { name, args } => {
                // `time(6) without time zone`: arguments follow the first word.
                let (head, tail) = name.split_once(' ').unwrap_or((name, ""));
                write!(f, "{head}({})", args.join(","))?;
                if tail.is_empty() {
                    Ok(())
                } else {
                    write!(f, " {tail}")
                }
            }
            DataTypeKind::Quoted { name, args } => {
                write!(f, "`{}`", name.replace('`', "``"))?;
                if args.is_empty() {
                    Ok(())
                } else {
                    write!(f, "({})", args.join(","))
                }
            }
            DataTypeKind::Array(None) => f.write_str("array"),
            DataTypeKind::Array(Some(e)) => write!(f, "array<{e}>"),
            DataTypeKind::Map(None) => f.write_str("map"),
            DataTypeKind::Map(Some(kv)) => write!(f, "map<{},{}>", kv.0, kv.1),
            DataTypeKind::Struct(None) => f.write_str("struct"),
            DataTypeKind::Struct(Some(fields)) => {
                f.write_str("struct<")?;
                for (i, field) in fields.iter().enumerate() {
                    let sep = if i == 0 { "" } else { "," };
                    let name = field.name.value.replace('`', "``");
                    write!(f, "{sep}`{name}`:{}", field.data_type)?;
                    if field.not_null {
                        f.write_str(" not null")?;
                    }
                    if let Some(comment) = &field.comment {
                        let escaped = comment.replace('\\', "\\\\").replace('\'', "\\'");
                        write!(f, " comment '{escaped}'")?;
                    }
                }
                f.write_str(">")
            }
        }?;
        if let Some(collation) = &self.collation {
            let name = collation
                .parts
                .iter()
                .map(|p| {
                    if p.quoted {
                        format!("`{}`", p.value.replace('`', "``"))
                    } else {
                        p.value.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(".");
            write!(f, " collate {name}")?;
        }
        Ok(())
    }
}

/// Whether a primitive type name is `CHAR`, `CHARACTER` or `VARCHAR` (a collation may follow the name).
fn is_char_name(name: &str) -> bool {
    matches!(
        name.split(' ').next().unwrap_or(name),
        "char" | "character" | "varchar"
    )
}

impl DataTypeSpec {
    /// Whether `CHAR(n)`, `CHARACTER(n)` or `VARCHAR(n)` occurs at any depth.
    #[must_use]
    pub fn has_char_or_varchar(&self) -> bool {
        match &self.kind {
            DataTypeKind::Primitive { name, .. } => is_char_name(name),
            DataTypeKind::Array(Some(e)) => e.has_char_or_varchar(),
            DataTypeKind::Map(Some(kv)) => kv.0.has_char_or_varchar() || kv.1.has_char_or_varchar(),
            DataTypeKind::Struct(Some(fields)) => {
                fields.iter().any(|f| f.data_type.has_char_or_varchar())
            }
            _ => false,
        }
    }

    /// This type with every `CHAR(n)` / `VARCHAR(n)` replaced by `STRING` (Spark's
    /// `CharVarcharUtils.replaceCharVarcharWithString`); a collation stays.
    #[must_use]
    pub fn without_char_varchar(&self) -> Self {
        let kind = match &self.kind {
            DataTypeKind::Primitive { name, .. } if is_char_name(name) => DataTypeKind::Primitive {
                name: "string".to_owned(),
                args: Vec::new(),
            },
            DataTypeKind::Array(Some(e)) => {
                DataTypeKind::Array(Some(Box::new(e.without_char_varchar())))
            }
            DataTypeKind::Map(Some(kv)) => DataTypeKind::Map(Some(Box::new((
                kv.0.without_char_varchar(),
                kv.1.without_char_varchar(),
            )))),
            DataTypeKind::Struct(Some(fields)) => DataTypeKind::Struct(Some(
                fields
                    .iter()
                    .map(|f| StructField {
                        data_type: f.data_type.without_char_varchar(),
                        ..f.clone()
                    })
                    .collect(),
            )),
            other => other.clone(),
        };
        Self {
            kind,
            collation: self.collation.clone(),
            span: self.span,
        }
    }

    /// The Spark catalog string of a type that holds `CHAR`/`VARCHAR` (`char(3)`, `array<varchar(2)>`;
    /// `CHARACTER` reads `char`), the value of a column's `__CHAR_VARCHAR_TYPE_STRING` metadata; `None` for a
    /// type that holds none.
    #[must_use]
    pub fn char_varchar_type_string(&self) -> Option<String> {
        self.has_char_or_varchar()
            .then(|| self.to_string().replace("character(", "char("))
    }

    /// The type of a table column written as this spec: `CHAR`/`VARCHAR` read as `STRING`, and the catalog
    /// string the length lives in (see [`Self::char_varchar_type_string`]).
    ///
    /// # Errors
    /// As [`Self::to_data_type`].
    pub fn to_column_type(
        &self,
    ) -> Result<(DataType, Option<String>), krishiv_common::error::KrishivError> {
        Ok((
            self.without_char_varchar().to_data_type()?,
            self.char_varchar_type_string(),
        ))
    }

    /// The type this spec denotes.
    ///
    /// # Errors
    /// The errors of [`krishiv_types::parse_data_type`] for the canonical
    /// type string: `UNSUPPORTED_DATATYPE` for an unknown name, the decimal
    /// bound errors, and `KRISHIV_UNSUPPORTED_FEATURE` for Spark types not
    /// supported yet.
    pub fn to_data_type(&self) -> Result<DataType, krishiv_common::error::KrishivError> {
        parse_data_type(&self.to_string())?.refuse_collations()
    }
}

#[cfg(test)]
mod tests {
    use krishiv_types::DataType;

    use crate::{ParserOptions, parse_data_type};

    fn spec(text: &str) -> crate::ast::DataTypeSpec {
        parse_data_type(text, &ParserOptions::default()).unwrap()
    }

    /// `CHAR`/`VARCHAR` read as `STRING` in a table column, and the length lives in the catalog string (Spark's
    /// `__CHAR_VARCHAR_TYPE_STRING`).
    #[test]
    fn char_and_varchar_columns_are_strings_with_a_catalog_string() {
        for (text, string, replaced) in [
            ("char(5)", "char(5)", "string"),
            ("CHARACTER(3)", "char(3)", "string"),
            ("varchar(6)", "varchar(6)", "string"),
            ("array<varchar(2)>", "array<varchar(2)>", "array<string>"),
            ("map<char(1),int>", "map<char(1),int>", "map<string,int>"),
            (
                "struct<a:char(2),b:int>",
                "struct<`a`:char(2),`b`:int>",
                "struct<`a`:string,`b`:int>",
            ),
        ] {
            let s = spec(text);
            assert!(s.has_char_or_varchar(), "{text}");
            assert_eq!(
                s.char_varchar_type_string().as_deref(),
                Some(string),
                "{text}"
            );
            assert_eq!(s.without_char_varchar().to_string(), replaced, "{text}");
            let (ty, meta) = s.to_column_type().unwrap();
            assert_eq!(ty, spec(replaced).to_data_type().unwrap(), "{text}");
            assert_eq!(meta.as_deref(), Some(string), "{text}");
        }
    }

    #[test]
    fn other_types_carry_no_catalog_string() {
        for text in [
            "string",
            "int",
            "array<string>",
            "decimal(10,2)",
            "struct<a:int>",
        ] {
            let s = spec(text);
            assert!(!s.has_char_or_varchar(), "{text}");
            assert_eq!(s.char_varchar_type_string(), None, "{text}");
            assert_eq!(s.without_char_varchar(), s, "{text}");
        }
        assert_eq!(
            spec("string").to_column_type().unwrap(),
            (DataType::String, None)
        );
    }
}
