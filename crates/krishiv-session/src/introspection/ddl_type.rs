// SPDX-License-Identifier: Apache-2.0
//! A column type as `SHOW CREATE TABLE` prints it: Spark's `DataType.sql` of the stored type, as
//! `StructType.toDDL` writes a table's columns (Spark 4.2.0). Every string is `STRING COLLATE UTF8_BINARY`, a
//! declared `CHAR(n)` / `VARCHAR(n)` prints as declared (also with its collation), struct fields are
//! `name: type[ NOT NULL][ COMMENT '..']` with the name in backquotes unless it is an identifier.

use std::fmt::Write as _;

use krishiv_sql::ast::{DataTypeKind, DataTypeSpec};
use krishiv_types::DataType;

/// `t` as `SHOW CREATE TABLE` prints it. `declared` is the column's `CHAR` / `VARCHAR` type string (the
/// store's `char_varchar`, e.g. `struct<a:varchar(3)>`); a string that does not parse is ignored, so the
/// column prints as its stored type.
pub(super) fn ddl_type(t: &DataType, declared: Option<&str>) -> String {
    let spec = declared
        .and_then(|d| krishiv_sql::parse_data_type(d, &krishiv_sql::ParserOptions::default()).ok());
    let mut out = String::new();
    render(t, spec.as_ref(), &mut out);
    out
}

/// Writes `t` to `out`; `spec` is the declared type at the same place, read only for `CHAR` / `VARCHAR`.
/// Recursion follows the type, which nests at most `DataType::MAX_NESTING_DEPTH` levels.
fn render(t: &DataType, spec: Option<&DataTypeSpec>, out: &mut String) {
    if let Some(DataTypeKind::Primitive { name, args }) = spec.map(|s| &s.kind)
        && matches!(name.as_str(), "char" | "character" | "varchar")
        && let [n] = args.as_slice()
        && matches!(t, DataType::String | DataType::CollatedString(_))
    {
        let kw = if name == "varchar" { "VARCHAR" } else { "CHAR" };
        let _ = write!(out, "{kw}({n}) COLLATE {}", collation(t));
        return;
    }
    match t {
        DataType::String | DataType::CollatedString(_) => {
            let _ = write!(out, "STRING COLLATE {}", collation(t));
        }
        DataType::Array(a) => {
            let inner = match spec.map(|s| &s.kind) {
                Some(DataTypeKind::Array(Some(e))) => Some(&**e),
                _ => None,
            };
            out.push_str("ARRAY<");
            render(a.element_type(), inner, out);
            out.push('>');
        }
        DataType::Map(m) => {
            let (k, v) = match spec.map(|s| &s.kind) {
                Some(DataTypeKind::Map(Some(kv))) => (Some(&kv.0), Some(&kv.1)),
                _ => (None, None),
            };
            out.push_str("MAP<");
            render(m.key_type(), k, out);
            out.push_str(", ");
            render(m.value_type(), v, out);
            out.push('>');
        }
        DataType::Struct(s) => {
            let declared = match spec.map(|s| &s.kind) {
                Some(DataTypeKind::Struct(Some(fields))) if fields.len() == s.fields().len() => {
                    Some(fields)
                }
                _ => None,
            };
            out.push_str("STRUCT<");
            for (i, f) in s.fields().iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&quote_field(f.name()));
                out.push_str(": ");
                render(
                    f.data_type(),
                    declared.and_then(|d| d.get(i)).map(|d| &d.data_type),
                    out,
                );
                if !f.nullable() {
                    out.push_str(" NOT NULL");
                }
                if let Some(c) = f.metadata().comment() {
                    let _ = write!(out, " COMMENT '{}'", c.replace('\'', "\\'"));
                }
            }
            out.push('>');
        }
        other => out.push_str(&other.sql()),
    }
}

/// The collation name of a string type: `UTF8_BINARY` for a plain `STRING`.
fn collation(t: &DataType) -> String {
    match t {
        DataType::CollatedString(c) => c.to_string(),
        _ => "UTF8_BINARY".to_owned(),
    }
}

/// A column or struct field name as `StructField.toDDL` / `StructField.sql` write it (Spark's `QuotingUtils.quoteIfNeeded`): bare when it is
/// an identifier (a letter or `_`, then letters, digits or `_`), else in backquotes with backquotes doubled.
pub(super) fn quote_field(name: &str) -> String {
    let mut chars = name.chars();
    let plain = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if plain {
        name.to_owned()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}

#[cfg(test)]
mod tests {
    use krishiv_types::parse_data_type;

    use super::ddl_type;

    fn shown(t: &str, declared: Option<&str>) -> String {
        ddl_type(&parse_data_type(t).unwrap(), declared)
    }

    #[test]
    fn strings_carry_their_collation_at_every_level() {
        assert_eq!(shown("string", None), "STRING COLLATE UTF8_BINARY");
        assert_eq!(
            shown("map<string,array<string>>", None),
            "MAP<STRING COLLATE UTF8_BINARY, ARRAY<STRING COLLATE UTF8_BINARY>>"
        );
        assert_eq!(shown("decimal(10,2)", None), "DECIMAL(10,2)");
    }

    #[test]
    fn declared_char_varchar_print_in_place() {
        assert_eq!(
            shown(
                "struct<a:string,b:string,c:int>",
                Some("struct<a:varchar(3),b:string,c:int>")
            ),
            "STRUCT<a: VARCHAR(3) COLLATE UTF8_BINARY, b: STRING COLLATE UTF8_BINARY, c: INT>"
        );
        assert_eq!(
            shown("map<string,string>", Some("map<char(2),varchar(5)>")),
            "MAP<CHAR(2) COLLATE UTF8_BINARY, VARCHAR(5) COLLATE UTF8_BINARY>"
        );
        // A declared string that does not parse or does not match is ignored.
        assert_eq!(
            shown("array<string>", Some("array<")),
            "ARRAY<STRING COLLATE UTF8_BINARY>"
        );
        assert_eq!(
            shown("struct<a:string>", Some("struct<a:char(1),b:char(1)>")),
            "STRUCT<a: STRING COLLATE UTF8_BINARY>"
        );
    }

    #[test]
    fn field_names_quote_like_spark() {
        assert_eq!(
            shown("struct<`9a`:int,_x:int,`a``b`:int,`é`:int>", None),
            "STRUCT<`9a`: INT, _x: INT, `a``b`: INT, `é`: INT>"
        );
        assert_eq!(
            shown("struct<a:int not null comment 'it''s'>", None),
            "STRUCT<a: INT NOT NULL COMMENT 'it\\'s'>"
        );
    }
}
