// SPDX-License-Identifier: Apache-2.0
//! Data types written in SQL, resolved through `krishiv-types`' DDL type
//! parser (`CatalystSqlParser.parseDataType`), so `CAST(x AS t)` and the
//! type string `t` mean the same type by construction.

use std::fmt;

use krishiv_types::{DataType, parse_data_type};

use crate::ast::{DataTypeKind, DataTypeSpec};

impl fmt::Display for DataTypeSpec {
    /// The type as a canonical DDL string (`decimal(10,2)`,
    /// `array<struct<a:int>>`).
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
                }
                f.write_str(">")
            }
        }
    }
}

impl DataTypeSpec {
    /// The type this spec denotes.
    ///
    /// # Errors
    /// The errors of [`krishiv_types::parse_data_type`] for the canonical
    /// type string: `UNSUPPORTED_DATATYPE` for an unknown name, the decimal
    /// bound errors, and `KRISHIV_UNSUPPORTED_FEATURE` for Spark types not
    /// supported yet.
    pub fn to_data_type(&self) -> Result<DataType, krishiv_common::error::KrishivError> {
        parse_data_type(&self.to_string())
    }
}
