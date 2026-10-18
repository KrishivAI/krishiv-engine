// SPDX-License-Identifier: Apache-2.0
//! Typed values read from the Spark-generated fixtures (`tests/fixtures/spark-4.2.0`): a
//! parser for Spark's `catalogString` types ([`SparkType`]) and the decoder from the fixtures'
//! JSON encoding to [`Cell`]s.
//!
//! The encoding is fixed by `tests/scripts/goldentables/dump_fixtures.scala`: integers, dates
//! (days), timestamps (microseconds) and interval counts are JSON numbers; floats and doubles
//! are Java `toString` text; decimals are the unscaled integer as text; binary is lower-case hex;
//! arrays are lists, maps lists of `[key, value]` pairs in storage order, structs lists in field
//! order, calendar intervals `{months, days, microseconds}` objects. `null` is NULL at any type.

use serde_json::Value;

use super::cell::Cell;

/// A fixture value or type that cannot be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ValueError(pub String);

fn fail<T>(msg: impl Into<String>) -> Result<T, ValueError> {
    Err(ValueError(msg.into()))
}

/// A Spark data type as `DataType.catalogString` writes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparkType {
    /// `void`: only NULL.
    Void,
    /// `boolean`.
    Boolean,
    /// `tinyint`, `smallint`, `int` or `bigint`.
    Integral,
    /// `float`.
    Float,
    /// `double`.
    Double,
    /// `decimal(p,s)`, with its scale.
    Decimal(u32),
    /// `string`, `char(n)` or `varchar(n)`.
    Text,
    /// `binary`.
    Binary,
    /// `date`.
    Date,
    /// `timestamp`.
    Timestamp,
    /// `timestamp_ntz`.
    TimestampNtz,
    /// `interval year`, `interval month` or `interval year to month`.
    YearMonth,
    /// Any `interval day ... second` type.
    DayTime,
    /// `interval`: months, days and microseconds.
    Calendar,
    /// `array<T>`.
    Array(Box<SparkType>),
    /// `map<K,V>`.
    Map(Box<SparkType>, Box<SparkType>),
    /// `struct<name:T,...>`.
    Struct(Vec<(String, SparkType)>),
}

/// Splits `s` at the commas outside `<>` and `()`.
fn split_top(s: &str) -> Vec<&str> {
    let (mut depth, mut start, mut parts) = (0_i32, 0, Vec::new());
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

impl SparkType {
    /// Parse a `catalogString`.
    pub fn parse(s: &str) -> Result<Self, ValueError> {
        let t = s.trim();
        let lower = t.to_ascii_lowercase();
        let inner = |prefix: &str, suffix: char| {
            lower
                .strip_prefix(prefix)
                .and_then(|r| r.strip_suffix(suffix))
                .map(|_| &t[prefix.len()..t.len() - 1])
        };
        if let Some(body) = inner("array<", '>') {
            return Ok(Self::Array(Box::new(Self::parse(body)?)));
        }
        if let Some(body) = inner("map<", '>') {
            return match split_top(body)[..] {
                [k, v] => Ok(Self::Map(
                    Box::new(Self::parse(k)?),
                    Box::new(Self::parse(v)?),
                )),
                _ => fail(format!("bad map type `{t}`")),
            };
        }
        if let Some(body) = inner("struct<", '>') {
            let field = |f: &str| {
                f.split_once(':')
                    .map(|(n, ty)| Self::parse(ty).map(|ty| (n.to_owned(), ty)))
                    .unwrap_or_else(|| fail(format!("bad struct field `{f}`")))
            };
            let fields = if body.is_empty() {
                Vec::new()
            } else {
                split_top(body)
            };
            return fields
                .into_iter()
                .map(field)
                .collect::<Result<_, _>>()
                .map(Self::Struct);
        }
        if let Some(args) = inner("decimal(", ')') {
            return match args.split_once(',').map(|(_, s)| s.trim().parse()) {
                Some(Ok(scale)) => Ok(Self::Decimal(scale)),
                _ => fail(format!("bad decimal type `{t}`")),
            };
        }
        if lower.starts_with("char(") || lower.starts_with("varchar(") {
            return Ok(Self::Text);
        }
        if let Some(fields) = lower.strip_prefix("interval") {
            let fields = fields.trim();
            return Ok(if fields.is_empty() {
                Self::Calendar
            } else if matches!(fields, "year" | "month" | "year to month") {
                Self::YearMonth
            } else {
                Self::DayTime
            });
        }
        match lower.as_str() {
            "void" => Ok(Self::Void),
            "boolean" => Ok(Self::Boolean),
            "tinyint" | "smallint" | "int" | "bigint" => Ok(Self::Integral),
            "float" => Ok(Self::Float),
            "double" => Ok(Self::Double),
            "string" => Ok(Self::Text),
            "binary" => Ok(Self::Binary),
            "date" => Ok(Self::Date),
            "timestamp" => Ok(Self::Timestamp),
            "timestamp_ntz" => Ok(Self::TimestampNtz),
            _ => fail(format!("unknown type `{t}`")),
        }
    }
}

fn mismatch<T>(v: &Value, ty: &SparkType) -> Result<T, ValueError> {
    fail(format!("{v} is not a {ty:?}"))
}

impl Cell {
    /// Decode the fixture encoding of a value of type `ty`.
    pub fn from_json(v: &Value, ty: &SparkType) -> Result<Self, ValueError> {
        if v.is_null() {
            return Ok(Self::Null);
        }
        let num = || {
            v.as_i64()
                .ok_or_else(|| ValueError(format!("{v} is not an integer")))
        };
        let text = || {
            v.as_str()
                .ok_or_else(|| ValueError(format!("{v} is not text")))
        };
        let list = || {
            v.as_array()
                .ok_or_else(|| ValueError(format!("{v} is not a list")))
        };
        let narrow = |n: i64| i32::try_from(n).map_err(|_| ValueError(format!("{n} out of range")));
        match ty {
            SparkType::Void => mismatch(v, ty),
            SparkType::Boolean => v
                .as_bool()
                .map(Self::Bool)
                .map_or_else(|| mismatch(v, ty), Ok),
            SparkType::Integral => num().map(Self::Int),
            SparkType::Float => text()?
                .parse()
                .map(Self::Float)
                .or_else(|_| mismatch(v, ty)),
            SparkType::Double => text()?
                .parse()
                .map(Self::Double)
                .or_else(|_| mismatch(v, ty)),
            SparkType::Decimal(scale) => text()?
                .parse()
                .map(|unscaled| Self::Decimal(unscaled, *scale))
                .or_else(|_| mismatch(v, ty)),
            SparkType::Text => text().map(|s| Self::Str(s.to_owned())),
            SparkType::Binary => hex(text()?).map(Self::Binary),
            SparkType::Date => narrow(num()?).map(Self::Date),
            SparkType::Timestamp => num().map(Self::Timestamp),
            SparkType::TimestampNtz => num().map(Self::TimestampNtz),
            SparkType::YearMonth => narrow(num()?).map(Self::YearMonth),
            SparkType::DayTime => num().map(Self::DayTime),
            SparkType::Calendar => {
                let field = |k: &str| v.get(k).and_then(Value::as_i64);
                match (field("months"), field("days"), field("microseconds")) {
                    (Some(m), Some(d), Some(micros)) => Ok(Self::Calendar {
                        months: narrow(m)?,
                        days: narrow(d)?,
                        micros,
                    }),
                    _ => mismatch(v, ty),
                }
            }
            SparkType::Array(elem) => list()?
                .iter()
                .map(|e| Self::from_json(e, elem))
                .collect::<Result<_, _>>()
                .map(Self::Array),
            SparkType::Map(kt, vt) => list()?
                .iter()
                .map(|pair| match pair.as_array().map(Vec::as_slice) {
                    Some([k, v]) => Ok((Self::from_json(k, kt)?, Self::from_json(v, vt)?)),
                    _ => mismatch(pair, ty),
                })
                .collect::<Result<_, _>>()
                .map(Self::Map),
            SparkType::Struct(fields) => {
                let values = list()?;
                if values.len() != fields.len() {
                    return mismatch(v, ty);
                }
                fields
                    .iter()
                    .zip(values)
                    .map(|((name, ft), fv)| Ok((name.clone(), Self::from_json(fv, ft)?)))
                    .collect::<Result<_, _>>()
                    .map(Self::Struct)
            }
        }
    }
}

fn hex(s: &str) -> Result<Vec<u8>, ValueError> {
    let pair = |i: usize| s.get(i..i + 2).and_then(|p| u8::from_str_radix(p, 16).ok());
    if !s.len().is_multiple_of(2) {
        return fail(format!("odd-length hex `{s}`"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| pair(i).ok_or_else(|| ValueError(format!("bad hex `{s}`"))))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;

    fn ty(s: &str) -> SparkType {
        SparkType::parse(s).unwrap_or(SparkType::Void)
    }

    #[test]
    fn parses_catalog_strings() {
        assert_eq!(ty("int"), SparkType::Integral);
        assert_eq!(ty(" DECIMAL(10, 2) "), SparkType::Decimal(2));
        assert_eq!(ty("varchar(5)"), SparkType::Text);
        assert_eq!(ty("char(5)"), SparkType::Text);
        assert_eq!(ty("interval day to second"), SparkType::DayTime);
        assert_eq!(ty("interval hour"), SparkType::DayTime);
        assert_eq!(ty("interval year to month"), SparkType::YearMonth);
        assert_eq!(ty("interval month"), SparkType::YearMonth);
        assert_eq!(ty("interval"), SparkType::Calendar);
        assert_eq!(
            ty("array<array<int>>"),
            SparkType::Array(Box::new(SparkType::Array(Box::new(SparkType::Integral))))
        );
        assert_eq!(
            ty("map<int,struct<a:decimal(3,1),b:array<string>>>"),
            SparkType::Map(
                Box::new(SparkType::Integral),
                Box::new(SparkType::Struct(vec![
                    ("a".into(), SparkType::Decimal(1)),
                    ("b".into(), SparkType::Array(Box::new(SparkType::Text))),
                ]))
            )
        );
        assert_eq!(ty("struct<>"), SparkType::Struct(Vec::new()));
        for simple in [
            ("void", SparkType::Void),
            ("boolean", SparkType::Boolean),
            ("float", SparkType::Float),
            ("double", SparkType::Double),
            ("binary", SparkType::Binary),
            ("date", SparkType::Date),
            ("timestamp", SparkType::Timestamp),
            ("timestamp_ntz", SparkType::TimestampNtz),
        ] {
            assert_eq!(ty(simple.0), simple.1);
        }
    }

    #[test]
    fn rejects_malformed_types() {
        for bad in [
            "blob",
            "map<int>",
            "struct<a>",
            "decimal(5)",
            "decimal(a,b)",
            "array<foo>",
        ] {
            assert!(SparkType::parse(bad).is_err(), "{bad}");
        }
    }

    fn cell(v: &Value, t: &str) -> Result<Cell, ValueError> {
        Cell::from_json(v, &SparkType::parse(t)?)
    }

    #[test]
    fn decodes_scalars() -> R {
        assert_eq!(cell(&json!(null), "int")?, Cell::Null);
        assert_eq!(cell(&json!(true), "boolean")?, Cell::Bool(true));
        assert_eq!(cell(&json!(-7), "tinyint")?, Cell::Int(-7));
        assert_eq!(cell(&json!("1.4E-45"), "float")?, Cell::Float(1.4e-45));
        assert!(matches!(cell(&json!("NaN"), "double")?, Cell::Double(d) if d.is_nan()));
        assert_eq!(
            cell(&json!("-12345"), "decimal(10,3)")?,
            Cell::Decimal(-12345, 3)
        );
        assert_eq!(cell(&json!("a"), "varchar(3)")?, Cell::Str("a".into()));
        assert_eq!(cell(&json!("00ff"), "binary")?, Cell::Binary(vec![0, 255]));
        assert_eq!(cell(&json!(19000), "date")?, Cell::Date(19000));
        assert_eq!(cell(&json!(-5), "timestamp")?, Cell::Timestamp(-5));
        assert_eq!(cell(&json!(5), "timestamp_ntz")?, Cell::TimestampNtz(5));
        assert_eq!(
            cell(&json!(14), "interval year to month")?,
            Cell::YearMonth(14)
        );
        assert_eq!(cell(&json!(9), "interval day to second")?, Cell::DayTime(9));
        let cal = json!({"months": 1, "days": 2, "microseconds": 3});
        assert_eq!(
            cell(&cal, "interval")?,
            Cell::Calendar {
                months: 1,
                days: 2,
                micros: 3
            }
        );
        Ok(())
    }

    #[test]
    fn decodes_nested() -> R {
        let arr = cell(&json!([[1, null], []]), "array<array<int>>")?;
        assert_eq!(
            arr,
            Cell::Array(vec![
                Cell::Array(vec![Cell::Int(1), Cell::Null]),
                Cell::Array(vec![])
            ])
        );
        let map = cell(&json!([[1, "a"], [2, null]]), "map<int,string>")?;
        assert_eq!(
            map,
            Cell::Map(vec![
                (Cell::Int(1), Cell::Str("a".into())),
                (Cell::Int(2), Cell::Null)
            ])
        );
        let st = cell(&json!([1, "x"]), "struct<a:int,b:string>")?;
        assert_eq!(
            st,
            Cell::Struct(vec![
                ("a".into(), Cell::Int(1)),
                ("b".into(), Cell::Str("x".into()))
            ])
        );
        Ok(())
    }

    #[test]
    fn rejects_values_of_the_wrong_shape() {
        let bad = [
            (json!("x"), "int"),
            (json!(1), "string"),
            (json!(1), "boolean"),
            (json!(1), "void"),
            (json!(1), "float"),
            (json!("x"), "float"),
            (json!("x"), "double"),
            (json!("1.5"), "decimal(3,1)"),
            (json!(5_000_000_000_i64), "date"),
            (json!(5_000_000_000_i64), "interval year to month"),
            (json!("0"), "binary"),
            (json!("zz"), "binary"),
            (json!({"months": 1}), "interval"),
            (
                json!({"months": 5_000_000_000_i64, "days": 0, "microseconds": 0}),
                "interval",
            ),
            (json!(1), "array<int>"),
            (json!([1]), "map<int,int>"),
            (json!([[1]]), "map<int,int>"),
            (json!([1, 2]), "struct<a:int>"),
        ];
        for (v, t) in bad {
            assert!(cell(&v, t).is_err(), "{v} as {t}");
        }
    }
}
