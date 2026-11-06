// SPDX-License-Identifier: Apache-2.0
//! `schema_of_variant` (Spark 4.2.0 `SchemaOfVariant`): the type of a
//! variant value, merged over the elements of an array with
//! `JsonInferSchema.compatibleType(_, _, VariantType)`, printed as
//! `DataType.sql` with objects as `OBJECT<name: type, ...>`.
//!
//! Integers are `BIGINT`; a decimal is `DECIMAL(p,s)` of its value without
//! trailing zeros (`Decimal(getDecimal)`); a UUID is `UUID`; a variant null
//! is `VOID`. Merging: equal types stay, `VOID` takes the other side,
//! numbers follow Spark's precedence (`BIGINT` < `FLOAT` < `DOUBLE`), a
//! decimal meets `DOUBLE` or `FLOAT` as `DOUBLE` and another decimal at the
//! wider range and scale (`DOUBLE` past 38 digits), `BIGINT` as
//! `DECIMAL(20,0)`, `DATE` and the timestamps widen, objects merge field by
//! field (sorted names), arrays by element, and anything else is `VARIANT`.
//! [`Schema::merge`] is also the merge `schema_of_variant_agg` needs. Known
//! gap: Spark first merges two objects of the same size whose names match
//! case-insensitively as one object (`findTypeForComplex`); here they keep
//! both spellings.

use krishiv_common::error::{KrishivError, MalformedVariant};
use krishiv_types::variant::{VariantKind, VariantRef, digits, strip_trailing_zeros};

use std::cmp::Ordering;

/// The schema of a variant value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Schema {
    /// `VOID`: a variant null (or no value at all).
    Null,
    /// `BOOLEAN`.
    Boolean,
    /// `BIGINT` (every integer width).
    Long,
    /// `FLOAT`.
    Float,
    /// `DOUBLE`.
    Double,
    /// `DECIMAL(precision, scale)`.
    Decimal(u8, u8),
    /// `STRING`.
    String,
    /// `BINARY`.
    Binary,
    /// `DATE`.
    Date,
    /// `TIMESTAMP`.
    Timestamp,
    /// `TIMESTAMP_NTZ`.
    TimestampNtz,
    /// `UUID` (no Spark type; printed by name).
    Uuid,
    /// `VARIANT`: types that do not merge.
    Variant,
    /// `OBJECT<...>`: fields sorted by name.
    Object(Vec<(String, Schema)>),
    /// `ARRAY<...>`.
    Array(Box<Schema>),
}

/// Java's `String.compareTo`.
fn java_order(a: &str, b: &str) -> Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

impl Schema {
    /// The schema of `v`.
    ///
    /// # Errors
    /// `MALFORMED_VARIANT` for bad bytes or object keys out of order, and
    /// `UNKNOWN_PRIMITIVE_TYPE_IN_VARIANT` for the types Spark 4.2.0 does not
    /// read.
    pub fn of(v: &VariantRef<'_>) -> Result<Self, KrishivError> {
        Ok(match v.kind() {
            VariantKind::Object => {
                let n = v.len()?;
                let mut fields: Vec<(String, Schema)> = Vec::with_capacity(n);
                for i in 0..n {
                    let key = v.field_key_at(i)?.to_owned();
                    if let Some((prev, _)) = fields.last()
                        && java_order(prev, &key) != Ordering::Less
                    {
                        return Err(MalformedVariant {}.into());
                    }
                    let schema = Self::of(&v.field_at(i)?)?;
                    fields.push((key, schema));
                }
                Self::Object(fields)
            }
            VariantKind::Array => {
                let mut element = Self::Null;
                for i in 0..v.len()? {
                    element = element.merge(Self::of(&v.get_index(i)?)?);
                }
                Self::Array(Box::new(element))
            }
            VariantKind::Null => Self::Null,
            VariantKind::Bool => Self::Boolean,
            VariantKind::Int8 | VariantKind::Int16 | VariantKind::Int32 | VariantKind::Int64 => {
                Self::Long
            }
            VariantKind::String => Self::String,
            VariantKind::Double => Self::Double,
            VariantKind::Float => Self::Float,
            VariantKind::Decimal { .. } => {
                let (u, s) = v.as_decimal()?;
                let (u, s) = strip_trailing_zeros(u, s);
                Self::Decimal(digits(u).max(s), s)
            }
            VariantKind::Date => Self::Date,
            VariantKind::Timestamp => Self::Timestamp,
            VariantKind::TimestampNtz => Self::TimestampNtz,
            VariantKind::Binary => Self::Binary,
            VariantKind::Uuid => Self::Uuid,
            _ => return Err(super::unknown_type(v)),
        })
    }

    /// `compatibleType(self, other, VariantType)`.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        use Schema as S;
        match (self, other) {
            (a, b) if a == b => a,
            (S::Null, t) | (t, S::Null) => t,
            // `findTightestCommonType`: a decimal that holds every `BIGINT`.
            (S::Long, S::Decimal(p, s)) | (S::Decimal(p, s), S::Long)
                if p.saturating_sub(s) >= 20 =>
            {
                S::Decimal(p, s)
            }
            (S::Long, S::Float) | (S::Float, S::Long) => S::Float,
            (S::Long | S::Float, S::Double) | (S::Double, S::Long | S::Float) => S::Double,
            (S::Date | S::TimestampNtz, S::Timestamp)
            | (S::Timestamp, S::Date | S::TimestampNtz) => S::Timestamp,
            (S::Date, S::TimestampNtz) | (S::TimestampNtz, S::Date) => S::TimestampNtz,
            // `compatibleType`.
            (S::Double | S::Float, S::Decimal(..)) | (S::Decimal(..), S::Double | S::Float) => {
                S::Double
            }
            (S::Decimal(p1, s1), S::Decimal(p2, s2)) => {
                let scale = s1.max(s2);
                let range = p1.saturating_sub(s1).max(p2.saturating_sub(s2));
                match range.checked_add(scale).filter(|p| *p <= 38) {
                    Some(p) => S::Decimal(p, scale),
                    None => S::Double,
                }
            }
            (S::Long, d @ S::Decimal(..)) | (d @ S::Decimal(..), S::Long) => {
                S::Decimal(20, 0).merge(d)
            }
            (S::Object(a), S::Object(b)) => S::Object(merge_fields(a, b)),
            (S::Array(a), S::Array(b)) => S::Array(Box::new(a.merge(*b))),
            _ => S::Variant,
        }
    }

    /// `printSchema`: `DataType.sql`, objects as `OBJECT<name: type, ...>`.
    #[must_use]
    pub fn sql(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        let name = match self {
            Self::Object(fields) => {
                out.push_str("OBJECT<");
                for (i, (k, t)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&quote_if_needed(k));
                    out.push_str(": ");
                    t.write(out);
                }
                out.push('>');
                return;
            }
            Self::Array(e) => {
                out.push_str("ARRAY<");
                e.write(out);
                out.push('>');
                return;
            }
            Self::Decimal(p, s) => {
                out.push_str(&format!("DECIMAL({p},{s})"));
                return;
            }
            Self::Null => "VOID",
            Self::Boolean => "BOOLEAN",
            Self::Long => "BIGINT",
            Self::Float => "FLOAT",
            Self::Double => "DOUBLE",
            Self::String => "STRING",
            Self::Binary => "BINARY",
            Self::Date => "DATE",
            Self::Timestamp => "TIMESTAMP",
            Self::TimestampNtz => "TIMESTAMP_NTZ",
            Self::Uuid => "UUID",
            Self::Variant => "VARIANT",
        };
        out.push_str(name);
    }
}

/// The merge of two sorted field lists: shared names merge their types.
fn merge_fields(a: Vec<(String, Schema)>, b: Vec<(String, Schema)>) -> Vec<(String, Schema)> {
    let mut out = Vec::with_capacity(a.len().max(b.len()));
    let mut a = a.into_iter().peekable();
    let mut b = b.into_iter().peekable();
    loop {
        let order = match (a.peek(), b.peek()) {
            (Some(x), Some(y)) => java_order(&x.0, &y.0),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => return out,
        };
        let next = match order {
            Ordering::Equal => a
                .next()
                .zip(b.next())
                .map(|((k, x), (_, y))| (k, x.merge(y))),
            Ordering::Less => a.next(),
            Ordering::Greater => b.next(),
        };
        out.extend(next);
    }
}

/// `QuotingUtils.quoteIfNeeded`: backquoted unless letters, digits and `_`
/// (and not only digits).
fn quote_if_needed(name: &str) -> String {
    let bare = !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.chars().all(|c| c.is_ascii_digit());
    if bare {
        name.to_owned()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}
