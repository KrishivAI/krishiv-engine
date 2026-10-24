// SPDX-License-Identifier: Apache-2.0
//! Spark's nested types: `ArrayType`, `MapType`, `StructType` and its
//! `StructField`s, with the field `Metadata`.
//!
//! Equality is Spark's: element and value nullability, field names (case
//! sensitive), field nullability and metadata all take part. The looser
//! comparisons Spark uses for coercion are [`DataType::equals_ignore_nullability`],
//! [`DataType::equals_ignore_case_and_nullability`] and
//! [`DataType::equals_structurally`].
//!
//! Nesting is bounded by [`DataType::MAX_NESTING_DEPTH`] wherever a type
//! comes from outside (a type string, Arrow, JSON, Spark Connect), so the
//! recursive walks over a type never exhaust the stack.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::DataType;

/// Spark's `ArrayType(elementType, containsNull)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayType {
    element_type: DataType,
    contains_null: bool,
}

impl ArrayType {
    /// `array<element_type>`; `contains_null` says whether elements may be
    /// null (Spark's default, and a DDL `array<...>`'s, is `true`).
    pub fn new(element_type: DataType, contains_null: bool) -> Self {
        Self {
            element_type,
            contains_null,
        }
    }

    /// The element type.
    pub fn element_type(&self) -> &DataType {
        &self.element_type
    }

    /// Whether an element may be null.
    pub fn contains_null(&self) -> bool {
        self.contains_null
    }
}

/// Spark's `MapType(keyType, valueType, valueContainsNull)`. Keys are never
/// null.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MapType {
    key_type: DataType,
    value_type: DataType,
    value_contains_null: bool,
}

impl MapType {
    /// `map<key_type, value_type>`; `value_contains_null` says whether a
    /// value may be null (Spark's default, and a DDL `map<...>`'s, is `true`).
    pub fn new(key_type: DataType, value_type: DataType, value_contains_null: bool) -> Self {
        Self {
            key_type,
            value_type,
            value_contains_null,
        }
    }

    /// The key type.
    pub fn key_type(&self) -> &DataType {
        &self.key_type
    }

    /// The value type.
    pub fn value_type(&self) -> &DataType {
        &self.value_type
    }

    /// Whether a value may be null.
    pub fn value_contains_null(&self) -> bool {
        self.value_contains_null
    }
}

/// Spark's `Metadata` of a struct field: a JSON object (a DDL `COMMENT 'c'`
/// is `{"comment":"c"}`). Keys are kept sorted; Spark prints a map of up to
/// four keys in insertion order, so the printed order of a field with
/// several keys can differ from Spark's.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Metadata(Map<String, Value>);

impl Metadata {
    /// The metadata key of a field's comment.
    pub const COMMENT: &'static str = "comment";

    /// No entries.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The metadata of the JSON object `map`.
    pub fn from_map(map: Map<String, Value>) -> Self {
        Self(map)
    }

    /// `{"comment": comment}`.
    pub fn with_comment(comment: impl Into<String>) -> Self {
        let mut map = Map::new();
        map.insert(Self::COMMENT.to_owned(), Value::String(comment.into()));
        Self(map)
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The entries.
    pub fn as_map(&self) -> &Map<String, Value> {
        &self.0
    }

    /// The comment, if the field has a string comment.
    pub fn comment(&self) -> Option<&str> {
        self.0.get(Self::COMMENT).and_then(Value::as_str)
    }
}

/// Spark's `StructField(name, dataType, nullable, metadata)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField {
    name: String,
    data_type: DataType,
    nullable: bool,
    metadata: Metadata,
}

impl StructField {
    /// A field without metadata.
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
            metadata: Metadata::empty(),
        }
    }

    /// This field with `metadata`.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// The name, as written (field names are case sensitive in the type).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The type.
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Whether the field may be null.
    pub fn nullable(&self) -> bool {
        self.nullable
    }

    /// The metadata.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }
}

/// Spark's `StructType(fields)`; field names may repeat, as in Spark.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructType {
    fields: Arc<[StructField]>,
}

impl StructType {
    /// A struct of `fields`, in order.
    pub fn new(fields: impl IntoIterator<Item = StructField>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
        }
    }

    /// The fields, in order.
    pub fn fields(&self) -> &[StructField] {
        &self.fields
    }
}

impl From<ArrayType> for DataType {
    fn from(a: ArrayType) -> Self {
        Self::Array(Arc::new(a))
    }
}

impl From<MapType> for DataType {
    fn from(m: MapType) -> Self {
        Self::Map(Arc::new(m))
    }
}

impl From<StructType> for DataType {
    fn from(s: StructType) -> Self {
        Self::Struct(s)
    }
}

/// Java's `String.equalsIgnoreCase`, which Spark compares field names
/// with: as many UTF-16 units, and each character equal, equal upper-cased
/// or equal lower-cased after upper-casing, by Java's simple (one character)
/// case mapping.
pub fn equals_ignore_case(a: &str, b: &str) -> bool {
    // Rust's mappings are the full ones; one that is not a single character
    // (`ß` to `SS`) leaves the character as Java's simple mapping does.
    fn simple(mut m: impl Iterator<Item = char>, c: char) -> char {
        match (m.next(), m.next()) {
            (Some(x), None) => x,
            _ => c,
        }
    }
    let upper = |c: char| simple(c.to_uppercase(), c);
    let lower = |c: char| simple(c.to_lowercase(), c);
    a.encode_utf16().count() == b.encode_utf16().count()
        && a.chars()
            .zip(b.chars())
            .all(|(x, y)| x == y || upper(x) == upper(y) || lower(upper(x)) == lower(upper(y)))
}

/// How two types are compared.
#[derive(Clone, Copy)]
struct Loose {
    /// Field names compared case-insensitively ([`equals_ignore_case`]).
    ignore_case: bool,
    /// Field names not compared at all.
    ignore_names: bool,
}

impl DataType {
    /// The deepest nesting of array, map and struct types: 0 for an atomic
    /// type, 1 for `array<int>`.
    pub fn nesting_depth(&self) -> usize {
        // Iterative: a type built in code may be deeper than the stack allows
        // a recursive walk to go.
        let mut deepest = 0;
        let mut stack = vec![(self, 0_usize)];
        while let Some((t, depth)) = stack.pop() {
            deepest = deepest.max(depth);
            match t {
                Self::Array(a) => stack.push((&a.element_type, depth + 1)),
                Self::Map(m) => {
                    stack.push((&m.key_type, depth + 1));
                    stack.push((&m.value_type, depth + 1));
                }
                Self::Struct(s) => stack.extend(s.fields.iter().map(|f| (&f.data_type, depth + 1))),
                _ => {}
            }
        }
        deepest
    }

    /// Whether this is an array, map or struct type.
    pub fn is_nested(&self) -> bool {
        matches!(self, Self::Array(_) | Self::Map(_) | Self::Struct(_))
    }

    /// Spark's `DataType.equalsIgnoreNullability`: equal but for element,
    /// value and field nullability and field metadata.
    pub fn equals_ignore_nullability(&self, other: &Self) -> bool {
        self.loose_eq(
            other,
            Loose {
                ignore_case: false,
                ignore_names: false,
            },
        )
    }

    /// Spark's `DataType.equalsIgnoreCaseAndNullability`: as
    /// [`Self::equals_ignore_nullability`], with field names compared
    /// case-insensitively.
    pub fn equals_ignore_case_and_nullability(&self, other: &Self) -> bool {
        self.loose_eq(
            other,
            Loose {
                ignore_case: true,
                ignore_names: false,
            },
        )
    }

    /// Spark's `DataType.equalsStructurally(_, _, ignoreNullability)`: the
    /// same shape and atomic types, whatever the field names (and the
    /// nullability, when `ignore_nullability`).
    pub fn equals_structurally(&self, other: &Self, ignore_nullability: bool) -> bool {
        if !ignore_nullability && !self.same_nullability(other) {
            return false;
        }
        self.loose_eq(
            other,
            Loose {
                ignore_case: true,
                ignore_names: true,
            },
        )
    }

    fn loose_eq(&self, other: &Self, how: Loose) -> bool {
        match (self, other) {
            (Self::Array(a), Self::Array(b)) => a.element_type.loose_eq(&b.element_type, how),
            (Self::Map(a), Self::Map(b)) => {
                a.key_type.loose_eq(&b.key_type, how) && a.value_type.loose_eq(&b.value_type, how)
            }
            (Self::Struct(a), Self::Struct(b)) => {
                a.fields.len() == b.fields.len()
                    && a.fields.iter().zip(b.fields.iter()).all(|(x, y)| {
                        let names = how.ignore_names
                            || if how.ignore_case {
                                equals_ignore_case(&x.name, &y.name)
                            } else {
                                x.name == y.name
                            };
                        names && x.data_type.loose_eq(&y.data_type, how)
                    })
            }
            (a, b) => a == b,
        }
    }

    /// Whether every nested nullability flag agrees (shapes assumed equal).
    fn same_nullability(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Array(a), Self::Array(b)) => {
                a.contains_null == b.contains_null
                    && a.element_type.same_nullability(&b.element_type)
            }
            (Self::Map(a), Self::Map(b)) => {
                a.value_contains_null == b.value_contains_null
                    && a.key_type.same_nullability(&b.key_type)
                    && a.value_type.same_nullability(&b.value_type)
            }
            (Self::Struct(a), Self::Struct(b)) => {
                a.fields.len() == b.fields.len()
                    && a.fields.iter().zip(b.fields.iter()).all(|(x, y)| {
                        x.nullable == y.nullable && x.data_type.same_nullability(&y.data_type)
                    })
            }
            _ => true,
        }
    }

    /// Spark's `StructType.unionLikeMerge` of two types that are equal
    /// structurally (ignoring nullability): this type's names and metadata,
    /// with each nullability flag set when either side's is. Types of other
    /// shapes give this type unchanged.
    #[must_use]
    pub fn union_like_merge(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Array(a), Self::Array(b)) => ArrayType::new(
                a.element_type.union_like_merge(&b.element_type),
                a.contains_null || b.contains_null,
            )
            .into(),
            (Self::Map(a), Self::Map(b)) => MapType::new(
                a.key_type.union_like_merge(&b.key_type),
                a.value_type.union_like_merge(&b.value_type),
                a.value_contains_null || b.value_contains_null,
            )
            .into(),
            (Self::Struct(a), Self::Struct(b)) if a.fields.len() == b.fields.len() => {
                StructType::new(
                    a.fields
                        .iter()
                        .zip(b.fields.iter())
                        .map(|(x, y)| StructField {
                            name: x.name.clone(),
                            data_type: x.data_type.union_like_merge(&y.data_type),
                            nullable: x.nullable || y.nullable,
                            metadata: x.metadata.clone(),
                        }),
                )
                .into()
            }
            (a, _) => a.clone(),
        }
    }
}
