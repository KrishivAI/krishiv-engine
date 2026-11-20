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

/// Spark's `UserDefinedType` (P3-T079): a value stored as its underlying SQL type, with the names of the classes
/// that turn it into and out of that storage. The engine never runs the hooks itself: a Python client
/// serializes before and deserializes after, so a UDT column is, inside the engine, its `sql_type`.
///
/// Spark compares UDTs by class (`UserDefinedType.equals`), so two UDTs are equal when their classes are. Only the two
/// shapes Spark's JSON has can be built: [`UdtType::jvm`] and [`UdtType::python`].
#[derive(Debug, Clone)]
pub struct UdtType {
    sql_type: DataType,
    python_class: Option<String>,
    jvm_class: Option<String>,
    serialized_python_class: Option<String>,
}

impl UdtType {
    /// A JVM UDT (Spark's `UserDefinedType`): named by its JVM class, with the Python class a PySpark client
    /// pairs with it, if any (`"pyClass":null` in Spark's JSON when there is none).
    pub fn jvm(sql_type: DataType, jvm_class: String, python_class: Option<String>) -> Self {
        Self {
            sql_type,
            python_class,
            jvm_class: Some(jvm_class),
            serialized_python_class: None,
        }
    }

    /// A Python-only UDT (Spark's `PythonUserDefinedType`): its Python class and the pickled class a PySpark
    /// client sends. Spark's JSON has no form for one without both.
    pub fn python(
        sql_type: DataType,
        python_class: String,
        serialized_python_class: String,
    ) -> Self {
        Self {
            sql_type,
            python_class: Some(python_class),
            jvm_class: None,
            serialized_python_class: Some(serialized_python_class),
        }
    }

    /// The type the values are stored as.
    pub fn sql_type(&self) -> &DataType {
        &self.sql_type
    }

    /// The Python class (`module.Class`), if the type has one.
    pub fn python_class(&self) -> Option<&str> {
        self.python_class.as_deref()
    }

    /// The JVM class, if the type has one.
    pub fn jvm_class(&self) -> Option<&str> {
        self.jvm_class.as_deref()
    }

    /// The pickled Python class a PySpark client sent, if any.
    pub fn serialized_python_class(&self) -> Option<&str> {
        self.serialized_python_class.as_deref()
    }

    /// Spark's `UserDefinedType.typeName`: the class's simple name without a trailing `$`, `Type` and `UDT`, lower
    /// case (`ExamplePointUDT` is `examplepoint`). From the JVM class; `pythonuserdefined` for a Python-only one; `udt` for none.
    pub fn type_name(&self) -> String {
        // A Python-only UDT is Spark's `PythonUserDefinedType`, whose type name is `pythonuserdefined` (4.2.0 run).
        if self.jvm_class().is_none() && self.python_class().is_some() {
            return "pythonuserdefined".to_owned();
        }
        let class = self.jvm_class().unwrap_or("");
        let simple = class
            .rsplit(['.', '$'])
            .find(|s| !s.is_empty())
            .unwrap_or("");
        let name = simple.strip_suffix("Type").unwrap_or(simple);
        let name = name.strip_suffix("UDT").unwrap_or(name).to_lowercase();
        if name.is_empty() {
            "udt".to_owned()
        } else {
            name
        }
    }

    /// The class that names this type for equality: the JVM class (Spark's `UserDefinedType.equals` compares
    /// classes), else the Python class of a Python-only type.
    /// A Python-only type is kept apart from a JVM one that happens to share the class name
    /// (`PythonUserDefinedType` is its own Spark class).
    fn identity(&self) -> (bool, Option<&str>) {
        (
            self.jvm_class.is_some(),
            self.jvm_class().or_else(|| self.python_class()),
        )
    }
}

impl PartialEq for UdtType {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity() && self.sql_type == other.sql_type
    }
}

impl Eq for UdtType {}

impl std::hash::Hash for UdtType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity().hash(state);
        self.sql_type.hash(state);
    }
}

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
                Self::Udt(u) => stack.push((u.sql_type(), depth + 1)),
                _ => {}
            }
        }
        deepest
    }

    /// Whether this type or any type nested in it (array elements, map keys
    /// and values, struct fields) satisfies `pred`. Nesting is bounded by
    /// [`Self::MAX_NESTING_DEPTH`], so the recursion is too.
    pub fn any_type(&self, pred: &impl Fn(&Self) -> bool) -> bool {
        pred(self)
            || match self {
                Self::Array(a) => a.element_type().any_type(pred),
                Self::Map(m) => m.key_type().any_type(pred) || m.value_type().any_type(pred),
                Self::Struct(s) => s.fields().iter().any(|f| f.data_type().any_type(pred)),
                Self::Udt(u) => u.sql_type().any_type(pred),
                _ => false,
            }
    }

    /// This type with every array element, map value and struct field
    /// nullable (Spark's `DataType.asNullable`).
    #[must_use]
    pub fn as_nullable(&self) -> Self {
        match self {
            Self::Array(a) => ArrayType::new(a.element_type().as_nullable(), true).into(),
            Self::Map(m) => MapType::new(
                m.key_type().as_nullable(),
                m.value_type().as_nullable(),
                true,
            )
            .into(),
            Self::Struct(s) => StructType::new(s.fields().iter().map(|f| {
                StructField::new(f.name(), f.data_type().as_nullable(), true)
                    .with_metadata(f.metadata().clone())
            }))
            .into(),
            other => other.clone(),
        }
    }

    /// This type with every user-defined type, at any depth, replaced by its storage type: what the engine
    /// computes with (a cast to a UDT is a cast to its storage type).
    #[must_use]
    pub fn without_udts(&self) -> Self {
        match self {
            Self::Udt(u) => u.sql_type().without_udts(),
            Self::Array(a) => {
                ArrayType::new(a.element_type().without_udts(), a.contains_null()).into()
            }
            Self::Map(m) => MapType::new(
                m.key_type().without_udts(),
                m.value_type().without_udts(),
                m.value_contains_null(),
            )
            .into(),
            Self::Struct(s) => StructType::new(s.fields().iter().map(|f| {
                StructField::new(f.name(), f.data_type().without_udts(), f.nullable())
                    .with_metadata(f.metadata().clone())
            }))
            .into(),
            other => other.clone(),
        }
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
