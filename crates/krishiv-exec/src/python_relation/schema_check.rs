// SPDX-License-Identifier: Apache-2.0
//! Spark's result-schema check of the map kinds (`mapInPandas` 205, `mapInArrow` 207) under
//! `spark.sql.execution.arrow.pyspark.validateSchema.enabled` (Spark 4.1+, default `false`;
//! `SQLConf.scala` L4765-4772 at v4.2.0).
//!
//! `MapInBatchEvaluatorFactory.scala` L81-91 (v4.2.0): each result batch's struct type, as the JVM reads it
//! from the Arrow schema, must equal the declared output type up to *compatible nullability*
//! (`DataType.equalsIgnoreCompatibleNullability`, `DataType.scala` L451-474: names equal, a non-nullable
//! declared field takes only a non-nullable result field, recursively through arrays, maps and structs);
//! otherwise `ARROW_TYPE_MISMATCH` with the eval type's name (`PythonEvalType.toString`) and both types'
//! Scala `toString`. With the setting off Spark has no check and Krishiv refuses a result unlike the
//! declared schema with `PYTHON_EXCEPTION` instead, a recorded divergence (the crate's `CLAUDE.md`).

use arrow_schema::Schema;
use krishiv_common::error::{ArrowTypeMismatch, KrishivError};
use krishiv_types::{DataType, StructField, StructType};

/// The settings key (canonical, and Spark's) that turns the check on.
pub const VALIDATE_SCHEMA: [&str; 2] = [
    "krishiv.sql.execution.arrow.pyspark.validateSchema.enabled",
    "spark.sql.execution.arrow.pyspark.validateSchema.enabled",
];

/// The check of one map call site.
#[derive(Debug, Clone)]
pub struct SchemaCheck {
    /// `PythonEvalType.toString` of the call's eval type.
    operation: &'static str,
    /// The declared output, as a struct.
    expected: DataType,
}

impl SchemaCheck {
    /// The check of a call of `eval_type` declaring `expected`, when `eval_type` is a map kind and `enabled`.
    pub fn new(eval_type: i32, expected: StructType, enabled: bool) -> Option<Self> {
        let operation = match eval_type {
            205 => "SQL_MAP_PANDAS_ITER_UDF",
            207 => "SQL_MAP_ARROW_ITER_UDF",
            _ => return None,
        };
        enabled.then_some(SchemaCheck {
            operation,
            expected: DataType::Struct(expected),
        })
    }

    /// `ARROW_TYPE_MISMATCH` unless a batch of `actual` (the result stream's schema) is what Spark accepts.
    pub fn check(&self, actual: &Schema) -> Result<(), KrishivError> {
        let fields: Option<Vec<StructField>> = actual
            .fields()
            .iter()
            .map(|f| {
                DataType::from_arrow_field(f)
                    .ok()
                    .map(|t| StructField::new(f.name().clone(), t, f.is_nullable()))
            })
            .collect();
        let actual_text = match &fields {
            Some(fields) => {
                let actual = DataType::Struct(StructType::new(fields.clone()));
                if compatible(&actual, &self.expected) {
                    return Ok(());
                }
                scala_string(&actual)
            }
            // A column with no Spark type: never compatible.
            None => format!("{actual}"),
        };
        Err(ArrowTypeMismatch {
            operation: self.operation.to_owned(),
            output_types: scala_string(&self.expected),
            actual_data_types: actual_text,
        }
        .into())
    }
}

/// Spark's `DataType.equalsIgnoreCompatibleNullability(from, to)` (names compared).
fn compatible(from: &DataType, to: &DataType) -> bool {
    match (from, to) {
        (DataType::Array(f), DataType::Array(t)) => {
            (t.contains_null() || !f.contains_null())
                && compatible(f.element_type(), t.element_type())
        }
        (DataType::Map(f), DataType::Map(t)) => {
            (t.value_contains_null() || !f.value_contains_null())
                && compatible(f.key_type(), t.key_type())
                && compatible(f.value_type(), t.value_type())
        }
        (DataType::Struct(f), DataType::Struct(t)) => {
            f.fields().len() == t.fields().len()
                && f.fields().iter().zip(t.fields()).all(|(f, t)| {
                    f.name() == t.name()
                        && (t.nullable() || !f.nullable())
                        && compatible(f.data_type(), t.data_type())
                })
        }
        (f, t) => f == t,
    }
}

/// A type as Spark's Scala `toString` prints it (`LongType`, `DecimalType(10,2)`,
/// `StructType(StructField(a,LongType,true))`): the case objects' names and the case classes' fields, as the
/// `org.apache.spark.sql.types` sources at v4.2.0 define them. A type Spark's Arrow reader has no form for
/// prints as its SQL name.
fn scala_string(t: &DataType) -> String {
    match t {
        DataType::Null => "NullType".to_owned(),
        DataType::Boolean => "BooleanType".to_owned(),
        DataType::Byte => "ByteType".to_owned(),
        DataType::Short => "ShortType".to_owned(),
        DataType::Integer => "IntegerType".to_owned(),
        DataType::Long => "LongType".to_owned(),
        DataType::Float => "FloatType".to_owned(),
        DataType::Double => "DoubleType".to_owned(),
        DataType::String => "StringType".to_owned(),
        DataType::CollatedString(c) if c.is_utf8_binary() => "StringType".to_owned(),
        DataType::CollatedString(c) => format!("StringType({c})"),
        DataType::Binary => "BinaryType".to_owned(),
        DataType::Decimal(d) => format!("DecimalType({},{})", d.precision(), d.scale()),
        DataType::Date => "DateType".to_owned(),
        DataType::Timestamp => "TimestampType".to_owned(),
        DataType::TimestampNtz => "TimestampNTZType".to_owned(),
        DataType::CalendarInterval => "CalendarIntervalType".to_owned(),
        DataType::Variant => "VariantType".to_owned(),
        DataType::Array(a) => format!(
            "ArrayType({},{})",
            scala_string(a.element_type()),
            a.contains_null()
        ),
        DataType::Map(m) => format!(
            "MapType({},{},{})",
            scala_string(m.key_type()),
            scala_string(m.value_type()),
            m.value_contains_null()
        ),
        DataType::Struct(s) => {
            let fields: Vec<String> = s
                .fields()
                .iter()
                .map(|f| {
                    format!(
                        "StructField({},{},{})",
                        f.name(),
                        scala_string(f.data_type()),
                        f.nullable()
                    )
                })
                .collect();
            format!("StructType({})", fields.join(","))
        }
        other => other.sql(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType as ArrowType, Field};
    use krishiv_common::error::ErrorClass;
    use krishiv_types::ArrayType;

    use super::*;

    fn declared(nullable: bool) -> StructType {
        StructType::new([
            StructField::new("id", DataType::Long, nullable),
            StructField::new("s", DataType::String, true),
        ])
    }

    fn result(fields: Vec<Field>) -> Schema {
        Schema::new(fields)
    }

    #[test]
    fn only_the_map_kinds_with_the_setting_on_are_checked() {
        assert!(SchemaCheck::new(207, declared(true), true).is_some());
        assert!(SchemaCheck::new(205, declared(true), true).is_some());
        assert!(SchemaCheck::new(207, declared(true), false).is_none());
        assert!(SchemaCheck::new(201, declared(true), true).is_none());
    }

    #[test]
    fn a_result_of_the_declared_types_passes_in_any_string_form() {
        let check = SchemaCheck::new(207, declared(true), true).unwrap();
        for s in [ArrowType::Utf8, ArrowType::LargeUtf8, ArrowType::Utf8View] {
            let ok = result(vec![
                Field::new("id", ArrowType::Int64, true),
                Field::new("s", s, true),
            ]);
            check.check(&ok).unwrap();
        }
    }

    #[test]
    fn another_type_is_arrow_type_mismatch_with_spark_s_parameters() {
        let check = SchemaCheck::new(207, declared(true), true).unwrap();
        let bad = result(vec![
            Field::new("id", ArrowType::Int32, true),
            Field::new("s", ArrowType::Utf8, true),
        ]);
        let e = check.check(&bad).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::ArrowTypeMismatch);
        assert_eq!(e.sql_state(), Some("42K0G"));
        assert_eq!(
            e.message(),
            "Invalid schema from SQL_MAP_ARROW_ITER_UDF: expected \
             StructType(StructField(id,LongType,true),StructField(s,StringType,true)), got \
             StructType(StructField(id,IntegerType,true),StructField(s,StringType,true))."
        );
    }

    #[test]
    fn nullability_is_compared_as_spark_compares_it() {
        // A non-nullable result fits a nullable declaration; a nullable one does not fit a non-nullable one.
        let nullable = SchemaCheck::new(205, declared(true), true).unwrap();
        let strict = SchemaCheck::new(205, declared(false), true).unwrap();
        let fields = |id_nullable| {
            result(vec![
                Field::new("id", ArrowType::Int64, id_nullable),
                Field::new("s", ArrowType::Utf8View, true),
            ])
        };
        nullable.check(&fields(false)).unwrap();
        nullable.check(&fields(true)).unwrap();
        strict.check(&fields(false)).unwrap();
        let e = strict.check(&fields(true)).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::ArrowTypeMismatch);
        assert!(
            e.message().contains("SQL_MAP_PANDAS_ITER_UDF"),
            "{}",
            e.message()
        );
    }

    #[test]
    fn names_and_nested_nullability_count() {
        let check = SchemaCheck::new(207, declared(true), true).unwrap();
        let renamed = result(vec![
            Field::new("x", ArrowType::Int64, true),
            Field::new("s", ArrowType::Utf8, true),
        ]);
        assert!(check.check(&renamed).is_err());
        let list = StructType::new([StructField::new(
            "a",
            DataType::Array(Arc::new(ArrayType::new(DataType::Long, false))),
            true,
        )]);
        let check = SchemaCheck::new(207, list, true).unwrap();
        let item = |nullable| {
            result(vec![Field::new(
                "a",
                ArrowType::List(Arc::new(Field::new("element", ArrowType::Int64, nullable))),
                true,
            )])
        };
        check.check(&item(false)).unwrap();
        let e = check.check(&item(true)).unwrap_err();
        assert!(
            e.message().contains("ArrayType(LongType,false)")
                && e.message().contains("ArrayType(LongType,true)"),
            "{}",
            e.message()
        );
    }
}
