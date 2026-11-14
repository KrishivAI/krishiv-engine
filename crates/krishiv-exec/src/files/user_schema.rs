// SPDX-License-Identifier: Apache-2.0
//! The schema of a read with a user schema (`spark.read.schema(..)`, KN-1):
//! its data columns, its partition columns typed as the schema names them and
//! the text format's checks, as Spark 4.2.0 combines them.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_schema::{FieldRef, SchemaRef};
use krishiv_common::error::{
    InvalidPartitionValue, LegacyErrorTemp1290, UnsupportedDataTypeForDatasource,
};
use krishiv_io::{Segment, partition_columns};
use krishiv_types::StructType;

use super::unsupported;
use crate::error::Result;

/// The text format's checks of a user schema's data columns (Spark's
/// `TextFileFormat.verifySchema`, after the partition columns are taken
/// out): exactly one column, a string.
///
/// # Errors
/// `_LEGACY_ERROR_TEMP_1290` for another number of columns,
/// `UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE` for a column of another type.
pub(super) fn check_user_schema(format: &str, data: &[&krishiv_types::StructField]) -> Result<()> {
    if format != "text" {
        return Ok(());
    }
    let [field] = data else {
        return Err(LegacyErrorTemp1290 {
            schema_size: data.len().to_string(),
        }
        .into());
    };
    if !field.data_type().is_string_type() {
        return Err(UnsupportedDataTypeForDatasource {
            column_name: format!("`{}`", field.name().replace('`', "``")),
            column_type: format!("\"{}\"", field.data_type().sql()),
            format: "Text".to_owned(),
        }
        .into());
    }
    Ok(())
}

/// How a read with a user schema types its partition columns.
#[derive(Debug, Clone, Copy)]
pub(super) struct UserPartitions {
    /// `spark.sql.sources.partitionColumnTypeInference.enabled`, for the
    /// partition columns the schema does not name.
    pub(super) infer: bool,
    /// `spark.sql.caseSensitive`: how the schema's names meet the
    /// directories' and each other.
    pub(super) case_sensitive: bool,
    /// `spark.sql.sources.validatePartitionColumns` (default true): a value
    /// that does not read is `INVALID_PARTITION_VALUE`, else NULL.
    pub(super) validate: bool,
    /// The format is one Spark reads vectorized (Parquet, ORC): a decimal
    /// partition value reads only when written at the column's scale.
    pub(super) exact_scale_decimals: bool,
}

/// The schema of a read with the user's `schema` and the partition columns
/// `discovered` under its paths, as Spark 4.2.0 combines them: the user's
/// columns that are not partition columns, in the user's order, then the
/// partition columns in directory order. A partition column the schema
/// names takes the schema's name and type, and each of its values must read
/// as that type the way Spark converts it (`krishiv_io::user_partition_value`;
/// the field is marked so the scan reads it so); one it does not name is
/// typed by inference. Every column is nullable, whatever the schema says.
/// Returns the schema, the partition columns and the data columns.
///
/// # Errors
/// `COLUMN_ALREADY_EXISTS` for a name the schema repeats, the partition
/// discovery errors, `INVALID_PARTITION_VALUE` for a value that does not
/// read (unless `validate` is off), and `KRISHIV_UNSUPPORTED_FEATURE` for a
/// value or type whose reading is not pinned to Spark.
pub(super) fn user_schema_columns<'a>(
    schema: &'a StructType,
    discovered: &[(String, Vec<Segment>)],
    rules: UserPartitions,
) -> Result<(
    SchemaRef,
    Vec<FieldRef>,
    Vec<&'a krishiv_types::StructField>,
)> {
    krishiv_common::names::check_column_name_duplication(
        schema.fields().iter().map(krishiv_types::StructField::name),
        rules.case_sensitive,
    )?;
    let same = |a: &str, b: &str| {
        if rules.case_sensitive {
            a == b
        } else {
            krishiv_types::equals_ignore_case(a, b)
        }
    };
    // Names and layout only: the types come from the schema or below.
    let found = partition_columns(discovered, false)?;
    let mut partitions: Vec<FieldRef> = Vec::with_capacity(found.len());
    for (i, column) in found.iter().enumerate() {
        let Some(named) = schema
            .fields()
            .iter()
            .find(|f| same(f.name(), column.name()))
        else {
            // Only this column's segments are needed to infer its type.
            let one: Vec<(String, Vec<Segment>)> = discovered
                .iter()
                .map(|(path, segs)| (path.clone(), vec![segs[i].clone()]))
                .collect();
            let mut typed = partition_columns(&one, rules.infer)?;
            partitions.push(
                typed.pop().ok_or_else(|| {
                    unsupported(format!("the partition column {}", column.name()))
                })?,
            );
            continue;
        };
        let data_type = named.data_type();
        let field = krishiv_io::user_partition_field(
            data_type.to_arrow_field(named.name(), true),
            rules.exact_scale_decimals,
        );
        let mut checked: HashSet<&str> = HashSet::new();
        for (_, segs) in discovered {
            let value = segs[i].1.as_deref();
            if !value.is_none_or(|v| checked.insert(v)) {
                continue;
            }
            use krishiv_io::UserPartitionValue as V;
            match krishiv_io::user_partition_value(&field, value, rules.exact_scale_decimals) {
                V::Null | V::Text(_) => {}
                V::Invalid if !rules.validate => {}
                V::Invalid => {
                    return Err(InvalidPartitionValue {
                        value: format!("'{}'", value.unwrap_or_default()),
                        data_type: format!("\"{}\"", data_type.sql()),
                        column_name: format!("`{}`", named.name().replace('`', "``")),
                    }
                    .into());
                }
                V::Unsupported => {
                    return Err(unsupported(format!(
                        "the partition value '{}' as {} (column {})",
                        value.unwrap_or_default(),
                        data_type.sql(),
                        named.name()
                    )));
                }
            }
        }
        partitions.push(Arc::new(field));
    }
    let data: Vec<&krishiv_types::StructField> = schema
        .fields()
        .iter()
        .filter(|f| !partitions.iter().any(|p| same(f.name(), p.name())))
        .collect();
    let fields: Vec<FieldRef> = data
        .iter()
        .map(|f| Arc::new(f.data_type().to_arrow_field(f.name(), true)))
        .chain(partitions.iter().cloned())
        .collect();
    Ok((
        Arc::new(arrow_schema::Schema::new(fields)),
        partitions,
        data,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_types::DataType;

    fn discovered(values: &[&str]) -> Vec<(String, Vec<Segment>)> {
        values
            .iter()
            .map(|v| {
                (
                    format!("/t/k={v}/f"),
                    vec![krishiv_io::parse_segment(&format!("k={v}")).unwrap()],
                )
            })
            .collect()
    }

    fn ddl(columns: &[(&str, DataType, bool)]) -> StructType {
        StructType::new(
            columns
                .iter()
                .map(|(n, t, nl)| krishiv_types::StructField::new(*n, t.clone(), *nl)),
        )
    }

    fn shape(schema: &SchemaRef) -> Vec<(String, String, bool)> {
        schema
            .fields()
            .iter()
            .map(|f| {
                let t = DataType::from_arrow_field(f).unwrap();
                (f.name().clone(), t.simple_string(), f.is_nullable())
            })
            .collect()
    }

    fn refs(v: &[krishiv_types::StructField]) -> Vec<&krishiv_types::StructField> {
        v.iter().collect()
    }

    fn rules(case_sensitive: bool, validate: bool) -> UserPartitions {
        UserPartitions {
            infer: true,
            case_sensitive,
            validate,
            exact_scale_decimals: false,
        }
    }

    #[test]
    fn a_user_schema_puts_partition_columns_last_typed_as_it_names_them() {
        let named = ddl(&[("K", DataType::Long, false), ("v", DataType::String, false)]);
        let (schema, parts, data) =
            user_schema_columns(&named, &discovered(&["1", "2"]), rules(false, true)).unwrap();
        // Moved last, the schema's name and type, nullable.
        assert_eq!(
            shape(&schema),
            [
                ("v".to_owned(), "string".to_owned(), true),
                ("K".to_owned(), "bigint".to_owned(), true),
            ]
        );
        assert_eq!(parts.len(), 1);
        assert_eq!(data.len(), 1);
        // The scan reads it as Spark converts a partition value.
        assert_eq!(krishiv_io::user_value_mode(&parts[0]), Some(false));
        // Case-sensitive, `K` is no partition column: `k` is appended, inferred.
        let (schema, _, _) =
            user_schema_columns(&named, &discovered(&["1", "2"]), rules(true, true)).unwrap();
        assert_eq!(
            shape(&schema),
            [
                ("K".to_owned(), "bigint".to_owned(), true),
                ("v".to_owned(), "string".to_owned(), true),
                ("k".to_owned(), "int".to_owned(), true),
            ]
        );
        // A value that does not read is Spark's error, or NULL without validation.
        let k_int = ddl(&[("k", DataType::Integer, true)]);
        for bad in ["abc", "1.5"] {
            let e = user_schema_columns(&k_int, &discovered(&["1", bad]), rules(false, true))
                .unwrap_err();
            assert_eq!(e.error_class().name(), "INVALID_PARTITION_VALUE");
            assert_eq!(
                e.message_parameters()
                    .iter()
                    .map(|(k, v)| (*k, v.clone()))
                    .collect::<Vec<_>>(),
                [
                    ("value", format!("'{bad}'")),
                    ("dataType", "\"INT\"".to_owned()),
                    ("columnName", "`k`".to_owned())
                ]
            );
            assert!(
                user_schema_columns(&k_int, &discovered(&["1", bad]), rules(false, false)).is_ok()
            );
        }
        // A reading Spark does not pin is refused, never cast another way.
        let e = user_schema_columns(
            &ddl(&[("k", DataType::Byte, true)]),
            &discovered(&["300"]),
            rules(false, true),
        )
        .unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    }

    #[test]
    fn a_user_schema_repeating_a_name_is_refused() {
        let twice = ddl(&[("v", DataType::String, true), ("V", DataType::String, true)]);
        let e = user_schema_columns(&twice, &[], rules(false, true)).unwrap_err();
        assert_eq!(e.error_class().name(), "COLUMN_ALREADY_EXISTS");
        assert!(user_schema_columns(&twice, &[], rules(true, true)).is_ok());
    }

    #[test]
    fn the_text_format_takes_one_string_data_column() {
        let one = ddl(&[("line", DataType::String, false)]);
        let fields = |s: &StructType| -> Vec<krishiv_types::StructField> { s.fields().to_vec() };
        assert!(check_user_schema("text", &refs(&fields(&one))).is_ok());
        let two = fields(&ddl(&[
            ("a", DataType::String, true),
            ("b", DataType::String, true),
        ]));
        assert_eq!(
            check_user_schema("text", &refs(&two))
                .unwrap_err()
                .error_class()
                .name(),
            "_LEGACY_ERROR_TEMP_1290"
        );
        let int = fields(&ddl(&[("line", DataType::Integer, true)]));
        assert_eq!(
            check_user_schema("text", &refs(&int))
                .unwrap_err()
                .error_class()
                .name(),
            "UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE"
        );
        assert!(check_user_schema("csv", &refs(&two)).is_ok());
        // The partition column is no data column: `line STRING, k INT` over `k=1`.
        let with_k = ddl(&[
            ("line", DataType::String, true),
            ("k", DataType::Integer, true),
        ]);
        let (_, _, data) =
            user_schema_columns(&with_k, &discovered(&["1"]), rules(false, true)).unwrap();
        assert!(check_user_schema("text", &data).is_ok());
    }
}
