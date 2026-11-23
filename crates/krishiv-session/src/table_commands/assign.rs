// SPDX-License-Identifier: Apache-2.0
//! Matching an `INSERT`'s query columns to the table's (by position, column list or name), Spark's ANSI
//! store-assignment rule and the `CAST_OVERFLOW_IN_TABLE_INSERT` rewrite of a write's overflow.

use krishiv_common::error::{
    CastOverflowInTableInsert, ErrorClass, IncompatibleDataForTableCannotSafelyCast,
    InsertColumnArityMismatchNotEnoughDataColumns, InsertColumnArityMismatchTooManyDataColumns,
    KrishivError, UnresolvedColumnWithSuggestion,
};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;

use super::{CatalogTable, quoted};

/// Where a table column's value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Source {
    /// The query's column at this position.
    Position(usize),
    /// The column's `DEFAULT`, else NULL.
    Default,
}

/// Matches the query's `output` to the columns of `table`: by position, by
/// the column list, or by name. Spark's arity and store-assignment errors.
pub(super) fn assign(
    table: &CatalogTable,
    listed: &[String],
    by_name: bool,
    output: &[TableColumn],
    case_sensitive: bool,
    legacy_store: bool,
) -> Result<Vec<Source>, KrishivError> {
    let eq = |a: &str, b: &str| {
        if case_sensitive {
            a == b
        } else {
            a.eq_ignore_ascii_case(b)
        }
    };
    let quoted_names = |names: Vec<&str>| {
        names
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let table_name = quoted(&table.namespace, &table.name);
    let table_cols = || quoted_names(table.columns.iter().map(|c| c.name.as_str()).collect());
    let data_cols = || quoted_names(output.iter().map(|c| c.name.as_str()).collect());
    let mut out = vec![Source::Default; table.columns.len()];
    if by_name {
        for (i, o) in output.iter().enumerate() {
            let Some(at) = table.columns.iter().position(|c| eq(&c.name, &o.name)) else {
                // A data column the table does not have: Spark counts the table's columns against the data's.
                return Err(InsertColumnArityMismatchTooManyDataColumns {
                    table_name,
                    table_columns: table_cols(),
                    data_columns: data_cols(),
                }
                .into());
            };
            out[at] = Source::Position(i);
        }
    } else if !listed.is_empty() {
        if listed.len() != output.len() {
            let (not_enough, table_cols_text) = (
                listed.len() > output.len(),
                quoted_names(listed.iter().map(String::as_str).collect()),
            );
            return Err(if not_enough {
                InsertColumnArityMismatchNotEnoughDataColumns {
                    table_name,
                    table_columns: table_cols_text,
                    data_columns: data_cols(),
                }
                .into()
            } else {
                InsertColumnArityMismatchTooManyDataColumns {
                    table_name,
                    table_columns: table_cols_text,
                    data_columns: data_cols(),
                }
                .into()
            });
        }
        for (i, name) in listed.iter().enumerate() {
            let Some(at) = table.columns.iter().position(|c| eq(&c.name, name)) else {
                return Err(UnresolvedColumnWithSuggestion {
                    object_name: format!("`{name}`"),
                    proposal: table_cols(),
                }
                .into());
            };
            out[at] = Source::Position(i);
        }
    } else {
        match output.len().cmp(&table.columns.len()) {
            std::cmp::Ordering::Less => {
                return Err(InsertColumnArityMismatchNotEnoughDataColumns {
                    table_name,
                    table_columns: table_cols(),
                    data_columns: data_cols(),
                }
                .into());
            }
            std::cmp::Ordering::Greater => {
                return Err(InsertColumnArityMismatchTooManyDataColumns {
                    table_name,
                    table_columns: table_cols(),
                    data_columns: data_cols(),
                }
                .into());
            }
            std::cmp::Ordering::Equal => {
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = Source::Position(i);
                }
            }
        }
    }
    for (column, from) in table.columns.iter().zip(&out) {
        // `spark.sql.storeAssignmentPolicy=LEGACY` stores through any cast Spark can run: no ANSI check, but
        // a cast Spark refuses in every mode (an `ARRAY` into an `INT`) is refused here, before a first
        // clause commits.
        if let Source::Position(i) = from
            && !(if legacy_store {
                !legacy_refused(&output[*i].data_type, &column.data_type)
            } else {
                can_store(&output[*i].data_type, &column.data_type)
            })
        {
            return Err(IncompatibleDataForTableCannotSafelyCast {
                table_name,
                col_name: format!("`{}`", column.name),
                src_type: format!(
                    "\"{}\"",
                    output[*i].data_type.catalog_string().to_uppercase()
                ),
                target_type: format!("\"{}\"", column.data_type.catalog_string().to_uppercase()),
            }
            .into());
        }
    }
    Ok(out)
}

/// Whether `storeAssignmentPolicy=LEGACY` refuses to store `from` into `to`. Spark recurses only through a
/// (MAP, MAP), (ARRAY, ARRAY) or (STRUCT, STRUCT of the same arity) pair, one leaf at a time
/// (`TableOutputResolver.resolveMapType`, `resolveArrayType`, `resolveStructType`), so a map's key pair and value
/// pair are checked apart; every other pair, a non-map source into a map column included, is one cast
/// (`checkField`), and a cast Spark refuses in every mode is refused here, before a first clause commits.
pub(super) fn legacy_refused(from: &DataType, to: &DataType) -> bool {
    match (from, to) {
        (DataType::Map(f), DataType::Map(t)) => {
            legacy_refused(f.key_type(), t.key_type())
                || legacy_refused(f.value_type(), t.value_type())
        }
        (DataType::Array(f), DataType::Array(t)) => {
            legacy_refused(f.element_type(), t.element_type())
        }
        (DataType::Struct(f), DataType::Struct(t)) if f.fields().len() == t.fields().len() => f
            .fields()
            .iter()
            .zip(t.fields().iter())
            .any(|(f, t)| legacy_refused(f.data_type(), t.data_type())),
        _ => krishiv_plan::analyzer::cast_always_refused(from, to),
    }
}

/// The LEGACY store cast of `value` (of type `from`) to `to`, one leaf at a time as Spark does: a map's keys and
/// values are cast apart (a non-ANSI cast each, so `'x'` stores `NULL` and an overflow wraps or saturates as
/// `CAST` does), through array elements and struct fields. `depth` keeps nested lambda names apart.
pub(super) fn legacy_store_cast(
    value: crate::Column,
    from: &DataType,
    to: &DataType,
    depth: usize,
) -> crate::Column {
    use crate::functions::{call_function, col, lambda, lit};
    if from == to {
        return value;
    }
    match (from, to) {
        (DataType::Map(f), DataType::Map(t)) => {
            let leaf = |arr: crate::Column, f: &DataType, t: &DataType| {
                let name = format!("__krishiv_x{depth}");
                let body = legacy_store_cast(col(&name), f, t, depth + 1);
                call_function("transform", [arr, lambda([name], body)])
            };
            let keys = leaf(
                call_function("map_keys", [value.clone()]),
                f.key_type(),
                t.key_type(),
            );
            let values = leaf(
                call_function("map_values", [value]),
                f.value_type(),
                t.value_type(),
            );
            call_function("map_from_arrays", [keys, values])
        }
        (DataType::Array(f), DataType::Array(t)) => {
            let name = format!("__krishiv_x{depth}");
            let body = legacy_store_cast(col(&name), f.element_type(), t.element_type(), depth + 1);
            call_function("transform", [value, lambda([name], body)])
        }
        (DataType::Struct(f), DataType::Struct(t)) if f.fields().len() == t.fields().len() => {
            let mut args = Vec::with_capacity(t.fields().len() * 2);
            for (f, t) in f.fields().iter().zip(t.fields().iter()) {
                args.push(lit(t.name()));
                args.push(legacy_store_cast(
                    value.clone().get_field(f.name()),
                    f.data_type(),
                    t.data_type(),
                    depth,
                ));
            }
            let built = call_function("named_struct", args);
            call_function(
                "if",
                [
                    call_function("isnull", [value]),
                    crate::functions::lit_null().cast(to.clone()),
                    built,
                ],
            )
        }
        _ => value.cast_legacy(to.clone()),
    }
}

/// Spark's ANSI store assignment (`Cast.canANSIStoreAssign`): whether a
/// value of type `from` may be stored into a column of type `to`.
pub(super) fn can_store(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    if from == to {
        return true;
    }
    let numeric = |t: &T| {
        matches!(
            t,
            T::Byte | T::Short | T::Integer | T::Long | T::Float | T::Double | T::Decimal(_)
        )
    };
    // Anything atomic may be stored as a string.
    let to_string = matches!(to, T::String | T::CollatedString(_))
        && (numeric(from)
            || matches!(
                from,
                T::Boolean | T::Date | T::Timestamp | T::TimestampNtz | T::Binary
            ));
    match (from, to) {
        (T::Null, _) | (T::String | T::CollatedString(_), T::String | T::CollatedString(_)) => true,
        (a, b) if (numeric(a) && numeric(b)) || to_string => true,
        // Dates and timestamps among themselves.
        (T::Date, T::Timestamp | T::TimestampNtz)
        | (T::Timestamp | T::TimestampNtz, T::Date | T::Timestamp | T::TimestampNtz) => true,
        (T::Array(a), T::Array(b)) => {
            can_store(a.element_type(), b.element_type())
                && (b.contains_null() || !a.contains_null())
        }
        (T::Map(a), T::Map(b)) => {
            can_store(a.key_type(), b.key_type())
                && can_store(a.value_type(), b.value_type())
                && (b.value_contains_null() || !a.value_contains_null())
        }
        (T::Struct(a), T::Struct(b)) => {
            a.fields().len() == b.fields().len()
                && a.fields().iter().zip(b.fields()).all(|(x, y)| {
                    can_store(x.data_type(), y.data_type()) && (y.nullable() || !x.nullable())
                })
        }
        _ => false,
    }
}

/// A `CAST_OVERFLOW` raised while a column is written is `CAST_OVERFLOW_IN_TABLE_INSERT`, naming the column,
/// whatever `spark.sql.ansi.enabled` is; any other error is returned as it came.
pub(super) fn overflow_in_table_insert(
    e: KrishivError,
    table: &CatalogTable,
    output: &[TableColumn],
    assignment: &[Source],
) -> KrishivError {
    let e = crate::char_write::map_error(e);
    if e.error_class() != ErrorClass::CastOverflow {
        return e;
    }
    let params = e.message_parameters();
    let param = |key: &str| {
        params
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, v)| v.clone())
    };
    let (Some(source_type), Some(target_type)) = (param("sourceType"), param("targetType")) else {
        return e;
    };
    let sql = |t: &DataType| format!("\"{}\"", t.catalog_string().to_uppercase());
    let column = table
        .columns
        .iter()
        .zip(assignment)
        .find(|(c, from)| {
            matches!(from, Source::Position(i)
                if sql(&c.data_type) == target_type && sql(&output[*i].data_type) == source_type)
        })
        .map(|(c, _)| c.name.clone());
    // No column matches: the overflow is the query's own, and Spark reports it as it is.
    let Some(column) = column else {
        return e;
    };
    CastOverflowInTableInsert {
        source_type,
        target_type,
        column_name: format!("`{column}`"),
    }
    .into()
}
