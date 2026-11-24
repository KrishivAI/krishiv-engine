// SPDX-License-Identifier: Apache-2.0
//! Matching an `INSERT`'s query columns to the table's (by position, column list or name), Spark's ANSI
//! store-assignment rule and the `CAST_OVERFLOW_IN_TABLE_INSERT` rewrite of a write's overflow.

use krishiv_common::error::{
    CastOverflowInTableInsert, ErrorClass, IncompatibleDataForTableAmbiguousColumnName,
    IncompatibleDataForTableCannotFindData, IncompatibleDataForTableCannotSafelyCast,
    IncompatibleDataForTableExtraStructFields, InsertColumnArityMismatchNotEnoughDataColumns,
    InsertColumnArityMismatchTooManyDataColumns, KrishivError, UnresolvedColumnWithSuggestion,
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
    // `INSERT .. BY NAME` also matches the fields of a nested struct by name (Spark's `reorderColumnsByName`).
    let names = by_name.then_some(case_sensitive);
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
        // `INSERT .. BY NAME` resolves a struct's fields by name under every policy (Spark's
        // `reorderColumnsByName`), before any leaf is checked.
        if let (Source::Position(i), Some(case)) = (from, names)
            && let Some(problem) = struct_problem(
                &output[*i].data_type,
                &column.data_type,
                &[column.name.as_str()],
                case,
            )
        {
            return Err(problem.into_error(table_name));
        }
        // `spark.sql.storeAssignmentPolicy=LEGACY` stores through any cast Spark can run: no ANSI check, but
        // a cast Spark refuses in every mode (an `ARRAY` into an `INT`) is refused here, before a first
        // clause commits.
        if let Source::Position(i) = from
            && !(if legacy_store {
                !legacy_refused(&output[*i].data_type, &column.data_type, names)
            } else {
                can_store_by(&output[*i].data_type, &column.data_type, names)
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

/// What `INSERT .. BY NAME` found wrong when it paired a struct's fields by name.
enum StructProblem {
    /// A target field no source field matches (`CANNOT_FIND_DATA`).
    CannotFind(String),
    /// Source fields no target field takes (`EXTRA_STRUCT_FIELDS`): the struct's path and the fields.
    Extra(String, String),
    /// Two source fields match one target field (`AMBIGUOUS_COLUMN_NAME`).
    Ambiguous(String),
}

impl StructProblem {
    fn into_error(self, table_name: String) -> KrishivError {
        match self {
            Self::CannotFind(col_name) => IncompatibleDataForTableCannotFindData {
                table_name,
                col_name,
            }
            .into(),
            Self::Extra(col_name, extra_fields) => IncompatibleDataForTableExtraStructFields {
                table_name,
                col_name,
                extra_fields,
            }
            .into(),
            Self::Ambiguous(col_name) => IncompatibleDataForTableAmbiguousColumnName {
                table_name,
                col_name,
            }
            .into(),
        }
    }
}

/// The backquoted dotted path of a nested column (Spark's `UnresolvedAttribute.quoted`).
fn quoted_path(path: &[&str]) -> String {
    path.iter()
        .map(|p| format!("`{p}`"))
        .collect::<Vec<_>>()
        .join(".")
}

/// The first struct, at any depth of `from` and `to` (through arrays and maps), whose fields do not pair by
/// name under `case_sensitive`: Spark's `TableOutputResolver.reorderColumnsByName` errors, in its order (each
/// target field in turn, ambiguity then absence, then source fields left over).
fn struct_problem(
    from: &DataType,
    to: &DataType,
    path: &[&str],
    case_sensitive: bool,
) -> Option<StructProblem> {
    match (from, to) {
        (DataType::Array(f), DataType::Array(t)) => {
            let path = [path, &["element"]].concat();
            struct_problem(f.element_type(), t.element_type(), &path, case_sensitive)
        }
        (DataType::Map(f), DataType::Map(t)) => {
            let (key, value) = ([path, &["key"]].concat(), [path, &["value"]].concat());
            struct_problem(f.key_type(), t.key_type(), &key, case_sensitive)
                .or_else(|| struct_problem(f.value_type(), t.value_type(), &value, case_sensitive))
        }
        (DataType::Struct(f), DataType::Struct(t)) => {
            let same = |a: &str, b: &str| {
                if case_sensitive {
                    a == b
                } else {
                    a.eq_ignore_ascii_case(b)
                }
            };
            let mut used = vec![false; f.fields().len()];
            for target in t.fields() {
                let at = [path, &[target.name()]].concat();
                let matches: Vec<usize> = (0..f.fields().len())
                    .filter(|&i| same(f.fields()[i].name(), target.name()))
                    .collect();
                match matches.as_slice() {
                    [] => return Some(StructProblem::CannotFind(quoted_path(&at))),
                    [one] => {
                        used[*one] = true;
                        if let Some(p) = struct_problem(
                            f.fields()[*one].data_type(),
                            target.data_type(),
                            &at,
                            case_sensitive,
                        ) {
                            return Some(p);
                        }
                    }
                    _ => return Some(StructProblem::Ambiguous(quoted_path(&at))),
                }
            }
            let extra: Vec<String> = f
                .fields()
                .iter()
                .zip(&used)
                .filter(|(_, used)| !**used)
                .map(|(f, _)| format!("`{}`", f.name()))
                .collect();
            (!extra.is_empty()).then(|| StructProblem::Extra(quoted_path(path), extra.join(", ")))
        }
        _ => None,
    }
}

/// Whether `storeAssignmentPolicy=LEGACY` refuses to store `from` into `to`. Spark recurses only through a
/// (MAP, MAP), (ARRAY, ARRAY) or (STRUCT, STRUCT of the same arity) pair, one leaf at a time
/// (`TableOutputResolver.resolveMapType`, `resolveArrayType`, `resolveStructType`), so a map's key pair and value
/// pair are checked apart; every other pair, a non-map source into a map column included, is one cast
/// (`checkField`), and a cast Spark refuses in every mode is refused here, before a first clause commits.
///
/// `by_name` is `Some(case_sensitive)` for `INSERT .. BY NAME`, where a struct's fields are paired by name and a
/// target field the source lacks is refused.
pub(super) fn legacy_refused(from: &DataType, to: &DataType, by_name: Option<bool>) -> bool {
    match (from, to) {
        (DataType::Map(f), DataType::Map(t)) => {
            legacy_refused(f.key_type(), t.key_type(), by_name)
                || legacy_refused(f.value_type(), t.value_type(), by_name)
        }
        (DataType::Array(f), DataType::Array(t)) => {
            legacy_refused(f.element_type(), t.element_type(), by_name)
        }
        (DataType::Struct(f), DataType::Struct(t)) if f.fields().len() == t.fields().len() => {
            match by_name {
                None => f
                    .fields()
                    .iter()
                    .zip(t.fields().iter())
                    .any(|(f, t)| legacy_refused(f.data_type(), t.data_type(), by_name)),
                // Source names that repeat under the rule have no name to pair by: refused.
                Some(case) => {
                    !unique_names(f, case)
                        || t.fields().iter().any(|t| {
                            field_named(f, t.name(), case).is_none_or(|f| {
                                legacy_refused(f.data_type(), t.data_type(), by_name)
                            })
                        })
                }
            }
        }
        _ => krishiv_plan::analyzer::cast_always_refused(from, to),
    }
}

/// The field of `st` named `name`, under the session's case rule.
fn field_named<'a>(
    st: &'a krishiv_types::StructType,
    name: &str,
    case_sensitive: bool,
) -> Option<&'a krishiv_types::StructField> {
    st.fields().iter().find(|f| {
        if case_sensitive {
            f.name() == name
        } else {
            f.name().eq_ignore_ascii_case(name)
        }
    })
}

/// Whether a struct's field names are all different, ignoring case unless `case_sensitive`.
fn unique_names(st: &krishiv_types::StructType, case_sensitive: bool) -> bool {
    let mut seen = std::collections::HashSet::new();
    st.fields().iter().all(|f| {
        seen.insert(if case_sensitive {
            f.name().to_owned()
        } else {
            f.name().to_lowercase()
        })
    })
}

/// The LEGACY store cast of `value` (of type `from`) to `to`, one leaf at a time as Spark does: a map's keys and
/// values are cast apart (a non-ANSI cast each, so `'x'` stores `NULL` and an overflow wraps or saturates as
/// `CAST` does), through array elements and struct fields. `depth` keeps nested lambda names apart.
pub(super) fn legacy_store_cast(
    value: crate::Column,
    from: &DataType,
    to: &DataType,
    depth: usize,
    by_name: Option<bool>,
) -> crate::Column {
    store_cast(value, from, to, depth, by_name, false)
}

/// The ANSI store cast of an `INSERT .. BY NAME` value whose type holds a struct with its fields in another
/// order than the column's (Spark's `reorderColumnsByName`, under every policy): the same per-field rebuild as
/// the LEGACY cast, with ANSI casts at the leaves.
pub(super) fn ansi_store_cast_by_name(
    value: crate::Column,
    from: &DataType,
    to: &DataType,
    case_sensitive: bool,
) -> crate::Column {
    if needs_reorder(from, to, case_sensitive) {
        store_cast(value, from, to, 0, Some(case_sensitive), true)
    } else {
        value.cast_ansi(to.clone())
    }
}

/// Whether a struct inside `from`/`to` has fields that pair by name in another order.
fn needs_reorder(from: &DataType, to: &DataType, case_sensitive: bool) -> bool {
    match (from, to) {
        (DataType::Map(f), DataType::Map(t)) => {
            needs_reorder(f.key_type(), t.key_type(), case_sensitive)
                || needs_reorder(f.value_type(), t.value_type(), case_sensitive)
        }
        (DataType::Array(f), DataType::Array(t)) => {
            needs_reorder(f.element_type(), t.element_type(), case_sensitive)
        }
        (DataType::Struct(f), DataType::Struct(t)) if f.fields().len() == t.fields().len() => {
            f.fields().iter().zip(t.fields().iter()).any(|(f, t)| {
                let same_place = if case_sensitive {
                    f.name() == t.name()
                } else {
                    f.name().eq_ignore_ascii_case(t.name())
                };
                !same_place || needs_reorder(f.data_type(), t.data_type(), case_sensitive)
            })
        }
        _ => false,
    }
}

/// `legacy_store_cast`, or with `ansi` the same walk with ANSI leaf casts.
fn store_cast(
    value: crate::Column,
    from: &DataType,
    to: &DataType,
    depth: usize,
    by_name: Option<bool>,
    ansi: bool,
) -> crate::Column {
    use crate::functions::{call_function, col, lambda, lit};
    if from == to {
        return value;
    }
    let leaf = |value: crate::Column, to: &DataType| {
        if ansi {
            value.cast_ansi(to.clone())
        } else {
            value.cast_legacy(to.clone())
        }
    };
    match (from, to) {
        (DataType::Map(f), DataType::Map(t)) => {
            let leaf = |arr: crate::Column, f: &DataType, t: &DataType| {
                let name = format!("__krishiv_x{depth}");
                let body = store_cast(col(&name), f, t, depth + 1, by_name, ansi);
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
            let body = store_cast(
                col(&name),
                f.element_type(),
                t.element_type(),
                depth + 1,
                by_name,
                ansi,
            );
            call_function("transform", [value, lambda([name], body)])
        }
        // A field read by name (`get_field`) needs unique source names; Spark reads by ordinal, so a source
        // struct whose names repeat (ignoring case) is cast as one value, by position.
        (DataType::Struct(f), DataType::Struct(t))
            if f.fields().len() == t.fields().len()
                && unique_names(f, by_name.unwrap_or(false)) =>
        {
            let mut args = Vec::with_capacity(t.fields().len() * 2);
            for (i, t) in t.fields().iter().enumerate() {
                // `BY NAME` pairs the fields by name; `legacy_refused` has refused a target field with no match.
                let f = match by_name {
                    Some(case) => field_named(f, t.name(), case),
                    None => f.fields().get(i),
                };
                let Some(f) = f else {
                    return leaf(value, to);
                };
                args.push(lit(t.name()));
                args.push(store_cast(
                    value.clone().get_field(f.name()),
                    f.data_type(),
                    t.data_type(),
                    depth,
                    by_name,
                    ansi,
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
        _ => leaf(value, to),
    }
}

/// Spark's ANSI store assignment (`Cast.canANSIStoreAssign`): whether a
/// value of type `from` may be stored into a column of type `to`.
pub(super) fn can_store(from: &DataType, to: &DataType) -> bool {
    can_store_by(from, to, None)
}

/// [`can_store`], where `by_name` (`Some(case_sensitive)`, for `INSERT .. BY NAME`) pairs a struct's fields by
/// name and refuses a target field the source lacks.
pub(super) fn can_store_by(from: &DataType, to: &DataType, by_name: Option<bool>) -> bool {
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
            can_store_by(a.element_type(), b.element_type(), by_name)
                && (b.contains_null() || !a.contains_null())
        }
        (T::Map(a), T::Map(b)) => {
            can_store_by(a.key_type(), b.key_type(), by_name)
                && can_store_by(a.value_type(), b.value_type(), by_name)
                && (b.value_contains_null() || !a.value_contains_null())
        }
        (T::Struct(a), T::Struct(b)) if a.fields().len() == b.fields().len() => match by_name {
            None => a.fields().iter().zip(b.fields()).all(|(x, y)| {
                can_store_by(x.data_type(), y.data_type(), by_name)
                    && (y.nullable() || !x.nullable())
            }),
            Some(case) => {
                unique_names(a, case)
                    && b.fields().iter().all(|y| {
                        field_named(a, y.name(), case).is_some_and(|x| {
                            can_store_by(x.data_type(), y.data_type(), by_name)
                                && (y.nullable() || !x.nullable())
                        })
                    })
            }
        },
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
