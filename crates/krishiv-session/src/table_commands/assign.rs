// SPDX-License-Identifier: Apache-2.0
//! Matching an `INSERT`'s query columns to the table's (by position, column list or name), Spark's ANSI
//! store-assignment rule and the `CAST_OVERFLOW_IN_TABLE_INSERT` rewrite of a write's overflow.

use krishiv_common::error::{
    CastOverflowInTableInsert, ErrorClass, IncompatibleDataForTableAmbiguousColumnName,
    IncompatibleDataForTableCannotFindData, IncompatibleDataForTableCannotSafelyCast,
    IncompatibleDataForTableExtraColumns, IncompatibleDataForTableExtraStructFields,
    InsertColumnArityMismatchNotEnoughDataColumns, InsertColumnArityMismatchTooManyDataColumns,
    KrishivError, UnresolvedColumnWithSuggestion,
};
use krishiv_plan::analyzer::{SparkProfile, TableColumn};
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
    profile: SparkProfile,
) -> Result<Vec<Source>, KrishivError> {
    // `INSERT .. BY NAME` also matches the fields of a nested struct by name (Spark's `reorderColumnsByName`).
    let names = by_name.then_some(case_sensitive);
    let eq = |a: &str, b: &str| names_match(a, b, case_sensitive);
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
    // What one table column checks about the data column paired with it.
    let check_column =
        |column: &crate::store::StoreColumn, data: &TableColumn, by_name: Option<bool>| {
            // `INSERT .. BY NAME` resolves a struct's fields by name under every policy (Spark's
            // `reorderColumnsByName`), before any leaf is checked.
            if let Some(case) = by_name
                && let Some(problem) = struct_problem(
                    &data.data_type,
                    &column.data_type,
                    &[column.name.as_str()],
                    case,
                    legacy_store,
                )
            {
                return Err(problem.into_error(table_name.clone()));
            }
            // `spark.sql.storeAssignmentPolicy=LEGACY` stores through any cast Spark can run: no ANSI check, but
            // a cast Spark refuses in every mode (an `ARRAY` into an `INT`) is refused here, before a first
            // clause commits.
            let stores = if legacy_store {
                !legacy_refused(&data.data_type, &column.data_type, by_name)
            } else {
                can_store_by(&data.data_type, &column.data_type, by_name)
            };
            if !stores {
                return Err(KrishivError::from(
                    IncompatibleDataForTableCannotSafelyCast {
                        table_name: table_name.clone(),
                        col_name: format!("`{}`", column.name.replace('`', "``")),
                        src_type: format!("\"{}\"", data.data_type.catalog_string().to_uppercase()),
                        target_type: format!(
                            "\"{}\"",
                            column.data_type.catalog_string().to_uppercase()
                        ),
                    },
                ));
            }
            Ok(())
        };
    let mut out = vec![Source::Default; table.columns.len()];
    if by_name {
        // Spark's `reorderColumnsByName`: the arity, then each table column in turn (its matches, then its
        // struct fields and store cast), then the data columns no table column took.
        if output.len() > table.columns.len() {
            return Err(InsertColumnArityMismatchTooManyDataColumns {
                table_name,
                table_columns: table_cols(),
                data_columns: data_cols(),
            }
            .into());
        }
        let mut taken = vec![false; output.len()];
        for (column, slot) in table.columns.iter().zip(&mut out) {
            let mut matches = output
                .iter()
                .enumerate()
                .filter(|(_, o)| eq(&column.name, &o.name));
            let Some((i, _)) = matches.next() else {
                continue;
            };
            if matches.next().is_some() {
                return Err(IncompatibleDataForTableAmbiguousColumnName {
                    table_name,
                    col_name: format!("`{}`", column.name.replace('`', "``")),
                }
                .into());
            }
            check_column(column, &output[i], names)?;
            taken[i] = true;
            *slot = Source::Position(i);
        }
        // Spark 4.3.0 quotes each raw name as one part (`toSQLId(Seq(name))`). 3.5 to 4.2 parse it first
        // (`toSQLId(name)`), so one that does not parse is `INVALID_ATTRIBUTE_NAME_SYNTAX`.
        let extra = output
            .iter()
            .zip(&taken)
            .filter(|(_, taken)| !**taken)
            .map(|(o, _)| {
                let parts = if profile >= SparkProfile::V4_3 {
                    vec![o.name.clone()]
                } else {
                    krishiv_common::names::parse_attribute_name(&o.name)?
                };
                Ok(quoted_path(
                    &parts.iter().map(String::as_str).collect::<Vec<_>>(),
                ))
            })
            .collect::<Result<Vec<String>, KrishivError>>()?;
        if !extra.is_empty() {
            return Err(IncompatibleDataForTableExtraColumns {
                table_name,
                extra_columns: extra.join(", "),
            }
            .into());
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
    if !by_name {
        for (column, from) in table.columns.iter().zip(&out) {
            if let Source::Position(i) = from {
                check_column(column, &output[*i], None)?;
            }
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
    /// A leaf the ANSI store assignment refuses (`CANNOT_SAFELY_CAST`): its path and both types.
    CannotSafelyCast(String, DataType, DataType),
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
            Self::CannotSafelyCast(col_name, from, to) => {
                IncompatibleDataForTableCannotSafelyCast {
                    table_name,
                    col_name,
                    src_type: format!("\"{}\"", from.catalog_string().to_uppercase()),
                    target_type: format!("\"{}\"", to.catalog_string().to_uppercase()),
                }
                .into()
            }
        }
    }
}

/// A name in backquotes, an embedded backquote doubled (Spark's `quoteIfNeeded` form).
fn quote_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// The backquoted dotted path of a nested column (Spark's `UnresolvedAttribute.quoted`).
fn quoted_path(path: &[&str]) -> String {
    path.iter()
        .map(|p| quote_ident(p))
        .collect::<Vec<_>>()
        .join(".")
}

/// Whether two field names are the same under the session's case rule: Spark's `equalsIgnoreCase` when it is
/// off, which compares code point by code point after `toUpperCase` and then `toLowerCase` (so `é` matches `É`
/// and `İ` matches `i`), the one rule for matching, uniqueness and reorder detection.
fn names_match(a: &str, b: &str, case_sensitive: bool) -> bool {
    if case_sensitive {
        return a == b;
    }
    let single = |mut it: Box<dyn Iterator<Item = char>>, c: char| match (it.next(), it.next()) {
        (Some(one), None) => one,
        _ => c,
    };
    let upper = |c: char| single(Box::new(c.to_uppercase()), c);
    let lower = |c: char| {
        // `Character.toLowerCase(U+0130)` is `i`, where Unicode's full mapping is two code points.
        if c == '\u{130}' {
            'i'
        } else {
            single(Box::new(c.to_lowercase()), c)
        }
    };
    let (mut x, mut y) = (a.chars(), b.chars());
    loop {
        match (x.next(), y.next()) {
            (None, None) => return true,
            (Some(p), Some(q)) => {
                if p != q && upper(p) != upper(q) && lower(upper(p)) != lower(upper(q)) {
                    return false;
                }
            }
            _ => return false,
        }
    }
}

/// The first struct, at any depth of `from` and `to` (through arrays and maps), whose fields do not pair by
/// name under `case_sensitive`: Spark's `TableOutputResolver.reorderColumnsByName` errors, in its order (each
/// target field in turn, ambiguity then absence, then source fields left over). Under the ANSI and STRICT
/// store policies a leaf is checked as it is reached (`checkField`), so the first failing leaf or struct wins;
/// under LEGACY Spark's `canWrite` passes, and only the struct errors are raised here.
fn struct_problem(
    from: &DataType,
    to: &DataType,
    path: &[&str],
    case_sensitive: bool,
    legacy: bool,
) -> Option<StructProblem> {
    match (from, to) {
        (DataType::Array(f), DataType::Array(t)) => {
            let path = [path, &["element"]].concat();
            struct_problem(
                f.element_type(),
                t.element_type(),
                &path,
                case_sensitive,
                legacy,
            )
        }
        (DataType::Map(f), DataType::Map(t)) => {
            let (key, value) = ([path, &["key"]].concat(), [path, &["value"]].concat());
            struct_problem(f.key_type(), t.key_type(), &key, case_sensitive, legacy).or_else(|| {
                struct_problem(
                    f.value_type(),
                    t.value_type(),
                    &value,
                    case_sensitive,
                    legacy,
                )
            })
        }
        (DataType::Struct(f), DataType::Struct(t)) => {
            let same = |a: &str, b: &str| names_match(a, b, case_sensitive);
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
                            legacy,
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
                .map(|(f, _)| quote_ident(f.name()))
                .collect();
            (!extra.is_empty()).then(|| StructProblem::Extra(quoted_path(path), extra.join(", ")))
        }
        _ if !legacy && !can_store(from, to) => Some(StructProblem::CannotSafelyCast(
            quoted_path(path),
            from.clone(),
            to.clone(),
        )),
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
    st.fields()
        .iter()
        .find(|f| names_match(f.name(), name, case_sensitive))
}

/// Whether a struct's field names are all different, ignoring case unless `case_sensitive`.
fn unique_names(st: &krishiv_types::StructType, case_sensitive: bool) -> bool {
    let fields = st.fields();
    fields.iter().enumerate().all(|(i, a)| {
        fields[..i]
            .iter()
            .all(|b| !names_match(a.name(), b.name(), case_sensitive))
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
    // A map whose keys change type is rebuilt (`map_from_arrays`) in either field order, so the
    // `spark.sql.mapKeyDedupPolicy` applies to colliding cast keys whatever the order of the struct's fields.
    if needs_reorder(from, to, case_sensitive) || map_keys_change(from, to) {
        store_cast(value, from, to, 0, Some(case_sensitive), true)
    } else {
        value.cast_ansi(to.clone())
    }
}

/// Whether a map inside `from`/`to` has keys of another type: the cast can make two keys collide.
fn map_keys_change(from: &DataType, to: &DataType) -> bool {
    match (from, to) {
        (DataType::Map(f), DataType::Map(t)) => {
            f.key_type() != t.key_type()
                || map_keys_change(f.key_type(), t.key_type())
                || map_keys_change(f.value_type(), t.value_type())
        }
        (DataType::Array(f), DataType::Array(t)) => {
            map_keys_change(f.element_type(), t.element_type())
        }
        (DataType::Struct(f), DataType::Struct(t)) if f.fields().len() == t.fields().len() => f
            .fields()
            .iter()
            .zip(t.fields())
            .any(|(f, t)| map_keys_change(f.data_type(), t.data_type())),
        _ => false,
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
                let same_place = names_match(f.name(), t.name(), case_sensitive);
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
