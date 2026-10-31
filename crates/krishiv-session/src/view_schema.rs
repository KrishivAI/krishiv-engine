// SPDX-License-Identifier: Apache-2.0
//! The schema binding modes of permanent views (Spark 4.2's `ViewSchemaMode`).
//!
//! A permanent view stores its columns (names, types) and the names its query output had at creation. A read
//! inlines the view's query (`catalog.rs`) and, by the view's mode:
//! - `BINDING`: finds each stored column by its query name and up-casts it to the stored type; a type that does
//!   not up-cast is `CANNOT_UP_CAST_DATATYPE` (Spark's `UpCast`);
//! - `COMPENSATION`: finds the columns by name and casts them with ANSI casts;
//! - `TYPE EVOLUTION`: finds the columns by name and keeps the query's types, which the view then stores;
//! - `EVOLUTION`: is the query's output, whose schema (names, types, count) the view then stores.
//!
//! A stored column whose name the query no longer outputs (as often as it did) is
//! `INCOMPATIBLE_VIEW_SCHEMA_CHANGE` in every mode but `EVOLUTION`. With `spark.sql.legacy.viewSchemaBindingMode`
//! off a view has no mode: it compensates, or up-casts when `spark.sql.legacy.viewSchemaCompensation` is off.
//!
//! The checks need the types of the view query's output, which the inlining (before binding) does not know: the
//! session analyzes each inlined view query once more ([`check_read`]) before the whole query is analyzed, and
//! stores the evolved schemas once that analysis succeeded (Spark's `ViewSyncSchemaToMetaStore`).

use std::sync::Arc;

use krishiv_common::config::{
    Config, ConfigValue, LEGACY_VIEW_SCHEMA_BINDING_MODE, LEGACY_VIEW_SCHEMA_COMPENSATION,
};
use krishiv_common::error::{
    CannotUpCastDatatype, ColumnAlreadyExists, IncompatibleViewSchemaChange, KrishivError,
};
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::{AttrId, BoundColumn, LogicalPlan, PlanNode, ReadSource};
use krishiv_sql::ast::SchemaBinding;
use krishiv_types::DataType;

use crate::store::{CatalogTable, StoreColumn, ViewSchemaMode};

/// `spark.sql.legacy.viewSchemaBindingMode` (default true).
pub(crate) fn binding_enabled(config: &Config) -> bool {
    config
        .get(&LEGACY_VIEW_SCHEMA_BINDING_MODE)
        .and_then(ConfigValue::as_bool)
        .unwrap_or(true)
}

/// `spark.sql.legacy.viewSchemaCompensation` (default true).
pub(crate) fn compensation(config: &Config) -> bool {
    config
        .get(&LEGACY_VIEW_SCHEMA_COMPENSATION)
        .and_then(ConfigValue::as_bool)
        .unwrap_or(true)
}

/// The mode `CREATE VIEW` stores (Spark's `visitSchemaBinding` and `visitCreateView`): the written one, `EVOLUTION`
/// with a column list being `TYPE EVOLUTION`; unwritten, `COMPENSATION` or `BINDING` by
/// `viewSchemaCompensation`; none at all while the feature is off (the parser refused a written one then).
pub(crate) fn create_mode(
    written: Option<SchemaBinding>,
    has_column_list: bool,
    config: &Config,
) -> Option<ViewSchemaMode> {
    if !binding_enabled(config) {
        return None;
    }
    Some(match written {
        None if compensation(config) => ViewSchemaMode::Compensation,
        None => ViewSchemaMode::Binding,
        Some(mode) => match from_ast(mode) {
            ViewSchemaMode::Evolution if has_column_list => ViewSchemaMode::TypeEvolution,
            mode => mode,
        },
    })
}

/// The stored mode of a written `WITH SCHEMA ...`.
pub(crate) fn from_ast(mode: SchemaBinding) -> ViewSchemaMode {
    match mode {
        SchemaBinding::Compensation => ViewSchemaMode::Compensation,
        SchemaBinding::TypeEvolution => ViewSchemaMode::TypeEvolution,
        SchemaBinding::Evolution => ViewSchemaMode::Evolution,
        // `SchemaBinding` is non-exhaustive; Spark's last case is `BINDING`.
        _ => ViewSchemaMode::Binding,
    }
}

/// The mode a view reads by (`CatalogTable.viewSchemaMode`): `None` (Spark's `SchemaUnsupported`) while
/// `viewSchemaBindingMode` is off, else the stored one, `BINDING` for a view stored without one.
pub(crate) fn effective_mode(
    stored: Option<ViewSchemaMode>,
    enabled: bool,
) -> Option<ViewSchemaMode> {
    enabled.then(|| stored.unwrap_or(ViewSchemaMode::Binding))
}

/// How a read projects the stored columns from the view query's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadProjection {
    /// By query name, cast to the stored type (an up-cast or an ANSI cast: an up-cast cannot fail).
    Cast,
    /// By query name, renamed to the stored name, in the query's type.
    Rename,
    /// The query's output as it is.
    Output,
}

/// The projection of a read by mode.
pub(crate) const fn projection(mode: Option<ViewSchemaMode>) -> ReadProjection {
    match mode {
        Some(ViewSchemaMode::TypeEvolution) => ReadProjection::Rename,
        Some(ViewSchemaMode::Evolution) => ReadProjection::Output,
        _ => ReadProjection::Cast,
    }
}

/// An inlined read of a stored view: what [`check_read`] needs.
#[derive(Debug, Clone)]
pub(crate) struct StoredRead {
    /// The view as the read found it.
    pub(crate) entry: Arc<CatalogTable>,
    /// The view's query, its own views inlined, before the read's projection.
    pub(crate) query: Arc<LogicalPlan>,
    /// The mode the read went by.
    pub(crate) mode: Option<ViewSchemaMode>,
}

/// Checks `read` against `analyzed`, the analysis of its query: the errors Spark raises resolving the view's
/// projection, and for the evolution modes the entry to store if the schema changed.
///
/// # Errors
/// `INCOMPATIBLE_VIEW_SCHEMA_CHANGE`, `CANNOT_UP_CAST_DATATYPE`, or `COLUMN_ALREADY_EXISTS` for an evolved schema
/// with a repeated name.
pub(crate) fn check_read(
    read: &StoredRead,
    analyzed: &AnalyzedPlan,
    case_sensitive: bool,
    compensation: bool,
) -> Result<Option<CatalogTable>, KrishivError> {
    let entry = &read.entry;
    let Some(view) = &entry.view else {
        return Ok(None);
    };
    let output = &analyzed.output;
    let fold = |n: &str| {
        if case_sensitive {
            n.to_owned()
        } else {
            n.to_lowercase()
        }
    };
    if read.mode == Some(ViewSchemaMode::Evolution) {
        let redo = output.len() != entry.columns.len()
            || output.iter().zip(&entry.columns).any(|(o, c)| {
                o.name != c.name || o.data_type != c.data_type || o.nullable != c.nullable
            });
        if !redo {
            return Ok(None);
        }
        let mut seen = std::collections::BTreeSet::new();
        for o in output {
            let key = fold(&o.name);
            if !seen.insert(key.clone()) {
                return Err(ColumnAlreadyExists {
                    column_name: format!("`{key}`"),
                }
                .into());
            }
        }
        let mut changed = (**entry).clone();
        changed.columns = output
            .iter()
            .map(|o| StoreColumn {
                name: o.name.clone(),
                data_type: o.data_type.clone(),
                nullable: o.nullable,
                default: None,
                comment: None,
            })
            .collect();
        if let Some(v) = &mut changed.view {
            v.query_columns = output.iter().map(|o| o.name.clone()).collect();
        }
        return Ok(Some(changed));
    }
    let names: Vec<String> = if view.query_columns.len() == entry.columns.len() {
        view.query_columns.clone()
    } else {
        entry.columns.iter().map(|c| c.name.clone()).collect()
    };
    // `GetViewColumnByNameAndOrdinal`: the n-th of the columns of that name, as many as at creation.
    let mut matched = Vec::with_capacity(names.len());
    let mut ordinals = std::collections::BTreeMap::<String, usize>::new();
    for name in &names {
        let key = fold(name);
        let expected = names.iter().filter(|n| fold(n) == key).count();
        let candidates: Vec<&BoundColumn> =
            output.iter().filter(|o| fold(&o.name) == key).collect();
        if candidates.len() != expected {
            return Err(IncompatibleViewSchemaChange {
                view_name: format!(
                    "`{}`.`{}`.`{}`",
                    crate::catalog::CATALOG,
                    entry.namespace,
                    entry.name
                ),
                col_name: name.clone(),
                expected_num: expected.to_string(),
                actual_cols: format!(
                    "[{}]",
                    candidates
                        .iter()
                        .map(|c| c.name.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                suggestion: view_ddl(entry, &names),
            }
            .into());
        }
        let ordinal = ordinals.entry(key).or_insert(0);
        matched.push(candidates[*ordinal]);
        *ordinal += 1;
    }
    let up_cast = match read.mode {
        Some(ViewSchemaMode::Binding) => true,
        None => !compensation,
        _ => false,
    };
    if up_cast {
        for (column, stored) in matched.iter().zip(&entry.columns) {
            if !can_up_cast(&column.data_type, &stored.data_type) {
                return Err(CannotUpCastDatatype {
                    expression: attribute_sql(analyzed.plan.root(), column.id, &column.name, &view.namespace),
                    source_type: format!("\"{}\"", column.data_type.sql()),
                    target_type: format!("\"{}\"", stored.data_type.sql()),
                    details: "The type path of the target object is:\n\nYou can either add an explicit cast to \
                              the input data or choose a higher precision type of the field in the target object"
                        .to_owned(),
                }
                .into());
            }
        }
    }
    if read.mode != Some(ViewSchemaMode::TypeEvolution) {
        return Ok(None);
    }
    let redo = matched
        .iter()
        .zip(&entry.columns)
        .any(|(o, c)| o.data_type != c.data_type || o.nullable != c.nullable);
    if !redo {
        return Ok(None);
    }
    let mut changed = (**entry).clone();
    for (column, o) in changed.columns.iter_mut().zip(&matched) {
        column.data_type = o.data_type.clone();
        column.nullable = o.nullable;
    }
    Ok(Some(changed))
}

/// Spark's `buildViewDDL`: the statement that would recreate the view over its current query, the column list
/// written only when the stored names are not the query's.
fn view_ddl(entry: &CatalogTable, query_names: &[String]) -> String {
    let sql = entry.view.as_ref().map_or("", |v| v.sql.as_str());
    let names: Vec<&str> = entry.columns.iter().map(|c| c.name.as_str()).collect();
    let columns = if names
        .iter()
        .copied()
        .eq(query_names.iter().map(String::as_str))
    {
        " ".to_owned()
    } else {
        format!(" ({}) ", names.join(", "))
    };
    format!(
        "CREATE OR REPLACE VIEW {}.{}.{}{columns}AS {sql}",
        crate::catalog::CATALOG,
        entry.namespace,
        entry.name
    )
}

/// The attribute `id` (named `name`) as Spark's `Attribute.sql` prints it: qualified by the table or subquery it
/// was read from (a catalog table's qualifier is `spark_catalog.namespace.table`), bare when it was computed.
fn attribute_sql(root: &Arc<LogicalPlan>, id: AttrId, name: &str, namespace: &str) -> String {
    let column = crate::introspection::quote_if_needed(name);
    match qualifier(root, id, namespace) {
        Some(parts) => {
            let mut out: Vec<String> = parts
                .iter()
                .map(|p| crate::introspection::quote_if_needed(p))
                .collect();
            out.push(column);
            out.join(".")
        }
        None => column,
    }
}

/// The qualifier of attribute `id` in `root`: the first subquery alias or table scan (in pre-order) that outputs
/// it. Iterative, so a deep plan cannot overflow the stack.
fn qualifier(root: &Arc<LogicalPlan>, id: AttrId, namespace: &str) -> Option<Vec<String>> {
    let outputs = |p: &LogicalPlan| {
        p.resolved_output()
            .is_some_and(|o| o.iter().any(|c| c.id == id))
    };
    let mut stack = vec![Arc::clone(root)];
    let mut seen = std::collections::HashSet::new();
    while let Some(p) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        match &p.node {
            PlanNode::SubqueryAlias { input, alias, .. } if outputs(&p) => {
                // A table read keeps its full name.
                if let PlanNode::Scan(scan) = &input.node
                    && let Some(parts) = table_parts(&scan.source, namespace)
                {
                    return Some(parts);
                }
                return Some(vec![alias.clone()]);
            }
            PlanNode::Scan(scan) if scan.output.iter().any(|c| c.id == id) => {
                return table_parts(&scan.source, namespace);
            }
            _ => {}
        }
        let mut children = Vec::new();
        p.for_each_child(&mut |c| children.push(Arc::clone(c)));
        // Pre-order, first child first.
        stack.extend(children.into_iter().rev());
    }
    None
}

/// The qualifier of a catalog table read: `spark_catalog`, its namespace and its name.
fn table_parts(source: &ReadSource, namespace: &str) -> Option<Vec<String>> {
    let ReadSource::Table { name, .. } = source else {
        return None;
    };
    let catalog = crate::catalog::CATALOG.to_owned();
    match name.as_slice() {
        [table] => Some(vec![catalog, namespace.to_owned(), table.clone()]),
        [ns, table] => Some(vec![catalog, ns.clone(), table.clone()]),
        [c, ns, table] => Some(vec![c.clone(), ns.clone(), table.clone()]),
        _ => None,
    }
}

/// Spark's `UpCastRule.canUpCast`: `from` converts to `to` with no truncation, precision loss or runtime failure.
pub(crate) fn can_up_cast(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    // `numericPrecedence`: the integral and floating types widen in this order.
    let rank = |t: &T| match t {
        T::Byte => Some(0),
        T::Short => Some(1),
        T::Integer => Some(2),
        T::Long => Some(3),
        T::Float => Some(4),
        T::Double => Some(5),
        _ => None,
    };
    // `DecimalType.forType` of an integral type.
    let integral = |t: &T| match t {
        T::Byte => Some((3u8, 0u8)),
        T::Short => Some((5, 0)),
        T::Integer => Some((10, 0)),
        T::Long => Some((20, 0)),
        _ => None,
    };
    let fits = |from_null: bool, to_null: bool| !from_null || to_null;
    match (from, to) {
        _ if from == to => true,
        // `to.isWiderThan(from)`.
        (T::Decimal(a), T::Decimal(b)) => {
            b.precision() - b.scale() >= a.precision() - a.scale() && b.scale() >= a.scale()
        }
        (f, T::Decimal(d)) if integral(f).is_some() => {
            integral(f).is_some_and(|(p, s)| d.precision() - d.scale() >= p - s && d.scale() >= s)
        }
        // `from.isTighterThan(to)`.
        (T::Decimal(d), t) if integral(t).is_some() => {
            integral(t).is_some_and(|(p, s)| d.precision() - d.scale() <= p - s && d.scale() <= s)
        }
        (f, t) if rank(f).zip(rank(t)).is_some_and(|(f, t)| f < t) => true,
        (T::Date | T::TimestampNtz | T::Long, T::Timestamp)
        | (T::Date | T::Timestamp, T::TimestampNtz)
        | (T::Timestamp, T::Long)
        | (T::Null, _) => true,
        // An atomic type to a plain string.
        (f, T::String) => !f.is_nested(),
        (T::Array(a), T::Array(b)) => {
            fits(a.contains_null(), b.contains_null())
                && can_up_cast(a.element_type(), b.element_type())
        }
        (T::Map(a), T::Map(b)) => {
            fits(a.value_contains_null(), b.value_contains_null())
                && can_up_cast(a.key_type(), b.key_type())
                && can_up_cast(a.value_type(), b.value_type())
        }
        (T::Struct(a), T::Struct(b)) => {
            a.fields().len() == b.fields().len()
                && a.fields().iter().zip(b.fields()).all(|(f, t)| {
                    fits(f.nullable(), t.nullable()) && can_up_cast(f.data_type(), t.data_type())
                })
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_types::DecimalType;

    fn dec(p: u32, s: u32) -> DataType {
        DataType::Decimal(DecimalType::try_new(p, s).unwrap())
    }

    #[test]
    fn up_casts_follow_sparks_rule() {
        use DataType as T;
        assert!(can_up_cast(&T::Integer, &T::Long));
        assert!(can_up_cast(&T::Short, &T::Integer));
        assert!(!can_up_cast(&T::Long, &T::Integer));
        assert!(!can_up_cast(&T::String, &T::Integer));
        assert!(can_up_cast(&T::Integer, &T::String));
        assert!(can_up_cast(&T::Long, &T::Double));
        assert!(!can_up_cast(&T::Double, &T::Float));
        assert!(can_up_cast(&T::Integer, &dec(10, 0)));
        assert!(!can_up_cast(&T::Integer, &dec(9, 0)));
        assert!(can_up_cast(&dec(5, 0), &T::Integer));
        assert!(!can_up_cast(&dec(5, 1), &T::Integer));
        assert!(can_up_cast(&dec(5, 2), &dec(6, 2)));
        assert!(!can_up_cast(&dec(5, 2), &dec(5, 1)));
        assert!(can_up_cast(&T::Date, &T::Timestamp));
        assert!(!can_up_cast(&T::Timestamp, &T::Date));
        assert!(can_up_cast(&T::Null, &T::Integer));
    }

    #[test]
    fn nested_types_up_cast_element_wise() {
        use DataType as T;
        use krishiv_types::{ArrayType, MapType, StructField, StructType};
        let array = |t: T, n| T::Array(std::sync::Arc::new(ArrayType::new(t, n)));
        assert!(can_up_cast(
            &array(T::Integer, false),
            &array(T::Long, true)
        ));
        assert!(!can_up_cast(
            &array(T::Integer, true),
            &array(T::Long, false)
        ));
        assert!(!can_up_cast(
            &array(T::Long, true),
            &array(T::Integer, true)
        ));
        let map = |k: T, v: T, n| T::Map(std::sync::Arc::new(MapType::new(k, v, n)));
        assert!(can_up_cast(
            &map(T::Integer, T::Integer, true),
            &map(T::Long, T::Long, true)
        ));
        assert!(!can_up_cast(
            &map(T::Long, T::Integer, true),
            &map(T::Integer, T::Integer, true)
        ));
        assert!(!can_up_cast(
            &map(T::Integer, T::Integer, true),
            &map(T::Integer, T::Integer, false)
        ));
        let strukt = |fields: Vec<(&str, T)>| {
            T::Struct(StructType::new(
                fields
                    .into_iter()
                    .map(|(n, t)| StructField::new(n, t, true)),
            ))
        };
        assert!(can_up_cast(
            &strukt(vec![("a", T::Integer)]),
            &strukt(vec![("a", T::Long)])
        ));
        assert!(!can_up_cast(
            &strukt(vec![("a", T::Integer)]),
            &strukt(vec![("a", T::Integer), ("b", T::Integer)])
        ));
        assert!(!can_up_cast(&array(T::Integer, true), &T::String));
    }

    #[test]
    fn a_column_list_makes_evolution_type_evolution() {
        let config = Config::new();
        assert_eq!(
            create_mode(Some(SchemaBinding::Evolution), true, &config),
            Some(ViewSchemaMode::TypeEvolution)
        );
        assert_eq!(
            create_mode(Some(SchemaBinding::Evolution), false, &config),
            Some(ViewSchemaMode::Evolution)
        );
        assert_eq!(
            create_mode(None, false, &config),
            Some(ViewSchemaMode::Compensation)
        );
    }
}
