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
use krishiv_common::error::{CannotUpCastDatatype, IncompatibleViewSchemaChange, KrishivError};
use krishiv_common::names::check_column_name_duplication;
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::{
    AttrId, BoundColumn, ColumnRef, Expr, ExprNode, LogicalPlan, Origin, PlanNode, ReadSource,
};
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

/// The names a view read finds its stored columns by in `input`, the inlined view query: the query's output names
/// (`query`). A query that repeats a name (ignoring case) is renamed by position to names of its own
/// ([`repeats_names`]), so the stored types and comments still apply; once the query is analyzed, [`check_read`]
/// finds each stored column by name and ordinal (Spark's `GetViewColumnByNameAndOrdinal`) and [`apply_renames`]
/// replaces that rename by one of the query's current width, so a base table whose columns changed order or count
/// reads as Spark reads it. Not yet for the views a SQL function's body reads (`sql_functions.rs` inlines them
/// without the checks): those are still read by position (open item in `ops/queue-opus-b2.md`).
pub(crate) fn read_names(
    input: Arc<LogicalPlan>,
    query: &[String],
    alias: &str,
    origin: Origin,
) -> (Arc<LogicalPlan>, Vec<String>) {
    if !repeats_names(query) {
        return (input, query.to_vec());
    }
    let names: Vec<String> = (0..query.len()).map(read_name).collect();
    (
        rename_by_position(input, alias, names.clone(), origin),
        names,
    )
}

/// Whether `query` repeats a name, ignoring case: a view read then renames the query's columns ([`read_names`]).
pub(crate) fn repeats_names(query: &[String]) -> bool {
    let mut seen = std::collections::HashSet::new();
    !query.iter().all(|c| seen.insert(c.to_lowercase()))
}

/// The name a renamed view read gives the query column its `i`-th stored column is read from.
fn read_name(i: usize) -> String {
    format!("__krishiv_view_column_{i}")
}

/// The rename of a repeated-name view read, once its query is analyzed: the `SubqueryAlias` over `query` (the
/// inlined query, by identity) takes `columns`, one name per column the query outputs now.
#[derive(Debug, Clone)]
pub(crate) struct ReadRename {
    query: Arc<LogicalPlan>,
    columns: Vec<String>,
}

/// The rename of `read` when its query output (`width` columns) holds the `i`-th stored column at `positions[i]`:
/// those take the names the read's projection reads, the others names it never reads.
fn read_rename(read: &StoredRead, positions: &[usize], width: usize) -> ReadRename {
    let mut columns: Vec<String> = (0..width)
        .map(|i| format!("__krishiv_view_unread_{i}"))
        .collect();
    for (i, &p) in positions.iter().enumerate() {
        if let Some(c) = columns.get_mut(p) {
            *c = read_name(i);
        }
    }
    ReadRename {
        query: Arc::clone(&read.query),
        columns,
    }
}

/// `plan` with the renames of `renames` in place: each `SubqueryAlias` whose input is a rename's query (the same
/// `Arc`, as inlined) takes the rename's columns; subquery plans included.
pub(crate) fn apply_renames(plan: &Arc<LogicalPlan>, renames: &[ReadRename]) -> Arc<LogicalPlan> {
    if renames.is_empty() {
        return Arc::clone(plan);
    }
    let found = match &plan.node {
        PlanNode::SubqueryAlias { input, .. } => {
            renames.iter().find(|r| Arc::ptr_eq(&r.query, input))
        }
        _ => None,
    };
    let rebuilt: Result<_, std::convert::Infallible> =
        Arc::clone(plan).map_children(|child| Ok(apply_renames(&child, renames)));
    let Ok(rebuilt) = rebuilt.and_then(|p| p.map_expressions(|e| Ok(rename_expr(&e, renames))));
    match (found, &rebuilt.node) {
        (Some(rename), PlanNode::SubqueryAlias { input, alias, .. }) => Arc::new(
            LogicalPlan::new(PlanNode::SubqueryAlias {
                input: Arc::clone(input),
                alias: alias.clone(),
                columns: rename.columns.clone(),
            })
            .with_origin(rebuilt.origin),
        ),
        _ => rebuilt,
    }
}

/// `expr` with [`apply_renames`] applied to the subquery plans inside it.
fn rename_expr(expr: &Arc<Expr>, renames: &[ReadRename]) -> Arc<Expr> {
    let mapped: Result<_, std::convert::Infallible> =
        Arc::clone(expr).map_children(|child| Ok(rename_expr(&child, renames)));
    let Ok(expr) = mapped;
    if let ExprNode::Subquery { kind, plan } = &expr.node {
        let renamed = apply_renames(plan, renames);
        if !Arc::ptr_eq(&renamed, plan) {
            return Arc::new(Expr {
                node: ExprNode::Subquery {
                    kind: kind.clone(),
                    plan: renamed,
                },
                origin: expr.origin,
            });
        }
    }
    expr
}

/// `input` with its columns renamed by position to `names` (a `SubqueryAlias` with a column list).
pub(crate) fn rename_by_position(
    input: Arc<LogicalPlan>,
    alias: &str,
    names: Vec<String>,
    origin: Origin,
) -> Arc<LogicalPlan> {
    Arc::new(
        LogicalPlan::new(PlanNode::SubqueryAlias {
            input,
            alias: alias.to_owned(),
            columns: names,
        })
        .with_origin(origin),
    )
}

/// One column of a view read: the input column `name` (read as one part: `col(name)` would split a dot or read a
/// trailing `*`), ANSI-cast to `cast` when given. With `comment` (`Some`), the field metadata it holds (a comment,
/// or none) replaces what the column under it has and the column is named `output`, as Spark's view read alias with
/// `explicitMetadata` does; without, the column is read as it is.
pub(crate) fn read_column(
    name: &str,
    cast: Option<DataType>,
    comment: Option<Option<String>>,
    output: &str,
) -> Arc<Expr> {
    let column = crate::column::Column::new(Expr::new(ExprNode::Column(ColumnRef::new(vec![
        name.to_owned(),
    ]))));
    let typed = match cast {
        Some(t) => column.cast_ansi(t),
        None => column,
    };
    match comment {
        Some(comment) => {
            let field = crate::column::Column::new(Expr::new(ExprNode::StoredField {
                expr: Arc::new(typed.expr),
                comment,
            }));
            Arc::new(field.alias(output).expr)
        }
        None => Arc::new(typed.expr),
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
    /// Whether the read renamed its query's columns by position ([`read_names`]), to be replaced by a rename by
    /// name and ordinal once the query is analyzed.
    pub(crate) renamed: bool,
}

/// What `Session::check_stored_views` returns: the evolved view entries to store (as found, as changed) and the
/// renames of repeated-name view reads.
pub(crate) type StoredViewChecks = (Vec<(Arc<CatalogTable>, CatalogTable)>, Vec<ReadRename>);

/// What [`check_read`] found: the entry to store for the evolution modes, and the rename of a read that renamed
/// its query ([`StoredRead::renamed`]).
#[derive(Debug, Default)]
pub(crate) struct ReadCheck {
    /// The evolved entry, when its schema changed.
    pub(crate) changed: Option<CatalogTable>,
    /// The rename by name and ordinal.
    pub(crate) rename: Option<ReadRename>,
}

/// Checks `read` against `analyzed`, the analysis of its query: the errors Spark raises resolving the view's
/// projection, for the evolution modes the entry to store if the schema changed, and for a renamed read the rename
/// that finds each stored column by name and ordinal.
///
/// # Errors
/// `INCOMPATIBLE_VIEW_SCHEMA_CHANGE`, `CANNOT_UP_CAST_DATATYPE`, or `COLUMN_ALREADY_EXISTS` for an evolved schema
/// with a repeated name.
pub(crate) fn check_read(
    read: &StoredRead,
    analyzed: &AnalyzedPlan,
    case_sensitive: bool,
    compensation: bool,
) -> Result<ReadCheck, KrishivError> {
    let entry = &read.entry;
    let Some(view) = &entry.view else {
        return Ok(ReadCheck::default());
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
        // A changed column comment is a schema change too (Spark compares the fields with their metadata).
        let comment = |i: usize| analyzed.output_comments.get(i).cloned().flatten();
        let redo = output.len() != entry.columns.len()
            || output
                .iter()
                .zip(&entry.columns)
                .enumerate()
                .any(|(i, (o, c))| {
                    o.name != c.name
                        || o.data_type != c.data_type
                        || o.nullable != c.nullable
                        || comment(i) != c.comment
                });
        if !redo {
            return Ok(ReadCheck::default());
        }
        check_column_name_duplication(output.iter().map(|o| o.name.as_str()), case_sensitive)?;
        let mut changed = (**entry).clone();
        changed.columns = output
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let mut c = StoreColumn::plain(o.name.clone(), o.data_type.clone(), o.nullable);
                c.char_varchar = analyzed.output_declared.get(i).cloned().flatten();
                c.comment = analyzed.output_comments.get(i).cloned().flatten();
                c
            })
            .collect();
        if let Some(v) = &mut changed.view {
            v.query_columns = output.iter().map(|o| o.name.clone()).collect();
        }
        return Ok(ReadCheck {
            changed: Some(changed),
            rename: None,
        });
    }
    let names: Vec<String> = if view.query_columns.len() == entry.columns.len() {
        view.query_columns.clone()
    } else {
        entry.columns.iter().map(|c| c.name.clone()).collect()
    };
    // `GetViewColumnByNameAndOrdinal`: the n-th of the columns of that name, as many as at creation.
    let mut matched = Vec::with_capacity(names.len());
    let mut positions = Vec::with_capacity(names.len());
    let mut ordinals = std::collections::BTreeMap::<String, usize>::new();
    for name in &names {
        let key = fold(name);
        let expected = names.iter().filter(|n| fold(n) == key).count();
        let candidates: Vec<(usize, &BoundColumn)> = output
            .iter()
            .enumerate()
            .filter(|(_, o)| fold(&o.name) == key)
            .collect();
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
                        .map(|(_, c)| c.name.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                suggestion: view_ddl(entry, &names),
            }
            .into());
        }
        let ordinal = ordinals.entry(key).or_insert(0);
        // Unreachable: `candidates` holds `expected` columns and `ordinal` counts this name's earlier ones.
        let Some(&(position, column)) = candidates.get(*ordinal) else {
            return Err(krishiv_common::error::InternalError {
                message: format!("view column `{name}` has no ordinal {ordinal}"),
            }
            .into());
        };
        matched.push(column);
        positions.push(position);
        *ordinal += 1;
    }
    let rename = read
        .renamed
        .then(|| read_rename(read, &positions, output.len()));
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
        return Ok(ReadCheck {
            changed: None,
            rename,
        });
    }
    let redo = matched
        .iter()
        .zip(&entry.columns)
        .any(|(o, c)| o.data_type != c.data_type || o.nullable != c.nullable);
    if !redo {
        return Ok(ReadCheck {
            changed: None,
            rename,
        });
    }
    let mut changed = (**entry).clone();
    for (column, o) in changed.columns.iter_mut().zip(&matched) {
        column.data_type = o.data_type.clone();
        column.nullable = o.nullable;
    }
    Ok(ReadCheck {
        changed: Some(changed),
        rename,
    })
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
