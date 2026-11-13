// SPDX-License-Identifier: Apache-2.0
//! Plan-building helpers: carrier casts, projections, naming and error constructors.

use super::*;

/// The largest attribute id in `root` (expressions and subqueries
/// included), to number the planner's own columns above it.
pub(super) fn max_id(root: &LogicalPlan) -> u64 {
    let mut max = 0;
    let mut stack = vec![root];
    let mut seen = std::collections::HashSet::new();
    while let Some(p) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(p)) {
            continue;
        }
        if let Some(out) = p.resolved_output() {
            max = out.iter().map(|c| c.id.0).fold(max, u64::max);
        }
        if let PlanNode::Shift { offset, input } = &p.node
            && let Some(out) = input.resolved_output()
        {
            max = out
                .iter()
                .map(|c| c.id.0.saturating_add(*offset))
                .fold(max, u64::max);
        }
        p.for_each_child(&mut |c| stack.push(c));
        p.for_each_expr(&mut |e| {
            let mut exprs = vec![e.as_ref()];
            while let Some(e) = exprs.pop() {
                if let ExprNode::Subquery { plan, .. } = &e.node {
                    stack.push(plan);
                }
                e.for_each_child(&mut |c| exprs.push(c));
            }
        });
    }
    max
}

/// The Arrow type each column of a union has in every input, and whether
/// every branch converts to it: the type the inputs all store it in (kept
/// as is), or, where they differ, the canonical Arrow type of its Spark
/// type — except a `string` column with a binary-safe branch (non-UTF-8
/// bytes in a binary carrier), which rides on `BinaryView`: the canonical
/// `Utf8View` would route the branch through the validating `STRING` cast
/// and raise `INVALID_UTF8_STRING`. An input column of another Spark type
/// than the union's is an error.
pub(super) fn union_carriers(
    out: &[BoundColumn],
    branches: &[(Vec<BoundColumn>, Planned)],
) -> Result<Vec<(arrow_schema::DataType, bool)>> {
    let mut carriers = Vec::with_capacity(out.len());
    for (i, o) in out.iter().enumerate() {
        let mut stored = Vec::with_capacity(branches.len());
        for (cols, planned) in branches {
            let c = cols
                .get(i)
                .ok_or_else(|| internal("union inputs of different widths"))?;
            stored.push((c, planned.stored(planned.column(c.id)?)?));
        }
        let first = stored.first().map(|(_, t)| *t);
        let (carrier, mixed) = if stored.iter().all(|(_, t)| Some(*t) == first) {
            (first.cloned(), false)
        } else if stored.iter().all(|(c, _)| c.data_type == o.data_type) {
            let binary = is_binary_safe_string(&o.data_type, stored.iter().map(|(_, t)| *t));
            (
                Some(if binary {
                    arrow_schema::DataType::BinaryView
                } else {
                    o.data_type.to_arrow()
                }),
                binary,
            )
        } else {
            return Err(internal("union inputs of different physical types"));
        };
        carriers.push((
            carrier.ok_or_else(|| internal("a union without inputs"))?,
            mixed,
        ));
    }
    Ok(carriers)
}

/// Whether the `string` column stored as `stored` needs a binary union
/// carrier: some branch holds it in a binary Arrow type, whose bytes are not
/// necessarily valid UTF-8, so the validating `STRING` cast is not a way to
/// meet the branches.
pub(super) fn is_binary_safe_string<'a>(
    spark: &krishiv_types::DataType,
    mut stored: impl Iterator<Item = &'a arrow_schema::DataType>,
) -> bool {
    *spark == krishiv_types::DataType::String
        && stored.any(|t| {
            matches!(
                t,
                arrow_schema::DataType::BinaryView
                    | arrow_schema::DataType::Binary
                    | arrow_schema::DataType::LargeBinary
            )
        })
}

/// The Spark type a union branch casts to so its column lands on `carrier`:
/// the column's own Spark type — unless the carrier is a binary-safe
/// `string` union's `BinaryView`, whose Spark type is `binary`, so the cast
/// is a byte-preserving reinterpretation, never a validating string cast.
pub(super) fn carrier_cast_target(
    spark: &krishiv_types::DataType,
    carrier: &arrow_schema::DataType,
) -> krishiv_types::DataType {
    if *spark == krishiv_types::DataType::String && *carrier == arrow_schema::DataType::BinaryView {
        krishiv_types::DataType::Binary
    } else {
        spark.clone()
    }
}

/// The id of a plain (or aliased plain) column expression.
pub(crate) fn column_of(e: &Expr) -> Option<AttrId> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => column_of(expr),
        _ => None,
    }
}

/// Exactly `output`'s columns of `planned`, in order, under their names.
pub(crate) fn select(planned: Planned, output: &[BoundColumn], name: &str) -> Result<Planned> {
    let ids: Vec<AttrId> = output.iter().map(|c| c.id).collect();
    let named = planned.schema.fields().len() == output.len()
        && planned
            .schema
            .fields()
            .iter()
            .zip(output)
            .all(|(f, c)| f.name() == &c.name);
    if planned.layout == ids && named {
        return Ok(planned);
    }
    let mut items = Vec::with_capacity(output.len());
    let mut columns_only = Vec::with_capacity(output.len());
    for c in output {
        let index = planned.column(c.id)?;
        items.push(ProjectItemPlan::Column {
            index,
            name: Some(c.name.clone()),
        });
        columns_only.push(ProjectItem::aliased(index, &c.name));
    }
    // Columns only: the schema needs no expression environment.
    let schema = Project::try_new(&planned.schema, columns_only)?.schema();
    let detail = columns(output);
    let input_layout = planned.layout.clone();
    Ok(planned.wrap(
        |input| PhysNode::Project {
            input,
            layout: input_layout,
            items,
            nullable: None,
            subqueries: Vec::new(),
        },
        schema,
        ids,
        (name, detail),
    ))
}

/// `range(start, end, step, numSlices)`: one `id BIGINT NOT NULL` column.
pub(super) fn range_node(
    start: i64,
    end: i64,
    step: i64,
    num_slices: Option<u32>,
    output: &krishiv_plan::BoundColumn,
) -> Planned {
    use arrow_schema::{DataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    Planned {
        ordering: Vec::new(),
        schema,
        node: PhysNode::Range(crate::physical::RangeNode {
            start,
            end,
            step,
            num_slices,
            output: output.id,
        }),
        layout: vec![output.id],
        explain: ExplainNode::leaf(
            "Range",
            &match num_slices {
                Some(n) => format!("[{start}, {end}, step {step}, slices {n}]"),
                None => format!("[{start}, {end}, step {step}]"),
            },
        ),
        sites: Vec::new(),
        placement: Placement::Unknown,
    }
}

/// One row without columns (`SELECT` without `FROM`).
pub(super) fn one_row() -> Planned {
    Planned {
        ordering: Vec::new(),
        schema: Arc::new(Schema::empty()),
        node: PhysNode::OneRow,
        layout: Vec::new(),
        explain: ExplainNode::leaf("OneRowRelation", ""),
        sites: Vec::new(),
        placement: Placement::Single,
    }
}

/// The schema of `l`'s columns then `r`'s.
pub(crate) fn concat_schemas(l: &SchemaRef, r: &SchemaRef) -> SchemaRef {
    let fields: Vec<_> = l.fields().iter().chain(r.fields()).cloned().collect();
    Arc::new(Schema::new(fields))
}

/// A `LIMIT` or `OFFSET`'s literal row count.
pub(super) fn limit_value(e: &Expr) -> Result<u64> {
    match &e.node {
        ExprNode::Literal(Literal::Integer(v)) => u64::try_from(*v)
            .map_err(|_| internal("a negative LIMIT or OFFSET reached the planner")),
        _ => Err(eval::unsupported_expr(e)),
    }
}

pub(crate) fn source_name(scan: &krishiv_plan::Scan) -> String {
    match &scan.source {
        krishiv_plan::ReadSource::Table { name, .. } => name.join("."),
        krishiv_plan::ReadSource::Files { format, .. } => format!("{format} files"),
        _ => "source".to_owned(),
    }
}

pub(crate) fn columns(cols: &[BoundColumn]) -> String {
    cols.iter()
        .map(|c| format!("{}#{}", c.name, c.id.0))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn list(exprs: &[Arc<Expr>]) -> String {
    format!(
        "[{}]",
        exprs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(crate) fn orders(order: &[SortOrder]) -> String {
    let text = order.iter().map(|o| {
        let dir = match o.direction {
            SortDirection::Ascending => "ASC",
            SortDirection::Descending => "DESC",
        };
        let nulls = match o.nulls {
            NullOrdering::NullsFirst => "NULLS FIRST",
            NullOrdering::NullsLast => "NULLS LAST",
        };
        format!("{} {dir} {nulls}", o.expr)
    });
    format!("[{}]", text.collect::<Vec<_>>().join(", "))
}

pub(crate) fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

pub(crate) fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

/// The bounds of a loop: `MAX RECURSION LEVEL` over
/// `spark.sql.cteRecursionLevelLimit`, and `spark.sql.cteRecursionRowLimit`
/// (a negative value: none).
pub(super) fn loop_limits(
    config: &krishiv_common::config::Config,
    max_level: Option<u32>,
) -> LoopLimits {
    use krishiv_common::config::{CTE_RECURSION_LEVEL_LIMIT, CTE_RECURSION_ROW_LIMIT, ConfigValue};
    let conf = |entry| config.get(entry).and_then(ConfigValue::as_int);
    let bound = |v: Option<i64>, default: i64| match v.unwrap_or(default) {
        n if n < 0 => i64::MAX,
        n => n,
    };
    LoopLimits {
        level: max_level.unwrap_or_else(|| {
            u32::try_from(bound(conf(&CTE_RECURSION_LEVEL_LIMIT), 100)).unwrap_or(u32::MAX)
        }),
        rows: u64::try_from(bound(conf(&CTE_RECURSION_ROW_LIMIT), 1_000_000)).unwrap_or(u64::MAX),
        pushed_limit: None,
    }
}
