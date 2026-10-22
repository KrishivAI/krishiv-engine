// SPDX-License-Identifier: Apache-2.0
//! `krishiv.plan.v1` to `PhysicalPlan`. The bytes are untrusted: sizes are
//! bounded first, every reference must point to an earlier entry, every
//! entry may be used once (nodes) and counts must agree. The tables are
//! flat, so nothing here recurses and protobuf nesting stays shallow.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::secret::{OptionValue, SecretValue};
use krishiv_exec::JoinType;
use krishiv_exec::aggregate::{AggInput, AggregateExpr, AggregateFunction, AggregateOutput};
use krishiv_exec::physical::{
    AggregatePlan, ExprEnv, HashJoinPlan, JoinFilterPlan, NestedLoopPlan, PhysNode, PhysicalPlan,
    ProjectItemPlan, Residual, ScanFilter, ScanId, ScanNode, ScanRuntimeFilter, SubqueryId,
    SubqueryPlan, SubqueryValueKind,
};
use krishiv_exec::planner::ExplainNode;
use krishiv_exec::{RowStageSettings, SortKey};
use krishiv_io::{ColumnPath, FilterId, KeyFilterOptions};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, CastMode, DayTimeField, Expr, ExprNode, FunctionId, FunctionKind,
    Literal, Origin, PlanId, ReadSource, ResolvedExpr, ResolvedFunction, Scan, SourceId, Span,
    TableVersion, YearMonthField,
};
use krishiv_types::{DataType, DecimalType};

use crate::arrow::{schema_from, type_from};
use crate::error::{CodecError, Result, corrupt};
use crate::proto as pb;
use crate::{FORMAT_VERSION, Limits};

/// Decodes `krishiv.plan.v1` bytes under the default [`Limits`].
///
/// # Errors
/// [`CodecError::Corrupt`] for bytes that are not a valid plan,
/// [`CodecError::TooNew`] for a plan that needs a newer reader,
/// [`CodecError::Limit`] for a plan beyond the limits.
pub fn decode(bytes: &[u8]) -> Result<PhysicalPlan> {
    decode_with(bytes, &Limits::default())
}

/// [`decode`] under `limits`.
///
/// # Errors
/// As [`decode`].
pub fn decode_with(bytes: &[u8], limits: &Limits) -> Result<PhysicalPlan> {
    use prost::Message as _;
    if bytes.len() > limits.max_bytes {
        return Err(CodecError::Limit(format!(
            "{} bytes (at most {})",
            bytes.len(),
            limits.max_bytes
        )));
    }
    // The messages are flat tables: protobuf nesting stays under a dozen
    // levels, whatever the plan's depth, so prost's recursion is bounded.
    #[expect(
        clippy::disallowed_methods,
        reason = "the plan format has constant nesting depth (flat tables)"
    )]
    let message = pb::PhysicalPlan::decode(bytes).map_err(|_| corrupt("malformed protobuf"))?;
    from_message(&message, limits)
}

/// [`decode_with`] from a parsed message.
///
/// # Errors
/// As [`decode`].
pub fn from_message(m: &pb::PhysicalPlan, limits: &Limits) -> Result<PhysicalPlan> {
    if m.format_version == 0 {
        return Err(corrupt("no format version"));
    }
    if m.min_reader_version > FORMAT_VERSION {
        return Err(CodecError::TooNew {
            required: m.min_reader_version,
            supported: FORMAT_VERSION,
        });
    }
    for (what, n, max) in [
        ("nodes", m.nodes.len(), limits.max_nodes),
        ("explain nodes", m.explain.len(), limits.max_nodes),
        ("expressions", m.exprs.len(), limits.max_exprs),
    ] {
        if n > max {
            return Err(CodecError::Limit(format!("{n} {what} (at most {max})")));
        }
    }
    let mut d = Decoder {
        exprs: Vec::with_capacity(m.exprs.len()),
        expr_depths: Vec::with_capacity(m.exprs.len()),
        slots: Vec::with_capacity(m.nodes.len()),
        depths: Vec::with_capacity(m.nodes.len()),
        child_depth: 0,
        max_depth: limits.max_depth,
    };
    for e in &m.exprs {
        let expr = d.expr(e)?;
        // `d.expr` checked every reference, so each child has a depth.
        let depth = 1 + expr_children(e)
            .map(|c| d.expr_depths[c as usize])
            .max()
            .unwrap_or(0);
        if depth > d.max_depth {
            return Err(CodecError::Limit(format!(
                "expression deeper than {} nodes",
                d.max_depth
            )));
        }
        d.expr_depths.push(depth);
        d.exprs.push(Arc::new(expr));
    }
    for n in &m.nodes {
        let node = d.node(n)?;
        d.slots.push(Some(node));
        let depth = d.child_depth + 1;
        if depth > d.max_depth {
            return Err(CodecError::Limit(format!(
                "plan deeper than {} nodes",
                d.max_depth
            )));
        }
        d.depths.push(depth);
        d.child_depth = 0;
    }
    let root = d.take(m.root, d.slots.len())?;
    if d.slots.iter().any(Option::is_some) {
        return Err(corrupt("a node no other node or the root uses"));
    }
    let explain = explain(&m.explain, m.explain_root, limits.max_depth)?;
    let env = m
        .env
        .as_ref()
        .ok_or_else(|| corrupt("no expression environment"))?;
    let env = ExprEnv {
        ansi: env.ansi,
        zone: env.zone,
        query_start_micros: env.query_start_micros,
        user: Arc::from(env.user.as_str()),
    };
    Ok(
        PhysicalPlan::new(root, schema_from(&m.schema)?, explain, env)
            .with_warnings(m.warnings.clone()),
    )
}

fn explain(table: &[pb::ExplainNode], root: u32, max_depth: usize) -> Result<ExplainNode> {
    let mut slots: Vec<Option<ExplainNode>> = Vec::with_capacity(table.len());
    let mut depths: Vec<usize> = Vec::with_capacity(table.len());
    for m in table {
        let mut children = Vec::with_capacity(m.children.len());
        let mut depth = 0;
        for c in &m.children {
            let at = slot(*c, slots.len())?;
            children.push(
                slots[at]
                    .take()
                    .ok_or_else(|| corrupt("an explain node used twice"))?,
            );
            depth = depth.max(depths[at]);
        }
        if depth + 1 > max_depth {
            return Err(CodecError::Limit(format!(
                "explain tree deeper than {max_depth} nodes"
            )));
        }
        depths.push(depth + 1);
        slots.push(Some(ExplainNode::new(&m.name, &m.detail, children)));
    }
    let at = slot(root, slots.len())?;
    let root = slots[at].take().ok_or_else(|| corrupt("no explain root"))?;
    if slots.iter().any(Option::is_some) {
        return Err(corrupt("an explain node nothing uses"));
    }
    Ok(root)
}

/// The table entries `m` refers to.
fn expr_children(m: &pb::Expr) -> impl Iterator<Item = u32> + '_ {
    use pb::expr::Kind;
    let (one, many): (Option<u32>, &[u32]) = match &m.kind {
        Some(Kind::Cast(c)) => (Some(c.expr), &[]),
        Some(Kind::Alias(a)) => (Some(a.expr), &[]),
        Some(Kind::Function(f)) => (f.filter, &f.args),
        _ => (None, &[]),
    };
    one.into_iter().chain(many.iter().copied())
}

/// `i` as an index into a table of `len` entries.
fn slot(i: u32, len: usize) -> Result<usize> {
    let at = usize::try_from(i).map_err(|_| corrupt("an index out of range"))?;
    if at < len {
        Ok(at)
    } else {
        Err(corrupt("a reference to a later or missing entry"))
    }
}

fn usize_of(v: u64) -> Result<usize> {
    usize::try_from(v).map_err(|_| corrupt("a count or index out of range"))
}

fn usizes(v: &[u64]) -> Result<Vec<usize>> {
    v.iter().map(|i| usize_of(*i)).collect()
}

fn layout(v: &[u64]) -> Vec<AttrId> {
    v.iter().map(|a| AttrId(*a)).collect()
}

fn required<'a, T>(field: &'a Option<T>, what: &str) -> Result<&'a T> {
    field
        .as_ref()
        .ok_or_else(|| corrupt(format!("{what} is missing")))
}

macro_rules! enum_of {
    ($ty:ty, $value:expr, $what:literal) => {
        match <$ty>::try_from($value) {
            Ok(<$ty>::Unspecified) | Err(_) => Err(corrupt(concat!("an unknown ", $what))),
            Ok(v) => Ok(v),
        }
    };
}

struct Decoder {
    /// Decoded expressions.
    exprs: Vec<Arc<Expr>>,
    /// The depth of each decoded expression (a shared subexpression counts
    /// once per path, since evaluation recurses along paths).
    expr_depths: Vec<usize>,
    /// Decoded nodes not yet used by a parent.
    slots: Vec<Option<PhysNode>>,
    /// The depth of each decoded node.
    depths: Vec<usize>,
    /// The deepest input of the node being decoded.
    child_depth: usize,
    max_depth: usize,
}

impl Decoder {
    /// The node at `i`, which must be one of the first `before` entries
    /// and not used yet.
    fn take(&mut self, i: u32, before: usize) -> Result<PhysNode> {
        let at = slot(i, before)?;
        self.child_depth = self.child_depth.max(self.depths[at]);
        self.slots[at]
            .take()
            .ok_or_else(|| corrupt("a node used by two parents"))
    }

    fn boxed(&mut self, i: u32) -> Result<Box<PhysNode>> {
        self.take(i, self.slots.len()).map(Box::new)
    }

    fn expr_ref(&self, i: u32) -> Result<Arc<Expr>> {
        Ok(Arc::clone(&self.exprs[slot(i, self.exprs.len())?]))
    }

    fn exprs_of(&self, v: &[u32]) -> Result<Vec<Arc<Expr>>> {
        v.iter().map(|i| self.expr_ref(*i)).collect()
    }

    fn node(&mut self, m: &pb::Node) -> Result<PhysNode> {
        use pb::node::Kind;
        Ok(match required(&m.kind, "a node kind")? {
            Kind::Scan(s) => PhysNode::Scan(self.scan(s)?),
            Kind::OneRow(_) => PhysNode::OneRow,
            Kind::Filter(f) => PhysNode::Filter {
                input: self.boxed(f.input)?,
                layout: layout(&f.layout),
                predicate: self.expr_ref(f.predicate)?,
                subqueries: self.subqueries(&f.subqueries)?,
            },
            Kind::Project(p) => PhysNode::Project {
                input: self.boxed(p.input)?,
                layout: layout(&p.layout),
                items: p
                    .items
                    .iter()
                    .map(|i| self.item(i))
                    .collect::<Result<_>>()?,
                nullable: p.nullable.as_ref().map(|n| n.values.clone()),
                subqueries: self.subqueries(&p.subqueries)?,
            },
            Kind::Limit(l) => PhysNode::Limit {
                input: self.boxed(l.input)?,
                limit: l.limit,
            },
            Kind::ReplicateRows(r) => PhysNode::ReplicateRows {
                input: self.boxed(r.input)?,
                layout: layout(&r.layout),
                count: self.expr_ref(r.count)?,
                subqueries: self.subqueries(&r.subqueries)?,
            },
            Kind::Sort(s) => PhysNode::Sort {
                input: self.boxed(s.input)?,
                keys: sort_keys(&s.keys)?,
            },
            Kind::TopN(t) => PhysNode::TopN {
                input: self.boxed(t.input)?,
                keys: sort_keys(&t.keys)?,
                limit: t.limit,
            },
            Kind::Aggregate(a) => PhysNode::Aggregate {
                input: self.boxed(a.input)?,
                aggregate: aggregate(a)?,
            },
            Kind::HashJoin(j) => PhysNode::HashJoin(self.hash_join(j)?),
            Kind::NestedLoopJoin(j) => PhysNode::NestedLoopJoin(NestedLoopPlan {
                probe: self.boxed(j.probe)?,
                build: self.boxed(j.build)?,
                join_type: join_type(j.join_type)?,
                residual: self.residual(j.residual.as_ref())?,
            }),
            Kind::Union(u) => PhysNode::Union {
                inputs: u
                    .inputs
                    .iter()
                    .map(|i| self.take(*i, self.slots.len()))
                    .collect::<Result<_>>()?,
            },
            Kind::WriteFiles(w) => PhysNode::WriteFiles {
                input: self.boxed(w.input)?,
            },
        })
    }

    fn subqueries(&mut self, v: &[pb::SubqueryPlan]) -> Result<Vec<SubqueryPlan>> {
        v.iter()
            .map(|s| {
                Ok(SubqueryPlan {
                    id: SubqueryId(s.id),
                    kind: match enum_of!(pb::SubqueryValueKind, s.kind, "subquery kind")? {
                        pb::SubqueryValueKind::Exists => SubqueryValueKind::Exists,
                        _ => SubqueryValueKind::Scalar,
                    },
                    data_type: type_from(&s.data_type)?,
                    plan: self.take(s.plan, self.slots.len())?,
                })
            })
            .collect()
    }

    fn item(&self, m: &pb::ProjectItem) -> Result<ProjectItemPlan> {
        use pb::project_item::Kind;
        Ok(match required(&m.kind, "a project item kind")? {
            Kind::Column(c) => ProjectItemPlan::Column {
                index: usize_of(c.index)?,
                name: c.name.clone(),
            },
            Kind::Expr(e) => ProjectItemPlan::Expr {
                name: e.name.clone(),
                expr: self.expr_ref(e.expr)?,
            },
        })
    }

    fn residual(&self, m: Option<&pb::Residual>) -> Result<Option<Residual>> {
        m.map(|r| {
            Ok(Residual {
                layout: layout(&r.layout),
                conjuncts: self.exprs_of(&r.conjuncts)?,
            })
        })
        .transpose()
    }

    fn hash_join(&mut self, j: &pb::HashJoinNode) -> Result<HashJoinPlan> {
        let (probe_keys, build_keys) = (usizes(&j.probe_keys)?, usizes(&j.build_keys)?);
        if probe_keys.len() != build_keys.len() || probe_keys.len() != j.null_safe.len() {
            return Err(corrupt("a hash join whose key lists differ in length"));
        }
        Ok(HashJoinPlan {
            probe: self.boxed(j.probe)?,
            build: self.boxed(j.build)?,
            join_type: join_type(j.join_type)?,
            probe_keys,
            build_keys,
            null_safe: j.null_safe.clone(),
            residual: self.residual(j.residual.as_ref())?,
            runtime_filters: j
                .runtime_filters
                .iter()
                .map(join_filter)
                .collect::<Result<_>>()?,
        })
    }

    fn scan(&self, s: &pb::ScanNode) -> Result<ScanNode> {
        let filter = s
            .filter
            .as_ref()
            .map(|f| {
                Ok(ScanFilter {
                    layout: layout(&f.layout),
                    predicate: self.expr_ref(f.predicate)?,
                    exact: f.exact,
                })
            })
            .transpose()?;
        let row_stage = required(&s.row_stage, "the row stage settings")?;
        Ok(ScanNode {
            id: ScanId(s.id),
            scan: Scan {
                source: read_source(required(&s.source, "a scan source")?)?,
                output: s.output.iter().map(bound_column).collect::<Result<_>>()?,
                version: s.version.map(TableVersion),
            },
            projection: s
                .projection
                .iter()
                .map(|p| ColumnPath(p.parts.clone()))
                .collect(),
            filter,
            runtime_filters: s
                .runtime_filters
                .iter()
                .map(|f| {
                    Ok(ScanRuntimeFilter {
                        id: FilterId(f.id),
                        column: ColumnPath(required(&f.column, "a filter column")?.parts.clone()),
                        key_type: type_from(&f.key_type)?,
                    })
                })
                .collect::<Result<_>>()?,
            row_stage: RowStageSettings {
                min_rows: row_stage.min_rows,
                max_pass_rate: row_stage.max_pass_rate,
            },
        })
    }

    fn expr(&self, m: &pb::Expr) -> Result<Expr> {
        use pb::expr::Kind;
        let node = match required(&m.kind, "an expression kind")? {
            Kind::Literal(l) => ExprNode::Literal(literal(l)?),
            Kind::Column(c) => ExprNode::Resolved(ResolvedExpr::Column(bound_column(c)?)),
            Kind::OuterColumn(c) => ExprNode::Resolved(ResolvedExpr::OuterColumn(bound_column(c)?)),
            Kind::Cast(c) => ExprNode::Cast {
                expr: self.expr_ref(c.expr)?,
                to: spark_type(required(&c.to, "a cast type")?)?,
                mode: match enum_of!(pb::CastMode, c.mode, "cast mode")? {
                    pb::CastMode::Legacy => CastMode::Legacy,
                    pb::CastMode::Ansi => CastMode::Ansi,
                    _ => CastMode::Try,
                },
            },
            Kind::Alias(a) => ExprNode::Resolved(ResolvedExpr::Alias {
                expr: self.expr_ref(a.expr)?,
                output: bound_column(required(&a.output, "an alias output")?)?,
            }),
            Kind::Function(f) => ExprNode::Resolved(ResolvedExpr::Function(self.function(f)?)),
        };
        let mut origin = Origin::UNKNOWN;
        if let Some(o) = &m.origin {
            if let Some(p) = o.plan_id {
                origin = origin.with_plan_id(PlanId(p));
            }
            if let Some(s) = &o.sql {
                origin = origin.with_sql(SourceId(s.source), Span::new(s.start, s.end));
            }
        }
        Ok(Expr::new(node).with_origin(origin))
    }

    fn function(&self, f: &pb::Function) -> Result<ResolvedFunction> {
        let id = match required(&f.id, "a function identity")? {
            pb::function::Id::Builtin(name) => FunctionId::Builtin(
                Builtin::from_name(name).ok_or_else(|| corrupt("an unknown canonical operator"))?,
            ),
            pb::function::Id::Registry(r) => FunctionId::Registry {
                name: r.name.clone(),
                id: r.id,
            },
        };
        Ok(ResolvedFunction {
            id,
            kind: match enum_of!(pb::FunctionKind, f.kind, "function kind")? {
                pb::FunctionKind::Scalar => FunctionKind::Scalar,
                pb::FunctionKind::Aggregate => FunctionKind::Aggregate,
                pb::FunctionKind::Window => FunctionKind::Window,
                _ => FunctionKind::Generator,
            },
            args: self.exprs_of(&f.args)?,
            distinct: f.distinct,
            filter: f.filter.map(|i| self.expr_ref(i)).transpose()?,
            data_type: spark_type(required(&f.data_type, "a function type")?)?,
            nullable: f.nullable,
            deterministic: f.deterministic,
        })
    }
}

fn sort_keys(v: &[pb::SortKey]) -> Result<Vec<SortKey>> {
    v.iter()
        .map(|k| {
            Ok(SortKey {
                column: usize_of(k.column)?,
                descending: k.descending,
                nulls_first: k.nulls_first,
            })
        })
        .collect()
}

fn aggregate(a: &pb::AggregateNode) -> Result<AggregatePlan> {
    if a.distinct.len() != a.aggregates.len() {
        return Err(corrupt(
            "an aggregate whose distinct flags do not match its functions",
        ));
    }
    Ok(AggregatePlan {
        group_by: usizes(&a.group_by)?,
        aggregates: a
            .aggregates
            .iter()
            .map(aggregate_expr)
            .collect::<Result<_>>()?,
        distinct: a.distinct.clone(),
        output: match enum_of!(pb::AggregateOutput, a.output, "aggregate output")? {
            pb::AggregateOutput::Final => AggregateOutput::Final,
            _ => AggregateOutput::Partial,
        },
    })
}

fn aggregate_expr(a: &pb::AggregateExpr) -> Result<AggregateExpr> {
    use pb::AggregateFunction as F;
    let ignore = a.ignore_nulls;
    let function = match enum_of!(F, a.function, "aggregate function")? {
        F::First => AggregateFunction::First {
            ignore_nulls: ignore,
        },
        F::Last => AggregateFunction::Last {
            ignore_nulls: ignore,
        },
        _ if ignore => return Err(corrupt("`ignore_nulls` on a function without it")),
        F::Sum => AggregateFunction::Sum,
        F::Avg => AggregateFunction::Avg,
        F::Min => AggregateFunction::Min,
        F::Max => AggregateFunction::Max,
        _ => AggregateFunction::Count,
    };
    Ok(AggregateExpr {
        function,
        args: usizes(&a.args)?,
        input: match enum_of!(pb::AggregateInput, a.input, "aggregate input")? {
            pb::AggregateInput::Raw => AggInput::Raw,
            _ => AggInput::State,
        },
        arg_type: type_from(&a.arg_type)?,
        name: a.name.clone(),
    })
}

fn join_type(t: i32) -> Result<JoinType> {
    Ok(match enum_of!(pb::JoinType, t, "join type")? {
        pb::JoinType::Inner => JoinType::Inner,
        pb::JoinType::LeftOuter => JoinType::LeftOuter,
        pb::JoinType::LeftSemi => JoinType::LeftSemi,
        pb::JoinType::LeftAnti => JoinType::LeftAnti,
        _ => JoinType::NullAwareAnti,
    })
}

fn join_filter(f: &pb::JoinFilter) -> Result<JoinFilterPlan> {
    let o = required(&f.options, "filter options")?;
    Ok(JoinFilterPlan {
        id: FilterId(f.id),
        key: usize_of(f.key)?,
        options: KeyFilterOptions {
            max_values: usize_of(o.max_values)?,
            exact_max_values: usize_of(o.exact_max_values)?,
            bloom: o.bloom,
            bloom_max_items: o.bloom_max_items,
            bloom_max_bytes: usize_of(o.bloom_max_bytes)?,
            bloom_fpp: o.bloom_fpp,
            bloom_max_fpp: o.bloom_max_fpp,
        },
    })
}

fn bound_column(c: &pb::BoundColumn) -> Result<BoundColumn> {
    Ok(BoundColumn {
        id: AttrId(c.id),
        name: c.name.clone(),
        data_type: spark_type(required(&c.data_type, "a column type")?)?,
        nullable: c.nullable,
    })
}

fn read_source(m: &pb::ReadSource) -> Result<ReadSource> {
    use pb::read_source::Kind;
    let options = |m: &BTreeMap<String, pb::OptionValue>| {
        m.iter()
            .map(|(k, v)| Ok((k.clone(), option(v)?)))
            .collect::<Result<BTreeMap<_, _>>>()
    };
    Ok(match required(&m.kind, "a source kind")? {
        Kind::Table(t) => ReadSource::Table {
            name: t.name.clone(),
            options: options(&t.options)?,
        },
        Kind::Files(f) => ReadSource::Files {
            format: f.format.clone(),
            paths: f.paths.iter().map(option).collect::<Result<_>>()?,
            options: options(&f.options)?,
        },
    })
}

fn option(m: &pb::OptionValue) -> Result<OptionValue> {
    Ok(match required(&m.kind, "an option value")? {
        pb::option_value::Kind::Plain(s) => OptionValue::Plain(s.clone()),
        pb::option_value::Kind::Secret(r) => OptionValue::Secret(SecretValue::Reference {
            scope: r.scope.clone(),
            key: r.key.clone(),
        }),
    })
}

fn spark_type(t: &pb::DataType) -> Result<DataType> {
    use pb::data_type::Kind;
    let kind = enum_of!(Kind, t.kind, "type")?;
    if kind != Kind::Decimal && (t.precision != 0 || t.scale != 0) {
        return Err(corrupt("a precision or scale on a type that has none"));
    }
    Ok(match kind {
        Kind::Boolean => DataType::Boolean,
        Kind::Byte => DataType::Byte,
        Kind::Short => DataType::Short,
        Kind::Integer => DataType::Integer,
        Kind::Long => DataType::Long,
        Kind::Float => DataType::Float,
        Kind::Double => DataType::Double,
        Kind::String => DataType::String,
        Kind::Binary => DataType::Binary,
        Kind::Decimal => DataType::Decimal(decimal(t)?),
        Kind::Date => DataType::Date,
        Kind::Timestamp => DataType::Timestamp,
        Kind::TimestampNtz => DataType::TimestampNtz,
        _ => DataType::Null,
    })
}

fn decimal(t: &pb::DataType) -> Result<DecimalType> {
    DecimalType::try_new(t.precision, t.scale).map_err(|_| corrupt("an invalid decimal type"))
}

fn narrow<T: TryFrom<i32>>(v: i32) -> Result<T> {
    T::try_from(v).map_err(|_| corrupt("a literal out of range for its type"))
}

fn literal(l: &pb::Literal) -> Result<Literal> {
    use pb::literal::Kind;
    Ok(match required(&l.kind, "a literal kind")? {
        Kind::Null(n) => Literal::Null(n.data_type.as_ref().map(spark_type).transpose()?),
        Kind::Boolean(v) => Literal::Boolean(*v),
        Kind::Byte(v) => Literal::Byte(narrow(*v)?),
        Kind::Short(v) => Literal::Short(narrow(*v)?),
        Kind::Integer(v) => Literal::Integer(*v),
        Kind::Long(v) => Literal::Long(*v),
        Kind::Float(v) => Literal::Float(*v),
        Kind::Double(v) => Literal::Double(*v),
        Kind::Decimal(d) => {
            let bytes: [u8; 16] = d
                .unscaled
                .as_slice()
                .try_into()
                .map_err(|_| corrupt("a decimal literal that is not 16 bytes"))?;
            let unscaled = i128::from_le_bytes(bytes);
            let data_type = decimal(required(&d.data_type, "a decimal type")?)?;
            // Spark's Decimal holds |unscaled| < 10^precision; 10^38 fits i128.
            if unscaled.unsigned_abs() >= 10_u128.pow(u32::from(data_type.precision())) {
                return Err(corrupt("a decimal literal beyond its precision"));
            }
            Literal::Decimal {
                unscaled,
                data_type,
            }
        }
        Kind::String(v) => Literal::String(v.clone()),
        Kind::Binary(v) => Literal::Binary(v.clone()),
        Kind::Date(v) => Literal::Date(*v),
        Kind::Timestamp(v) => Literal::Timestamp(*v),
        Kind::TimestampNtz(v) => Literal::TimestampNtz(*v),
        Kind::YearMonth(v) => Literal::YearMonthInterval {
            months: v.months,
            start: year_month_field(v.start)?,
            end: year_month_field(v.end)?,
        },
        Kind::DayTime(v) => Literal::DayTimeInterval {
            micros: v.micros,
            start: day_time_field(v.start)?,
            end: day_time_field(v.end)?,
        },
    })
}

fn year_month_field(v: i32) -> Result<YearMonthField> {
    Ok(match enum_of!(pb::YearMonthField, v, "interval field")? {
        pb::YearMonthField::Year => YearMonthField::Year,
        _ => YearMonthField::Month,
    })
}

fn day_time_field(v: i32) -> Result<DayTimeField> {
    Ok(match enum_of!(pb::DayTimeField, v, "interval field")? {
        pb::DayTimeField::Day => DayTimeField::Day,
        pb::DayTimeField::Hour => DayTimeField::Hour,
        pb::DayTimeField::Minute => DayTimeField::Minute,
        _ => DayTimeField::Second,
    })
}
