// SPDX-License-Identifier: Apache-2.0
//! `krishiv.plan.v1` to `PhysicalPlan`. The bytes are untrusted: sizes are
//! bounded first, every reference must point to an earlier entry, every
//! entry may be used once (nodes) and counts must agree. The tables are
//! flat, so nothing here recurses and protobuf nesting stays shallow.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use krishiv_common::secret::{OptionValue, SecretValue};
use krishiv_exec::JoinType;
use krishiv_exec::aggregate::{AggInput, AggregateExpr, AggregateFunction, AggregateOutput};
use krishiv_exec::physical::{
    AggregatePlan, ExchangeId, ExchangePlan, ExprEnv, HashJoinPlan, JoinDistribution,
    JoinFilterPlan, NestedLoopPlan, Partitioning, PhysNode, PhysicalPlan, ProjectItemPlan,
    Residual, ScanFilter, ScanId, ScanNode, ScanRuntimeFilter, SubqueryId, SubqueryPlan,
    SubqueryValueKind,
};
use krishiv_exec::planner::ExplainNode;
use krishiv_exec::window::{
    FrameBound, FrameUnits, GroupLimitPlan, RankLike, WindowFrame, WindowFunction,
    WindowFunctionPlan, WindowPlan,
};
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
use crate::features::{Feature, ReaderFeatures, legacy_level, plan_features};
use crate::proto as pb;
use crate::{EPOCH, FORMAT_VERSION, Limits};

/// Decodes `krishiv.plan.v1` bytes under the default [`Limits`], with every
/// feature this build implements.
///
/// # Errors
/// [`CodecError::Corrupt`] for bytes that are not a valid plan (a header
/// that does not match the plan's constructs included),
/// [`CodecError::UnsupportedEpoch`] and [`CodecError::UnsupportedFeatures`]
/// for a plan this reader cannot read, [`CodecError::TooNew`] for a plan
/// written before the feature header that needs a newer reader,
/// [`CodecError::Limit`] for a plan beyond the limits.
pub fn decode(bytes: &[u8]) -> Result<PhysicalPlan> {
    decode_with(bytes, &Limits::default(), &ReaderFeatures::all())
}

/// [`decode`] under `limits`, as a reader that implements `reader`.
///
/// # Errors
/// As [`decode`].
pub fn decode_with(bytes: &[u8], limits: &Limits, reader: &ReaderFeatures) -> Result<PhysicalPlan> {
    check_size(bytes, limits)?;
    from_message(&parse(bytes)?, limits, reader)
}

fn check_size(bytes: &[u8], limits: &Limits) -> Result<()> {
    if bytes.len() > limits.max_bytes {
        return Err(CodecError::Limit(format!(
            "{} bytes (at most {})",
            bytes.len(),
            limits.max_bytes
        )));
    }
    Ok(())
}

fn parse<M: prost::Message + Default>(bytes: &[u8]) -> Result<M> {
    // The messages are flat tables: protobuf nesting stays under a dozen
    // levels, whatever the plan's depth, so prost's recursion is bounded.
    #[expect(
        clippy::disallowed_methods,
        reason = "the plan format has constant nesting depth (flat tables)"
    )]
    M::decode(bytes).map_err(|_| corrupt("malformed protobuf"))
}

/// The features an encoded plan needs, from its header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanFeatures {
    /// The plan's wire epoch.
    pub epoch: u32,
    /// The ids of the features it uses, sorted.
    pub features: Vec<String>,
}

impl PlanFeatures {
    /// The features `reader` lacks to read the plan.
    #[must_use]
    pub fn missing(&self, reader: &ReaderFeatures) -> Vec<String> {
        reader.missing(self.features.iter().map(String::as_str))
    }
}

/// The epoch and features of an encoded plan, read from its header without
/// decoding the plan (the header is checked against the plan on decode, so
/// it can be trusted). A plan written before the header existed is decoded
/// under the default [`Limits`] to find its features.
///
/// # Errors
/// [`CodecError::Corrupt`] for a malformed header,
/// [`CodecError::UnsupportedEpoch`] for an epoch this build does not know,
/// and for an older plan, as [`decode`].
pub fn features_of(bytes: &[u8]) -> Result<PlanFeatures> {
    check_size(bytes, &Limits::default())?;
    let h: pb::PlanHeader = parse(bytes)?;
    let features = match header(h.format_version, h.min_reader_version, h.epoch, &h.features)? {
        Header::Current => h.features,
        Header::Legacy => plan_features(&decode(bytes)?)
            .iter()
            .map(|f| f.id().to_owned())
            .collect(),
    };
    Ok(PlanFeatures {
        epoch: EPOCH,
        features,
    })
}

/// What a valid header says about the rest of the plan.
enum Header {
    /// A plan with a feature header: the ids are well formed and sorted.
    Current,
    /// A plan written before the feature header: its features follow from
    /// its content.
    Legacy,
}

/// The longest feature id a reader accepts.
const MAX_FEATURE_ID: usize = 64;

/// The most feature ids a header may list.
const MAX_FEATURES: usize = 256;

fn header(format_version: u32, min_reader: u32, epoch: u32, ids: &[String]) -> Result<Header> {
    if format_version == 0 {
        return Err(corrupt("no format version"));
    }
    if epoch == 0 {
        if !ids.is_empty() {
            return Err(corrupt("a feature list without an epoch"));
        }
        if min_reader > FORMAT_VERSION {
            return Err(CodecError::TooNew {
                required: min_reader,
                supported: FORMAT_VERSION,
            });
        }
        return Ok(Header::Legacy);
    }
    if epoch != EPOCH {
        return Err(CodecError::UnsupportedEpoch {
            epoch,
            supported: EPOCH,
        });
    }
    if ids.len() > MAX_FEATURES {
        return Err(corrupt("more feature ids than any plan uses"));
    }
    for id in ids {
        let well_formed = id.len() <= MAX_FEATURE_ID
            && id.starts_with(|c: char| c.is_ascii_lowercase())
            && id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if !well_formed {
            return Err(corrupt("a malformed feature id"));
        }
    }
    if ids.windows(2).any(|w| w[0] >= w[1]) {
        return Err(corrupt("feature ids that are not sorted or repeat"));
    }
    Ok(Header::Current)
}

/// [`decode_with`] from a parsed message.
///
/// # Errors
/// As [`decode`].
pub fn from_message(
    m: &pb::PhysicalPlan,
    limits: &Limits,
    reader: &ReaderFeatures,
) -> Result<PhysicalPlan> {
    let kind = header(m.format_version, m.min_reader_version, m.epoch, &m.features)?;
    if let Header::Current = kind {
        // Before any node is read.
        let missing = reader.missing(m.features.iter().map(String::as_str));
        if !missing.is_empty() {
            return Err(CodecError::UnsupportedFeatures { missing });
        }
    }
    let plan = plan(m, limits)?;
    let used = plan_features(&plan);
    match kind {
        Header::Current => {
            let declared: BTreeSet<Feature> = m
                .features
                .iter()
                .filter_map(|id| Feature::from_id(id))
                .collect();
            if declared != used {
                return Err(corrupt(
                    "the feature header does not match the plan's constructs",
                ));
            }
            if m.min_reader_version != legacy_level(&used) {
                return Err(corrupt("a reader version that does not match the features"));
            }
        }
        Header::Legacy => {
            if used.iter().any(|f| f.legacy_level() > m.min_reader_version) {
                return Err(corrupt("a construct newer than the plan's reader version"));
            }
            let missing: Vec<String> = used
                .iter()
                .filter(|f| !reader.contains(**f))
                .map(|f| f.id().to_owned())
                .collect();
            if !missing.is_empty() {
                return Err(CodecError::UnsupportedFeatures { missing });
            }
        }
    }
    Ok(plan)
}

fn plan(m: &pb::PhysicalPlan, limits: &Limits) -> Result<PhysicalPlan> {
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
        Some(Kind::Literal(_) | Kind::Column(_) | Kind::OuterColumn(_)) | None => (None, &[]),
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

/// Maps a protobuf enum value with an exhaustive match: a value added to the
/// schema does not compile here until it is mapped.
macro_rules! enum_map {
    ($ty:ty, $value:expr, $what:literal, { $($from:ident => $to:expr),+ $(,)? }) => {
        match <$ty>::try_from($value) {
            $(Ok(<$ty>::$from) => Ok($to),)+
            Ok(<$ty>::Unspecified) | Err(_) => Err(corrupt(concat!("an unknown ", $what))),
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
                offset: l.offset,
                limit: l.limit,
            },
            Kind::ReplicateRows(r) => PhysNode::ReplicateRows {
                input: self.boxed(r.input)?,
                layout: layout(&r.layout),
                count: self.expr_ref(r.count)?,
                subqueries: self.subqueries(&r.subqueries)?,
            },
            Kind::Expand(e) => {
                expand_shape(e)?;
                PhysNode::Expand {
                    input: self.boxed(e.input)?,
                    layout: layout(&e.layout),
                    projections: e
                        .projections
                        .iter()
                        .map(|p| p.items.iter().map(|i| self.item(i)).collect())
                        .collect::<Result<_>>()?,
                    nullable: e.nullable.clone(),
                    subqueries: self.subqueries(&e.subqueries)?,
                }
            }
            Kind::Sort(s) => PhysNode::Sort {
                input: self.boxed(s.input)?,
                keys: sort_keys(&s.keys)?,
            },
            Kind::TopN(t) => PhysNode::TopN {
                input: self.boxed(t.input)?,
                keys: sort_keys(&t.keys)?,
                limit: t.limit,
            },
            Kind::Window(w) => PhysNode::Window {
                input: self.boxed(w.input)?,
                window: window_plan(w)?,
            },
            Kind::WindowGroupLimit(g) => PhysNode::WindowGroupLimit {
                input: self.boxed(g.input)?,
                limit: group_limit(g)?,
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
            Kind::Exchange(e) => PhysNode::Exchange(self.exchange(e)?),
        })
    }

    fn subqueries(&mut self, v: &[pb::SubqueryPlan]) -> Result<Vec<SubqueryPlan>> {
        v.iter()
            .map(|s| {
                Ok(SubqueryPlan {
                    id: SubqueryId(s.id),
                    kind: enum_map!(pb::SubqueryValueKind, s.kind, "subquery kind", {
                        Scalar => SubqueryValueKind::Scalar,
                        Exists => SubqueryValueKind::Exists,
                    })?,
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
            Kind::NullOf(n) => ProjectItemPlan::NullOf {
                index: usize_of(n.index)?,
                name: n.name.clone(),
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
            distribution: distribution(j.distribution)?,
        })
    }

    fn exchange(&mut self, e: &pb::ExchangeNode) -> Result<ExchangePlan> {
        Ok(ExchangePlan {
            id: ExchangeId(e.id),
            input: self.boxed(e.input)?,
            partitioning: partitioning(required(&e.partitioning, "the partitioning")?)?,
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
                mode: enum_map!(pb::CastMode, c.mode, "cast mode", {
                    Legacy => CastMode::Legacy,
                    Ansi => CastMode::Ansi,
                    Try => CastMode::Try,
                })?,
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
            kind: enum_map!(pb::FunctionKind, f.kind, "function kind", {
                Scalar => FunctionKind::Scalar,
                Aggregate => FunctionKind::Aggregate,
                Window => FunctionKind::Window,
                Generator => FunctionKind::Generator,
            })?,
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
        output: enum_map!(pb::AggregateOutput, a.output, "aggregate output", {
            Final => AggregateOutput::Final,
            Partial => AggregateOutput::Partial,
        })?,
    })
}

/// An expand has a projection, and every projection has one item per output
/// column.
fn expand_shape(e: &pb::ExpandNode) -> Result<()> {
    if e.projections.is_empty() {
        return Err(corrupt("an expand without a projection"));
    }
    if e.projections
        .iter()
        .any(|p| p.items.len() != e.nullable.len())
    {
        return Err(corrupt(
            "an expand projection whose items do not match the nullability",
        ));
    }
    Ok(())
}

fn window_plan(w: &pb::WindowNode) -> Result<WindowPlan> {
    Ok(WindowPlan {
        partition_by: usizes(&w.partition_by)?,
        order_by: sort_keys(&w.order_by)?,
        functions: w
            .functions
            .iter()
            .map(window_function)
            .collect::<Result<_>>()?,
    })
}

fn group_limit(g: &pb::WindowGroupLimitNode) -> Result<GroupLimitPlan> {
    if g.limit == 0 {
        return Err(corrupt("a window group limit of 0 rows"));
    }
    Ok(GroupLimitPlan {
        partition_by: usizes(&g.partition_by)?,
        order_by: sort_keys(&g.order_by)?,
        function: enum_map!(pb::RankLikeFunction, g.function, "rank-like function", {
            RowNumber => RankLike::RowNumber,
            Rank => RankLike::Rank,
            DenseRank => RankLike::DenseRank,
        })?,
        limit: g.limit,
    })
}

fn window_function(f: &pb::WindowFunctionPlan) -> Result<WindowFunctionPlan> {
    use pb::window_function_plan::Function;
    let function = match required(&f.function, "a window function")? {
        Function::RowNumber(_) => WindowFunction::RowNumber,
        Function::Rank(_) => WindowFunction::Rank,
        Function::DenseRank(_) => WindowFunction::DenseRank,
        Function::PercentRank(_) => WindowFunction::PercentRank,
        Function::CumeDist(_) => WindowFunction::CumeDist,
        Function::Ntile(buckets) => {
            if *buckets <= 0 {
                return Err(corrupt("an ntile with a bucket count below one"));
            }
            WindowFunction::Ntile(*buckets)
        }
        Function::Aggregate(a) => WindowFunction::Aggregate(aggregate_expr(a)?),
    };
    let frame = required(&f.frame, "a window frame")?;
    let bound = |b: i32| {
        enum_map!(pb::FrameBound, b, "window frame bound", {
            UnboundedPreceding => FrameBound::UnboundedPreceding,
            CurrentRow => FrameBound::CurrentRow,
            UnboundedFollowing => FrameBound::UnboundedFollowing,
        })
    };
    Ok(WindowFunctionPlan {
        function,
        frame: WindowFrame {
            units: enum_map!(pb::FrameUnits, frame.units, "window frame unit", {
                Rows => FrameUnits::Rows,
                Range => FrameUnits::Range,
            })?,
            lower: bound(frame.lower)?,
            upper: bound(frame.upper)?,
        },
        filter: f.filter.map(usize_of).transpose()?,
        name: f.name.clone(),
    })
}

fn aggregate_expr(a: &pb::AggregateExpr) -> Result<AggregateExpr> {
    use pb::AggregateFunction as F;
    let ignore = a.ignore_nulls;
    let function = enum_map!(F, a.function, "aggregate function", {
        Count => AggregateFunction::Count,
        Sum => AggregateFunction::Sum,
        Avg => AggregateFunction::Avg,
        Min => AggregateFunction::Min,
        Max => AggregateFunction::Max,
        First => AggregateFunction::First { ignore_nulls: ignore },
        Last => AggregateFunction::Last { ignore_nulls: ignore },
    })?;
    let takes_ignore = matches!(
        function,
        AggregateFunction::First { .. } | AggregateFunction::Last { .. }
    );
    if ignore && !takes_ignore {
        return Err(corrupt("`ignore_nulls` on a function without it"));
    }
    Ok(AggregateExpr {
        function,
        args: usizes(&a.args)?,
        input: enum_map!(pb::AggregateInput, a.input, "aggregate input", {
            Raw => AggInput::Raw,
            State => AggInput::State,
        })?,
        arg_type: type_from(&a.arg_type)?,
        name: a.name.clone(),
    })
}

fn join_type(t: i32) -> Result<JoinType> {
    enum_map!(pb::JoinType, t, "join type", {
        Inner => JoinType::Inner,
        LeftOuter => JoinType::LeftOuter,
        LeftSemi => JoinType::LeftSemi,
        LeftAnti => JoinType::LeftAnti,
        NullAwareAnti => JoinType::NullAwareAnti,
        RightSemi => JoinType::RightSemi,
        RightAnti => JoinType::RightAnti,
        RightOuter => JoinType::RightOuter,
    })
}

fn distribution(d: i32) -> Result<JoinDistribution> {
    // v1 bytes carry no distribution (`UNSPECIFIED`): they predate the
    // partitioned join, when every hash join broadcast its build.
    Ok(match pb::JoinDistribution::try_from(d) {
        Ok(pb::JoinDistribution::Unspecified | pb::JoinDistribution::Broadcast) => {
            JoinDistribution::Broadcast
        }
        Ok(pb::JoinDistribution::Partitioned) => JoinDistribution::Partitioned,
        Err(_) => return Err(corrupt("an unknown join distribution")),
    })
}

fn partitioning(p: &pb::Partitioning) -> Result<Partitioning> {
    use pb::partitioning::Kind;
    Ok(match required(&p.kind, "a partitioning kind")? {
        Kind::Hash(h) => Partitioning::Hash {
            keys: usizes(&h.keys)?,
        },
        Kind::Single(_) => Partitioning::Single,
        Kind::Broadcast(_) => Partitioning::Broadcast,
        Kind::RoundRobin(_) => Partitioning::RoundRobin,
        Kind::Range(r) => Partitioning::Range {
            keys: sort_keys(&r.keys)?,
        },
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
    let data_type = enum_map!(Kind, t.kind, "type", {
        Null => DataType::Null,
        Boolean => DataType::Boolean,
        Byte => DataType::Byte,
        Short => DataType::Short,
        Integer => DataType::Integer,
        Long => DataType::Long,
        Float => DataType::Float,
        Double => DataType::Double,
        String => DataType::String,
        Binary => DataType::Binary,
        Decimal => DataType::Decimal(decimal(t)?),
        Date => DataType::Date,
        Timestamp => DataType::Timestamp,
        TimestampNtz => DataType::TimestampNtz,
        Nested => nested(t)?,
    })?;
    if !matches!(data_type, DataType::Decimal(_)) && (t.precision != 0 || t.scale != 0) {
        return Err(corrupt("a precision or scale on a type that has none"));
    }
    if !data_type.is_nested() && !t.json.is_empty() {
        return Err(corrupt("a JSON form on a type that is not nested"));
    }
    Ok(data_type)
}

/// A nested type from its Spark JSON (depth-bounded by `from_json`).
fn nested(t: &pb::DataType) -> Result<DataType> {
    match DataType::from_json(&t.json) {
        Ok(d) if d.is_nested() => Ok(d),
        _ => Err(corrupt("an invalid nested type")),
    }
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
    enum_map!(pb::YearMonthField, v, "interval field", {
        Year => YearMonthField::Year,
        Month => YearMonthField::Month,
    })
}

fn day_time_field(v: i32) -> Result<DayTimeField> {
    enum_map!(pb::DayTimeField, v, "interval field", {
        Day => DayTimeField::Day,
        Hour => DayTimeField::Hour,
        Minute => DayTimeField::Minute,
        Second => DayTimeField::Second,
    })
}
