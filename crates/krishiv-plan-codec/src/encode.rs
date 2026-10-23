// SPDX-License-Identifier: Apache-2.0
//! `PhysicalPlan` to `krishiv.plan.v1`. Nothing here recurses: nodes,
//! expressions and explain nodes are collected in post-order with an
//! explicit stack, and each is written after the entries it refers to.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::secret::{OptionValue, SecretValue};
use krishiv_exec::JoinType;
use krishiv_exec::aggregate::{AggInput, AggregateExpr, AggregateFunction, AggregateOutput};
use krishiv_exec::physical::{
    AggregatePlan, ExchangePlan, ExprEnv, HashJoinPlan, JoinDistribution, JoinFilterPlan,
    NestedLoopPlan, Partitioning, PhysNode, PhysicalPlan, ProjectItemPlan, Residual, ScanFilter,
    ScanNode, ScanRuntimeFilter, SubqueryPlan, SubqueryValueKind,
};
use krishiv_exec::planner::ExplainNode;
use krishiv_exec::{RowStageSettings, SortKey};
use krishiv_io::{ColumnPath, KeyFilterOptions};
use krishiv_plan::{
    AttrId, BoundColumn, CastMode, DayTimeField, Expr, ExprNode, FunctionId, FunctionKind, Literal,
    Origin, ReadSource, ResolvedExpr, ResolvedFunction, Scan, YearMonthField,
};
use krishiv_types::DataType;

use crate::arrow::{schema_bytes, type_bytes};
use crate::error::{CodecError, Result};
use crate::proto as pb;
use crate::{FORMAT_VERSION, MIN_READER_VERSION};

/// Encodes `plan` as `krishiv.plan.v1` bytes. Deterministic: the same plan
/// always gives the same bytes.
///
/// # Errors
/// [`CodecError::Unsupported`] for an unresolved expression, an inline
/// secret or a URL with credentials in a source's options or paths (a plan
/// carries `secret(scope, key)` references only, ADR 0038), or a node kind
/// this version does not know.
pub fn encode(plan: &PhysicalPlan) -> Result<Vec<u8>> {
    use prost::Message as _;
    Ok(to_message(plan)?.encode_to_vec())
}

/// [`encode`] without the final serialization.
pub fn to_message(plan: &PhysicalPlan) -> Result<pb::PhysicalPlan> {
    let mut e = Encoder {
        min_reader: MIN_READER_VERSION,
        ..Encoder::default()
    };
    let order = post_order(plan.root(), PhysNode::children);
    for node in &order {
        let message = e.node(node)?;
        e.nodes.insert(std::ptr::from_ref(*node), e.table.len());
        e.table.push(message);
    }
    let root = e.node_index(plan.root())?;
    let explain_order = post_order(plan.explain(), |n| n.children.iter().collect());
    let mut explain = Vec::with_capacity(explain_order.len());
    let mut explain_ids: HashMap<*const ExplainNode, u32> = HashMap::new();
    for node in &explain_order {
        let children = node
            .children
            .iter()
            .map(|c| explain_ids[&std::ptr::from_ref(c)])
            .collect();
        explain_ids.insert(std::ptr::from_ref(*node), index(explain.len())?);
        explain.push(pb::ExplainNode {
            name: node.name.clone(),
            detail: node.detail.clone(),
            children,
        });
    }
    Ok(pb::PhysicalPlan {
        format_version: FORMAT_VERSION,
        min_reader_version: e.min_reader,
        nodes: e.table,
        root,
        schema: schema_bytes(plan.schema()),
        explain,
        explain_root: explain_ids[&std::ptr::from_ref(plan.explain())],
        warnings: plan.warnings().to_vec(),
        env: Some(env(plan.env())),
        exprs: e.exprs,
    })
}

/// `root` and everything below it, children before parents.
fn post_order<'a, T>(root: &'a T, children: impl Fn(&'a T) -> Vec<&'a T>) -> Vec<&'a T> {
    let mut out = Vec::new();
    let mut stack = vec![(root, false)];
    while let Some((node, done)) = stack.pop() {
        if done {
            out.push(node);
        } else {
            stack.push((node, true));
            // Reversed, so the first child is written first.
            stack.extend(children(node).into_iter().rev().map(|c| (c, false)));
        }
    }
    out
}

fn index(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| CodecError::Unsupported("more than 2^32 entries".to_owned()))
}

fn env(env: &ExprEnv) -> pb::ExprEnv {
    pb::ExprEnv {
        ansi: env.ansi,
        zone: env.zone,
        query_start_micros: env.query_start_micros,
        user: env.user.to_string(),
    }
}

#[derive(Default)]
struct Encoder {
    /// Written nodes.
    table: Vec<pb::Node>,
    /// The index of each written node.
    nodes: HashMap<*const PhysNode, usize>,
    /// Written expressions.
    exprs: Vec<pb::Expr>,
    /// The index of each written expression, by `Arc` identity, so a
    /// shared subexpression is written once.
    expr_ids: HashMap<*const Expr, u32>,
    /// The oldest reader version that understands every field written so
    /// far (a `Limit` with a nonzero offset needs version 3).
    min_reader: u32,
}

fn layout(layout: &[AttrId]) -> Vec<u64> {
    layout.iter().map(|a| a.0).collect()
}

fn indexes(v: &[usize]) -> Vec<u64> {
    v.iter().map(|i| *i as u64).collect()
}

impl Encoder {
    fn node_index(&self, node: &PhysNode) -> Result<u32> {
        let at = self.nodes.get(&std::ptr::from_ref(node)).copied();
        at.ok_or_else(|| CodecError::Unsupported("a node written out of order".to_owned()))
            .and_then(index)
    }

    fn node(&mut self, node: &PhysNode) -> Result<pb::Node> {
        use pb::node::Kind;
        let kind = match node {
            PhysNode::Window { .. } => {
                return Err(CodecError::Unsupported(
                    "a Window node (the next commit adds it)".to_owned(),
                ));
            }
            PhysNode::Scan(scan) => Kind::Scan(self.scan(scan)?),
            PhysNode::OneRow => Kind::OneRow(pb::Empty {}),
            PhysNode::Filter {
                input,
                layout: l,
                predicate,
                subqueries,
            } => Kind::Filter(pb::FilterNode {
                input: self.node_index(input)?,
                layout: layout(l),
                predicate: self.expr(predicate)?,
                subqueries: self.subqueries(subqueries)?,
            }),
            PhysNode::Project {
                input,
                layout: l,
                items,
                nullable,
                subqueries,
            } => Kind::Project(pb::ProjectNode {
                input: self.node_index(input)?,
                layout: layout(l),
                items: items
                    .iter()
                    .map(|i| self.item(i))
                    .collect::<Result<Vec<_>>>()?,
                nullable: nullable
                    .as_ref()
                    .map(|v| pb::BoolList { values: v.clone() }),
                subqueries: self.subqueries(subqueries)?,
            }),
            PhysNode::Limit {
                input,
                offset,
                limit,
            } => {
                if *offset != 0 {
                    // An older reader would silently ignore the offset.
                    self.min_reader = self.min_reader.max(3);
                }
                Kind::Limit(pb::LimitNode {
                    input: self.node_index(input)?,
                    limit: *limit,
                    offset: *offset,
                })
            }
            PhysNode::ReplicateRows {
                input,
                layout: l,
                count,
                subqueries,
            } => Kind::ReplicateRows(pb::ReplicateRowsNode {
                input: self.node_index(input)?,
                layout: layout(l),
                count: self.expr(count)?,
                subqueries: self.subqueries(subqueries)?,
            }),
            PhysNode::Sort { input, keys } => Kind::Sort(pb::SortNode {
                input: self.node_index(input)?,
                keys: keys.iter().map(sort_key).collect(),
            }),
            PhysNode::TopN { input, keys, limit } => Kind::TopN(pb::TopNNode {
                input: self.node_index(input)?,
                keys: keys.iter().map(sort_key).collect(),
                limit: *limit,
            }),
            PhysNode::Aggregate { input, aggregate } => {
                Kind::Aggregate(self.aggregate(input, aggregate)?)
            }
            PhysNode::HashJoin(join) => Kind::HashJoin(self.hash_join(join)?),
            PhysNode::NestedLoopJoin(join) => Kind::NestedLoopJoin(self.nested_loop(join)?),
            PhysNode::Union { inputs } => Kind::Union(pb::UnionNode {
                inputs: inputs
                    .iter()
                    .map(|i| self.node_index(i))
                    .collect::<Result<_>>()?,
            }),
            PhysNode::WriteFiles { input } => Kind::WriteFiles(pb::WriteFilesNode {
                input: self.node_index(input)?,
            }),
            PhysNode::Exchange(exchange) => Kind::Exchange(self.exchange(exchange)?),
        };
        Ok(pb::Node { kind: Some(kind) })
    }

    fn subqueries(&mut self, subqueries: &[SubqueryPlan]) -> Result<Vec<pb::SubqueryPlan>> {
        subqueries
            .iter()
            .map(|s| {
                Ok(pb::SubqueryPlan {
                    id: s.id.0,
                    kind: match s.kind {
                        SubqueryValueKind::Scalar => pb::SubqueryValueKind::Scalar,
                        SubqueryValueKind::Exists => pb::SubqueryValueKind::Exists,
                    } as i32,
                    data_type: type_bytes(&s.data_type),
                    plan: self.node_index(&s.plan)?,
                })
            })
            .collect()
    }

    fn item(&mut self, item: &ProjectItemPlan) -> Result<pb::ProjectItem> {
        use pb::project_item::Kind;
        Ok(pb::ProjectItem {
            kind: Some(match item {
                ProjectItemPlan::Column { index, name } => Kind::Column(pb::ProjectColumn {
                    index: *index as u64,
                    name: name.clone(),
                }),
                ProjectItemPlan::Expr { name, expr } => Kind::Expr(pb::ProjectExpr {
                    name: name.clone(),
                    expr: self.expr(expr)?,
                }),
            }),
        })
    }

    fn aggregate(&self, input: &PhysNode, plan: &AggregatePlan) -> Result<pb::AggregateNode> {
        Ok(pb::AggregateNode {
            input: self.node_index(input)?,
            group_by: indexes(&plan.group_by),
            aggregates: plan.aggregates.iter().map(aggregate_expr).collect(),
            distinct: plan.distinct.clone(),
            output: match plan.output {
                AggregateOutput::Final => pb::AggregateOutput::Final,
                AggregateOutput::Partial => pb::AggregateOutput::Partial,
            } as i32,
        })
    }

    fn residual(&mut self, residual: Option<&Residual>) -> Result<Option<pb::Residual>> {
        residual
            .map(|r| {
                Ok(pb::Residual {
                    layout: layout(&r.layout),
                    conjuncts: r
                        .conjuncts
                        .iter()
                        .map(|c| self.expr(c))
                        .collect::<Result<_>>()?,
                })
            })
            .transpose()
    }

    fn hash_join(&mut self, join: &HashJoinPlan) -> Result<pb::HashJoinNode> {
        Ok(pb::HashJoinNode {
            probe: self.node_index(&join.probe)?,
            build: self.node_index(&join.build)?,
            join_type: join_type(join.join_type) as i32,
            probe_keys: indexes(&join.probe_keys),
            build_keys: indexes(&join.build_keys),
            null_safe: join.null_safe.clone(),
            residual: self.residual(join.residual.as_ref())?,
            runtime_filters: join.runtime_filters.iter().map(join_filter).collect(),
            distribution: distribution(join.distribution) as i32,
        })
    }

    fn nested_loop(&mut self, join: &NestedLoopPlan) -> Result<pb::NestedLoopJoinNode> {
        Ok(pb::NestedLoopJoinNode {
            probe: self.node_index(&join.probe)?,
            build: self.node_index(&join.build)?,
            join_type: join_type(join.join_type) as i32,
            residual: self.residual(join.residual.as_ref())?,
        })
    }

    fn exchange(&self, exchange: &ExchangePlan) -> Result<pb::ExchangeNode> {
        Ok(pb::ExchangeNode {
            id: exchange.id.0,
            input: self.node_index(&exchange.input)?,
            partitioning: Some(partitioning(&exchange.partitioning)),
        })
    }

    fn scan(&mut self, node: &ScanNode) -> Result<pb::ScanNode> {
        let Scan {
            source,
            output,
            version,
        } = &node.scan;
        let filter = node
            .filter
            .as_ref()
            .map(|f: &ScanFilter| {
                Ok(pb::ScanFilter {
                    layout: layout(&f.layout),
                    predicate: self.expr(&f.predicate)?,
                    exact: f.exact,
                })
            })
            .transpose()?;
        Ok(pb::ScanNode {
            id: node.id.0,
            source: Some(read_source(source)?),
            output: output.iter().map(bound_column).collect::<Result<_>>()?,
            version: version.map(|v| v.0),
            projection: node.projection.iter().map(column_path).collect(),
            filter,
            runtime_filters: node.runtime_filters.iter().map(scan_filter).collect(),
            row_stage: Some(row_stage(&node.row_stage)),
        })
    }

    /// The index of `root`'s entry in the expression table, writing it and
    /// the entries below it first.
    fn expr(&mut self, root: &Arc<Expr>) -> Result<u32> {
        let mut stack = vec![(root, false)];
        while let Some((e, ready)) = stack.pop() {
            let key = Arc::as_ptr(e);
            if self.expr_ids.contains_key(&key) {
                continue;
            }
            if ready {
                let message = self.expr_message(e)?;
                self.expr_ids.insert(key, index(self.exprs.len())?);
                self.exprs.push(message);
            } else {
                stack.push((e, true));
                e.for_each_child(&mut |c| stack.push((c, false)));
            }
        }
        Ok(self.expr_ids[&Arc::as_ptr(root)])
    }

    /// The index of an expression whose entry is already written.
    fn child(&self, e: &Arc<Expr>) -> u32 {
        self.expr_ids[&Arc::as_ptr(e)]
    }

    fn expr_message(&self, e: &Expr) -> Result<pb::Expr> {
        use pb::expr::Kind;
        let kind = match &e.node {
            ExprNode::Literal(l) => Kind::Literal(literal(l)?),
            ExprNode::Cast { expr, to, mode } => Kind::Cast(pb::Cast {
                expr: self.child(expr),
                to: Some(spark_type(to)?),
                mode: match mode {
                    CastMode::Legacy => pb::CastMode::Legacy,
                    CastMode::Ansi => pb::CastMode::Ansi,
                    CastMode::Try => pb::CastMode::Try,
                    _ => return Err(unresolved("a cast without a mode")),
                } as i32,
            }),
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Kind::Column(bound_column(c)?),
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => Kind::OuterColumn(bound_column(c)?),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => Kind::Function(self.function(f)?),
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                Kind::Alias(pb::AliasExpr {
                    expr: self.child(expr),
                    output: Some(bound_column(output)?),
                })
            }
            ExprNode::Column(_) => return Err(unresolved("an unresolved column")),
            ExprNode::Star(_) => return Err(unresolved("an unresolved star")),
            ExprNode::Function(_) => return Err(unresolved("an unresolved function call")),
            ExprNode::Alias { .. } => return Err(unresolved("an unresolved alias")),
            ExprNode::Subquery { .. } => return Err(unresolved("a subquery expression")),
            _ => return Err(unresolved("an expression kind this version does not know")),
        };
        Ok(pb::Expr {
            origin: origin(e.origin),
            kind: Some(kind),
        })
    }

    fn function(&self, f: &ResolvedFunction) -> Result<pb::Function> {
        let id = match &f.id {
            FunctionId::Builtin(b) => pb::function::Id::Builtin(b.name().to_owned()),
            FunctionId::Registry { name, id } => pb::function::Id::Registry(pb::RegistryFunction {
                name: name.clone(),
                id: *id,
            }),
            _ => return Err(unresolved("a function identity this version does not know")),
        };
        Ok(pb::Function {
            id: Some(id),
            kind: match f.kind {
                FunctionKind::Scalar => pb::FunctionKind::Scalar,
                FunctionKind::Aggregate => pb::FunctionKind::Aggregate,
                FunctionKind::Window => pb::FunctionKind::Window,
                FunctionKind::Generator => pb::FunctionKind::Generator,
                _ => return Err(unresolved("a function kind this version does not know")),
            } as i32,
            args: f.args.iter().map(|a| self.child(a)).collect(),
            distinct: f.distinct,
            filter: f.filter.as_ref().map(|a| self.child(a)),
            data_type: Some(spark_type(&f.data_type)?),
            nullable: f.nullable,
            deterministic: f.deterministic,
        })
    }
}

fn unresolved(what: &str) -> CodecError {
    CodecError::Unsupported(format!(
        "{what} (a physical plan holds resolved expressions)"
    ))
}

fn origin(o: Origin) -> Option<pb::Origin> {
    (o != Origin::UNKNOWN).then(|| pb::Origin {
        plan_id: o.plan_id.map(|p| p.0),
        sql: o.sql.map(|(source, span)| pb::SqlSpan {
            source: source.0,
            start: span.start,
            end: span.end,
        }),
    })
}

fn bound_column(c: &BoundColumn) -> Result<pb::BoundColumn> {
    Ok(pb::BoundColumn {
        id: c.id.0,
        name: c.name.clone(),
        data_type: Some(spark_type(&c.data_type)?),
        nullable: c.nullable,
    })
}

fn column_path(p: &ColumnPath) -> pb::ColumnPath {
    pb::ColumnPath { parts: p.0.clone() }
}

fn sort_key(k: &SortKey) -> pb::SortKey {
    pb::SortKey {
        column: k.column as u64,
        descending: k.descending,
        nulls_first: k.nulls_first,
    }
}

fn join_type(t: JoinType) -> pb::JoinType {
    match t {
        JoinType::Inner => pb::JoinType::Inner,
        JoinType::LeftOuter => pb::JoinType::LeftOuter,
        JoinType::LeftSemi => pb::JoinType::LeftSemi,
        JoinType::LeftAnti => pb::JoinType::LeftAnti,
        JoinType::NullAwareAnti => pb::JoinType::NullAwareAnti,
    }
}

fn distribution(d: JoinDistribution) -> pb::JoinDistribution {
    match d {
        JoinDistribution::Broadcast => pb::JoinDistribution::Broadcast,
        JoinDistribution::Partitioned => pb::JoinDistribution::Partitioned,
    }
}

fn partitioning(p: &Partitioning) -> pb::Partitioning {
    use pb::partitioning::Kind;
    pb::Partitioning {
        kind: Some(match p {
            Partitioning::Hash { keys } => Kind::Hash(pb::HashPartitioning {
                keys: indexes(keys),
            }),
            Partitioning::Single => Kind::Single(pb::Empty {}),
            Partitioning::Broadcast => Kind::Broadcast(pb::Empty {}),
            Partitioning::RoundRobin => Kind::RoundRobin(pb::Empty {}),
            Partitioning::Range { keys } => Kind::Range(pb::RangePartitioning {
                keys: keys.iter().map(sort_key).collect(),
            }),
        }),
    }
}

fn join_filter(f: &JoinFilterPlan) -> pb::JoinFilter {
    let KeyFilterOptions {
        max_values,
        exact_max_values,
        bloom,
        bloom_max_items,
        bloom_max_bytes,
        bloom_fpp,
        bloom_max_fpp,
    } = f.options;
    pb::JoinFilter {
        id: f.id.0,
        key: f.key as u64,
        options: Some(pb::KeyFilterOptions {
            max_values: max_values as u64,
            exact_max_values: exact_max_values as u64,
            bloom,
            bloom_max_items,
            bloom_max_bytes: bloom_max_bytes as u64,
            bloom_fpp,
            bloom_max_fpp,
        }),
    }
}

fn scan_filter(f: &ScanRuntimeFilter) -> pb::ScanRuntimeFilter {
    pb::ScanRuntimeFilter {
        id: f.id.0,
        column: Some(column_path(&f.column)),
        key_type: type_bytes(&f.key_type),
    }
}

fn row_stage(s: &RowStageSettings) -> pb::RowStageSettings {
    pb::RowStageSettings {
        min_rows: s.min_rows,
        max_pass_rate: s.max_pass_rate,
    }
}

fn aggregate_expr(a: &AggregateExpr) -> pb::AggregateExpr {
    let (function, ignore_nulls) = match a.function {
        AggregateFunction::Count => (pb::AggregateFunction::Count, false),
        AggregateFunction::Sum => (pb::AggregateFunction::Sum, false),
        AggregateFunction::Avg => (pb::AggregateFunction::Avg, false),
        AggregateFunction::Min => (pb::AggregateFunction::Min, false),
        AggregateFunction::Max => (pb::AggregateFunction::Max, false),
        AggregateFunction::First { ignore_nulls } => (pb::AggregateFunction::First, ignore_nulls),
        AggregateFunction::Last { ignore_nulls } => (pb::AggregateFunction::Last, ignore_nulls),
    };
    pb::AggregateExpr {
        function: function as i32,
        ignore_nulls,
        args: indexes(&a.args),
        input: match a.input {
            AggInput::Raw => pb::AggregateInput::Raw,
            AggInput::State => pb::AggregateInput::State,
        } as i32,
        arg_type: type_bytes(&a.arg_type),
        name: a.name.clone(),
    }
}

fn read_source(source: &ReadSource) -> Result<pb::ReadSource> {
    use pb::read_source::Kind;
    let options = |options: &std::collections::BTreeMap<String, OptionValue>| {
        options
            .iter()
            .map(|(k, v)| Ok((k.clone(), option(k, v)?)))
            .collect::<Result<_>>()
    };
    let kind = match source {
        ReadSource::Table { name, options: o } => Kind::Table(pb::TableSource {
            name: name.clone(),
            options: options(o)?,
        }),
        ReadSource::Files {
            format,
            paths,
            options: o,
        } => Kind::Files(pb::FilesSource {
            format: format.clone(),
            paths: paths
                .iter()
                .map(|p| option("path", p))
                .collect::<Result<_>>()?,
            options: options(o)?,
        }),
        _ => {
            return Err(CodecError::Unsupported(
                "a read source this version does not know".to_owned(),
            ));
        }
    };
    Ok(pb::ReadSource { kind: Some(kind) })
}

/// A plan carries plain text and `secret(scope, key)` references only.
fn option(key: &str, value: &OptionValue) -> Result<pb::OptionValue> {
    use pb::option_value::Kind;
    let kind = match value {
        OptionValue::Plain(s) => Kind::Plain(s.clone()),
        OptionValue::Secret(SecretValue::Reference { scope, key }) => Kind::Secret(pb::SecretRef {
            scope: scope.clone(),
            key: key.clone(),
        }),
        OptionValue::Secret(SecretValue::Inline(_)) => {
            return Err(CodecError::Unsupported(format!(
                "the option `{key}` is an inline secret (use a secret reference)"
            )));
        }
        OptionValue::Url(_) => {
            return Err(CodecError::Unsupported(format!(
                "the option `{key}` is a URL with credentials (use a secret reference)"
            )));
        }
    };
    Ok(pb::OptionValue { kind: Some(kind) })
}

fn spark_type(t: &DataType) -> Result<pb::DataType> {
    use pb::data_type::Kind;
    let (kind, precision, scale) = match t {
        DataType::Null => (Kind::Null, 0, 0),
        DataType::Boolean => (Kind::Boolean, 0, 0),
        DataType::Byte => (Kind::Byte, 0, 0),
        DataType::Short => (Kind::Short, 0, 0),
        DataType::Integer => (Kind::Integer, 0, 0),
        DataType::Long => (Kind::Long, 0, 0),
        DataType::Float => (Kind::Float, 0, 0),
        DataType::Double => (Kind::Double, 0, 0),
        DataType::String => (Kind::String, 0, 0),
        DataType::Binary => (Kind::Binary, 0, 0),
        DataType::Decimal(d) => (
            Kind::Decimal,
            u32::from(d.precision()),
            u32::from(d.scale()),
        ),
        DataType::Date => (Kind::Date, 0, 0),
        DataType::Timestamp => (Kind::Timestamp, 0, 0),
        DataType::TimestampNtz => (Kind::TimestampNtz, 0, 0),
        _ => {
            return Err(CodecError::Unsupported(
                "a Spark type this version does not know".to_owned(),
            ));
        }
    };
    Ok(pb::DataType {
        kind: kind as i32,
        precision,
        scale,
    })
}

fn literal(l: &Literal) -> Result<pb::Literal> {
    use pb::literal::Kind;
    let kind = match l {
        Literal::Null(t) => Kind::Null(pb::NullLiteral {
            data_type: t.as_ref().map(spark_type).transpose()?,
        }),
        Literal::Boolean(v) => Kind::Boolean(*v),
        Literal::Byte(v) => Kind::Byte(i32::from(*v)),
        Literal::Short(v) => Kind::Short(i32::from(*v)),
        Literal::Integer(v) => Kind::Integer(*v),
        Literal::Long(v) => Kind::Long(*v),
        Literal::Float(v) => Kind::Float(*v),
        Literal::Double(v) => Kind::Double(*v),
        Literal::Decimal {
            unscaled,
            data_type,
        } => Kind::Decimal(pb::DecimalLiteral {
            unscaled: unscaled.to_le_bytes().to_vec(),
            data_type: Some(spark_type(&DataType::Decimal(*data_type))?),
        }),
        Literal::String(v) => Kind::String(v.clone()),
        Literal::Binary(v) => Kind::Binary(v.clone()),
        Literal::Date(v) => Kind::Date(*v),
        Literal::Timestamp(v) => Kind::Timestamp(*v),
        Literal::TimestampNtz(v) => Kind::TimestampNtz(*v),
        Literal::YearMonthInterval { months, start, end } => {
            Kind::YearMonth(pb::YearMonthLiteral {
                months: *months,
                start: year_month_field(*start) as i32,
                end: year_month_field(*end) as i32,
            })
        }
        Literal::DayTimeInterval { micros, start, end } => Kind::DayTime(pb::DayTimeLiteral {
            micros: *micros,
            start: day_time_field(*start) as i32,
            end: day_time_field(*end) as i32,
        }),
        _ => {
            return Err(CodecError::Unsupported(
                "a literal kind this version does not know".to_owned(),
            ));
        }
    };
    Ok(pb::Literal { kind: Some(kind) })
}

fn year_month_field(f: YearMonthField) -> pb::YearMonthField {
    match f {
        YearMonthField::Year => pb::YearMonthField::Year,
        YearMonthField::Month => pb::YearMonthField::Month,
    }
}

fn day_time_field(f: DayTimeField) -> pb::DayTimeField {
    match f {
        DayTimeField::Day => pb::DayTimeField::Day,
        DayTimeField::Hour => pb::DayTimeField::Hour,
        DayTimeField::Minute => pb::DayTimeField::Minute,
        DayTimeField::Second => pb::DayTimeField::Second,
    }
}
