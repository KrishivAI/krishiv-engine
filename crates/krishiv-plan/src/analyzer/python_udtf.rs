// SPDX-License-Identifier: Apache-2.0
//! Python table functions (UDTFs, eval types 300, 301, 302; design P3-T085
//! §3.1, §3.3): a call of a registered UDTF in `FROM` (or on the right of
//! `LATERAL`) resolves to a [`PlanNode::PythonTableFunction`].
//!
//! As Spark 4.2.0 (`UserDefinedPythonFunction.scala` `builder`,
//! `Analyzer.scala` `ResolveFunctions`, `FunctionTableSubqueryArgumentExpression`):
//! - scalar arguments bind over no input (a lateral call's over the left
//!   side, as outer references the join turns into columns); a named
//!   argument (`x => 1`) keeps its name;
//! - at most one `TABLE(...)` argument
//!   (`TABLE_VALUED_FUNCTION_TOO_MANY_TABLE_ARGUMENTS`), its `PARTITION BY`
//!   and `ORDER BY` bound over its relation;
//! - a UDTF registered without a return type runs `analyze` now, through
//!   the registry ([`super::TableFunctionRegistry::analyze_python_udtf`]):
//!   the call's schema is its answer, its `withSinglePartition`,
//!   `partitionBy`, `orderBy` and `select` apply to the `TABLE` argument
//!   (`PythonUDTFAnalyzeResult.applyToTableArgument`, with its two
//!   `TABLE_VALUED_FUNCTION_REQUIRED_METADATA_*` refusals), each name parsed
//!   by the session's SQL front end as Spark's parser parses it and bound
//!   over the table, and the pickled result stays in the node.

use std::sync::Arc;

use krishiv_common::error::{
    InternalError, LateralJoinWithArrowUdtfUnsupported,
    TableValuedFunctionRequiredMetadataIncompatibleWithCall,
    TableValuedFunctionRequiredMetadataInvalid, TableValuedFunctionTooManyTableArguments,
    UdtfInvalidAliasInRequestedOrderingStringFromAnalyzeMethod,
    UdtfInvalidRequestedSelectedExpressionFromAnalyzeMethodRequiresAlias,
};
use krishiv_types::{DataType, StructField, StructType};

use super::AnalysisError;
use super::exprs::pretty;
use super::resolve::{Attr, Env, Rel, Resolver};
use crate::udtf::{ResolvedUdtf, UdtfAnalysis, UdtfAnalyzeArg, UdtfAnalyzeSpec};
use crate::{
    BoundColumn, Expr, ExprNode, LogicalPlan, NullOrdering, Origin, PlanNode, PythonEvalType,
    ResolvedExpr, SortDirection, SortOrder, SubqueryKind, TableArg, TableArgSpec,
};

/// The operator name errors cite.
const OP: &str = "PythonUDTF";

/// A bound `TABLE(...)` argument before `analyze` applies to it.
pub(super) struct Table {
    pub(super) rel: Rel,
    arg: TableArg,
}

/// Whether `e` is a `TABLE(...)` argument, named or not.
pub(super) fn is_table_argument(e: &Expr) -> bool {
    let inner = match &e.node {
        ExprNode::Alias { expr, .. } => expr.as_ref(),
        _ => e,
    };
    matches!(
        inner.node,
        ExprNode::Subquery {
            kind: SubqueryKind::Table(_),
            ..
        }
    )
}

impl Resolver<'_> {
    /// A call of the registered Python UDTF `entry` with `args` at `p`.
    #[inline(never)]
    pub(super) fn python_udtf_call(
        &mut self,
        p: &Arc<LogicalPlan>,
        entry: &ResolvedUdtf,
        args: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let tables = args.iter().filter(|a| is_table_argument(a)).count();
        if tables > 1 {
            return Err(AnalysisError::new(
                TableValuedFunctionTooManyTableArguments {
                    num: tables.to_string(),
                },
                o,
            ));
        }
        let mut bound: Vec<Arc<Expr>> = Vec::with_capacity(args.len());
        let mut analyze: Vec<UdtfAnalyzeArg> = Vec::with_capacity(args.len());
        let mut table: Option<Table> = None;
        for a in args {
            let (name, value) = match &a.node {
                ExprNode::Alias { expr, name } => (Some(name.clone()), expr),
                _ => (None, a),
            };
            if let ExprNode::Subquery {
                kind: SubqueryKind::Table(spec),
                plan,
            } = &value.node
            {
                let t = self.table_argument(plan, spec, env)?;
                analyze.push(UdtfAnalyzeArg {
                    data_type: row_type(&t.rel.out),
                    value: None,
                    is_table: true,
                    name: name.clone(),
                });
                let mut t = t;
                t.arg.position = bound.len();
                t.arg.name = name;
                table = Some(t);
                continue;
            }
            let cx = Self::cx(&[], &[], env, OP);
            let b = self.bind(value, &cx)?;
            if entry.return_type.is_none() {
                analyze.push(UdtfAnalyzeArg {
                    data_type: b.ty.data_type.clone().unwrap_or(DataType::Null),
                    value: self.constant_value(&b.expr, b.ty.data_type.as_ref(), o)?,
                    is_table: false,
                    name: name.clone(),
                });
            }
            let named = self.named_arg(b.expr, name);
            bound.push(named);
        }
        let (schema, analyzed) = match &entry.return_type {
            Some(t) => (t.clone(), None),
            None => {
                let spec = UdtfAnalyzeSpec {
                    name: entry.udf.name.clone(),
                    version: entry.udf.version,
                    args: analyze,
                };
                let analysis = self
                    .ctx
                    .table_functions
                    .analyze_python_udtf(&spec)
                    .map_err(|e| AnalysisError::new(e, o))?;
                if let Some(t) = &mut table {
                    self.apply_analysis(&entry.udf.name, &analysis, t, env, o)?;
                }
                (analysis.schema.clone(), Some(Arc::clone(&analysis.pickled)))
            }
        };
        let DataType::Struct(fields) = &schema else {
            return Err(AnalysisError::new(
                InternalError {
                    message: format!(
                        "the result of the Python UDTF {} is not a struct",
                        entry.udf.name
                    ),
                },
                o,
            ));
        };
        let qual: Arc<[String]> = Arc::from(Vec::new());
        let out: Vec<Attr> = fields
            .fields()
            .iter()
            .map(|f| Attr {
                col: BoundColumn {
                    id: self.fresh(),
                    name: f.name().to_owned(),
                    data_type: f.data_type().clone(),
                    nullable: f.nullable(),
                },
                qual: Arc::clone(&qual),
                source: None,
                duplicate: false,
            })
            .collect();
        let (kids, table_arg): (Vec<Rel>, Option<TableArg>) = match table {
            Some(t) => (vec![t.rel], Some(t.arg)),
            None => (Vec::new(), None),
        };
        let node = PlanNode::PythonTableFunction {
            args: bound,
            table_arg,
            udf: entry.udf.clone(),
            output: out.iter().map(|a| a.col.clone()).collect(),
            lateral: None,
            analyzed,
        };
        let refs: Vec<&Rel> = kids.iter().collect();
        Ok(self.make(o, node, out, &refs))
    }

    /// A `TABLE(...)` argument: its relation, and its partitioning bound
    /// over it.
    pub(super) fn table_argument(
        &mut self,
        plan: &Arc<LogicalPlan>,
        spec: &TableArgSpec,
        env: &Arc<Env>,
    ) -> Result<Table, AnalysisError> {
        let rel = self.rel(plan, env)?;
        let mut arg = TableArg {
            input: Arc::clone(&rel.plan),
            partition_by: Vec::with_capacity(spec.partition_by.len()),
            order_by: Vec::with_capacity(spec.order_by.len()),
            single_partition: spec.single_partition,
            select: Vec::new(),
            position: 0,
            name: None,
        };
        {
            let rels = [&rel];
            let cx = Self::cx(&rel.out, &rels, env, OP);
            for e in &spec.partition_by {
                arg.partition_by.push(self.bind(e, &cx)?.expr);
            }
            for s in &spec.order_by {
                let b = self.bind(&s.expr, &cx)?;
                arg.order_by.push(SortOrder {
                    expr: b.expr,
                    ..s.clone()
                });
            }
        }
        Ok(Table { rel, arg })
    }

    /// `PythonUDTFAnalyzeResult.applyToTableArgument` (v4.2.0): the
    /// partitioning, ordering and selection `analyze` asked for, on `t`.
    fn apply_analysis(
        &mut self,
        name: &str,
        a: &UdtfAnalysis,
        t: &mut Table,
        env: &Arc<Env>,
        o: Origin,
    ) -> Result<(), AnalysisError> {
        let invalid = |reason: &str| {
            AnalysisError::new(
                TableValuedFunctionRequiredMetadataInvalid {
                    function_name: name.to_owned(),
                    reason: reason.to_owned(),
                },
                o,
            )
        };
        if a.with_single_partition && !a.partition_by.is_empty() {
            return Err(invalid(
                "the 'with_single_partition' field cannot be assigned to true if the 'partition_by' list is non-empty",
            ));
        }
        if !a.order_by.is_empty() && !a.with_single_partition && a.partition_by.is_empty() {
            return Err(invalid(
                "the 'order_by' field cannot be non-empty unless the 'with_single_partition' field is set to true or the 'partition_by' list is non-empty",
            ));
        }
        let repartitioned = t.arg.single_partition || !t.arg.partition_by.is_empty();
        if (a.with_single_partition || !a.partition_by.is_empty()) && repartitioned {
            return Err(AnalysisError::new(
                TableValuedFunctionRequiredMetadataIncompatibleWithCall {
                    function_name: name.to_owned(),
                    requested_metadata: "specified its own required partitioning of the input table"
                        .to_owned(),
                    invalid_function_call_property: "specified the WITH SINGLE PARTITION or PARTITION BY clause; please remove these clauses and retry the query again."
                        .to_owned(),
                },
                o,
            ));
        }
        let rel = t.rel.clone();
        let rels = [&rel];
        let parse_bind =
            |r: &mut Self, text: &str| -> Result<(Arc<Expr>, Arc<Expr>), AnalysisError> {
                let parsed = r.parse_expression(text, o)?;
                let cx = Self::cx(&rel.out, &rels, env, OP);
                let inner = match &parsed.node {
                    ExprNode::Alias { expr, .. } => expr,
                    _ => &parsed,
                };
                Ok((Arc::clone(&parsed), r.bind(inner, &cx)?.expr))
            };
        if a.with_single_partition {
            t.arg.single_partition = true;
        }
        if !a.partition_by.is_empty() {
            let mut keys = Vec::with_capacity(a.partition_by.len());
            for text in &a.partition_by {
                keys.push(parse_bind(self, text)?.1);
            }
            t.arg.partition_by = keys;
        }
        if !a.order_by.is_empty() {
            let mut orders = Vec::with_capacity(a.order_by.len());
            for c in &a.order_by {
                let (parsed, expr) = parse_bind(self, &c.name)?;
                if let ExprNode::Alias { name, .. } = &parsed.node {
                    return Err(AnalysisError::new(
                        UdtfInvalidAliasInRequestedOrderingStringFromAnalyzeMethod {
                            alias_name: name.clone(),
                        },
                        o,
                    ));
                }
                let direction = if c.ascending {
                    SortDirection::Ascending
                } else {
                    SortDirection::Descending
                };
                let nulls = match (c.nulls_first, c.ascending) {
                    (Some(true), _) | (None, true) => NullOrdering::NullsFirst,
                    (Some(false), _) | (None, false) => NullOrdering::NullsLast,
                };
                orders.push(SortOrder {
                    expr,
                    direction,
                    nulls,
                });
            }
            t.arg.order_by = orders;
        }
        if !a.select.is_empty() {
            let mut select = Vec::with_capacity(a.select.len());
            for s in &a.select {
                let (_, expr) = parse_bind(self, &s.name)?;
                let shown = if s.alias.is_empty() {
                    match &expr.node {
                        ExprNode::Resolved(ResolvedExpr::Column(_)) => expr,
                        _ => {
                            return Err(AnalysisError::new(
                                UdtfInvalidRequestedSelectedExpressionFromAnalyzeMethodRequiresAlias {
                                    expression: pretty(&expr, self.ctx.conf.session_zone),
                                },
                                o,
                            ));
                        }
                    }
                } else {
                    self.named_arg(expr, Some(s.alias.clone()))
                };
                select.push(shown);
            }
            t.arg.select = select;
        }
        Ok(())
    }

    /// `text` parsed as Spark's `parser.parseExpression` parses it: the
    /// first item of `SELECT text` as the session's front end lowers it.
    fn parse_expression(&self, text: &str, o: Origin) -> Result<Arc<Expr>, AnalysisError> {
        let Some(frontend) = &self.ctx.sql else {
            return Err(super::bind::unsupported(
                "an 'analyze' result naming columns without a SQL front end",
                o,
            ));
        };
        let plan = frontend.lower(&format!("SELECT {text}"), crate::SourceId(u32::MAX))?;
        let mut p = Arc::clone(plan.root());
        loop {
            match &p.node {
                PlanNode::Project { exprs, .. } if exprs.len() == 1 => {
                    return Ok(Arc::clone(&exprs[0]));
                }
                PlanNode::SubqueryAlias { input, .. } => p = Arc::clone(input),
                _ => {
                    return Err(super::bind::unsupported(
                        "an 'analyze' result column that is not one expression",
                        o,
                    ));
                }
            }
        }
    }

    /// The value of a constant argument as `analyze` sees it (Spark sends
    /// `value.eval()` of a `foldable` argument): a one-row array of the
    /// type's Arrow form, `None` when the argument is not constant.
    fn constant_value(
        &self,
        e: &Arc<Expr>,
        ty: Option<&DataType>,
        o: Origin,
    ) -> Result<Option<arrow_array::ArrayRef>, AnalysisError> {
        // The literals folding leaves alone (it never folds a binary or timestamp value).
        if let ExprNode::Literal(l) = &e.node {
            let array: Option<arrow_array::ArrayRef> = match l {
                crate::Literal::Binary(v) => {
                    Some(Arc::new(arrow_array::BinaryArray::from(vec![v.as_slice()])))
                }
                crate::Literal::Timestamp(v) | crate::Literal::TimestampNtz(v) => {
                    let zone = match ty.map(DataType::to_arrow) {
                        Some(arrow_schema::DataType::Timestamp(_, z)) => z,
                        _ => None,
                    };
                    Some(Arc::new(
                        arrow_array::TimestampMicrosecondArray::from(vec![*v])
                            .with_timezone_opt(zone),
                    ))
                }
                _ => None,
            };
            if array.is_some() {
                return Ok(array);
            }
        }
        let Some(folded) = self.fold_array(e, self.ctx.conf.ansi) else {
            return Ok(None);
        };
        let array = folded.map_err(|err| AnalysisError::new(err, o))?;
        // An untyped `NULL`: Spark's `void`.
        if ty.is_none() {
            return Ok(Some(arrow_array::new_null_array(
                &arrow_schema::DataType::Null,
                1,
            )));
        }
        Ok(Some(array))
    }
}

/// The type of a `TABLE` argument's rows: a struct of its columns.
pub(super) fn row_type(out: &[Attr]) -> DataType {
    DataType::Struct(StructType::new(out.iter().map(|a| {
        StructField::new(a.col.name.clone(), a.col.data_type.clone(), a.col.nullable)
    })))
}

impl Resolver<'_> {
    /// `e` under the name of a named argument: a resolved alias.
    fn named_arg(&mut self, e: Arc<Expr>, name: Option<String>) -> Arc<Expr> {
        match name {
            Some(name) => {
                let o = e.origin;
                let output = BoundColumn {
                    id: self.fresh(),
                    name,
                    data_type: e.resolved_type().unwrap_or(DataType::Null),
                    nullable: true,
                };
                Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Alias { expr: e, output }))
                        .with_origin(o),
                )
            }
            None => e,
        }
    }
}

impl Resolver<'_> {
    /// `l JOIN LATERAL f(..)` where `f` is a Python UDTF (Spark's
    /// `Generate(PythonUDTF)` over the left side, `requiredChildOutput` all
    /// of it): one node over `l` whose rows are each left row's columns,
    /// then the rows the call gave for it. `None` when `r` is not such a
    /// call. A `LEFT JOIN LATERAL` gives the same rows (Spark 4.2.0's
    /// goldens: a row with no result shows nowhere).
    pub(super) fn lateral_udtf(
        &mut self,
        o: Origin,
        l: &Rel,
        r: &Rel,
        kind: crate::JoinKind,
        condition: &crate::JoinCondition,
        env: &Arc<Env>,
    ) -> Result<Option<Rel>, AnalysisError> {
        let mut node = &r.plan;
        while let PlanNode::SubqueryAlias { input, .. } = &node.node {
            node = input;
        }
        let PlanNode::PythonTableFunction {
            args,
            table_arg: None,
            udf,
            output,
            lateral: None,
            analyzed,
        } = &node.node
        else {
            return Ok(None);
        };
        check_lateral(udf, o)?;
        // An inner join's condition filters the joined rows (Spark 4.2.0 runs `JOIN LATERAL f(..) ON c` so);
        // an outer one's rows are not verified against a Spark run yet, nor are `USING` and `NATURAL`.
        let filter = match condition {
            crate::JoinCondition::None => None,
            crate::JoinCondition::On(c) if kind == crate::JoinKind::Inner => Some(c),
            _ => {
                return Err(super::bind::unsupported(
                    "LEFT JOIN LATERAL with a join condition, or USING or NATURAL, on a Python table function",
                    o,
                ));
            }
        };
        let left: std::collections::HashSet<crate::AttrId> = l
            .out
            .iter()
            .chain(l.hidden.iter())
            .map(|a| a.col.id)
            .collect();
        let mut columns = |e: Arc<Expr>| -> Result<Arc<Expr>, std::convert::Infallible> {
            Ok(match &e.node {
                ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) if left.contains(&c.id) => {
                    Arc::new(
                        Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c.clone())))
                            .with_origin(e.origin),
                    )
                }
                _ => e,
            })
        };
        let mut bound = Vec::with_capacity(args.len());
        for a in args {
            let Ok(e) = Arc::clone(a).transform_up(&mut columns);
            bound.push(e);
        }
        let node = PlanNode::PythonTableFunction {
            args: bound,
            table_arg: None,
            udf: udf.clone(),
            output: output.clone(),
            lateral: Some(Arc::clone(&l.plan)),
            analyzed: analyzed.clone(),
        };
        let out: Vec<Attr> = l.out.iter().chain(r.out.iter()).cloned().collect();
        let mut rel = self.make(o, node, out, &[l, r]);
        rel.outer = l.outer;
        let Some(c) = filter else {
            return Ok(Some(rel));
        };
        let rels = [&rel];
        let b = self.bind(c, &Self::cx(&rel.out, &rels, env, "Join"))?;
        if b.ty.data_type != Some(DataType::Boolean) {
            return Err(AnalysisError::new(
                krishiv_common::error::JoinConditionIsNotBooleanType {
                    join_condition: format!(
                        "\"{}\"",
                        pretty(&b.shown(), self.ctx.conf.session_zone)
                    ),
                    condition_type: super::coerce::quoted_type(b.ty.data_type.as_ref()),
                },
                o,
            ));
        }
        let out = rel.out.to_vec();
        let filtered = self.make(
            o,
            PlanNode::Filter {
                input: Arc::clone(&rel.plan),
                predicate: b.expr,
            },
            out,
            &[&rel],
        );
        Ok(Some(filtered))
    }
}

/// A `TABLE(...)` argument anywhere but a Python table function's call (which takes it before binding).
pub(super) fn refuse_table_argument(kind: &SubqueryKind, o: Origin) -> Result<(), AnalysisError> {
    match kind {
        SubqueryKind::Table(_) => Err(super::bind::unsupported(
            "a TABLE argument of this function",
            o,
        )),
        _ => Ok(()),
    }
}

/// Spark's refusal of an Arrow UDTF (302) on the right of `LATERAL`.
pub(super) fn check_lateral(udf: &crate::PythonUdfRef, o: Origin) -> Result<(), AnalysisError> {
    if udf.eval_type == PythonEvalType::ArrowUdtf {
        return Err(AnalysisError::new(
            LateralJoinWithArrowUdtfUnsupported {},
            o,
        ));
    }
    Ok(())
}
