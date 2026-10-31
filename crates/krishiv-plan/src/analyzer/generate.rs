// SPDX-License-Identifier: Apache-2.0
//! Generators (`explode`, `posexplode`, `inline` and their `_outer` forms)
//! as [`PlanNode::Generate`] relations: Spark's `Generate` with the
//! argument checks of its `ExplodeBase` and `Inline` expressions.
//!
//! Lowering builds the unresolved node: the generator's kind, its argument
//! and the names written after `AS`. Here the argument is bound over the
//! input, its type picks the generated columns, and the node outputs the
//! input's attributes followed by them.

use std::sync::Arc;

use krishiv_common::error::{
    DatatypeMismatchStackColumnDiffTypes, DatatypeMismatchUnexpectedInputType,
    UdtfAliasNumberMismatch, UnsupportedGeneratorMultiGenerator,
    UnsupportedGeneratorNestedInExpressions,
};
use krishiv_types::DataType;

use super::AnalysisError;
use super::bind::unsupported;
use super::coerce::quoted_type;
use super::exprs::{Bound, Cx, pretty};
use super::ops::col_expr;
use super::resolve::{Attr, Env, Rel, Resolver, Widen};
use crate::{
    BoundColumn, Expr, ExprNode, GeneratorKind, Literal, LogicalPlan, Origin, PlanNode,
    ResolvedExpr,
};

/// The most rows `stack` generates: its rows become expressions.
const STACK_MAX_ROWS: usize = 4096;

/// The generated columns' names and types for an argument of type `ty`, or
/// `None` when the generator does not take the type.
fn generated(kind: GeneratorKind, ty: &DataType) -> Option<Vec<(String, DataType, bool)>> {
    match (kind, ty) {
        (GeneratorKind::Explode | GeneratorKind::PosExplode, DataType::Array(a)) => {
            let mut cols = Vec::new();
            if kind == GeneratorKind::PosExplode {
                cols.push(("pos".to_owned(), DataType::Integer, false));
            }
            cols.push((
                "col".to_owned(),
                a.element_type().clone(),
                a.contains_null(),
            ));
            Some(cols)
        }
        (GeneratorKind::Explode | GeneratorKind::PosExplode, DataType::Map(m)) => {
            let mut cols = Vec::new();
            if kind == GeneratorKind::PosExplode {
                cols.push(("pos".to_owned(), DataType::Integer, false));
            }
            cols.push(("key".to_owned(), m.key_type().clone(), false));
            cols.push((
                "value".to_owned(),
                m.value_type().clone(),
                m.value_contains_null(),
            ));
            Some(cols)
        }
        (GeneratorKind::Inline, DataType::Array(a)) => match a.element_type() {
            DataType::Struct(s) => Some(
                s.fields()
                    .iter()
                    // A NULL element of the array gives NULL fields.
                    .map(|f| {
                        (
                            f.name().to_owned(),
                            f.data_type().clone(),
                            f.nullable() || a.contains_null(),
                        )
                    })
                    .collect(),
            ),
            _ => None,
        },
        _ => None,
    }
}

impl Resolver<'_> {
    /// A generator over `input`: unresolved (`output` empty: the argument
    /// is bound and the columns derived) or resolved and read again (the
    /// argument bound again, the columns as they were).
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the node's fields, as every resolution of a relation takes them"
    )]
    pub(super) fn generate_node(
        &mut self,
        origin: Origin,
        input: &Arc<LogicalPlan>,
        kind: GeneratorKind,
        outer: bool,
        arg: &Arc<Expr>,
        output: &[BoundColumn],
        names: &[String],
        qualifier: Option<&String>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let rels = [&c];
        let cx = Self::cx(&c.out, &rels, env, "Generate");
        let arg = &self.stack_rows(arg, &cx)?;
        let b = self.bind(arg, &cx)?;
        if super::ops::has_generator(&b.expr) {
            return Err(AnalysisError::new(
                UnsupportedGeneratorNestedInExpressions {
                    expression: format!("\"{}\"", pretty(&b.expr, self.ctx.conf.session_zone)),
                },
                b.expr.origin,
            ));
        }
        let resolved = !output.is_empty();
        let columns = if resolved {
            output.to_vec()
        } else {
            self.generated_columns(origin, kind, outer, &b, names)?
        };
        // The argument is a column of its own, which the node reads but
        // does not carry: an exploded array is not copied once per element.
        // A resolved node read again already has it as a column.
        let (below, arg_col) = match crate::generator_arg(&b.expr) {
            Some(_) if resolved => (c.clone(), Arc::clone(&b.expr)),
            _ => {
                let col = self.attr("_generate_arg", &b.ty);
                let mut items: Vec<Arc<Expr>> = c
                    .out
                    .iter()
                    .map(|a| col_expr(&a.col, Origin::UNKNOWN))
                    .collect();
                items.push(Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::Alias {
                        expr: Arc::clone(&b.expr),
                        output: col.clone(),
                    },
                ))));
                let mut out = c.out.to_vec();
                out.push(Attr {
                    col: col.clone(),
                    qual: super::ops::no_qual(),
                    source: None,
                });
                let below = self.projection(origin, &c, items, out);
                (below, col_expr(&col, Origin::UNKNOWN))
            }
        };
        let carried: Vec<Attr> = below
            .out
            .iter()
            .filter(|a| crate::generator_arg(&arg_col) != Some(a.col.id))
            .cloned()
            .collect();
        let qual: Arc<[String]> = Arc::from(qualifier.cloned().into_iter().collect::<Vec<_>>());
        let mut out = carried;
        out.extend(columns.iter().map(|col| Attr {
            col: col.clone(),
            qual: Arc::clone(&qual),
            source: None,
        }));
        let node = PlanNode::Generate {
            input: Arc::clone(&below.plan),
            kind,
            outer,
            arg: arg_col,
            output: columns,
            names: Vec::new(),
            qualifier: None,
        };
        Ok(self.make(origin, node, out, &[&below]))
    }

    /// `stack(n, e1, ..., ek)` as the array of `n` structs (`col0`, `col1`,
    /// ... the fields; a short last row is padded with NULL) `inline`
    /// generates its rows from, after Spark's `Stack` checks: the count is
    /// an INT literal and every column holds one type (NULLs aside). Any
    /// other argument is returned as it is.
    fn stack_rows(&mut self, arg: &Arc<Expr>, cx: &Cx<'_>) -> Result<Arc<Expr>, AnalysisError> {
        let ExprNode::Function(f) = &arg.node else {
            return Ok(Arc::clone(arg));
        };
        if f.unqualified() != Some("stack") || f.args.len() < 2 {
            return Ok(Arc::clone(arg));
        }
        let origin = arg.origin;
        let bound = f
            .args
            .iter()
            .map(|a| self.bind(a, cx))
            .collect::<Result<Vec<_>, _>>()?;
        let shown = format!(
            "\"stack({})\"",
            bound
                .iter()
                .map(|b| pretty(&b.expr, self.ctx.conf.session_zone))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let count = &bound[0];
        if count.ty.data_type != Some(DataType::Integer) {
            let input = pretty(&count.expr, self.ctx.conf.session_zone);
            return Err(AnalysisError::new(
                DatatypeMismatchUnexpectedInputType {
                    sql_expr: shown,
                    param_index: "first".to_owned(),
                    required_type: "\"INT\"".to_owned(),
                    input_sql: format!("\"{input}\""),
                    input_type: quoted_type(count.ty.data_type.as_ref()),
                },
                origin,
            ));
        }
        let rows = match &f.args[0].node {
            ExprNode::Literal(Literal::Integer(n)) => usize::try_from(*n)
                .ok()
                .filter(|n| (1..=STACK_MAX_ROWS).contains(n)),
            _ => None,
        };
        let Some(rows) = rows else {
            return Err(unsupported(
                "stack with a row count that is no INT literal from 1 to 4096",
                origin,
            ));
        };
        let values = &bound[1..];
        let width = values.len().div_ceil(rows);
        for col in 0..width {
            let mut previous: Option<(usize, &DataType)> = None;
            for (i, b) in values.iter().enumerate().skip(col).step_by(width) {
                let Some(ty) = b.ty.data_type.as_ref().filter(|t| **t != DataType::Null) else {
                    continue;
                };
                if let Some((left, left_ty)) = previous.filter(|(_, t)| *t != ty) {
                    return Err(AnalysisError::new(
                        DatatypeMismatchStackColumnDiffTypes {
                            sql_expr: shown,
                            column_index: col.to_string(),
                            left_type: quoted_type(Some(left_ty)),
                            left_param_index: (left + 1).to_string(),
                            right_type: quoted_type(Some(ty)),
                            right_param_index: (i + 1).to_string(),
                        },
                        origin,
                    ));
                }
                previous = Some((i, ty));
            }
        }
        let structs = (0..rows)
            .map(|row| {
                let fields = (0..width).flat_map(|col| {
                    let value = match f.args.get(1 + row * width + col) {
                        Some(v) => (**v).clone(),
                        None => Expr::literal(Literal::Null(None)),
                    };
                    [Expr::literal(Literal::String(format!("col{col}"))), value]
                });
                Expr::function("named_struct", fields.collect()).with_origin(origin)
            })
            .collect();
        Ok(Arc::new(
            Expr::function("array", structs).with_origin(origin),
        ))
    }

    fn generated_columns(
        &mut self,
        origin: Origin,
        kind: GeneratorKind,
        outer: bool,
        b: &Bound,
        names: &[String],
    ) -> Result<Vec<BoundColumn>, AnalysisError> {
        let function = format!("{}{}", kind.name(), if outer { "_outer" } else { "" });
        let Some(cols) = b.ty.data_type.as_ref().and_then(|ty| generated(kind, ty)) else {
            let required = match kind {
                GeneratorKind::Inline => "\"ARRAY<STRUCT>\"".to_owned(),
                _ => "(\"ARRAY\" or \"MAP\")".to_owned(),
            };
            let shown = pretty(&b.expr, None);
            return Err(AnalysisError::new(
                DatatypeMismatchUnexpectedInputType {
                    sql_expr: format!("\"{function}({shown})\""),
                    param_index: "first".to_owned(),
                    required_type: required,
                    input_sql: format!("\"{shown}\""),
                    input_type: quoted_type(b.ty.data_type.as_ref()),
                },
                origin,
            ));
        };
        let renamed = !names.is_empty() && names.len() == cols.len();
        if !names.is_empty() && !renamed {
            return Err(AnalysisError::new(
                UdtfAliasNumberMismatch {
                    aliases_size: cols.len().to_string(),
                    aliases_names: names.join(","),
                },
                origin,
            ));
        }
        Ok(cols
            .into_iter()
            .enumerate()
            .map(|(i, (name, data_type, nullable))| BoundColumn {
                id: self.fresh(),
                name: if renamed { names[i].clone() } else { name },
                data_type,
                // The first generated column of an outer generator is
                // null for an empty input; a position is never.
                nullable: nullable || outer,
            })
            .collect())
    }
}

/// The generator of an aggregate's select list, taken out as Spark's
/// `ExtractGenerator` does: the select item stands for the generator's
/// argument (named `_gen_input_N`) while the aggregate is built, and a
/// [`PlanNode::Generate`] over the aggregate's output generates the
/// columns in the item's place.
pub(super) struct ExtractedGenerator {
    /// The select item the generator stood at.
    index: usize,
    kind: GeneratorKind,
    outer: bool,
    /// The column holding the argument in the aggregate's output.
    input: BoundColumn,
    /// The item's alias, when it had one.
    name: Option<String>,
}

/// Whether `name` (as written) is one of the explode-family generators.
pub(super) fn is_generator_name(name: &str) -> bool {
    generator_kind(&name.to_ascii_lowercase()).is_some()
}

impl ExtractedGenerator {
    /// The select item the generator stood at.
    pub(super) const fn index(&self) -> usize {
        self.index
    }

    /// The id of the column holding the argument.
    pub(super) const fn input_id(&self) -> crate::AttrId {
        self.input.id
    }
}

fn generator_kind(name: &str) -> Option<(GeneratorKind, bool)> {
    let (base, outer) = match name.strip_suffix("_outer") {
        Some(base) => (base, true),
        None => (name, false),
    };
    let kind = match base {
        "explode" => GeneratorKind::Explode,
        "posexplode" => GeneratorKind::PosExplode,
        "inline" => GeneratorKind::Inline,
        _ => return None,
    };
    Some((kind, outer))
}

impl Resolver<'_> {
    /// Replaces the one top-level generator of `items` by its argument. A
    /// generator inside another expression is
    /// `UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS`; the generators other
    /// than the explode family, and several generators, are unsupported.
    pub(super) fn extract_generator(
        &mut self,
        origin: Origin,
        items: Vec<Bound>,
        alias: Option<String>,
    ) -> Result<(Vec<Bound>, Option<ExtractedGenerator>), AnalysisError> {
        let zone = self.ctx.conf.session_zone;
        let mut found = None;
        for (i, b) in items.iter().enumerate() {
            let inner = match &b.expr.node {
                ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => expr,
                _ => &b.expr,
            };
            if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &inner.node
                && f.kind == crate::FunctionKind::Generator
            {
                if let Some(first) = found {
                    let shown = |i: usize| {
                        let e = match &items[i].expr.node {
                            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => expr,
                            _ => &items[i].expr,
                        };
                        format!("\"{}\"", pretty(e, zone))
                    };
                    return Err(AnalysisError::new(
                        UnsupportedGeneratorMultiGenerator {
                            num: "2".to_owned(),
                            generators: format!("{}, {}", shown(first), shown(i)),
                        },
                        b.expr.origin,
                    ));
                }
                found = Some(i);
            } else if super::ops::has_generator(inner) {
                return Err(AnalysisError::new(
                    UnsupportedGeneratorNestedInExpressions {
                        expression: format!("\"{}\"", pretty(inner, zone)),
                    },
                    b.expr.origin,
                ));
            }
        }
        let Some(index) = found else {
            return Ok((items, None));
        };
        // The item's name is the generated one unless the user wrote an
        // alias (`select_items` names every item): only a written alias
        // renames the generated columns.
        let inner = match &items[index].expr.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
            _ => Arc::clone(&items[index].expr),
        };
        let name = alias;
        let ExprNode::Resolved(ResolvedExpr::Function(call)) = &inner.node else {
            return Err(super::bind::unsupported("a generator", origin));
        };
        let (Some((kind, outer)), [arg]) = (generator_kind(call.id.name()), &call.args[..]) else {
            return Err(super::bind::unsupported(
                format!(
                    "the generator {} in an aggregate SELECT list",
                    call.id.name()
                ),
                items[index].expr.origin,
            ));
        };
        let (data_type, nullable) = super::subquery::typed(arg);
        let ty = super::exprs::arg_type(data_type.unwrap_or(DataType::Null), nullable);
        // Spark names the hidden column after the generator's child index:
        // the one argument of the explode family is child 0.
        let input = self.attr("_gen_input_0", &ty);
        let mut items = items;
        let replaced = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                expr: Arc::clone(arg),
                output: input.clone(),
            }))
            .with_origin(items[index].expr.origin),
        );
        items[index] = Bound {
            expr: replaced,
            ty,
            ..items[index].clone()
        };
        let extracted = ExtractedGenerator {
            index,
            kind,
            outer,
            input,
            name,
        };
        Ok((items, Some(extracted)))
    }

    /// The generator over the aggregate `agg`, and a projection restoring
    /// the select list's order with the generated columns where the
    /// generator stood.
    pub(super) fn generate_over_aggregate(
        &mut self,
        origin: Origin,
        agg: &Rel,
        g: &ExtractedGenerator,
    ) -> Result<Rel, AnalysisError> {
        let arg_expr = col_expr(&g.input, origin);
        let arg = Bound {
            ty: super::exprs::arg_type(g.input.data_type.clone(), g.input.nullable),
            expr: arg_expr,
            agg: None,
            win: None,
            outer: None,
            lateral: Vec::new(),
            shown: None,
        };
        let names: Vec<String> = g.name.iter().cloned().collect();
        let columns = self.generated_columns(origin, g.kind, g.outer, &arg, &names)?;
        let qual: Arc<[String]> = Arc::from(Vec::<String>::new());
        let carried: Vec<Attr> = agg
            .out
            .iter()
            .filter(|a| a.col.id != g.input.id)
            .cloned()
            .collect();
        let generated: Vec<Attr> = columns
            .iter()
            .map(|col| Attr {
                col: col.clone(),
                qual: Arc::clone(&qual),
                source: None,
            })
            .collect();
        let mut out = carried;
        out.extend(generated.iter().cloned());
        let node = PlanNode::Generate {
            input: Arc::clone(&agg.plan),
            kind: g.kind,
            outer: g.outer,
            arg: col_expr(&g.input, origin),
            output: columns,
            names: Vec::new(),
            qualifier: None,
        };
        let mut generate = self.make(origin, node, out, &[agg]);
        // A grouping column the select list leaves out (`ORDER BY k`) comes
        // back through the aggregate below.
        generate.widen = Some(Arc::new(Widen::Through { child: agg.clone() }));
        // Select-list order: the aggregate's outputs, the generated
        // columns in the place of the argument.
        let mut exprs = Vec::new();
        let mut order = Vec::new();
        for (i, a) in agg.out.iter().enumerate() {
            if i == g.index {
                for a in &generated {
                    exprs.push(col_expr(&a.col, origin));
                    order.push(a.clone());
                }
            } else {
                exprs.push(col_expr(&a.col, origin));
                order.push(a.clone());
            }
        }
        Ok(self.projection(origin, &generate, exprs, order))
    }
}
