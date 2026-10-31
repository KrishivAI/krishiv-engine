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
    DatatypeMismatchUnexpectedInputType, UdtfAliasNumberMismatch,
    UnsupportedGeneratorNestedInExpressions,
};
use krishiv_types::DataType;

use super::AnalysisError;
use super::coerce::quoted_type;
use super::exprs::{Bound, pretty};
use super::ops::col_expr;
use super::resolve::{Attr, Env, Rel, Resolver};
use crate::{
    BoundColumn, Expr, ExprNode, GeneratorKind, LogicalPlan, Origin, PlanNode, ResolvedExpr,
};

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
