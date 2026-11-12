// SPDX-License-Identifier: Apache-2.0
//! Connect expressions to IR expressions, after Spark's
//! `SparkConnectPlanner.transformExpression`.
//!
//! Supported: literals, attributes (with their `plan_id`), functions and
//! operators (canonical names), `Alias`, `Cast`, `*`, and sort orders in
//! `Sort`. SQL strings (`selectExpr`, `where("...")`, `expr()`) need the
//! lowering of the SQL parser's syntax tree into the IR, which comes with the
//! analyzer (P2-T071a): until then they fail with
//! `KRISHIV_UNSUPPORTED_FEATURE`.

use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    Builtin, CastMode, ColumnRef, Expr, ExprNode, FunctionCall, Literal, NullOrdering, PlanId,
    SortDirection, SortOrder, Star,
};

use super::errors::{
    expression_not_set, field_empty, not_set, unsupported_field, unsupported_variant,
};
use super::literal::literal;
use super::names::{parse_attribute_name, parse_multipart_identifier};
use super::types::data_type;
use super::{Result, Translator};
use crate::proto::v4_2::Expression;
use crate::proto::v4_2::expression::{
    self, ExprType, cast::CastToType, cast::EvalMode, sort_order::NullOrdering as ProtoNulls,
    sort_order::SortDirection as ProtoDirection,
};

impl Translator<'_> {
    /// Translates an expression.
    pub(crate) fn expression(&mut self, e: &Expression) -> Result<Arc<Expr>> {
        self.count()?;
        // A leaf has nothing below it to recurse into, so it needs no thread
        // past the inline depth; it still is a level of the IR.
        if matches!(
            e.expr_type,
            None | Some(
                ExprType::Literal(_)
                    | ExprType::UnresolvedAttribute(_)
                    | ExprType::UnresolvedStar(_)
            )
        ) {
            if self.depth >= krishiv_plan::MAX_DEPTH {
                return Err(krishiv_common::error::FailedToParseTooComplex {}.into());
            }
            return self.expression_inner(e);
        }
        self.nested(|t| t.expression_inner(e))
    }

    /// Translates a required (`Option`al in the proto) expression.
    pub(crate) fn required_expression(&mut self, e: Option<&Expression>) -> Result<Arc<Expr>> {
        self.expression(e.ok_or_else(expression_not_set)?)
    }

    fn expression_inner(&mut self, e: &Expression) -> Result<Arc<Expr>> {
        let Some(kind) = &e.expr_type else {
            return Err(expression_not_set());
        };
        let node = match kind {
            ExprType::Literal(l) => ExprNode::Literal(literal(l)?),
            ExprType::UnresolvedAttribute(a) => {
                if a.is_metadata_column == Some(true) {
                    return Err(unsupported_field(
                        "Expression.UnresolvedAttribute",
                        "is_metadata_column",
                        "metadata columns",
                    ));
                }
                ExprNode::Column(ColumnRef {
                    name: parse_attribute_name(&a.unparsed_identifier)?,
                    plan_id: a.plan_id.map(PlanId),
                    by_identifier: false,
                })
            }
            ExprType::UnresolvedFunction(f) => self.function(f)?,
            ExprType::CallFunction(c) => self.call_function(c)?,
            ExprType::NamedArgumentExpression(_) => {
                // Only as an argument of a function call (`call_args`); Spark's analyzer
                // refuses it anywhere else.
                return Err(unsupported_field(
                    "Expression.NamedArgumentExpression",
                    "key",
                    "named arguments outside a function call's arguments",
                ));
            }
            ExprType::UnresolvedStar(s) => ExprNode::Star(star(s)?),
            ExprType::Alias(a) => {
                if a.metadata.as_deref().is_some_and(|m| !m.is_empty()) {
                    return Err(unsupported_field(
                        "Expression.Alias",
                        "metadata",
                        "column aliases with metadata",
                    ));
                }
                let [name] = a.name.as_slice() else {
                    return if a.name.is_empty() {
                        Err(field_empty("name", "spark.connect.Expression.Alias"))
                    } else {
                        Err(unsupported_field(
                            "Expression.Alias",
                            "name",
                            "aliases with several names",
                        ))
                    };
                };
                ExprNode::Alias {
                    expr: self.required_expression(a.expr.as_deref())?,
                    name: name.clone(),
                }
            }
            ExprType::Cast(c) => {
                let to = match &c.cast_to_type {
                    Some(CastToType::Type(t)) => data_type(t)?.refuse_collations()?,
                    Some(CastToType::TypeStr(s)) => {
                        krishiv_types::parse_data_type(s)?.refuse_collations()?
                    }
                    None => {
                        return Err(not_set(
                            "spark.connect.Expression.Cast",
                            "CAST_TO_TYPE_NOT_SET",
                        ));
                    }
                };
                let mode = match EvalMode::try_from(c.eval_mode) {
                    Ok(EvalMode::Legacy) => CastMode::Legacy,
                    Ok(EvalMode::Ansi) => CastMode::Ansi,
                    Ok(EvalMode::Try) => CastMode::Try,
                    Ok(EvalMode::Unspecified) | Err(_) => CastMode::Default,
                };
                ExprNode::Cast {
                    expr: self.required_expression(c.expr.as_deref())?,
                    to,
                    mode,
                }
            }
            other => {
                return Err(unsupported_variant(
                    "spark.connect.Expression",
                    other.variant(),
                ));
            }
        };
        Ok(Arc::new(Expr::new(node)))
    }

    /// A function call. Operators take their canonical name (see
    /// [`krishiv_plan::function`]): `=` is `==`, and `!=` and `<>` are
    /// `not(==(..))`; any other function keeps the name it was sent under.
    fn function(&mut self, f: &expression::UnresolvedFunction) -> Result<ExprNode> {
        if f.is_user_defined_function {
            return Err(unsupported_field(
                "Expression.UnresolvedFunction",
                "is_user_defined_function",
                "calls of functions by qualified name",
            ));
        }
        if f.is_internal == Some(true) {
            return Err(unsupported_field(
                "Expression.UnresolvedFunction",
                "is_internal",
                "calls of internal functions",
            ));
        }
        let (args, arg_names) = self.call_args(&f.arguments)?;
        let call = |name: &str, args: Vec<Arc<Expr>>, distinct: bool| FunctionCall {
            name: name.to_owned(),
            qualifier: Vec::new(),
            arg_names: arg_names.clone(),
            args,
            distinct,
            filter: None,
            within_group: Vec::new(),
        };
        let name = f.function_name.as_str();
        Ok(match name {
            "=" | "==" => ExprNode::Function(call(Builtin::Eq.name(), args, f.is_distinct)),
            "!=" | "<>" => {
                let equal = Expr::new(ExprNode::Function(call(Builtin::Eq.name(), args, false)));
                // `not` has one argument, so it carries none of the call's argument names.
                let mut not = call(Builtin::Not.name(), vec![Arc::new(equal)], f.is_distinct);
                not.arg_names = Vec::new();
                ExprNode::Function(not)
            }
            // `F.collate(col, "name")` / `Column.collate`: the function form of `expr COLLATE name`.
            _ if name.eq_ignore_ascii_case("collate")
                && !f.is_distinct
                && let [value, collation] = args.as_slice()
                && let ExprNode::Literal(Literal::String(text)) = &collation.node =>
            {
                ExprNode::Collate {
                    expr: Arc::clone(value),
                    collation: krishiv_types::Collation::parse(text)?,
                    name: text.clone(),
                }
            }
            _ => {
                let canonical = Builtin::from_name(name)
                    .map_or_else(|| name.to_owned(), |b| b.name().to_owned());
                ExprNode::Function(call(&canonical, args, f.is_distinct))
            }
        })
    }

    /// `CallFunction`: Spark's `UnresolvedFunction(parseMultipartIdentifier(name), args, false)`.
    /// The last part of the name is the function, the parts before it its qualifier.
    fn call_function(&mut self, c: &crate::proto::v4_2::CallFunction) -> Result<ExprNode> {
        let mut parts = parse_multipart_identifier(&c.function_name)?;
        let name = parts.pop().unwrap_or_default();
        let (args, arg_names) = self.call_args(&c.arguments)?;
        Ok(ExprNode::Function(FunctionCall {
            name,
            qualifier: parts,
            arg_names,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
        }))
    }

    /// The arguments of a call and their parameter names: a `NamedArgumentExpression` is
    /// `name => value`. The names are empty when every argument is positional.
    fn call_args(&mut self, arguments: &[Expression]) -> Result<CallArgs> {
        let mut args = Vec::with_capacity(arguments.len());
        let mut names = Vec::with_capacity(arguments.len());
        for a in arguments {
            if let Some(ExprType::NamedArgumentExpression(n)) = &a.expr_type {
                args.push(self.required_expression(n.value.as_deref())?);
                names.push(Some(n.key.clone()));
            } else {
                args.push(self.expression(a)?);
                names.push(None);
            }
        }
        if names.iter().all(Option::is_none) {
            names.clear();
        }
        Ok((args, names))
    }

    /// A sort key. As in Spark, anything but `SORT_DIRECTION_ASCENDING` is
    /// descending and anything but `SORT_NULLS_FIRST` is nulls last, so a
    /// key that leaves them unspecified sorts `DESC NULLS LAST`.
    pub(crate) fn sort_order(&mut self, o: &expression::SortOrder) -> Result<SortOrder> {
        let direction = match ProtoDirection::try_from(o.direction) {
            Ok(ProtoDirection::Ascending) => SortDirection::Ascending,
            _ => SortDirection::Descending,
        };
        let nulls = match ProtoNulls::try_from(o.null_ordering) {
            Ok(ProtoNulls::SortNullsFirst) => NullOrdering::NullsFirst,
            _ => NullOrdering::NullsLast,
        };
        Ok(SortOrder {
            expr: self.required_expression(o.child.as_deref())?,
            direction,
            nulls,
        })
    }
}

/// The arguments of a call and the parameter name of each (empty when all are positional).
type CallArgs = (Vec<Arc<Expr>>, Vec<Option<String>>);

/// `*`, `t.*` (the target ends in `.*`) or the star of one DataFrame
/// (`plan_id`); not both, as in Spark.
fn star(s: &expression::UnresolvedStar) -> Result<Star> {
    match (&s.unparsed_target, s.plan_id) {
        (None, plan_id) => Ok(Star {
            regex: None,
            replace: Vec::new(),
            position: None,
            qualifier: Vec::new(),
            except: Vec::new(),
            plan_id: plan_id.map(PlanId),
        }),
        (Some(target), None) => {
            let Some(path) = target.strip_suffix(".*") else {
                return Err(KrishivError::from(
                    krishiv_common::error::ConnectInvalidPlanUnresolvedStarTargetInvalid {
                        target: target.clone(),
                    },
                ));
            };
            Ok(Star {
                regex: None,
                replace: Vec::new(),
                position: None,
                qualifier: parse_attribute_name(path)?,
                except: Vec::new(),
                plan_id: None,
            })
        }
        (Some(_), Some(_)) => Err(KrishivError::from(
            krishiv_common::error::ConnectInvalidPlanUnresolvedStarWithBothTargetAndPlanId {},
        )),
    }
}

#[cfg(test)]
mod tests;
