// SPDX-License-Identifier: Apache-2.0
//! `IN (subquery)` and its `DATATYPE_MISMATCH` errors.

use super::*;

impl Resolver<'_> {
    /// `(values) IN (subquery)`: the values and the subquery's columns, each
    /// pair cast to the type it meets at (the subquery under a projection
    /// that casts). Returns the values and the subquery plan.
    pub(super) fn in_subquery(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        rel: &Rel,
        cx: &Cx<'_>,
    ) -> Result<(Vec<Bound>, Arc<LogicalPlan>), AnalysisError> {
        let shown = with_children(e, kids.iter().map(Bound::shown).collect());
        let zone = self.ctx.conf.session_zone;
        let sql_expr = format!("\"{}\"", pretty(&shown, zone));
        let texts = |v: Vec<String>| v.join(", ");
        if kids.len() != rel.out.len() {
            return Err(AnalysisError::new(
                DatatypeMismatchInSubqueryLengthMismatch {
                    sql_expr,
                    left_length: kids.len().to_string(),
                    left_columns: texts(
                        kids.iter()
                            .map(|k| format!("\"{}\"", pretty(&k.shown(), zone)))
                            .collect(),
                    ),
                    right_length: rel.out.len().to_string(),
                    right_columns: texts(
                        rel.out
                            .iter()
                            .map(|a| format!("\"{}\"", a.col.name))
                            .collect(),
                    ),
                },
                e.origin,
            ));
        }
        let qualified = |qual: &[String], name: &str, ty: Option<&DataType>| {
            let mut parts = qual.to_vec();
            parts.push(name.to_owned());
            let ty = ty.map_or_else(|| "void".to_owned(), DataType::simple_string);
            format!("{}:{ty}", parts.join("."))
        };
        let mut targets = Vec::new();
        let mut bad = Vec::new();
        for (k, a) in kids.iter().zip(rel.out.iter()) {
            let out = Arg {
                ty: Some(&a.col.data_type),
                nullable: a.col.nullable,
                literal: None,
            };
            match coerce::common_type(&arg_of(k), &out, self.ctx.conf.case_sensitive) {
                Some(t) => targets.push(t),
                None => {
                    let left = format!(
                        "{}:{}",
                        pretty(
                            &super::super::ops::qualified(&k.shown(), cx.scope),
                            self.ctx.conf.session_zone
                        ),
                        k.ty.data_type
                            .as_ref()
                            .map_or_else(|| "void".to_owned(), DataType::simple_string)
                    );
                    let right = qualified(&a.qual, &a.col.name, Some(&a.col.data_type));
                    bad.push(format!("{left}, {right}"));
                }
            }
        }
        if !bad.is_empty() {
            return Err(AnalysisError::new(
                DatatypeMismatchInSubqueryDataTypeMismatch {
                    sql_expr,
                    mismatched_columns: format!("({})", bad.join(", ")),
                    left_type: texts(
                        kids.iter()
                            .map(|k| quoted_type(k.ty.data_type.as_ref()))
                            .collect(),
                    ),
                    right_type: texts(
                        rel.out
                            .iter()
                            .map(|a| quoted_type(Some(&a.col.data_type)))
                            .collect(),
                    ),
                },
                e.origin,
            ));
        }
        let values = kids
            .iter()
            .zip(&targets)
            .map(|(k, t)| {
                self.implicit_cast(k, coerce::cast_to(k.ty.data_type.as_ref(), t.as_ref()))
            })
            .collect();
        let needs = rel
            .out
            .iter()
            .zip(&targets)
            .any(|(a, t)| t.as_ref().is_some_and(|t| *t != a.col.data_type));
        if !needs {
            return Ok((values, Arc::clone(&rel.plan)));
        }
        let mut exprs = Vec::new();
        let mut out = Vec::new();
        for (a, t) in rel.out.iter().zip(targets) {
            let column = Bound {
                expr: super::super::ops::col_expr(&a.col, Origin::UNKNOWN),
                ty: column(&a.col),
                agg: None,
                win: None,
                outer: None,
                lateral: Vec::new(),
                shown: None,
            };
            let to = coerce::cast_to(Some(&a.col.data_type), t.as_ref());
            if to.is_none() {
                exprs.push(column.expr);
                out.push(a.clone());
                continue;
            }
            let cast = self.implicit_cast(&column, to);
            let output = self.attr(&a.col.name, &cast.ty);
            exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: cast.expr,
                    output: output.clone(),
                },
            ))));
            out.push(Attr {
                col: output,
                ..a.clone()
            });
        }
        Ok((
            values,
            self.projection(Origin::UNKNOWN, rel, exprs, out).plan,
        ))
    }

    /// The analysis error of a call that does not type.
    pub(in crate::analyzer) fn mismatch(
        &self,
        m: Mismatch,
        call: &Arc<Expr>,
        kids: &[Bound],
        function: &str,
        origin: Origin,
    ) -> AnalysisError {
        let zone = self.ctx.conf.session_zone;
        // `mask` materializes its omitted arguments before typing (Spark's
        // constructor writes them into the call), so an input-type error
        // quotes the expanded call, as Spark does
        // (`mask(c1, replaceArg, x, n, NULL)`).
        let sql_expr = match function {
            "mask" if kids.len() == 5 => {
                let args: Vec<String> = kids.iter().map(|k| pretty(&k.shown(), zone)).collect();
                format!("\"{function}({})\"", args.join(", "))
            }
            _ => format!("\"{}\"", pretty(call, zone)),
        };
        let error: KrishivError = match m {
            Mismatch::DiffTypes(left, right) => DatatypeMismatchBinaryOpDiffTypes {
                sql_expr,
                left: quoted_type(left.as_ref()),
                right: quoted_type(right.as_ref()),
            }
            .into(),
            Mismatch::WrongType(actual, input) => DatatypeMismatchBinaryOpWrongType {
                sql_expr,
                input_type: input.to_owned(),
                actual_data_type: quoted_type(actual.as_ref()),
            }
            .into(),
            Mismatch::Unexpected {
                arg,
                param,
                ty,
                required,
            } => {
                // `extract(year FROM x)` and `date_part('year', x)` are
                // `year(x)` to Spark, which names that in the error.
                let (sql_expr, param) = match extract_function(function, call) {
                    Some(name) if param == 1 => (
                        format!(
                            "\"{name}({})\"",
                            kids.get(1)
                                .map(|k| pretty(&k.shown(), zone))
                                .unwrap_or_default()
                        ),
                        0,
                    ),
                    _ => (sql_expr, param),
                };
                DatatypeMismatchUnexpectedInputType {
                    sql_expr,
                    param_index: ordinal(param),
                    required_type: required,
                    input_sql: format!(
                        "\"{}\"",
                        kids.get(arg)
                            .map(|k| pretty(&k.shown(), zone))
                            .unwrap_or_default()
                    ),
                    input_type: quoted_type(ty.as_ref()),
                }
                .into()
            }
            Mismatch::DateDays { sub, date, ty } => {
                let shown = |i: usize| {
                    kids.get(i)
                        .map(|k| pretty(&k.shown(), zone))
                        .unwrap_or_default()
                };
                DatatypeMismatchUnexpectedInputType {
                    sql_expr: format!(
                        "\"{}({}, {})\"",
                        if sub { "date_sub" } else { "date_add" },
                        shown(date),
                        shown(1 - date)
                    ),
                    param_index: ordinal(1),
                    required_type: r#"("INT" or "SMALLINT" or "TINYINT")"#.to_owned(),
                    input_sql: format!("\"{}\"", shown(1 - date)),
                    input_type: quoted_type(ty.as_ref()),
                }
                .into()
            }

            Mismatch::DateAddText => {
                let shown = |i: usize| {
                    kids.get(i)
                        .map(|k| pretty(&k.shown(), zone))
                        .unwrap_or_default()
                };
                DatatypeMismatchUnexpectedInputType {
                    sql_expr: format!("\"date_add({}, {})\"", shown(0), shown(1)),
                    param_index: ordinal(1),
                    required_type: r#"("INT" or "SMALLINT" or "TINYINT")"#.to_owned(),
                    input_sql: format!("\"{}\"", shown(1)),
                    input_type: r#""DATE""#.to_owned(),
                }
                .into()
            }

            Mismatch::DataDiff(types, list) => {
                let types: Vec<String> = types.iter().map(|t| quoted_type(t.as_ref())).collect();
                let data_type = if list {
                    format!("[{}]", types.join(", "))
                } else {
                    format!("({})", types.join(" or "))
                };
                let function_name = match function {
                    "when" => "casewhen",
                    other => other,
                };
                DatatypeMismatchDataDiffTypes {
                    sql_expr,
                    function_name: quoted(&[function_name.to_owned()]),
                    data_type,
                }
                .into()
            }
            Mismatch::CreateMapKeyDiff(types) => DatatypeMismatchCreateMapKeyDiffTypes {
                sql_expr,
                function_name: "`map`".to_owned(),
                data_type: list_types(&types),
            }
            .into(),
            Mismatch::CreateMapValueDiff(types) => DatatypeMismatchCreateMapValueDiffTypes {
                sql_expr,
                function_name: "`map`".to_owned(),
                data_type: list_types(&types),
            }
            .into(),
            Mismatch::MapConcatDiffTypes(types) => DatatypeMismatchMapConcatDiffTypes {
                sql_expr,
                function_name: "`map_concat`".to_owned(),
                data_type: list_types(&types),
            }
            .into(),
            Mismatch::InvalidOrdering(t) => DatatypeMismatchInvalidOrderingType {
                sql_expr,
                function_name: quoted(&[function.to_owned()]),
                data_type: quoted_type(Some(&t)),
            }
            .into(),
            Mismatch::Unsupported(what) => return unsupported(what, origin),
        };
        AnalysisError::new(error, origin)
    }
}
