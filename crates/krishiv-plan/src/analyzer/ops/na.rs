// SPDX-License-Identifier: Apache-2.0
//! `na.drop` and `na.fill` (`ColumnNamesOp::DropNa`, `ColumnNamesOp::FillNa`): lowered over the
//! input's columns and their types, as Spark's `DataFrameNaFunctions` does.

use super::*;
use crate::Builtin;

/// Which columns a fill value reaches (Spark's `fillValue` target type).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fills {
    Numeric,
    Text,
    Boolean,
}

impl Fills {
    fn of(value: &Literal) -> Option<Self> {
        match value {
            Literal::Byte(_)
            | Literal::Short(_)
            | Literal::Integer(_)
            | Literal::Long(_)
            | Literal::Float(_)
            | Literal::Double(_)
            | Literal::Decimal { .. } => Some(Self::Numeric),
            Literal::String(_) => Some(Self::Text),
            Literal::Boolean(_) => Some(Self::Boolean),
            _ => None,
        }
    }

    fn reaches(self, ty: &DataType) -> bool {
        match self {
            Self::Numeric => matches!(
                ty,
                DataType::Byte
                    | DataType::Short
                    | DataType::Integer
                    | DataType::Long
                    | DataType::Float
                    | DataType::Double
                    | DataType::Decimal(_)
            ),
            Self::Text => matches!(ty, DataType::String),
            Self::Boolean => matches!(ty, DataType::Boolean),
        }
    }
}

fn call(name: &str, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Function(FunctionCall {
        name: name.to_owned(),
        qualifier: Vec::new(),
        args,
        arg_names: Vec::new(),
        distinct: false,
        filter: None,
        within_group: Vec::new(),
    })))
}

fn is_floating(ty: &DataType) -> bool {
    matches!(ty, DataType::Float | DataType::Double)
}

impl Resolver<'_> {
    /// `na.drop`: a filter keeping the rows with at least `min_non_nulls` values among the
    /// `subset` columns that are neither NULL nor NaN.
    pub(in crate::analyzer) fn drop_na(
        &mut self,
        p: &LogicalPlan,
        c: Rel,
        subset: &[String],
        min_non_nulls: Option<u32>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let cols: Vec<Attr> = if subset.is_empty() {
            c.out.to_vec()
        } else {
            let mut found = Vec::new();
            for name in subset {
                match self.find(&c.out, std::slice::from_ref(name), p.origin, false)? {
                    Some((a, _)) => found.push(a),
                    None => {
                        return Err(crate::analyzer::exprs::unresolved(
                            std::slice::from_ref(name),
                            &c.out,
                            p.origin,
                        ));
                    }
                }
            }
            found
        };
        let needed = min_non_nulls.map_or(cols.len(), |n| n as usize);
        if needed == 0 {
            return Ok(c);
        }
        // One 0/1 term per column, summed: `AtLeastNNonNulls`.
        let one = || Arc::new(Expr::literal(Literal::Integer(1)));
        let zero = || Arc::new(Expr::literal(Literal::Integer(0)));
        let mut terms = cols.iter().map(|a| {
            let value = self.pass_expr(&a.col);
            let mut valid = call(Builtin::IsNotNull.name(), vec![Arc::clone(&value)]);
            if is_floating(&a.col.data_type) {
                let nan = call("isnan", vec![value]);
                valid = call(
                    Builtin::And.name(),
                    vec![valid, call(Builtin::Not.name(), vec![nan])],
                );
            }
            call(Builtin::When.name(), vec![valid, one(), zero()])
        });
        let first = terms.next();
        let Some(sum) =
            first.map(|f| terms.fold(f, |acc, t| call(Builtin::Add.name(), vec![acc, t])))
        else {
            return Ok(c);
        };
        let predicate = call(
            Builtin::GtEq.name(),
            vec![
                sum,
                Arc::new(Expr::literal(Literal::Integer(
                    i32::try_from(needed).unwrap_or(i32::MAX),
                ))),
            ],
        );
        let rels = [&c];
        let cx = Self::cx(&c.out, &rels, env, "Filter");
        let b = self.bind(&predicate, &cx)?;
        boolean_filter(&b, p.origin)?;
        let node = PlanNode::Filter {
            input: Arc::clone(&c.plan),
            predicate: b.expr,
        };
        let mut filtered = self.make(p.origin, node, c.out.to_vec(), &[&c]);
        filtered.widen = Some(Arc::new(Widen::Through { child: c }));
        Ok(filtered)
    }

    /// `na.fill`: a projection replacing NULL (and NaN in a float or double column) by the value,
    /// cast to the column's type, in the columns the value reaches.
    pub(in crate::analyzer) fn fill_na(
        &mut self,
        p: &LogicalPlan,
        c: Rel,
        cols: &[String],
        values: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let mut literals = Vec::with_capacity(values.len());
        for v in values {
            let ExprNode::Literal(l) = &v.node else {
                return Err(unsupported_fill(p.origin));
            };
            let Some(kind) = Fills::of(l) else {
                return Err(unsupported_fill(p.origin));
            };
            literals.push((kind, Arc::clone(v)));
        }
        self.check_fill_columns(p, &c, cols)?;
        // Several values paired with columns are Spark's `fillMap`: each value is cast to its
        // column's type whatever its kind. One value is `fill(value, subset)`, which only reaches
        // the columns its kind fits.
        let by_kind = literals.len() == 1;
        let value_for = |name: &str| -> Option<&(Fills, Arc<Expr>)> {
            if cols.is_empty() {
                return literals.first();
            }
            let at = cols
                .iter()
                // A name given twice takes the last value (Spark builds a `Map` from the pairs).
                .rposition(|n| crate::analyzer::bind::same(name, n, cs))?;
            if literals.len() == 1 {
                literals.first()
            } else {
                literals.get(at)
            }
        };
        let mut changed = false;
        let mut exprs = Vec::with_capacity(c.out.len());
        for a in c.out.iter() {
            let pass = self.pass_expr(&a.col);
            let fill = value_for(&a.col.name)
                .filter(|(kind, _)| !by_kind || kind.reaches(&a.col.data_type));
            let Some((_, value)) = fill else {
                exprs.push(pass);
                continue;
            };
            changed = true;
            let ty = a.col.data_type.clone();
            let base = if is_floating(&ty) {
                let null = Arc::new(Expr::literal(Literal::Null(Some(ty.clone()))));
                call("nanvl", vec![pass, null])
            } else {
                pass
            };
            let replacement = Arc::new(Expr::clone(value).cast(ty));
            let filled = call("coalesce", vec![base, replacement]);
            exprs.push(Arc::new(Expr::new(ExprNode::Alias {
                expr: filled,
                name: a.col.name.clone(),
            })));
        }
        if !changed {
            return Ok(c);
        }
        self.project_items(p, &c, &exprs, env)
    }
}

impl Resolver<'_> {
    /// The columns a fill or replace names resolve, each to one column: a name that is not a
    /// column is `UNRESOLVED_COLUMN`, an ambiguous one `AMBIGUOUS_REFERENCE` (Spark's
    /// `df.resolve`).
    fn check_fill_columns(
        &mut self,
        p: &LogicalPlan,
        c: &Rel,
        cols: &[String],
    ) -> Result<(), AnalysisError> {
        for name in cols {
            if self
                .find(&c.out, std::slice::from_ref(name), p.origin, false)?
                .is_none()
            {
                return Err(crate::analyzer::exprs::unresolved(
                    std::slice::from_ref(name),
                    &c.out,
                    p.origin,
                ));
            }
        }
        Ok(())
    }

    /// `na.replace`: a projection of `CASE WHEN col = old THEN new ... ELSE col END` per reached
    /// column, the literals cast to the column's type (Spark's `DataFrameNaFunctions.replaceCol`;
    /// a NaN old value matches NaN because SQL equality does). A NULL old value is refused, as
    /// Spark's `convertToDouble` does.
    pub(in crate::analyzer) fn replace_na(
        &mut self,
        p: &LogicalPlan,
        c: Rel,
        cols: &[String],
        pairs: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let mut literals = Vec::with_capacity(pairs.len());
        for v in pairs {
            match &v.node {
                ExprNode::Literal(l) => {
                    literals.push((Fills::of(l), Arc::clone(v), matches!(l, Literal::Null(_))));
                }
                _ => return Err(unsupported_replace(p.origin)),
            }
        }
        // Spark resolves the named columns before it looks at the pairs.
        self.check_fill_columns(p, &c, cols)?;
        let Some((kind, _, _)) = literals.first() else {
            return Ok(c);
        };
        let Some(kind) = *kind else {
            return Err(unsupported_replace(p.origin));
        };
        for (i, (k, _, null)) in literals.iter().enumerate() {
            if i % 2 == 0 && (*null || *k != Some(kind)) {
                return Err(unsupported_replace(p.origin));
            }
        }
        let mut changed = false;
        let mut exprs = Vec::with_capacity(c.out.len());
        for a in c.out.iter() {
            let pass = self.pass_expr(&a.col);
            let named = cols.is_empty()
                || cols
                    .iter()
                    .any(|n| crate::analyzer::bind::same(&a.col.name, n, cs));
            if !named || !kind.reaches(&a.col.data_type) {
                exprs.push(pass);
                continue;
            }
            changed = true;
            let ty = a.col.data_type.clone();
            let mut args = Vec::with_capacity(pairs.len() + 1);
            // `CASE` takes the first match: backwards, the last pair of a repeated old value wins (Spark's `Map`).
            for pair in literals.chunks(2).rev() {
                let [(_, old, _), (_, new, _)] = pair else {
                    return Err(unsupported_replace(p.origin));
                };
                // Spark's `CaseKeyWhen` compares the key as written (SQL coercion between the column
                // and the literal; `1.5` never matches an INT `1`); only the replacement is cast.
                args.push(call(
                    Builtin::Eq.name(),
                    vec![Arc::clone(&pass), Arc::clone(old)],
                ));
                args.push(Arc::new(Expr::clone(new).cast(ty.clone())));
            }
            args.push(Arc::clone(&pass));
            exprs.push(Arc::new(Expr::new(ExprNode::Alias {
                expr: call(Builtin::When.name(), args),
                name: a.col.name.clone(),
            })));
        }
        if !changed {
            return Ok(c);
        }
        self.project_items(p, &c, &exprs, env)
    }
}

fn unsupported_replace(origin: Origin) -> AnalysisError {
    AnalysisError::new(
        krishiv_common::error::KrishivUnsupportedFeature {
            feature: "na.replace with a NULL or mixed-type old value, or a value that is not a number, a string or a boolean".to_owned(),
        },
        origin,
    )
}

fn unsupported_fill(origin: Origin) -> AnalysisError {
    AnalysisError::new(
        krishiv_common::error::KrishivUnsupportedFeature {
            feature: "na.fill with a value that is not a number, a string or a boolean".to_owned(),
        },
        origin,
    )
}
