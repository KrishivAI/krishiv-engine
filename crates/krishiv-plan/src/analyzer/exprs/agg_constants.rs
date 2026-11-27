// SPDX-License-Identifier: Apache-2.0
//! The constant arguments of `histogram_numeric` and `count_min_sketch` (Spark's `checkInputDataTypes`, after the
//! implicit casts): each must be foldable, not NULL and in range. The call keeps the folded literals, which the
//! executor reads, and is still named as written.

use super::*;

impl Resolver<'_> {
    /// Checks and folds the constants of the built-in aggregate `name` in `call_kids`; `kids` are the arguments
    /// as written. Any other function is left alone.
    pub(super) fn aggregate_constants(
        &mut self,
        name: &str,
        e: &Arc<Expr>,
        kids: &[Bound],
        call_kids: &mut [Bound],
    ) -> Result<(), AnalysisError> {
        match (name, call_kids.len()) {
            ("histogram_numeric", 2) => self.histogram_constant(e, kids, call_kids),
            ("count_min_sketch", 4) => self.count_min_constants(e, kids, call_kids),
            _ => Ok(()),
        }
    }

    /// `count_min_sketch` of a string of another collation than `UTF8_BINARY`: Spark's collection takes none.
    /// The collation pass asks, before it hands the call plain strings.
    pub(super) fn refuse_collated_sketch_value(
        &self,
        name: &str,
        e: &Arc<Expr>,
        kids: &[Bound],
    ) -> Result<(), AnalysisError> {
        // The value is the first positional argument, or the one named `column`.
        let at = match &e.node {
            ExprNode::Function(f) => f
                .arg_names
                .iter()
                .position(|n| {
                    n.as_deref()
                        .is_some_and(|n| n.eq_ignore_ascii_case("column"))
                })
                .unwrap_or(0),
            _ => 0,
        };
        let (true, Some(value)) = (name == "count_min_sketch", kids.get(at)) else {
            return Ok(());
        };
        let Some(t @ DataType::CollatedString(c)) = &value.ty.data_type else {
            return Ok(());
        };
        if c.is_utf8_binary() {
            return Ok(());
        }
        Err(AnalysisError::new(
            DatatypeMismatchUnexpectedInputType {
                sql_expr: self.quoted_call(e, kids),
                param_index: coerce::ordinal(0),
                required_type: InputType::StrictIntegralStringOrBinary.sql(),
                input_sql: format!("\"{}\"", pretty(&value.shown(), self.ctx.conf.session_zone)),
                input_type: format!("\"{}\"", t.sql()),
            },
            e.origin,
        ))
    }

    /// The call `e` over `kids` as Spark's errors quote it.
    fn quoted_call(&self, e: &Arc<Expr>, kids: &[Bound]) -> String {
        let shown = with_children(e, kids.iter().map(Bound::shown).collect());
        format!("\"{}\"", pretty(&shown, self.ctx.conf.session_zone))
    }

    fn non_foldable_constant(
        &self,
        e: &Arc<Expr>,
        kids: &[Bound],
        at: usize,
        input_name: &str,
        input_type: &str,
    ) -> AnalysisError {
        AnalysisError::new(
            DatatypeMismatchNonFoldableInput {
                sql_expr: self.quoted_call(e, kids),
                input_name: format!("`{input_name}`"),
                input_type: format!("\"{input_type}\""),
                input_expr: format!(
                    "\"{}\"",
                    pretty(&kids[at].shown(), self.ctx.conf.session_zone)
                ),
            },
            e.origin,
        )
    }

    fn null_constant(&self, e: &Arc<Expr>, kids: &[Bound], name: &str) -> AnalysisError {
        AnalysisError::new(
            DatatypeMismatchUnexpectedNull {
                expr_name: name.to_owned(),
                sql_expr: self.quoted_call(e, kids),
            },
            e.origin,
        )
    }

    fn out_of_range(
        &self,
        e: &Arc<Expr>,
        kids: &[Bound],
        name: &str,
        range: String,
        current: String,
    ) -> AnalysisError {
        AnalysisError::new(
            DatatypeMismatchValueOutOfRange {
                expr_name: name.to_owned(),
                value_range: range,
                current_value: current,
                sql_expr: self.quoted_call(e, kids),
            },
            e.origin,
        )
    }

    /// `histogram_numeric(x, nb)`: `nb` is at least 2.
    fn histogram_constant(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        call_kids: &mut [Bound],
    ) -> Result<(), AnalysisError> {
        match self.fold(&call_kids[1].expr, self.ctx.conf.ansi) {
            Some(Constant::Integral(nb)) if (2..=i64::from(i32::MAX)).contains(&nb) => {
                let nb = i32::try_from(nb).unwrap_or(i32::MAX);
                fold_into(&mut call_kids[1], Literal::Integer(nb), e.origin);
                Ok(())
            }
            Some(Constant::Integral(nb)) => {
                Err(self.out_of_range(e, kids, "nb", format!("[2, {}]", i32::MAX), nb.to_string()))
            }
            Some(Constant::Null) => Err(self.null_constant(e, kids, "nb")),
            Some(Constant::Error(err)) => Err(AnalysisError::new(err, e.origin)),
            _ => Err(self.non_foldable_constant(e, kids, 1, "nb", "INT")),
        }
    }

    /// The folded `double` argument at `at`, `None` for a NULL.
    fn double_constant(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        call_kids: &[Bound],
        at: usize,
        name: &str,
    ) -> Result<Option<f64>, AnalysisError> {
        let conf = self.fold_conf(self.ctx.conf.ansi);
        let folded = fold::doubles(
            &call_kids[at].expr,
            conf,
            &mut self.folds.borrow_mut(),
            self.ctx.functions.as_ref(),
        );
        match folded {
            None => Err(self.non_foldable_constant(e, kids, at, name, "DOUBLE")),
            Some(Err(err)) => Err(AnalysisError::new(err, e.origin)),
            Some(Ok(values)) => Ok(values.first().copied().flatten()),
        }
    }

    /// `count_min_sketch(x, eps, confidence, seed)`: `eps > 0`, `confidence` in `(0, 1)` and any seed, of which
    /// the low 32 bits count. A NaN passes both ranges, as in Spark.
    fn count_min_constants(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        call_kids: &mut [Bound],
    ) -> Result<(), AnalysisError> {
        // Spark checks that all three are foldable before it reads any.
        let eps = self.double_constant(e, kids, call_kids, 1, "eps")?;
        let confidence = self.double_constant(e, kids, call_kids, 2, "confidence")?;
        let seed = match self.fold(&call_kids[3].expr, self.ctx.conf.ansi) {
            Some(Constant::Integral(seed)) => Some(seed),
            Some(Constant::Null) => None,
            Some(Constant::Error(err)) => return Err(AnalysisError::new(err, e.origin)),
            _ => {
                let long = call_kids[3].ty.data_type == Some(DataType::Long);
                let ty = if long { "BIGINT" } else { "INT" };
                return Err(self.non_foldable_constant(e, kids, 3, "seed", ty));
            }
        };
        let eps = eps.ok_or_else(|| self.null_constant(e, kids, "eps"))?;
        let confidence = confidence.ok_or_else(|| self.null_constant(e, kids, "confidence"))?;
        let seed = seed.ok_or_else(|| self.null_constant(e, kids, "seed"))?;
        // Java prints a non-finite double as `Infinity` / `-Infinity` / `NaN`; `spark_fmt` takes finite values only.
        let shown = |v: f64| {
            if v.is_finite() {
                format!("{}D", crate::spark_fmt::double(v))
            } else if v.is_nan() {
                "NaND".to_owned()
            } else if v > 0.0 {
                "InfinityD".to_owned()
            } else {
                "-InfinityD".to_owned()
            }
        };
        if eps <= 0.0 {
            return Err(self.out_of_range(
                e,
                kids,
                "eps",
                "(0.0, 1.7976931348623157E308]".to_owned(),
                shown(eps),
            ));
        }
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(self.out_of_range(
                e,
                kids,
                "confidence",
                "(0.0, 1.0]".to_owned(),
                shown(confidence),
            ));
        }
        // Java's `Number.intValue()` of a `bigint` seed keeps its low 32 bits.
        let low = seed.cast_unsigned() & u64::from(u32::MAX);
        let seed = u32::try_from(low).map_or(0, u32::cast_signed);
        fold_into(&mut call_kids[1], Literal::Double(eps), e.origin);
        fold_into(&mut call_kids[2], Literal::Double(confidence), e.origin);
        fold_into(&mut call_kids[3], Literal::Integer(seed), e.origin);
        Ok(())
    }
}

/// Replaces `kid` by its folded value, keeping the name it is shown by.
fn fold_into(kid: &mut Bound, value: Literal, origin: Origin) {
    let shown = kid.shown();
    kid.expr = Arc::new(Expr::new(ExprNode::Literal(value)).with_origin(origin));
    kid.shown = Some(shown);
}
