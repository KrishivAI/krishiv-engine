// SPDX-License-Identifier: Apache-2.0
//! Parsing queries: set operations, pipes, SELECT, FROM, joins, PIVOT and the clauses around them.

use super::*;

impl Parser<'_> {
    /// A query; a failure at a position is remembered, so the subquery
    /// attempts of nested parentheses never repeat work.
    pub(crate) fn query(&mut self) -> PR<Query> {
        let pos = self.pos;
        if let Some(&far) = self.failed_queries.get(&pos) {
            self.far = self.far.max(far);
            return Err(Fail::Syntax);
        }
        let r = self.query_at();
        if matches!(r, Err(Fail::Syntax)) {
            self.failed_queries.insert(pos, self.far);
        }
        r
    }

    pub(super) fn query_at(&mut self) -> PR<Query> {
        self.nested(|p| {
            let start = p.start();
            let with = if p.kw_at(0, "WITH") {
                Some(p.with_clause()?)
            } else {
                None
            };
            let body = p.query_term()?;
            let mut q = Self::bare_query(body, start);
            q.with = with;
            p.organization(&mut q)?;
            q.span = p.span_from(start);
            Ok(q)
        })
    }

    pub(super) fn bare_query(body: SetExpr, start: usize) -> Query {
        Query {
            with: None,
            span: Span::new(start, body.span.end),
            body,
            order_by: vec![],
            order_span: None,
            cluster_by: vec![],
            distribute_by: vec![],
            sort_by: vec![],
            windows: vec![],
            limit: None,
            offset: None,
            organization_span: None,
        }
    }

    /// `queryOrganization`, onto `q`.
    pub(super) fn organization(&mut self, q: &mut Query) -> PR<()> {
        let o = self.query_organization()?;
        let any = !(o.order_by.is_empty()
            && o.cluster_by.is_empty()
            && o.distribute_by.is_empty()
            && o.sort_by.is_empty()
            && o.windows.is_empty()
            && o.limit.is_none()
            && o.offset.is_none());
        q.organization_span = any.then_some(o.span);
        q.order_by = o.order_by;
        q.order_span = o.order_span;
        q.cluster_by = o.cluster_by;
        q.distribute_by = o.distribute_by;
        q.sort_by = o.sort_by;
        q.windows = o.windows;
        q.limit = o.limit;
        q.offset = o.offset;
        Ok(())
    }

    /// `queryOrganization`: every clause is optional, so it may be empty
    /// (its span then is the empty range at the current token).
    #[inline(never)]
    pub(super) fn query_organization(&mut self) -> PR<QueryOrganization> {
        let begin = self.start();
        let mut o = QueryOrganization {
            order_by: vec![],
            order_span: None,
            cluster_by: vec![],
            distribute_by: vec![],
            sort_by: vec![],
            windows: vec![],
            limit: None,
            offset: None,
            span: Span::default(),
        };
        if self.kw_at(0, "ORDER") && self.kw_at(1, "BY") {
            let start = self.start();
            self.pos += 2;
            o.order_by = self.comma_sep(|p| p.sort_item(true))?;
            o.order_span = Some(self.span_from(start));
        }
        for (kw, list) in [
            ("CLUSTER", &mut o.cluster_by),
            ("DISTRIBUTE", &mut o.distribute_by),
        ] {
            if self.kw_at(0, kw) && self.kw_at(1, "BY") {
                self.pos += 2;
                *list = self.comma_sep(Self::expr)?;
            }
        }
        if self.kw_at(0, "SORT") && self.kw_at(1, "BY") {
            self.pos += 2;
            o.sort_by = self.comma_sep(|p| p.sort_item(false))?;
        }
        o.windows = self.window_clause()?;
        let start = self.start();
        if self.eat_kw("LIMIT") {
            let value = if self.eat_kw("ALL") {
                None
            } else {
                Some(self.expr()?)
            };
            o.limit = Some(Limit {
                value,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("OFFSET") {
            o.offset = Some(self.expr()?);
        }
        o.span = self.span_from(begin);
        Ok(o)
    }

    #[inline(never)]
    pub(super) fn with_clause(&mut self) -> PR<With> {
        let start = self.start();
        self.expect_kw("WITH")?;
        let recursive = self.eat_kw("RECURSIVE");
        let ctes = self.comma_sep(|p| {
            let start = p.start();
            let name = p.name_part(true)?;
            let columns = if p.sym_at(0, Sym::LParen) {
                p.ident_list()?
            } else {
                vec![]
            };
            let mut max_recursion_level = None;
            if p.kw_at(0, "MAX") && p.kw_at(1, "RECURSION") && p.kw_at(2, "LEVEL") {
                p.pos += 3;
                if p.kind(0) != Some(TokenKind::Number(crate::ast::NumKind::Int)) {
                    return p.fail();
                }
                max_recursion_level = Some(p.text(0).to_owned());
                p.bump();
            }
            p.eat_kw("AS");
            p.expect_sym(Sym::LParen)?;
            let query = Box::new(p.query()?);
            p.expect_sym(Sym::RParen)?;
            Ok(Cte {
                name,
                columns,
                max_recursion_level,
                query,
                span: p.span_from(start),
            })
        })?;
        Ok(With {
            recursive,
            ctes,
            span: self.span_from(start),
        })
    }

    /// Set operations: `INTERSECT` binds tighter than `UNION` / `EXCEPT`
    /// unless `spark.sql.legacy.setopsPrecedence.enabled`; all are
    /// left-associative.
    pub(super) fn query_term(&mut self) -> PR<SetExpr> {
        self.set_ops(0)
    }

    pub(super) fn set_ops(&mut self, min_prec: u8) -> PR<SetExpr> {
        let (mut left, mut height) = self.measured(Self::query_primary)?;
        loop {
            if min_prec == 0 && self.pipe_follows() {
                let op = self.pipe_op()?;
                if let SetExprKind::Pipe(pipe) = &mut left.kind {
                    left.span = left.span.to(op.span);
                    pipe.span = left.span;
                    pipe.ops.push(op);
                } else {
                    height += 1;
                    self.grow(height)?;
                    let span = left.span.to(op.span);
                    let input = Box::new(left);
                    let kind = SetExprKind::Pipe(Pipe {
                        input,
                        ops: vec![op],
                        span,
                    });
                    left = SetExpr { kind, span };
                }
                continue;
            }
            let (op, prec) = if self.kw_at(0, "INTERSECT") {
                (SetOperator::Intersect, 2)
            } else if self.kw_at(0, "UNION") {
                (SetOperator::Union, 1)
            } else if self.any_kw_at(0, &["EXCEPT", "MINUS"]) {
                (SetOperator::Except, 1)
            } else {
                break;
            };
            let prec = if self.opts.legacy_setops_precedence {
                1
            } else {
                prec
            };
            if prec < min_prec {
                break;
            }
            self.bump();
            let quantifier = self.set_quantifier();
            let (right, right_height) = self.measured(|p| p.nested(|p| p.set_ops(prec + 1)))?;
            height = (height + 1).max(right_height);
            self.grow(height)?;
            let span = left.span.to(right.span);
            left = SetExpr {
                kind: SetExprKind::SetOp {
                    op,
                    quantifier,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    /// Whether a pipe operator starts here: `|>`, or `|` before a word that
    /// starts one (Spark's `isOperatorPipeStart`), unless the single
    /// character form is off.
    pub(crate) fn pipe_follows(&self) -> bool {
        self.sym_at(0, Sym::PipeGt) || (self.sym_at(0, Sym::Pipe) && self.single_pipe_word_at(1))
    }

    /// Whether `|` before token `n` is the pipe operator, not a bitwise or.
    pub(crate) fn single_pipe_word_at(&self, n: usize) -> bool {
        self.opts.single_character_pipe_operator
            && self.any_kw_at(
                n,
                &[
                    "SELECT",
                    "EXTEND",
                    "SET",
                    "DROP",
                    "AS",
                    "WHERE",
                    "PIVOT",
                    "UNPIVOT",
                    "TABLESAMPLE",
                    "INNER",
                    "CROSS",
                    "LEFT",
                    "RIGHT",
                    "FULL",
                    "NATURAL",
                    "SEMI",
                    "ANTI",
                    "JOIN",
                    "UNION",
                    "EXCEPT",
                    "MINUS",
                    "INTERSECT",
                    "ORDER",
                    "CLUSTER",
                    "DISTRIBUTE",
                    "SORT",
                    "LIMIT",
                    "OFFSET",
                    "AGGREGATE",
                    "WINDOW",
                    "LATERAL",
                ],
            )
    }

    /// One pipe operator, from its `|>` (or `|`).
    #[inline(never)]
    pub(super) fn pipe_op(&mut self) -> PR<PipeOp> {
        let start = self.start();
        let pipe_span = self.tok(0).map_or_else(Span::default, |t| t.span);
        self.bump();
        let kind = if self.kw_at(0, "SELECT") {
            let begin = self.start();
            let (hints, quantifier, projection) = self.select_head()?;
            let group_by = if self.kw_at(0, "GROUP") && self.kw_at(1, "BY") {
                Some(self.group_by()?)
            } else {
                None
            };
            let windows = self.window_clause()?;
            PipeOpKind::Select(Box::new(PipeSelect {
                hints,
                quantifier,
                projection,
                group_by,
                windows,
                span: self.span_from(begin),
            }))
        } else if self.eat_kw("EXTEND") {
            PipeOpKind::Extend(self.comma_sep(Self::named_item)?)
        } else if self.eat_kw("SET") {
            PipeOpKind::Set(self.comma_sep(|p| {
                let start = p.start();
                let mut key = vec![p.name_part(true)?];
                while p.eat_sym(Sym::Dot) {
                    key.push(p.name_part(true)?);
                }
                p.expect_sym(Sym::Eq)?;
                Ok(PipeAssignment {
                    key,
                    value: p.expr()?,
                    span: p.span_from(start),
                })
            })?)
        } else if self.eat_kw("DROP") {
            PipeOpKind::Drop(self.comma_sep(Self::table_name)?)
        } else if self.eat_kw("AS") {
            PipeOpKind::As(self.name_part(true)?)
        } else if self.kw_at(0, "WHERE") {
            let begin = self.start();
            self.bump();
            let condition = self.expr()?;
            let windows = self.window_clause()?;
            PipeOpKind::Where(PipeWhere {
                condition,
                windows,
                span: self.span_from(begin),
            })
        } else if self.kw_at(0, "PIVOT") {
            let pivot = Box::new(self.pivot()?);
            let unpivot = if self.unpivot_follows() {
                Some(Box::new(self.unpivot()?))
            } else {
                None
            };
            PipeOpKind::Pivot(pivot, unpivot)
        } else if self.unpivot_follows() {
            let unpivot = Box::new(self.unpivot()?);
            let pivot = if self.kw_at(0, "PIVOT") {
                Some(Box::new(self.pivot()?))
            } else {
                None
            };
            PipeOpKind::Unpivot(unpivot, pivot)
        } else if self.kw_at(0, "TABLESAMPLE") {
            PipeOpKind::Sample(self.table_sample()?)
        } else if self.any_kw_at(0, &["UNION", "EXCEPT", "MINUS", "INTERSECT"]) {
            let op = if self.kw_at(0, "UNION") {
                SetOperator::Union
            } else if self.kw_at(0, "INTERSECT") {
                SetOperator::Intersect
            } else {
                SetOperator::Except
            };
            self.bump();
            let quantifier = self.set_quantifier();
            let right = Box::new(self.nested(Self::query_primary)?);
            PipeOpKind::SetOp {
                op,
                quantifier,
                right,
            }
        } else if self.kw_at(0, "AGGREGATE") {
            let begin = self.start();
            self.bump();
            // `AGGREGATE group by` with nothing after `by` reads `group` as
            // an aggregate named `by`, as Spark does.
            let group_follows =
                self.kw_at(0, "GROUP") && self.kw_at(1, "BY") && self.expr_may_start_at(2);
            let aggregates = if group_follows || !self.expr_may_start_at(0) {
                vec![]
            } else if self.kw_at(0, "GROUP")
                && self.kw_at(1, "BY")
                && !self.opts.enforce_reserved_keywords
            {
                let item_start = self.start();
                let column = self.keyword_ident();
                let expr = Expr {
                    span: column.span,
                    kind: ExprKind::Column(vec![column]),
                };
                let alias = Some(Alias::Name(self.any_ident()?));
                let mut items = vec![SelectItem {
                    expr,
                    alias,
                    span: self.span_from(item_start),
                }];
                while self.eat_sym(Sym::Comma) {
                    items.push(self.select_item()?);
                }
                items
            } else {
                self.comma_sep(Self::select_item)?
            };
            let group_by = if self.kw_at(0, "GROUP") && self.kw_at(1, "BY") {
                Some(self.group_by_with(true)?)
            } else {
                None
            };
            PipeOpKind::Aggregate(Box::new(PipeAggregate {
                aggregates,
                group_by,
                span: self.span_from(begin),
            }))
        } else if let Some(join) = self.join()? {
            PipeOpKind::Join(Box::new(join))
        } else {
            let mut o = self.query_organization()?;
            if o.span.start == o.span.end {
                o.span = pipe_span;
            }
            PipeOpKind::Organization(o)
        };
        Ok(PipeOp {
            kind,
            span: self.span_from(start),
        })
    }

    pub(super) fn unpivot_follows(&self) -> bool {
        self.kw_at(0, "UNPIVOT")
            && (self.sym_at(1, Sym::LParen) || self.any_kw_at(1, &["INCLUDE", "EXCLUDE"]))
    }

    /// Whether an expression can start at token `n` (not the end of the
    /// operator: a closing bracket, a pipe or a query clause).
    pub(super) fn expr_may_start_at(&self, n: usize) -> bool {
        match self.kind(n) {
            None
            | Some(TokenKind::Sym(
                Sym::RParen | Sym::RBracket | Sym::Semi | Sym::Comma | Sym::PipeGt,
            )) => false,
            Some(TokenKind::Sym(Sym::Pipe)) => !self.single_pipe_word_at(n + 1),
            _ => !self.any_kw_at(
                n,
                &[
                    "ORDER",
                    "UNION",
                    "EXCEPT",
                    "MINUS",
                    "INTERSECT",
                    "LIMIT",
                    "OFFSET",
                    "WINDOW",
                    "CLUSTER",
                    "DISTRIBUTE",
                    "SORT",
                ],
            ),
        }
    }

    pub(super) fn set_quantifier(&mut self) -> Option<SetQuantifier> {
        if self.eat_kw("DISTINCT") {
            Some(SetQuantifier::Distinct)
        } else if self.eat_kw("ALL") {
            Some(SetQuantifier::All)
        } else {
            None
        }
    }

    pub(super) fn query_primary(&mut self) -> PR<SetExpr> {
        let start = self.start();
        let kind = if self.any_kw_at(0, &["SELECT", "MAP", "REDUCE"]) {
            return self.select_kind(start, true);
        } else if self.kw_at(0, "FROM") {
            return self.query_from_first();
        } else if self.kw_at(0, "VALUES") {
            SetExprKind::Values(self.values()?)
        } else if self.eat_kw("TABLE") {
            match self.ddl_name()? {
                DdlName::Parts(name) => SetExprKind::Table(name),
                DdlName::Dynamic(name) => SetExprKind::DynamicTable(name),
            }
        } else if self.eat_sym(Sym::LParen) {
            let q = self.query()?;
            self.expect_sym(Sym::RParen)?;
            SetExprKind::Query(Box::new(q))
        } else {
            return self.fail();
        };
        Ok(SetExpr {
            kind,
            span: self.span_from(start),
        })
    }

    #[inline(never)]
    pub(super) fn values(&mut self) -> PR<Values> {
        let start = self.start();
        self.expect_kw("VALUES")?;
        let rows = self.comma_sep(Self::expr)?;
        let alias = self.table_alias()?;
        Ok(Values {
            rows,
            alias,
            span: self.span_from(start),
        })
    }

    /// `FROM relations [SELECT ... [organization]]*`. The `FROM` clause is
    /// parsed and stored once, however many bodies read it.
    #[inline(never)]
    pub(super) fn query_from_first(&mut self) -> PR<SetExpr> {
        let start = self.start();
        let from = self.clause_from()?;
        let mut bodies = vec![];
        while self.any_kw_at(0, &["SELECT", "MAP", "REDUCE"]) {
            let body_start = self.start();
            let body = self.select_kind(body_start, false)?;
            let mut q = Self::bare_query(body, body_start);
            self.organization(&mut q)?;
            q.span = self.span_from(body_start);
            bodies.push(q);
        }
        let span = self.span_from(start);
        Ok(SetExpr {
            kind: SetExprKind::FromStatement(Box::new(FromStatement { from, bodies, span })),
            span,
        })
    }

    /// `FROM relation, ... lateralView* pivot? unpivot?`.
    #[inline(never)]
    pub(super) fn clause_from(&mut self) -> PR<FromClause> {
        let start = self.start();
        self.expect_kw("FROM")?;
        let relations = self.comma_sep(Self::relation)?;
        let lateral_views = self.lateral_views()?;
        let pivot = if self.kw_at(0, "PIVOT") {
            Some(self.pivot()?)
        } else {
            None
        };
        let unpivot = if self.kw_at(0, "UNPIVOT") {
            Some(self.unpivot()?)
        } else {
            None
        };
        Ok(FromClause {
            relations,
            lateral_views,
            pivot,
            unpivot,
            span: self.span_from(start),
        })
    }

    /// `SELECT [hints] [DISTINCT | ALL] items`.
    pub(super) fn select_head(
        &mut self,
    ) -> PR<(Vec<Hint>, Option<SetQuantifier>, Vec<SelectItem>)> {
        self.expect_kw("SELECT")?;
        let mut hints = vec![];
        while self.kind(0) == Some(TokenKind::HintStart) {
            self.hints(&mut hints)?;
        }
        // `SELECT all, b`: `ALL` / `DISTINCT` before `,` or `FROM` is a column.
        let column_next = !self.opts.enforce_reserved_keywords
            && (self.sym_at(1, Sym::Comma) || self.kw_at(1, "FROM") || self.tok(1).is_none());
        let quantifier = if column_next {
            None
        } else {
            self.set_quantifier()
        };
        let projection = self.comma_sep(Self::named_item)?;
        Ok((hints, quantifier, projection))
    }

    /// A query specification from `SELECT` on; `with_from` reads a `FROM`
    /// clause (a body of a `FROM ... SELECT` statement has none).
    /// A `SELECT` body: a query specification, or a `TRANSFORM` when its
    /// parentheses are followed by what only a transform has.
    pub(super) fn select_kind(&mut self, start: usize, with_from: bool) -> PR<SetExpr> {
        if self.any_kw_at(0, &["MAP", "REDUCE"]) || self.transform_follows() {
            let t = self.transform_select(start, with_from)?;
            let span = t.span;
            return Ok(SetExpr {
                kind: SetExprKind::Transform(Box::new(t)),
                span,
            });
        }
        let select = self.select(start, with_from)?;
        let span = select.span;
        Ok(SetExpr {
            kind: SetExprKind::Select(Box::new(select)),
            span,
        })
    }

    pub(super) fn select(&mut self, start: usize, with_from: bool) -> PR<Select> {
        let (hints, quantifier, projection) = self.select_head()?;
        self.select_tail(start, with_from, (hints, quantifier, projection), true)
    }

    /// The clauses of a query specification after its select list;
    /// `qualify` admits `QUALIFY` (a transform has none).
    pub(super) fn select_tail(
        &mut self,
        start: usize,
        with_from: bool,
        head: (Vec<Hint>, Option<SetQuantifier>, Vec<SelectItem>),
        qualify: bool,
    ) -> PR<Select> {
        let (hints, quantifier, projection) = head;
        let from = if with_from && self.kw_at(0, "FROM") {
            Some(self.clause_from()?)
        } else {
            None
        };
        let lateral_views = self.lateral_views()?;
        let selection = self.opt_clause("WHERE")?;
        let group_by = if self.kw_at(0, "GROUP") && self.kw_at(1, "BY") {
            Some(self.group_by()?)
        } else {
            None
        };
        let having = self.opt_clause("HAVING")?;
        let windows = self.window_clause()?;
        let qualify = if qualify {
            self.opt_clause("QUALIFY")?
        } else {
            None
        };
        Ok(Select {
            hints,
            quantifier,
            projection,
            from,
            lateral_views,
            selection,
            group_by,
            having,
            having_as_where: self.opts.having_without_group_by_as_where,
            windows,
            qualify,
            span: self.span_from(start),
        })
    }

    /// Whether the `SELECT` here starts a `TRANSFORM`: `TRANSFORM (...)`
    /// followed by a row format, a record writer or `USING` (else it calls
    /// the function `transform`).
    pub(super) fn transform_follows(&self) -> bool {
        if !(self.kw_at(0, "SELECT") && self.kw_at(1, "TRANSFORM") && self.sym_at(2, Sym::LParen)) {
            return false;
        }
        // The token after the parenthesis that opens at `pos + 2`.
        let close = self.closers[self.pos + 2];
        if close == usize::MAX {
            return false;
        }
        let n = close + 1 - self.pos;
        self.kw_at(n, "USING")
            || self.kw_at(n, "RECORDWRITER")
            || self.kw_at(n, "ROW") && self.kw_at(n + 1, "FORMAT")
    }

    /// `SELECT TRANSFORM (inputs)`, `MAP inputs` or `REDUCE inputs`, the
    /// script clauses, and the clauses of a query specification.
    #[inline(never)]
    pub(super) fn transform_select(
        &mut self,
        start: usize,
        with_from: bool,
    ) -> PR<TransformSelect> {
        let (kind, quantifier, inputs) = if self.eat_kw("SELECT") {
            self.expect_kw("TRANSFORM")?;
            self.expect_sym(Sym::LParen)?;
            let quantifier = self.set_quantifier();
            let inputs = self.comma_sep(Self::expr)?;
            self.expect_sym(Sym::RParen)?;
            (TransformKind::SelectTransform, quantifier, inputs)
        } else {
            let kind = if self.kw_at(0, "MAP") {
                TransformKind::Map
            } else {
                TransformKind::Reduce
            };
            self.bump();
            let quantifier = self.set_quantifier();
            (kind, quantifier, self.comma_sep(Self::expr)?)
        };
        let input_format = self.opt_script_row_format()?;
        let record_writer = self.opt_record_class("RECORDWRITER")?;
        self.expect_kw("USING")?;
        let script = self.string_after()?;
        let output = if self.eat_kw("AS") {
            Some(self.transform_output()?)
        } else {
            None
        };
        let output_format = self.opt_script_row_format()?;
        let record_reader = self.opt_record_class("RECORDREADER")?;
        let projection = inputs
            .into_iter()
            .map(|expr| SelectItem {
                span: expr.span,
                expr,
                alias: None,
            })
            .collect();
        let select = self.select_tail(start, with_from, (vec![], None, projection), false)?;
        Ok(TransformSelect {
            kind,
            quantifier,
            input_format,
            record_writer,
            script,
            output,
            output_format,
            record_reader,
            select,
            span: self.span_from(start),
        })
    }

    pub(super) fn opt_script_row_format(&mut self) -> PR<Option<RowFormat>> {
        if self.kw_at(0, "ROW") && self.kw_at(1, "FORMAT") {
            self.row_format().map(Some)
        } else {
            Ok(None)
        }
    }

    /// `RECORDWRITER 'class'` or `RECORDREADER 'class'`.
    pub(super) fn opt_record_class(&mut self, kw: &str) -> PR<Option<String>> {
        if self.eat_kw(kw) {
            self.string_after().map(Some)
        } else {
            Ok(None)
        }
    }

    /// The `AS` of a transform: names or typed columns, with or without
    /// parentheses.
    pub(super) fn transform_output(&mut self) -> PR<TransformOutput> {
        let paren = self.eat_sym(Sym::LParen);
        let mut names = vec![];
        let mut columns: Vec<StructField> = vec![];
        loop {
            let start = self.start();
            let name = self.name_part(true)?;
            // A type follows unless the clause ends or goes on.
            let typed = self.kind(0) == Some(TokenKind::Word)
                && !(self.kw_at(0, "ROW") && self.kw_at(1, "FORMAT"))
                && !self.any_kw_at(
                    0,
                    &[
                        "FROM",
                        "WHERE",
                        "GROUP",
                        "HAVING",
                        "WINDOW",
                        "LATERAL",
                        "RECORDREADER",
                        "UNION",
                        "EXCEPT",
                        "MINUS",
                        "INTERSECT",
                        "ORDER",
                        "LIMIT",
                        "OFFSET",
                        "CLUSTER",
                        "DISTRIBUTE",
                        "SORT",
                    ],
                );
            if typed {
                let data_type = self.data_type()?;
                let not_null = self.kw_at(0, "NOT") && self.kw_at(1, "NULL");
                self.pos += 2 * usize::from(not_null);
                let comment = if self.eat_kw("COMMENT") {
                    Some(self.string_after()?)
                } else {
                    None
                };
                columns.push(StructField {
                    name,
                    data_type,
                    not_null,
                    comment,
                    span: self.span_from(start),
                });
            } else {
                names.push(name);
            }
            if !self.eat_sym(Sym::Comma) {
                break;
            }
        }
        if paren {
            self.expect_sym(Sym::RParen)?;
        }
        if names.is_empty() {
            Ok(TransformOutput::Columns(columns))
        } else if columns.is_empty() {
            Ok(TransformOutput::Names(names))
        } else {
            self.fail()
        }
    }

    pub(super) fn opt_clause(&mut self, kw: &str) -> PR<Option<Expr>> {
        if self.eat_kw(kw) {
            self.expr().map(Some)
        } else {
            Ok(None)
        }
    }

    /// `/*+ name[(params)] [,] ... */`.
    #[inline(never)]
    pub(super) fn hints(&mut self, out: &mut Vec<Hint>) -> PR<()> {
        self.bump();
        loop {
            let start = self.start();
            // A hint name is a plain word: `IDENTIFIER('x')` is the hint
            // `IDENTIFIER` with the parameter `'x'`.
            let name = if self.identifier_literal_at(0) {
                let span = self.tok(0).map_or_else(Span::default, |t| t.span);
                self.bump();
                Ident {
                    value: span.text(self.src).to_owned(),
                    quoted: false,
                    span,
                }
            } else {
                self.any_ident()?
            };
            let params = if self.sym_at(0, Sym::LParen) {
                self.paren_list(Self::primary_expr)?
            } else {
                vec![]
            };
            out.push(Hint {
                name,
                params,
                span: self.span_from(start),
            });
            if self.kind(0) == Some(TokenKind::HintEnd) {
                self.bump();
                return Ok(());
            }
            // `COMMA? hintStatement`: a comma needs another hint.
            self.eat_sym(Sym::Comma);
        }
    }

    /// A select item of a select list: its names may be column regexes.
    pub(super) fn named_item(&mut self) -> PR<SelectItem> {
        self.in_named += 1;
        let item = self.select_item();
        self.in_named -= 1;
        item
    }

    pub(super) fn select_item(&mut self) -> PR<SelectItem> {
        let start = self.start();
        let expr = self.expr()?;
        let alias = self.select_alias()?;
        Ok(SelectItem {
            expr,
            alias,
            span: self.span_from(start),
        })
    }

    /// The `[AS] alias` after a select item's expression.
    pub(super) fn select_alias(&mut self) -> PR<Option<Alias>> {
        Ok(if !self.item_end_at(1) && self.eat_kw("AS") {
            Some(if self.sym_at(0, Sym::LParen) {
                Alias::List(self.ident_list()?)
            } else {
                Alias::Name(self.name_part(true)?)
            })
        } else if self.sym_at(0, Sym::LParen) {
            Some(Alias::List(self.ident_list()?))
        } else {
            self.implicit_alias(true)?.map(Alias::Name)
        })
    }

    pub(super) fn sort_item(&mut self, allow_all: bool) -> PR<SortItem> {
        let start = self.start();
        let expr = if allow_all
            && !self.opts.enforce_reserved_keywords
            && self.kw_at(0, "ALL")
            && !self.sym_at(1, Sym::Dot)
            && !self.sym_at(1, Sym::LParen)
        {
            self.column_named_all()
        } else {
            self.expr()?
        };
        let asc = if self.eat_kw("ASC") {
            Some(true)
        } else if self.eat_kw("DESC") {
            Some(false)
        } else {
            None
        };
        let mut nulls_first = None;
        if self.eat_kw("NULLS") {
            nulls_first = Some(self.eat_kw("FIRST"));
            if nulls_first == Some(false) {
                self.expect_kw("LAST")?;
            }
        }
        Ok(SortItem {
            expr,
            asc,
            nulls_first,
            span: self.span_from(start),
        })
    }

    /// `ALL` in `ORDER BY ALL` / `GROUP BY ALL`: a column named `ALL`.
    pub(super) fn column_named_all(&mut self) -> Expr {
        let span = self.tok(0).map_or_else(Span::default, |t| t.span);
        self.bump();
        Expr {
            kind: crate::ast::ExprKind::Column(vec![Ident {
                value: span.text(self.src).to_owned(),
                quoted: false,
                span,
            }]),
            span,
        }
    }

    pub(super) fn group_by(&mut self) -> PR<GroupBy> {
        self.group_by_with(false)
    }

    /// `GROUP BY items`; `named` lets an item carry an alias, as in the
    /// pipe operator `AGGREGATE`.
    #[inline(never)]
    pub(super) fn group_by_with(&mut self, named: bool) -> PR<GroupBy> {
        let start = self.start();
        self.pos += 2;
        let items = self.comma_sep(|p| p.group_item_with(named))?;
        let with = if self.kw_at(0, "WITH") && self.any_kw_at(1, &["ROLLUP", "CUBE"]) {
            let rollup = self.kw_at(1, "ROLLUP");
            self.pos += 2;
            Some(if rollup {
                GroupWith::Rollup
            } else {
                GroupWith::Cube
            })
        } else if self.kw_at(0, "GROUPING") && self.kw_at(1, "SETS") {
            self.pos += 2;
            Some(GroupWith::GroupingSets(
                self.paren_list(Self::grouping_set)?,
            ))
        } else {
            None
        };
        Ok(GroupBy {
            items,
            with,
            span: self.span_from(start),
        })
    }

    pub(super) fn group_item(&mut self) -> PR<GroupItem> {
        self.group_item_with(false)
    }

    pub(super) fn group_item_with(&mut self, named: bool) -> PR<GroupItem> {
        let start = self.start();
        let kind = if self.any_kw_at(0, &["ROLLUP", "CUBE"]) && self.sym_at(1, Sym::LParen) {
            let rollup = self.kw_at(0, "ROLLUP");
            self.bump();
            let sets = self.paren_list(Self::grouping_set)?;
            if rollup {
                GroupItemKind::Rollup(sets)
            } else {
                GroupItemKind::Cube(sets)
            }
        } else if self.kw_at(0, "GROUPING") && self.kw_at(1, "SETS") {
            self.pos += 2;
            GroupItemKind::GroupingSets(self.paren_list(|p| {
                let analytics = p.any_kw_at(0, &["ROLLUP", "CUBE"]) && p.sym_at(1, Sym::LParen)
                    || p.kw_at(0, "GROUPING") && p.kw_at(1, "SETS");
                if analytics {
                    p.nested(Self::group_item)
                } else {
                    let set = p.grouping_set()?;
                    Ok(GroupItem {
                        span: set.span,
                        kind: GroupItemKind::Set(set),
                    })
                }
            })?)
        } else if !self.opts.enforce_reserved_keywords
            && self.kw_at(0, "ALL")
            && !self.sym_at(1, Sym::Dot)
            && !self.sym_at(1, Sym::LParen)
        {
            GroupItemKind::Expr(self.column_named_all())
        } else if named {
            let expr = self.expr()?;
            // `GROUPING SETS` after the items is no alias (`GROUPING`).
            let alias = if self.kw_at(0, "GROUPING") && self.kw_at(1, "SETS") {
                None
            } else {
                self.select_alias()?
            };
            if alias.is_some() {
                let span = self.span_from(start);
                GroupItemKind::Named(SelectItem { expr, alias, span })
            } else {
                GroupItemKind::Expr(expr)
            }
        } else {
            GroupItemKind::Expr(self.expr()?)
        };
        Ok(GroupItem {
            kind,
            span: self.span_from(start),
        })
    }

    /// `(a, b)`, `()` or `a`. A parenthesized list is a set; `(a + b) * 2`
    /// is an expression, continued from `(a + b)` without re-parsing it.
    pub(super) fn grouping_set(&mut self) -> PR<GroupingSet> {
        let start = self.start();
        if !self.sym_at(0, Sym::LParen) || self.query_follows(1) {
            let e = self.expr()?;
            return Ok(self.grouping_set_of(vec![e], start));
        }
        self.bump();
        if self.eat_sym(Sym::RParen) {
            return Ok(self.grouping_set_of(vec![], start));
        }
        let (first, height) = self.measured(Self::expr)?;
        if self.eat_sym(Sym::Comma) {
            let mut exprs = vec![first];
            exprs.extend(self.comma_sep(Self::expr)?);
            self.expect_sym(Sym::RParen)?;
            return Ok(self.grouping_set_of(exprs, start));
        }
        self.expect_sym(Sym::RParen)?;
        if self.item_end_at(0) {
            return Ok(self.grouping_set_of(vec![first], start));
        }
        let nested = Expr {
            span: self.span_from(start),
            kind: crate::ast::ExprKind::Nested(Box::new(first)),
        };
        let e = self.expr_from(nested, height)?;
        Ok(self.grouping_set_of(vec![e], start))
    }

    pub(super) fn grouping_set_of(&self, exprs: Vec<Expr>, start: usize) -> GroupingSet {
        GroupingSet {
            exprs,
            span: self.span_from(start),
        }
    }

    pub(super) fn window_clause(&mut self) -> PR<Vec<NamedWindow>> {
        if !self.eat_kw("WINDOW") {
            return Ok(vec![]);
        }
        self.comma_sep(|p| {
            let start = p.start();
            let name = p.name_part(true)?;
            p.expect_kw("AS")?;
            let window = p.window_ref()?;
            Ok(NamedWindow {
                name,
                window,
                span: p.span_from(start),
            })
        })
    }

    pub(super) fn relation(&mut self) -> PR<Relation> {
        self.nested(|p| {
            let start = p.start();
            let lateral = p.eat_kw("LATERAL");
            let primary = p.table_factor()?;
            let mut extensions = vec![];
            loop {
                if p.kw_at(0, "PIVOT") && p.sym_at(1, Sym::LParen) {
                    extensions.push(RelationExt::Pivot(p.pivot()?));
                } else if p.kw_at(0, "UNPIVOT")
                    && (p.sym_at(1, Sym::LParen) || p.any_kw_at(1, &["INCLUDE", "EXCLUDE"]))
                {
                    extensions.push(RelationExt::Unpivot(p.unpivot()?));
                } else if let Some(j) = p.join()? {
                    extensions.push(RelationExt::Join(j));
                } else {
                    break;
                }
            }
            Ok(Relation {
                lateral,
                primary,
                extensions,
                span: p.span_from(start),
            })
        })
    }

    /// Whether token `n` and the parentheses after it open a query.
    pub(super) fn query_follows(&self, n: usize) -> bool {
        let at = self.pos + n;
        let after = self.after_parens.get(at).copied().unwrap_or(at);
        self.any_kw_at(after - self.pos, QUERY_START)
            || self.any_kw_at(after - self.pos, &["MAP", "REDUCE"])
    }

    pub(super) fn table_alias(&mut self) -> PR<Option<TableAlias>> {
        let start = self.start();
        // `AS (columns)`: a column list without a table name, as in
        // `VALUES (1) AS (c)`.
        if self.kw_at(0, "AS") && self.sym_at(1, Sym::LParen) && !self.query_follows(2) {
            self.pos += 1;
            let columns = self.ident_list()?;
            return Ok(Some(TableAlias {
                name: None,
                columns,
                span: self.span_from(start),
            }));
        }
        let Some(name) = self.opt_alias_name()? else {
            return Ok(None);
        };
        let columns = if self.sym_at(0, Sym::LParen) {
            self.ident_list()?
        } else {
            vec![]
        };
        Ok(Some(TableAlias {
            name: Some(name),
            columns,
            span: self.span_from(start),
        }))
    }

    pub(super) fn table_factor(&mut self) -> PR<TableFactor> {
        let start = self.start();
        let kind = if self.kw_at(0, "VALUES") && !self.item_end_at(1) {
            let values = self.values()?;
            return Ok(TableFactor {
                kind: TableFactorKind::Values(values.rows),
                sample: None,
                alias: values.alias,
                span: values.span,
            });
        } else if self.sym_at(0, Sym::LParen) {
            let save = self.pos;
            let mut sub = None;
            if self.query_follows(1) {
                self.bump();
                let nested = self.sym_at(0, Sym::LParen);
                match self.query() {
                    Ok(q) if self.sym_at(0, Sym::RParen) || !nested => {
                        // `(SELECT ...` is committed to by ANTLR's prediction.
                        self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
                        sub = Some(q);
                    }
                    Err(Fail::Fatal(e)) => return Err(Fail::Fatal(e)),
                    _ => self.pos = save,
                }
            }
            match sub {
                Some(q) => TableFactorKind::Subquery(Box::new(q)),
                None => {
                    self.expect_sym(Sym::LParen)?;
                    let r = self.relation()?;
                    self.expect_sym(Sym::RParen)?;
                    TableFactorKind::Nested(Box::new(r))
                }
            }
        } else {
            let name = self.ddl_name()?;
            if self.sym_at(0, Sym::LParen) {
                self.bump();
                let args = self.function_args(true)?;
                self.expect_sym(Sym::RParen)?;
                let (name, dynamic_name) = match name {
                    DdlName::Parts(name) => (name, None),
                    DdlName::Dynamic(d) => {
                        let word = Span::new(d.span.start, d.span.start + "IDENTIFIER".len());
                        (self.identifier_word(word), Some(d))
                    }
                };
                TableFactorKind::TableFunction {
                    name,
                    dynamic_name,
                    args,
                }
            } else {
                let temporal = self.temporal_clause()?;
                let options = self.table_options()?;
                match name {
                    DdlName::Parts(name) => TableFactorKind::Table {
                        name,
                        temporal,
                        options,
                    },
                    DdlName::Dynamic(name) => TableFactorKind::DynamicTable {
                        name,
                        temporal,
                        options,
                    },
                }
            }
        };
        let sample = if self.kw_at(0, "TABLESAMPLE")
            && matches!(
                kind,
                TableFactorKind::Table { .. }
                    | TableFactorKind::Subquery(_)
                    | TableFactorKind::Nested(_)
            ) {
            Some(self.table_sample()?)
        } else {
            None
        };
        let alias = self.table_alias()?;
        Ok(TableFactor {
            kind,
            sample,
            alias,
            span: self.span_from(start),
        })
    }

    /// `TABLESAMPLE [SYSTEM | BERNOULLI] ([method]) [REPEATABLE (seed)]`.
    #[inline(never)]
    pub(super) fn table_sample(&mut self) -> PR<TableSample> {
        let start = self.start();
        self.expect_kw("TABLESAMPLE")?;
        let sample_type = if self.eat_kw("SYSTEM") {
            Some(SampleType::System)
        } else if self.eat_kw("BERNOULLI") {
            Some(SampleType::Bernoulli)
        } else {
            None
        };
        self.expect_sym(Sym::LParen)?;
        let method = if self.sym_at(0, Sym::RParen) {
            None
        } else {
            Some(self.sample_method()?)
        };
        self.expect_sym(Sym::RParen)?;
        let mut seed = None;
        if self.kw_at(0, "REPEATABLE") && self.sym_at(1, Sym::LParen) {
            self.pos += 2;
            seed = Some(self.integer_text()?);
            self.expect_sym(Sym::RParen)?;
        }
        Ok(TableSample {
            sample_type,
            method,
            seed,
            span: self.span_from(start),
        })
    }

    /// An integer literal's text (`integerValue`).
    pub(super) fn integer_text(&mut self) -> PR<String> {
        if self.kind(0) != Some(TokenKind::Number(NumKind::Int)) {
            return self.fail();
        }
        let text = self.text(0).to_owned();
        self.bump();
        Ok(text)
    }

    pub(super) fn sample_method(&mut self) -> PR<SampleMethod> {
        let start = self.start();
        let negative = self.sym_at(0, Sym::Minus);
        let n = usize::from(negative);
        let kind = if self.kw_at(0, "BUCKET")
            && self.kind(1) == Some(TokenKind::Number(NumKind::Int))
            && self.kw_at(2, "OUT")
            && self.kw_at(3, "OF")
        {
            self.bump();
            let numerator = self.integer_text()?;
            self.pos += 2;
            let denominator = self.integer_text()?;
            let on = if self.eat_kw("ON") {
                let name = self.table_name()?;
                Some(if self.eat_sym(Sym::LParen) {
                    self.expect_sym(Sym::RParen)?;
                    BucketOn::Function(name)
                } else if let [ident] = &name.parts[..] {
                    BucketOn::Column(ident.clone())
                } else {
                    return self.fail();
                })
            } else {
                None
            };
            SampleMethodKind::Bucket {
                numerator,
                denominator,
                on,
            }
        } else if matches!(
            self.kind(n),
            Some(TokenKind::Number(NumKind::Int | NumKind::Decimal))
        ) && self.kw_at(n + 1, "PERCENT")
        {
            self.pos += n;
            let value = self.text(0).to_owned();
            self.pos += 2;
            SampleMethodKind::Percent { negative, value }
        } else {
            let e = self.expr()?;
            if self.eat_kw("ROWS") {
                SampleMethodKind::Rows(e)
            } else {
                SampleMethodKind::Bytes(e)
            }
        };
        Ok(SampleMethod {
            kind,
            span: self.span_from(start),
        })
    }

    /// `[FOR] (VERSION | SYSTEM_VERSION) AS OF version` or `[FOR]
    /// (TIMESTAMP | SYSTEM_TIME) AS OF expression`, if one starts here.
    #[inline(never)]
    pub(super) fn temporal_clause(&mut self) -> PR<Option<TemporalClause>> {
        let start = self.start();
        let n = usize::from(self.kw_at(0, "FOR"));
        let kind = if self.any_kw_at(n, &["VERSION", "SYSTEM_VERSION"]) {
            TemporalKind::Version
        } else if self.any_kw_at(n, &["TIMESTAMP", "SYSTEM_TIME"]) {
            TemporalKind::Timestamp
        } else {
            return Ok(None);
        };
        if !(self.kw_at(n + 1, "AS") && self.kw_at(n + 2, "OF")) {
            return Ok(None);
        }
        self.pos += n + 3;
        let value = match kind {
            TemporalKind::Version => {
                let at = self.start();
                let literal = match self.kind(0) {
                    Some(TokenKind::Number(NumKind::Int)) => {
                        let text = self.text(0).to_owned();
                        self.bump();
                        Literal::Number {
                            text,
                            kind: NumKind::Int,
                        }
                    }
                    _ => match self.string_lit() {
                        Some(v) => Literal::String(v),
                        None => return self.fail(),
                    },
                };
                Expr {
                    kind: ExprKind::Literal(literal),
                    span: self.span_from(at),
                }
            }
            _ => self.value()?,
        };
        Ok(Some(TemporalClause {
            kind,
            value,
            span: self.span_from(start),
        }))
    }

    /// `WITH (key [=] value, ...)`, the options of a table.
    #[inline(never)]
    pub(super) fn table_options(&mut self) -> PR<Vec<TableOption>> {
        if !(self.kw_at(0, "WITH") && self.sym_at(1, Sym::LParen)) {
            return Ok(vec![]);
        }
        self.bump();
        self.paren_list(|p| {
            let start = p.start();
            let key = p.option_key()?;
            let equals = p.eat_sym(Sym::Eq);
            let literal = matches!(
                p.kind(0),
                Some(TokenKind::Number(NumKind::Int | NumKind::Decimal) | TokenKind::String)
            ) || p.any_kw_at(0, &["TRUE", "FALSE"])
                || (p.kind(0) == Some(TokenKind::DoubleQuoted)
                    && !p.opts.double_quoted_identifiers);
            let value = if equals || literal {
                if !literal {
                    return p.fail();
                }
                Some(p.primary_expr()?)
            } else {
                None
            };
            Ok(TableOption {
                key,
                value,
                span: p.span_from(start),
            })
        })
    }

    /// `name(.name)*` or a string.
    pub(super) fn option_key(&mut self) -> PR<OptionKey> {
        let start = self.start();
        if let Some(text) = self.single_string() {
            let span = self.span_from(start);
            return Ok(OptionKey {
                parts: vec![Ident {
                    value: text,
                    quoted: true,
                    span,
                }],
                is_string: true,
                span,
            });
        }
        let mut parts = vec![self.name_part(true)?];
        while self.eat_sym(Sym::Dot) {
            parts.push(self.name_part(true)?);
        }
        Ok(OptionKey {
            parts,
            is_string: false,
            span: self.span_from(start),
        })
    }

    /// A join, if one starts here.
    #[inline(never)]
    pub(super) fn join(&mut self) -> PR<Option<Join>> {
        let start = self.start();
        let natural = self.kw_at(0, "NATURAL");
        let n = usize::from(natural);
        let at = |p: &Self, i: usize, kw: &str| p.kw_at(n + i, kw);
        let (kind, words) = if at(self, 0, "JOIN") {
            (JoinKind::Inner, 0)
        } else if at(self, 0, "INNER") {
            (JoinKind::Inner, 1)
        } else if at(self, 0, "CROSS") {
            (JoinKind::Cross, 1)
        } else if at(self, 0, "LEFT") && at(self, 1, "SEMI") || at(self, 0, "SEMI") {
            (JoinKind::LeftSemi, 1 + usize::from(at(self, 0, "LEFT")))
        } else if at(self, 0, "LEFT") && at(self, 1, "ANTI") || at(self, 0, "ANTI") {
            (JoinKind::LeftAnti, 1 + usize::from(at(self, 0, "LEFT")))
        } else if let Some(kind) = [
            ("LEFT", JoinKind::Left),
            ("RIGHT", JoinKind::Right),
            ("FULL", JoinKind::Full),
        ]
        .into_iter()
        .find_map(|(kw, k)| at(self, 0, kw).then_some(k))
        {
            (kind, 1 + usize::from(at(self, 1, "OUTER")))
        } else {
            if natural {
                self.bump();
                return self.fail();
            }
            return Ok(None);
        };
        self.pos += n + words;
        self.expect_kw("JOIN")?;
        let lateral = self.eat_kw("LATERAL");
        let right = Box::new(self.table_factor()?);
        let constraint = if natural {
            JoinConstraint::None
        } else if self.eat_kw("ON") {
            JoinConstraint::On(self.expr()?)
        } else if self.eat_kw("USING") {
            JoinConstraint::Using(self.ident_list()?)
        } else if self.any_kw_at(0, &["APPROX", "EXACT"]) && self.kw_at(1, "NEAREST") {
            JoinConstraint::NearestBy(self.nearest_by()?)
        } else {
            JoinConstraint::None
        };
        Ok(Some(Join {
            kind,
            natural,
            lateral,
            right,
            constraint,
            span: self.span_from(start),
        }))
    }

    /// `(APPROX | EXACT) NEAREST [k] BY (DISTANCE | SIMILARITY) expr`.
    pub(super) fn nearest_by(&mut self) -> PR<NearestBy> {
        let start = self.start();
        let approx = self.kw_at(0, "APPROX");
        self.pos += 2;
        let num_results = if self.kind(0) == Some(TokenKind::Number(NumKind::Int)) {
            Some(self.integer_text()?)
        } else {
            None
        };
        self.expect_kw("BY")?;
        let direction = if self.eat_kw("DISTANCE") {
            NearestDirection::Distance
        } else if self.eat_kw("SIMILARITY") {
            NearestDirection::Similarity
        } else {
            return self.fail();
        };
        let ranking = self.expr()?;
        Ok(NearestBy {
            approx,
            num_results,
            direction,
            ranking,
            span: self.span_from(start),
        })
    }

    #[inline(never)]
    pub(super) fn pivot(&mut self) -> PR<Pivot> {
        let start = self.start();
        self.expect_kw("PIVOT")?;
        self.expect_sym(Sym::LParen)?;
        let aggregates = self.comma_sep(Self::named_item)?;
        self.expect_kw("FOR")?;
        let columns = if self.sym_at(0, Sym::LParen) {
            self.paren_list(|p| p.name_part(true))?
        } else {
            vec![self.name_part(true)?]
        };
        self.expect_kw("IN")?;
        let values = self.paren_list(|p| {
            let start = p.start();
            let expr = p.expr()?;
            let alias = p.opt_alias_name()?.map(Alias::Name);
            Ok(SelectItem {
                expr,
                alias,
                span: p.span_from(start),
            })
        })?;
        self.expect_sym(Sym::RParen)?;
        let alias = self.opt_alias_name()?;
        Ok(Pivot {
            aggregates,
            columns,
            values,
            alias,
            span: self.span_from(start),
        })
    }

    #[inline(never)]
    pub(super) fn unpivot(&mut self) -> PR<Unpivot> {
        let start = self.start();
        self.expect_kw("UNPIVOT")?;
        let mut include_nulls = None;
        if self.any_kw_at(0, &["INCLUDE", "EXCLUDE"]) {
            include_nulls = Some(self.kw_at(0, "INCLUDE"));
            self.bump();
            self.expect_kw("NULLS")?;
        }
        self.expect_sym(Sym::LParen)?;
        let multi = self.sym_at(0, Sym::LParen);
        let value_columns = if multi {
            self.ident_list()?
        } else {
            vec![self.any_ident()?]
        };
        self.expect_kw("FOR")?;
        let name_column = self.any_ident()?;
        self.expect_kw("IN")?;
        let columns = self.paren_list(|p| {
            let start = p.start();
            let columns = if multi {
                p.paren_list(Self::object_name)?
            } else {
                vec![p.object_name()?]
            };
            let alias = p.opt_alias_name()?;
            Ok(UnpivotColumns {
                columns,
                alias,
                span: p.span_from(start),
            })
        })?;
        self.expect_sym(Sym::RParen)?;
        let alias = self.opt_alias_name()?;
        Ok(Unpivot {
            include_nulls,
            value_columns,
            name_column,
            columns,
            alias,
            span: self.span_from(start),
        })
    }

    /// `LATERAL VIEW [OUTER] f(args) table [AS] col, ...`*.
    #[inline(never)]
    pub(super) fn lateral_views(&mut self) -> PR<Vec<LateralView>> {
        let mut out = vec![];
        while self.kw_at(0, "LATERAL") && self.kw_at(1, "VIEW") {
            let start = self.start();
            self.pos += 2;
            let outer = self.eat_kw("OUTER");
            let function = self.object_name()?;
            self.expect_sym(Sym::LParen)?;
            let args = if self.sym_at(0, Sym::RParen) {
                vec![]
            } else {
                self.comma_sep(Self::expr)?
            };
            self.expect_sym(Sym::RParen)?;
            // Spark's grammar needs the table name: `AS t` names it `AS`.
            let table = self.any_ident()?;
            let columns = if self.eat_kw("AS") {
                self.comma_sep(Self::any_ident)?
            } else if let Some(first) = self.implicit_alias(false)? {
                let mut columns = vec![first];
                while self.eat_sym(Sym::Comma) {
                    columns.push(self.ident()?);
                }
                columns
            } else {
                vec![]
            };
            out.push(LateralView {
                outer,
                function,
                args,
                table,
                columns,
                span: self.span_from(start),
            });
        }
        Ok(out)
    }
}
