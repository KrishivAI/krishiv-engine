// SPDX-License-Identifier: Apache-2.0
//! Parsing INSERT, DELETE, UPDATE and MERGE.

use super::*;

impl Parser<'_> {
    /// `INSERT INTO|OVERWRITE ...`.
    pub(crate) fn insert_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("INSERT")?;
            if p.kw_at(0, "OVERWRITE")
                && (p.kw_at(1, "DIRECTORY") || (p.kw_at(1, "LOCAL") && p.kw_at(2, "DIRECTORY")))
            {
                return p.insert_directory(start);
            }
            let clause = p.insert_clause(start, false)?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Insert(Box::new(Insert {
                source: None,
                clauses: vec![clause],
                span,
            })))
        })
    }

    /// `OVERWRITE [LOCAL] DIRECTORY ['path'] USING provider [OPTIONS (...)] query` or
    /// `OVERWRITE [LOCAL] DIRECTORY 'path' [ROW FORMAT ...] [STORED AS ...] query`, after `INSERT`.
    fn insert_directory(&mut self, start: usize) -> PR<Statement> {
        self.expect_kw("OVERWRITE")?;
        let local = self.eat_kw("LOCAL");
        self.expect_kw("DIRECTORY")?;
        let path = self.string_lit();
        let format = if self.eat_kw("USING") {
            let provider = self.table_name()?;
            let options = if self.eat_kw("OPTIONS") {
                self.property_list(true)?
            } else {
                Vec::new()
            };
            DirectoryFormat::Provider { provider, options }
        } else {
            if path.is_none() {
                return self.fail();
            }
            let row_format = if self.kw_at(0, "ROW") && self.kw_at(1, "FORMAT") {
                Some(self.row_format()?)
            } else {
                None
            };
            let file_format = if self.kw_at(0, "STORED") {
                Some(self.file_format()?)
            } else {
                None
            };
            DirectoryFormat::Hive {
                row_format,
                file_format,
            }
        };
        let query = self.insert_query(false)?;
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::InsertDirectory(Box::new(InsertDirectory {
            local,
            path,
            format,
            query: Box::new(query),
            span,
        })))
    }

    /// `LOAD DATA [LOCAL] INPATH 'path' [OVERWRITE] INTO TABLE name [PARTITION (...)]`.
    pub(crate) fn load_data_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("LOAD")?;
            p.expect_kw("DATA")?;
            let local = p.eat_kw("LOCAL");
            p.expect_kw("INPATH")?;
            let path = p.string_after()?;
            let overwrite = p.eat_kw("OVERWRITE");
            p.expect_kw("INTO")?;
            p.expect_kw("TABLE")?;
            let table = p.ddl_name()?;
            let partition = if p.kw_at(0, "PARTITION") {
                Some(p.partition_spec()?)
            } else {
                None
            };
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::LoadData(Box::new(LoadData {
                local,
                path,
                overwrite,
                table,
                partition,
                span,
            })))
        })
    }

    /// `FROM ... INSERT INTO|OVERWRITE ... [INSERT ...]`: a multi-table
    /// insert (the fallback for a `FROM`-led statement that is no query).
    pub(crate) fn multi_insert_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            let mut source = p.clause_from()?;
            // `INSERT` after the source is the first insert clause, not a
            // table alias (`FROM s INSERT INTO ...`).
            if let Some(relation) = source.relations.last_mut() {
                p.strip_insert_alias(relation);
            }
            let mut clauses = Vec::new();
            let outer = std::mem::replace(&mut p.insert_bodies, true);
            let read: PR<()> = (|| loop {
                let clause_start = p.start();
                p.expect_kw("INSERT")?;
                clauses.push(p.insert_clause(clause_start, true)?);
                if !p.kw_at(0, "INSERT") {
                    return Ok(());
                }
            })();
            p.insert_bodies = outer;
            read?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Insert(Box::new(Insert {
                source: Some(source),
                clauses,
                span,
            })))
        })
    }

    /// A bare `INSERT` alias at the end of a relation is the next insert
    /// clause, not an alias (`FROM s INSERT INTO ...`): drop it and
    /// rewind past it. Only a bare alias (never `AS INSERT`) ending
    /// exactly where parsing stopped is dropped, so a mid-query `INSERT`
    /// is untouched.
    pub(super) fn strip_insert_alias(&mut self, relation: &mut Relation) {
        let mut factor = &mut relation.primary;
        if let Some(RelationExt::Join(join)) = relation.extensions.last_mut() {
            factor = join.right.as_mut();
        }
        if let Some(alias) = &factor.alias
            && let Some(name) = &alias.name
            && !name.quoted
            && alias.columns.is_empty()
            && alias.span.range() == name.span.range()
            && name.value.eq_ignore_ascii_case("INSERT")
            && self
                .toks
                .get(self.pos.wrapping_sub(1))
                .is_some_and(|t| t.span.range() == name.span.range())
        {
            factor.alias = None;
            self.pos -= 1;
        }
    }

    /// The same, for the top level of a clause query (never a subquery).
    pub(super) fn strip_query_insert_alias(&mut self, query: &mut Query) {
        let mut body = &mut query.body;
        loop {
            match &mut body.kind {
                SetExprKind::Select(select) => {
                    if let Some(from) = select.from.as_mut()
                        && let Some(relation) = from.relations.last_mut()
                    {
                        self.strip_insert_alias(relation);
                    }
                    return;
                }
                SetExprKind::FromStatement(from) => {
                    if let Some(relation) = from.from.relations.last_mut() {
                        self.strip_insert_alias(relation);
                    }
                    return;
                }
                SetExprKind::SetOp { right, .. } => {
                    body = right.as_mut();
                }
                _ => return,
            }
        }
    }

    /// After the `INSERT` at `clause_start`, Spark's `insertInto` then the
    /// query: `[(WITH SCHEMA EVOLUTION)] OVERWRITE [TABLE] name [WITH
    /// (...)] [PARTITION ... [IF NOT EXISTS]] [BY NAME | (columns)] query`,
    /// `[(WITH SCHEMA EVOLUTION)] INTO [TABLE] name [WITH (...)]
    /// [PARTITION ...] [IF NOT EXISTS] [BY NAME | (columns)] query`, or
    /// `[(WITH SCHEMA EVOLUTION)] INTO [TABLE] name [alias] [WITH (...)]
    /// [BY NAME] REPLACE (WHERE | ON) ... query` or `REPLACE USING
    /// (columns) query`. A `(` opening a query reads as the query (it
    /// consumes its own parens), anything else as a column list.
    pub(in crate::parser) fn insert_clause(
        &mut self,
        clause_start: usize,
        multi: bool,
    ) -> PR<InsertClause> {
        let with_schema_evolution =
            if self.kw_at(0, "WITH") && self.kw_at(1, "SCHEMA") && self.kw_at(2, "EVOLUTION") {
                self.pos += 3;
                true
            } else {
                false
            };
        let overwrite = if self.eat_kw("OVERWRITE") {
            true
        } else {
            self.expect_kw("INTO")?;
            false
        };
        self.eat_kw("TABLE");
        let name = self.ddl_name()?;
        if overwrite {
            let target_options = self.table_options()?;
            let partition = if self.kw_at(0, "PARTITION") {
                Some(self.partition_spec()?)
            } else {
                None
            };
            // `IF NOT EXISTS` follows a partition for `OVERWRITE`; for
            // `INTO` it parses anywhere here and fails after parsing.
            let if_not_exists = partition.is_some() && self.eat_if_not_exists();
            let by_name = if self.kw_at(0, "BY") && self.kw_at(1, "NAME") {
                self.pos += 2;
                true
            } else {
                false
            };
            let head_before_columns = self.span_from(clause_start);
            let (columns, paren_query) = if by_name {
                (Vec::new(), None)
            } else if multi && self.sym_at(0, Sym::LParen) && self.query_follows(1) {
                // A multi-table insert body is no parenthesized query.
                return self.fail();
            } else if self.sym_at(0, Sym::LParen) && self.query_follows(1) {
                let save = self.checkpoint();
                match self.query() {
                    Ok(query) => {
                        self.commit(save);
                        (Vec::new(), Some(query))
                    }
                    Err(Fail::Syntax) => {
                        self.restore(save);
                        (self.paren_list(Self::column_name)?, None)
                    }
                    Err(fatal) => return Err(fatal),
                }
            } else if self.sym_at(0, Sym::LParen) {
                (self.paren_list(Self::column_name)?, None)
            } else {
                (Vec::new(), None)
            };
            let head_span = if columns.is_empty() {
                head_before_columns
            } else {
                self.span_from(clause_start)
            };
            let query = match paren_query {
                Some(query) => query,
                None => self.insert_query(multi)?,
            };
            return Ok(InsertClause {
                overwrite,
                with_schema_evolution,
                name,
                target_alias: None,
                target_options,
                columns,
                partition,
                if_not_exists,
                replace_where: None,
                replace_on: None,
                replace_using: None,
                by_name,
                query: Box::new(query),
                head_span,
                span: self.span_from(clause_start),
            });
        }
        // `INTO`: try a `REPLACE` variant first (an alias, `WITH (...)`
        // options and `BY NAME` may precede `REPLACE`).
        let save = self.checkpoint();
        let target_alias: Option<Ident> = if self.eat_kw("AS") {
            Some(self.any_ident()?)
        } else if self.is_ident_at(0, true)
            && (self.kw_at(1, "WITH") || self.kw_at(1, "BY") || self.kw_at(1, "REPLACE"))
        {
            // A bare alias only when something a replace head takes
            // follows; otherwise the name stands alone.
            Some(self.any_ident()?)
        } else {
            None
        };
        let target_options = self.table_options()?;
        let by_name_before = if self.kw_at(0, "BY") && self.kw_at(1, "NAME") {
            self.pos += 2;
            true
        } else {
            false
        };
        if self.kw_at(0, "REPLACE") {
            let replace_kw_at = self.pos;
            self.pos += 1;
            if self.eat_kw("WHERE") {
                let where_at = replace_kw_at + 1;
                let replace_where = Some(self.expr()?);
                if !by_name_before && self.kw_at(0, "BY") && self.kw_at(1, "NAME") {
                    // `BY NAME` goes before `REPLACE`: ANTLR's prediction
                    // rejects the reverse order at the `WHERE`.
                    self.pos = where_at;
                    self.far = where_at;
                    self.matched = None;
                    self.extra = None;
                    self.missing = None;
                    return Err(Fail::Syntax);
                }
                self.commit(save);
                let head_span = self.span_from(clause_start);
                let query = self.insert_query(multi)?;
                return Ok(InsertClause {
                    overwrite: false,
                    with_schema_evolution,
                    name,
                    target_alias,
                    target_options,
                    columns: Vec::new(),
                    partition: None,
                    if_not_exists: false,
                    replace_where,
                    replace_on: None,
                    replace_using: None,
                    by_name: by_name_before,
                    query: Box::new(query),
                    head_span,
                    span: self.span_from(clause_start),
                });
            }
            if self.eat_kw("ON") {
                let on_at = replace_kw_at + 1;
                let replace_on = Some(self.expr()?);
                if !by_name_before && self.kw_at(0, "BY") && self.kw_at(1, "NAME") {
                    self.pos = on_at;
                    self.far = on_at;
                    self.matched = None;
                    self.extra = None;
                    self.missing = None;
                    return Err(Fail::Syntax);
                }
                self.commit(save);
                let head_span = self.span_from(clause_start);
                let query = self.insert_query(multi)?;
                return Ok(InsertClause {
                    overwrite: false,
                    with_schema_evolution,
                    name,
                    target_alias,
                    target_options,
                    columns: Vec::new(),
                    partition: None,
                    if_not_exists: false,
                    replace_where: None,
                    replace_on,
                    replace_using: None,
                    by_name: by_name_before,
                    query: Box::new(query),
                    head_span,
                    span: self.span_from(clause_start),
                });
            }
            if self.eat_kw("USING") {
                let using_at = replace_kw_at + 1;
                let replace_using = Some(self.ident_list()?);
                if !by_name_before && self.kw_at(0, "BY") && self.kw_at(1, "NAME") {
                    self.pos = using_at;
                    self.far = using_at;
                    self.matched = None;
                    self.extra = None;
                    self.missing = None;
                    return Err(Fail::Syntax);
                }
                self.commit(save);
                let head_span = self.span_from(clause_start);
                let query = self.insert_query(multi)?;
                return Ok(InsertClause {
                    overwrite: false,
                    with_schema_evolution,
                    name,
                    target_alias,
                    target_options,
                    columns: Vec::new(),
                    partition: None,
                    if_not_exists: false,
                    replace_where: None,
                    replace_on: None,
                    replace_using,
                    by_name: by_name_before,
                    query: Box::new(query),
                    head_span,
                    span: self.span_from(clause_start),
                });
            }
            // `REPLACE` with nothing Spark takes: a syntax error here.
            return self.fail();
        }
        self.restore(save);
        let target_options = self.table_options()?;
        let partition = if self.kw_at(0, "PARTITION") {
            Some(self.partition_spec()?)
        } else {
            None
        };
        // `IF NOT EXISTS` follows a partition for `OVERWRITE`; for `INTO`
        // it parses anywhere here and fails after parsing.
        let if_not_exists = self.eat_if_not_exists();
        let by_name = if self.kw_at(0, "BY") && self.kw_at(1, "NAME") {
            self.pos += 2;
            true
        } else {
            false
        };
        let head_before_columns = self.span_from(clause_start);
        let (columns, paren_query) = if by_name {
            (Vec::new(), None)
        } else if multi && self.sym_at(0, Sym::LParen) && self.query_follows(1) {
            // A multi-table insert body is no parenthesized query.
            return self.fail();
        } else if self.sym_at(0, Sym::LParen) && self.query_follows(1) {
            let save = self.checkpoint();
            match self.query() {
                Ok(query) => {
                    self.commit(save);
                    (Vec::new(), Some(query))
                }
                Err(Fail::Syntax) => {
                    self.restore(save);
                    (self.paren_list(Self::column_name)?, None)
                }
                Err(fatal) => return Err(fatal),
            }
        } else if self.sym_at(0, Sym::LParen) {
            (self.paren_list(Self::column_name)?, None)
        } else {
            (Vec::new(), None)
        };
        let head_span = if columns.is_empty() {
            head_before_columns
        } else {
            self.span_from(clause_start)
        };
        let query = match paren_query {
            Some(query) => query,
            None => self.insert_query(multi)?,
        };
        Ok(InsertClause {
            overwrite: false,
            with_schema_evolution,
            name,
            target_alias: None,
            target_options,
            columns,
            partition,
            if_not_exists,
            replace_where: None,
            replace_on: None,
            replace_using: None,
            by_name,
            query: Box::new(query),
            head_span,
            span: self.span_from(clause_start),
        })
    }

    /// The rows of an insert clause: a query, or for a multi-table insert
    /// (`multi`) Spark's `fromStatementBody` (a `SELECT` or `TRANSFORM`
    /// without `FROM` or set operators, then `queryOrganization`).
    pub(super) fn insert_query(&mut self, multi: bool) -> PR<Query> {
        if !multi {
            let mut query = self.query()?;
            self.strip_query_insert_alias(&mut query);
            return Ok(query);
        }
        let start = self.start();
        let body = self.select_kind(start, false)?;
        let mut query = Self::bare_query(body, start);
        self.organization(&mut query)?;
        query.span = self.span_from(start);
        Ok(query)
    }

    /// `DELETE FROM name [AS alias] [WHERE ...]`.
    pub(super) fn delete_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("DELETE")?;
            let follows = p.name_here();
            p.expect_kw_or_missing("FROM", "'FROM'", follows)?;
            let table = p.ddl_name()?;
            let alias = p.stmt_alias("WHERE")?;
            let where_clause = p.opt_clause("WHERE")?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Delete(Delete {
                table,
                alias,
                where_clause,
                span,
            }))
        })
    }

    /// `UPDATE name [AS alias] SET ... [WHERE ...]`.
    pub(super) fn update_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("UPDATE")?;
            let table = p.ddl_name()?;
            let alias = p.stmt_alias("SET")?;
            p.expect_kw("SET")?;
            let mut assignments = vec![p.update_assignment()?];
            while p.eat_sym(Sym::Comma) {
                assignments.push(p.update_assignment()?);
            }
            let where_clause = p.opt_clause("WHERE")?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Update(Box::new(Update {
                table,
                alias,
                assignments,
                where_clause,
                span,
            })))
        })
    }

    /// `target = expr` in an `UPDATE` or `MERGE` action.
    pub(super) fn update_assignment(&mut self) -> PR<UpdateAssignment> {
        let start = self.start();
        let target = self.table_name()?;
        self.expect_sym(Sym::Eq)?;
        let value = self.expr()?;
        Ok(UpdateAssignment {
            target,
            value,
            span: self.span_from(start),
        })
    }

    /// `MERGE INTO target USING source ON ... WHEN ...`.
    pub(super) fn merge_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("MERGE")?;
            p.expect_kw("INTO")?;
            let target = p.ddl_name()?;
            let target_alias = p.stmt_alias("USING")?;
            p.expect_kw("USING")?;
            let source = p.merge_source()?;
            p.expect_kw("ON")?;
            let on = p.expr()?;
            let mut clauses = Vec::new();
            while p.eat_kw("WHEN") {
                clauses.push(p.merge_clause()?);
            }
            let span = p.span_from(start);
            p.command_end()?;
            if clauses.is_empty() {
                // Spark's builder, after parsing: `MERGE_WITHOUT_WHEN`.
                return Self::fatal(ParseError {
                    error: MergeWithoutWhen {}.into(),
                    span,
                });
            }
            Ok(Statement::Merge(Box::new(Merge {
                target,
                target_alias,
                source,
                on,
                clauses,
                span,
            })))
        })
    }

    /// The source of a `MERGE`: a table or a subquery, each with an
    /// optional alias.
    pub(super) fn merge_source(&mut self) -> PR<MergeSource> {
        let start = self.start();
        if self.sym_at(0, Sym::LParen) && self.query_follows(1) {
            let query = Box::new(self.query()?);
            let alias = self.stmt_alias("ON")?;
            Ok(MergeSource::Query {
                query,
                alias,
                span: self.span_from(start),
            })
        } else {
            let name = self.ddl_name()?;
            let alias = self.stmt_alias("ON")?;
            Ok(MergeSource::Table {
                name,
                alias,
                span: self.span_from(start),
            })
        }
    }

    /// `MATCHED ... THEN ...` after `WHEN`.
    pub(super) fn merge_clause(&mut self) -> PR<MergeClause> {
        let start = self.start();
        let matched = if self.eat_kw("MATCHED") {
            MergeMatched::Matched
        } else if self.kw_at(0, "NOT") && self.kw_at(1, "MATCHED") {
            self.pos += 2;
            if self.kw_at(0, "BY") && self.kw_at(1, "TARGET") {
                self.pos += 2;
                MergeMatched::NotMatchedByTarget
            } else if self.kw_at(0, "BY") && self.kw_at(1, "SOURCE") {
                self.pos += 2;
                MergeMatched::NotMatchedBySource
            } else {
                MergeMatched::NotMatched
            }
        } else {
            return self.fail();
        };
        let condition = if self.eat_kw("AND") {
            Some(self.expr()?)
        } else {
            None
        };
        self.expect_kw("THEN")?;
        let action = self.merge_action(matched)?;
        Ok(MergeClause {
            matched,
            condition,
            action,
            span: self.span_from(start),
        })
    }

    /// The action of a `WHEN ... THEN ...` for `matched`: Spark allows
    /// only `UPDATE`/`DELETE` after `WHEN MATCHED` or `WHEN NOT MATCHED BY
    /// SOURCE`, and only `INSERT` after `WHEN NOT MATCHED [BY TARGET]`.
    /// Anything else is a `PARSE_SYNTAX_ERROR` at the action.
    pub(super) fn merge_action(&mut self, matched: MergeMatched) -> PR<MergeAction> {
        let allow_update_delete = matches!(
            matched,
            MergeMatched::Matched | MergeMatched::NotMatchedBySource
        );
        let allow_insert = matches!(
            matched,
            MergeMatched::NotMatched | MergeMatched::NotMatchedByTarget
        );
        if self.kw_at(0, "UPDATE") && !allow_update_delete {
            return self.fail();
        }
        if self.kw_at(0, "DELETE") && !allow_update_delete {
            return self.fail();
        }
        if self.kw_at(0, "INSERT") && !allow_insert {
            return self.fail();
        }
        let start = self.start();
        if self.kw_at(0, "UPDATE") && self.kw_at(1, "SET") {
            self.pos += 2;
            if matched == MergeMatched::Matched && self.eat_sym(Sym::Star) {
                return Ok(MergeAction::UpdateStar);
            }
            let mut assignments = vec![self.update_assignment()?];
            while self.eat_sym(Sym::Comma) {
                assignments.push(self.update_assignment()?);
            }
            return Ok(MergeAction::Update(assignments));
        }
        if self.eat_kw("DELETE") {
            return Ok(MergeAction::Delete);
        }
        if self.eat_kw("INSERT") {
            // `INSERT *` or `INSERT (columns) VALUES (...)`.
            if self.eat_sym(Sym::Star) {
                return Ok(MergeAction::Insert(MergeInsert::Star));
            }
            if !self.sym_at(0, Sym::LParen) {
                return self.fail();
            }
            let columns = self.paren_list(Self::table_name)?;
            self.expect_kw("VALUES")?;
            self.expect_sym(Sym::LParen)?;
            let values = self.comma_sep(Self::expr)?;
            self.expect_closing(Sym::RParen, Close::List(values.len()))?;
            return Ok(MergeAction::Insert(MergeInsert::Values {
                columns,
                values,
                span: self.span_from(start),
            }));
        }
        self.fail()
    }
}
