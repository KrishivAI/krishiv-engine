// SPDX-License-Identifier: Apache-2.0
//! Parsing CREATE and DROP statements: tables, columns, constraints, partitions, row formats and namespaces.

use super::*;

impl Parser<'_> {
    /// `CREATE ...`: a temporary view, a table (or `TABLE ... LIKE`), a
    /// function, or a namespace. Anything else (`CREATE VIEW`,
    /// `CREATE GLOBAL ...`, ...) fails where it stands, as in Spark.
    pub(super) fn create_command(&mut self) -> PR<Statement> {
        // The legacy shapes first (temporary views, tables, namespaces);
        // a view-shaped statement the legacy reading rejects reads as a
        // full view statement instead (a fatal error is the legacy
        // reading's own).
        if !self.create_is_view_shaped() {
            return self.create_command_inner();
        }
        let start = self.start();
        let save = self.checkpoint();
        match self.create_command_inner() {
            Ok(statement) => {
                self.commit(save);
                Ok(statement)
            }
            Err(Fail::Fatal(e)) => Err(Fail::Fatal(e)),
            Err(Fail::Syntax) => {
                self.restore(save);
                self.create_full_view_command(start)
            }
        }
    }

    /// Whether the statement at `CREATE` reads `CREATE [OR REPLACE]
    /// [GLOBAL] [TEMP|TEMPORARY] VIEW`.
    pub(super) fn create_is_view_shaped(&self) -> bool {
        let mut n = 1;
        if self.kw_at(n, "OR") && self.kw_at(n + 1, "REPLACE") {
            n += 2;
        }
        if self.kw_at(n, "GLOBAL") {
            n += 1;
        }
        if self.kw_at(n, "TEMP") || self.kw_at(n, "TEMPORARY") {
            n += 1;
        }
        self.kw_at(n, "VIEW")
    }

    pub(super) fn create_command_inner(&mut self) -> PR<Statement> {
        let start = self.start();
        self.expect_kw("CREATE")?;
        // `OR` is followed by `REPLACE` only: the error stands at the word after it.
        if self.kw_at(0, "OR") && !self.kw_at(1, "REPLACE") && self.tok(1).is_some() {
            self.pos += 1;
            return self.fail();
        }
        let or_replace = if self.kw_at(0, "OR") && self.kw_at(1, "REPLACE") {
            self.pos += 2;
            true
        } else {
            false
        };
        if or_replace {
            if self.eat_kw("TEMP") || self.eat_kw("TEMPORARY") {
                if self.kw_at(0, "FUNCTION") {
                    return self.create_function_command(start, true, true);
                }
                return self.create_temp_view_command(start, true);
            }
            if self.kw_at(0, "TABLE") {
                return self.create_table_command(start, true, false, false);
            }
            if self.kw_at(0, "FUNCTION") {
                return self.create_function_command(start, true, false);
            }
            return self.fail();
        }
        let temporary = self.eat_kw("TEMP") || self.eat_kw("TEMPORARY");
        if self.kw_at(0, "FUNCTION") {
            return self.create_function_command(start, false, temporary);
        }
        if !temporary && self.kw_at(0, "FLOW") {
            return self.create_flow_command(start);
        }
        if !temporary
            && ((self.kw_at(0, "MATERIALIZED") && self.kw_at(1, "VIEW"))
                || (self.kw_at(0, "STREAMING") && self.kw_at(1, "TABLE")))
        {
            return self.create_pipeline_dataset_command(start);
        }
        let external = self.eat_kw("EXTERNAL");
        if self.kw_at(0, "TABLE") {
            return self.create_table_command(start, false, temporary, external);
        }
        if self.kw_at(0, "VIEW") {
            if temporary && !external {
                return self.create_temp_view_command(start, false);
            }
            return self.fail();
        }
        if !temporary && !external {
            return self.create_namespace_command(start);
        }
        self.fail()
    }

    /// `CREATE [OR REPLACE] TEMPORARY VIEW name [(column [COMMENT '...'],
    /// ...)] [DEFAULT COLLATION name] AS query` (`TEMP` for `TEMPORARY`),
    /// after the `CREATE [OR REPLACE] TEMPORARY` at `start`. `IF NOT
    /// EXISTS` or a qualified name fails here, so the statement reads as a
    /// full view statement ([`Parser::create_full_view_command`]), whose
    /// `CREATE_VIEW_WITH_IF_NOT_EXISTS_AND_REPLACE`, `_LEGACY_ERROR_TEMP_0053`
    /// and `INVALID_TEMP_OBJ_QUALIFIER` validation raises in Spark's order.
    pub(super) fn create_temp_view_command(
        &mut self,
        start: usize,
        or_replace: bool,
    ) -> PR<Statement> {
        self.expect_kw("VIEW")?;
        // `IF NOT EXISTS` and a qualified name read as the full view
        // statement, whose checks run after parsing in Spark's order.
        if self.kw_at(0, "IF") {
            return self.fail();
        }
        let first = self.any_ident()?;
        if self.sym_at(0, Sym::Dot) {
            return self.fail();
        }
        let columns = if self.sym_at(0, Sym::LParen) {
            self.paren_list(Self::view_column)?
        } else {
            Vec::new()
        };
        let collation = if self.kw_at(0, "DEFAULT") && self.kw_at(1, "COLLATION") {
            self.pos += 2;
            Some(self.name_part(true)?)
        } else {
            None
        };
        self.expect_kw("AS")?;
        let query = Box::new(self.query()?);
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::CreateView(CreateView {
            or_replace,
            if_not_exists: false,
            name: first,
            columns,
            collation,
            query,
            span,
        }))
    }

    /// `name [COMMENT '...']` in a `CREATE VIEW` column list.
    pub(super) fn view_column(&mut self) -> PR<ViewColumn> {
        let start = self.start();
        let name = self.any_ident()?;
        let comment = if self.eat_kw("COMMENT") {
            match self.string_lit() {
                Some(comment) => Some(comment),
                None => return self.fail(),
            }
        } else {
            None
        };
        Ok(ViewColumn {
            name,
            comment,
            span: self.span_from(start),
        })
    }

    /// `DROP ...`: a view, a table, a function, a namespace, or a temporary variable.
    /// Anything else fails where it stands, as in Spark. A `DROP VIEW`
    /// the one-part reading rejects reads as a full view statement instead.
    pub(super) fn drop_command(&mut self) -> PR<Statement> {
        if !self.kw_at(1, "VIEW") {
            return self.drop_command_inner();
        }
        let start = self.start();
        let save = self.checkpoint();
        match self.drop_command_inner() {
            // A one-part reading that deferred an error (an identifier
            // literal of several parts, `DROP VIEW IDENTIFIER('a.b.c')`)
            // reads as the full view statement, which takes the whole name.
            Ok(statement) if self.deferred.is_none() => {
                self.commit(save);
                Ok(statement)
            }
            Err(Fail::Fatal(e)) => Err(Fail::Fatal(e)),
            Ok(_) | Err(Fail::Syntax) => {
                self.restore(save);
                self.expect_kw("DROP")?;
                self.drop_full_view_command(start)
            }
        }
    }

    /// `DROP ...`: a view, a table, a namespace, or a temporary variable.
    /// Anything else fails where it stands, as in Spark.
    pub(super) fn drop_command_inner(&mut self) -> PR<Statement> {
        let start = self.start();
        self.expect_kw("DROP")?;
        if self.kw_at(0, "FUNCTION")
            || (self.any_kw_at(0, &["TEMP", "TEMPORARY"]) && self.kw_at(1, "FUNCTION"))
        {
            return self.drop_function_command(start);
        }
        if self.kw_at(0, "VIEW") {
            return self.drop_view_command(start);
        }
        if self.kw_at(0, "TABLE") {
            return self.drop_table_command(start);
        }
        if self.kw_at(0, "TEMPORARY")
            || (self.kw_at(0, "TEMP") && (self.kw_at(1, "VAR") || self.kw_at(1, "VARIABLE")))
        {
            return self.drop_variable_command(start);
        }
        self.drop_namespace_command(start)
    }

    /// `DROP TEMP|TEMPORARY VAR[IABLE] [IF EXISTS] name` (`VAR` for
    /// `VARIABLE`), after the `DROP` at `start`. Any other word after
    /// `TEMPORARY` fails there, as Spark reports `DROP TEMPORARY`.
    pub(super) fn drop_variable_command(&mut self, start: usize) -> PR<Statement> {
        if !self.eat_kw("TEMP") {
            self.expect_kw("TEMPORARY")?;
        }
        if !(self.eat_kw("VAR") || self.eat_kw("VARIABLE")) {
            return self.fail();
        }
        let if_exists = if self.kw_at(0, "IF") && self.kw_at(1, "EXISTS") {
            self.pos += 2;
            true
        } else {
            false
        };
        let name = self.variable_reference()?;
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::DropVariable(DropVariable {
            if_exists,
            name,
            span,
        }))
    }

    /// `DROP VIEW [IF EXISTS] name`, after the `DROP` at `start`.
    pub(super) fn drop_view_command(&mut self, start: usize) -> PR<Statement> {
        self.expect_kw("VIEW")?;
        let if_exists = if self.kw_at(0, "IF") && self.kw_at(1, "EXISTS") {
            self.pos += 2;
            true
        } else {
            false
        };
        let name = self.any_ident()?;
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::DropView(DropView {
            if_exists,
            name,
            span,
        }))
    }

    /// A table, namespace or `USE` name: dotted parts or
    /// `IDENTIFIER(expr)`, which Spark's `identifierReference` allows in
    /// both identifier modes.
    pub(super) fn ddl_name(&mut self) -> PR<DdlName> {
        if self.kw_at(0, "IDENTIFIER") && self.sym_at(1, Sym::LParen) {
            let at = self.checkpoint();
            let name = self.identifier_name()?;
            // `IDENTIFIER('a').b` is a dotted name whose first part is an
            // identifier literal, unless only the legacy function syntax is on.
            if !self.sym_at(0, Sym::Dot) || self.opts.legacy_identifier_clause {
                self.commit(at);
                return Ok(DdlName::Dynamic(name));
            }
            self.restore(at);
        }
        self.table_name().map(DdlName::Parts)
    }

    /// A column name: `name` or `IDENTIFIER('name')` (never in legacy
    /// identifier mode, where it fails at the `(` as in Spark).
    pub(super) fn column_name(&mut self) -> PR<ColumnName> {
        if self.kw_at(0, "IDENTIFIER") && self.sym_at(1, Sym::LParen) {
            if self.opts.legacy_identifier_clause {
                self.pos += 1;
                return self.fail();
            }
            return self.identifier_name().map(ColumnName::Dynamic);
        }
        self.name_part(true).map(ColumnName::Ident)
    }

    /// The name `IDENTIFIER` as written at `word`: what a function named by
    /// `IDENTIFIER(expr)` holds as its name.
    pub(crate) fn identifier_word(&self, word: Span) -> ObjectName {
        ObjectName {
            parts: vec![Ident {
                value: word.text(self.src).to_owned(),
                quoted: false,
                span: word,
            }],
            span: word,
        }
    }

    /// `IDENTIFIER(expr)`.
    pub(super) fn identifier_name(&mut self) -> PR<DynamicName> {
        let start = self.start();
        self.expect_kw("IDENTIFIER")?;
        self.expect_sym(Sym::LParen)?;
        let expr = Box::new(self.expr()?);
        self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
        Ok(DynamicName {
            expr,
            span: self.span_from(start),
        })
    }

    /// `DATABASE`, `SCHEMA` or `NAMESPACE`.
    pub(super) fn namespace_kind(&mut self) -> PR<NamespaceKind> {
        if self.eat_kw("DATABASE") {
            Ok(NamespaceKind::Database)
        } else if self.eat_kw("SCHEMA") {
            Ok(NamespaceKind::Schema)
        } else if self.eat_kw("NAMESPACE") {
            Ok(NamespaceKind::Namespace)
        } else {
            self.fail()
        }
    }

    /// The string literal after `LOCATION`, `COMMENT` or `STORED BY`
    /// (adjacent strings concatenate, as in Spark).
    pub(super) fn string_after(&mut self) -> PR<String> {
        match self.string_lit() {
            Some(s) => Ok(s),
            None => self.fail(),
        }
    }

    /// `CREATE [OR REPLACE] [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS]
    /// name [(elements)] [USING provider] [clauses] [AS query]`, after the
    /// `CREATE [OR REPLACE]` at `start` (`OR REPLACE` takes no
    /// `TEMPORARY`/`EXTERNAL`/`IF NOT EXISTS`/`LIKE`, as in Spark).
    pub(super) fn create_table_command(
        &mut self,
        start: usize,
        or_replace: bool,
        temporary: bool,
        external: bool,
    ) -> PR<Statement> {
        self.nested(|p| {
            p.expect_kw("TABLE")?;
            p.create_table_after_keyword(start, or_replace, temporary, external)
        })
    }

    /// `CREATE MATERIALIZED VIEW | STREAMING TABLE [IF NOT EXISTS] name [(columns)] [clauses] [AS query]`,
    /// after the `CREATE` at `start`: the `CREATE TABLE` clauses over a pipeline dataset.
    pub(super) fn create_pipeline_dataset_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            let kind = if p.eat_kw("MATERIALIZED") {
                p.expect_kw("VIEW")?;
                PipelineDatasetKind::MaterializedView
            } else {
                p.expect_kw("STREAMING")?;
                p.expect_kw("TABLE")?;
                PipelineDatasetKind::StreamingTable
            };
            if p.kw_at(0, "LIKE") {
                return p.fail();
            }
            match p.create_table_after_keyword(start, false, false, false)? {
                Statement::CreateTable(table) => {
                    let span = table.span;
                    Ok(Statement::CreatePipelineDataset(Box::new(
                        CreatePipelineDataset {
                            kind,
                            table: *table,
                            span,
                        },
                    )))
                }
                _ => p.fail(),
            }
        })
    }

    /// `CREATE FLOW name [COMMENT '...'] AS INSERT ...`, after the `CREATE` at `start`.
    pub(super) fn create_flow_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            p.expect_kw("FLOW")?;
            let name = p.ddl_name()?;
            let comment = if p.eat_kw("COMMENT") {
                Some(p.string_after()?)
            } else {
                None
            };
            p.expect_kw("AS")?;
            let clause_start = p.start();
            p.expect_kw("INSERT")?;
            let clause = p.insert_clause(clause_start, false)?;
            let insert_span = p.span_from(clause_start);
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::CreateFlow(Box::new(CreateFlow {
                name,
                comment,
                insert: Insert {
                    source: None,
                    clauses: vec![clause],
                    span: insert_span,
                },
                span,
            })))
        })
    }

    /// The part of `CREATE TABLE` after the `TABLE` keyword.
    fn create_table_after_keyword(
        &mut self,
        start: usize,
        or_replace: bool,
        temporary: bool,
        external: bool,
    ) -> PR<Statement> {
        let p = self;
        {
            let if_not_exists =
                if !or_replace && p.kw_at(0, "IF") && p.kw_at(1, "NOT") && p.kw_at(2, "EXISTS") {
                    p.pos += 3;
                    true
                } else {
                    false
                };
            let name = p.ddl_name()?;
            // `... LIKE source ...`: no columns and none of the other
            // clauses.
            if !or_replace && p.eat_kw("LIKE") {
                let source = p.ddl_name()?;
                let mut like = CreateTableLike {
                    if_not_exists,
                    name,
                    source,
                    using: None,
                    row_format: None,
                    file_format: None,
                    location: None,
                    tblproperties: Vec::new(),
                    duplicate_clause: None,
                    span: Span::new(start, start),
                };
                // Occurrences per clause, in Spark's duplicate-check order.
                let mut seen = [0_usize; CREATE_TABLE_LIKE_CLAUSES.len()];
                loop {
                    let clause = if p.eat_kw("USING") {
                        like.using = Some(p.table_name()?);
                        "PROVIDER"
                    } else if p.kw_at(0, "ROW") && p.kw_at(1, "FORMAT") {
                        like.row_format = Some(p.row_format()?);
                        "ROW FORMAT"
                    } else if p.kw_at(0, "STORED") {
                        like.file_format = Some(p.file_format()?);
                        "STORED AS/BY"
                    } else if p.eat_kw("LOCATION") {
                        like.location = Some(p.string_after()?);
                        "LOCATION"
                    } else if p.eat_kw("TBLPROPERTIES") {
                        like.tblproperties.extend(p.property_list(false)?);
                        "TBLPROPERTIES"
                    } else {
                        break;
                    };
                    if let Some(i) = CREATE_TABLE_LIKE_CLAUSES.iter().position(|k| *k == clause) {
                        seen[i] += 1;
                    }
                }
                like.duplicate_clause = CREATE_TABLE_LIKE_CLAUSES
                    .iter()
                    .zip(seen)
                    .find(|(_, n)| *n > 1)
                    .map(|(c, _)| *c);
                like.span = p.span_from(start);
                p.command_end()?;
                return Ok(Statement::CreateTableLike(Box::new(like)));
            }
            // `(columns)` or `(query)`: a parenthesized query reads as the
            // CTAS query (the query consumes its own parens), anything else
            // as a column list. ANTLR predicts a query when the `(` opens
            // one and rewinds to the column list otherwise, so `WITH` still
            // reads as a column name. No clauses follow a parenthesized
            // query.
            let (elements, paren_query) = if p.sym_at(0, Sym::LParen) && p.query_follows(1) {
                let save = p.checkpoint();
                match p.query() {
                    Ok(query) => {
                        p.commit(save);
                        (Vec::new(), Some(query))
                    }
                    Err(Fail::Syntax) => {
                        p.restore(save);
                        (p.paren_list(Self::table_element)?, None)
                    }
                    Err(fatal) => return Err(fatal),
                }
            } else if p.sym_at(0, Sym::LParen) {
                (p.paren_list(Self::table_element)?, None)
            } else {
                (Vec::new(), None)
            };
            let using = if paren_query.is_none() && p.eat_kw("USING") {
                Some(p.table_name()?)
            } else {
                None
            };
            let mut options = Vec::new();
            let mut partitioned_by = Vec::new();
            let mut cluster_by = Vec::new();
            let mut bucket = None;
            let mut skew = None;
            let mut row_format = None;
            let mut file_format = None;
            let mut location = None;
            let mut comment = None;
            let mut collation = None;
            let mut tblproperties = Vec::new();
            // Occurrences per clause, in Spark's duplicate-check order.
            let mut seen = [0_usize; CREATE_TABLE_CLAUSES.len()];
            loop {
                if paren_query.is_some() {
                    break;
                }
                let clause = if p.kw_at(0, "TBLPROPERTIES") {
                    Some("TBLPROPERTIES")
                } else if p.kw_at(0, "OPTIONS") {
                    Some("OPTIONS")
                } else if p.kw_at(0, "PARTITIONED") && p.kw_at(1, "BY") {
                    Some("PARTITIONED BY")
                } else if p.kw_at(0, "STORED") {
                    Some("STORED AS/BY")
                } else if p.kw_at(0, "ROW") && p.kw_at(1, "FORMAT") {
                    Some("ROW FORMAT")
                } else if p.kw_at(0, "COMMENT") {
                    Some("COMMENT")
                } else if p.kw_at(0, "DEFAULT") && p.kw_at(1, "COLLATION") {
                    Some("DEFAULT COLLATION")
                } else if p.kw_at(0, "CLUSTERED") {
                    Some("CLUSTERED BY")
                } else if p.kw_at(0, "CLUSTER") && p.kw_at(1, "BY") {
                    Some("CLUSTER BY")
                } else if p.kw_at(0, "LOCATION") {
                    Some("LOCATION")
                } else {
                    None
                };
                if let Some(i) =
                    clause.and_then(|c| CREATE_TABLE_CLAUSES.iter().position(|k| *k == c))
                {
                    seen[i] += 1;
                }
                if p.eat_kw("OPTIONS") {
                    options.extend(p.property_list(true)?);
                } else if p.kw_at(0, "PARTITIONED") && p.kw_at(1, "BY") {
                    p.pos += 2;
                    partitioned_by.extend(p.paren_list(Self::partition_field)?);
                } else if p.kw_at(0, "SKEWED") {
                    skew = Some(p.skew_spec()?);
                } else if p.kw_at(0, "CLUSTERED") {
                    bucket = Some(p.bucket_spec()?);
                } else if p.kw_at(0, "CLUSTER") && p.kw_at(1, "BY") {
                    p.pos += 2;
                    cluster_by.extend(p.paren_list(Self::table_name)?);
                } else if p.kw_at(0, "ROW") && p.kw_at(1, "FORMAT") {
                    row_format = Some(p.row_format()?);
                } else if p.kw_at(0, "STORED") {
                    file_format = Some(p.file_format()?);
                } else if p.eat_kw("LOCATION") {
                    location = Some(p.string_after()?);
                } else if p.eat_kw("COMMENT") {
                    comment = Some(p.string_after()?);
                } else if p.kw_at(0, "DEFAULT") && p.kw_at(1, "COLLATION") {
                    p.pos += 2;
                    collation = Some(p.table_name()?);
                } else if p.eat_kw("TBLPROPERTIES") {
                    tblproperties.extend(p.property_list(false)?);
                } else {
                    break;
                }
            }
            // `[AS] query`: a CTAS (or the parenthesized query above).
            let as_query = (if let Some(query) = paren_query {
                Some(query)
            } else if p.eat_kw("AS")
                || p.kw_at(0, "SELECT")
                || p.kw_at(0, "WITH")
                || p.kw_at(0, "VALUES")
                || p.kw_at(0, "TABLE")
                || p.kw_at(0, "FROM")
                || (p.sym_at(0, Sym::LParen) && p.query_follows(1))
            {
                Some(p.query()?)
            } else {
                None
            })
            .map(Box::new);
            let duplicate_clause = CREATE_TABLE_CLAUSES
                .iter()
                .zip(seen)
                .find(|(_, n)| *n > 1)
                .map(|(c, _)| *c);
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::CreateTable(Box::new(CreateTable {
                or_replace,
                temporary,
                external,
                if_not_exists,
                name,
                elements,
                using,
                options,
                partitioned_by,
                cluster_by,
                bucket,
                skew,
                row_format,
                file_format,
                location,
                comment,
                collation,
                tblproperties,
                as_query,
                duplicate_clause,
                span,
            })))
        }
    }

    /// A `CREATE TABLE` element: a table constraint, or a column
    /// definition. The constraint keywords are never column names, so a
    /// constraint-starting element is committed to.
    pub(super) fn table_element(&mut self) -> PR<TableElement> {
        if self.kw_at(0, "CONSTRAINT")
            || self.any_kw_at(0, &["PRIMARY", "FOREIGN", "CHECK", "UNIQUE"])
        {
            return self.table_constraint().map(TableElement::Constraint);
        }
        self.table_column()
            .map(|c| TableElement::Column(Box::new(c)))
    }

    /// `name type [NOT NULL] [DEFAULT ...] [GENERATED ...] [COMMENT '...']
    /// [constraints]`.
    pub(super) fn table_column(&mut self) -> PR<ColumnDef> {
        let start = self.start();
        let name = self.column_name()?;
        let data_type = self.data_type()?;
        let mut not_null = false;
        let mut default = None;
        let mut generated = None;
        let mut comment = None;
        let mut constraints = Vec::new();
        // Descriptors seen so far, and the first one written twice.
        let mut seen: Vec<&'static str> = Vec::new();
        let mut duplicate_descriptor = None;
        loop {
            let descriptor = if self.kw_at(0, "NOT") {
                Some("NOT NULL")
            } else if self.kw_at(0, "DEFAULT") {
                Some("DEFAULT")
            } else if self.kw_at(0, "GENERATED") {
                Some("GENERATED ALWAYS AS")
            } else if self.kw_at(0, "COMMENT") {
                Some("COMMENT")
            } else if self.kw_at(0, "CONSTRAINT")
                || self.any_kw_at(0, &["PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "REFERENCES"])
            {
                Some("CONSTRAINT")
            } else {
                None
            };
            if let Some(d) = descriptor {
                if seen.contains(&d) {
                    duplicate_descriptor = duplicate_descriptor.or(Some(d));
                } else {
                    seen.push(d);
                }
            }
            if self.eat_kw("NOT") {
                not_null = true;
                if !self.eat_kw("NULL") {
                    // ANTLR inserts the missing `NULL` when what follows
                    // fits after it.
                    if self.kind(0).is_none()
                        || self.sym_at(0, Sym::Comma)
                        || self.sym_at(0, Sym::RParen)
                        || self.kw_at(0, "COMMENT")
                    {
                        self.missing = Some((self.pos, "'NULL'"));
                    }
                    return self.fail();
                }
            } else if self.eat_kw("DEFAULT") {
                default = Some(self.expr()?);
            } else if self.kw_at(0, "GENERATED") {
                generated = Some(self.generated_column()?);
            } else if self.eat_kw("COMMENT") {
                comment = Some(match self.string_lit() {
                    Some(comment) => comment,
                    None => return self.fail(),
                });
            } else if self.kw_at(0, "CONSTRAINT")
                || self.any_kw_at(0, &["PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "REFERENCES"])
            {
                constraints.push(self.column_constraint()?);
            } else {
                break;
            }
        }
        Ok(ColumnDef {
            name,
            data_type,
            not_null,
            default,
            generated,
            comment,
            constraints,
            duplicate_descriptor,
            span: self.span_from(start),
        })
    }

    /// `GENERATED ALWAYS AS (expr)` or `GENERATED (ALWAYS | BY DEFAULT)
    /// AS IDENTITY [(START WITH n INCREMENT BY m)]`.
    pub(super) fn generated_column(&mut self) -> PR<GeneratedColumn> {
        let start = self.start();
        self.expect_kw("GENERATED")?;
        let kind = if self.eat_kw("ALWAYS") {
            self.expect_kw("AS")?;
            if self.kw_at(0, "IDENTITY") {
                let ident = self.start();
                self.bump();
                GeneratedKind::Identity(self.identity_options(IdentityHow::Always, ident)?)
            } else {
                self.expect_sym(Sym::LParen)?;
                let expr = self.expr()?;
                self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
                GeneratedKind::Expr(expr)
            }
        } else if self.kw_at(0, "BY") {
            self.pos += 1;
            self.expect_kw("DEFAULT")?;
            self.expect_kw("AS")?;
            let ident = self.start();
            self.expect_kw("IDENTITY")?;
            GeneratedKind::Identity(self.identity_options(IdentityHow::ByDefault, ident)?)
        } else {
            return self.fail();
        };
        Ok(GeneratedColumn {
            kind,
            span: self.span_from(start),
        })
    }

    /// `[(START WITH n INCREMENT BY m)]` after `IDENTITY` at `ident`.
    pub(super) fn identity_options(
        &mut self,
        how: IdentityHow,
        ident: usize,
    ) -> PR<IdentityColumn> {
        let (mut start, mut step) = (None, None);
        let mut duplicate_option = None;
        let mut count = 0;
        if self.eat_sym(Sym::LParen) {
            loop {
                if self.kw_at(0, "START") && self.kw_at(1, "WITH") {
                    self.pos += 2;
                    if start.is_some() {
                        duplicate_option = duplicate_option.or(Some("START"));
                    }
                    start = Some(self.identity_count()?);
                    count += 1;
                } else if self.kw_at(0, "INCREMENT") && self.kw_at(1, "BY") {
                    self.pos += 2;
                    if step.is_some() {
                        duplicate_option = duplicate_option.or(Some("STEP"));
                    }
                    step = Some(self.identity_count()?);
                    count += 1;
                } else {
                    break;
                }
            }
            self.expect_closing(Sym::RParen, Close::List(count))?;
        }
        Ok(IdentityColumn {
            how,
            start,
            step,
            duplicate_option,
            span: self.span_from(ident),
        })
    }

    /// A `START WITH` / `INCREMENT BY` count: an integer, optionally
    /// negative.
    pub(super) fn identity_count(&mut self) -> PR<i64> {
        let negative = self.sym_at(0, Sym::Minus);
        let at = usize::from(negative);
        if self.kind(at) != Some(TokenKind::Number(NumKind::Int)) {
            return self.fail();
        }
        let text = if negative {
            format!("-{}", self.text(at))
        } else {
            self.text(at).to_owned()
        };
        match text.parse() {
            Ok(n) => {
                self.pos += at + 1;
                Ok(n)
            }
            Err(_) => self.fail(),
        }
    }

    /// `[CONSTRAINT name] (PRIMARY KEY ... | FOREIGN KEY ... | UNIQUE ...
    /// | CHECK ...) [characteristics]`.
    pub(super) fn table_constraint(&mut self) -> PR<TableConstraint> {
        let start = self.start();
        let name = if self.eat_kw("CONSTRAINT") {
            Some(self.name_part(true)?)
        } else {
            None
        };
        let kind = if self.kw_at(0, "PRIMARY") && self.kw_at(1, "KEY") {
            self.pos += 2;
            TableConstraintKind::PrimaryKey(self.paren_list(|p| p.name_part(true))?)
        } else if self.kw_at(0, "FOREIGN") && self.kw_at(1, "KEY") {
            let fk = self.start();
            self.pos += 2;
            let columns = self.paren_list(|p| p.name_part(true))?;
            let (table, ref_columns) = self.foreign_key_ref()?;
            TableConstraintKind::ForeignKey(ForeignKeyConstraint {
                columns,
                table,
                ref_columns,
                span: self.span_from(fk),
            })
        } else if self.eat_kw("UNIQUE") {
            TableConstraintKind::Unique(self.paren_list(|p| p.name_part(true))?)
        } else if self.eat_kw("CHECK") {
            self.expect_sym(Sym::LParen)?;
            let expr = self.expr()?;
            self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
            TableConstraintKind::Check(expr)
        } else {
            return self.fail();
        };
        let characteristics = self.constraint_characteristics()?;
        Ok(TableConstraint {
            name,
            kind,
            characteristics,
            span: self.span_from(start),
        })
    }

    /// `[CONSTRAINT name] (PRIMARY KEY | UNIQUE | REFERENCES ... | CHECK
    /// ...) [characteristics]` on a column.
    pub(super) fn column_constraint(&mut self) -> PR<ColumnConstraint> {
        let start = self.start();
        let name = if self.eat_kw("CONSTRAINT") {
            Some(self.name_part(true)?)
        } else {
            None
        };
        let kind = if self.kw_at(0, "PRIMARY") && self.kw_at(1, "KEY") {
            self.pos += 2;
            ColumnConstraintKind::PrimaryKey
        } else if self.eat_kw("UNIQUE") {
            ColumnConstraintKind::Unique
        } else if self.eat_kw("REFERENCES") {
            let start = self.start();
            let table = self.table_name()?;
            let columns = if self.sym_at(0, Sym::LParen) {
                self.paren_list(|p| p.name_part(true))?
            } else {
                Vec::new()
            };
            ColumnConstraintKind::References(ForeignKeyRef {
                table,
                columns,
                span: self.span_from(start),
            })
        } else if self.eat_kw("CHECK") {
            self.expect_sym(Sym::LParen)?;
            let expr = self.expr()?;
            self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
            ColumnConstraintKind::Check(expr)
        } else {
            return self.fail();
        };
        let characteristics = self.constraint_characteristics()?;
        Ok(ColumnConstraint {
            name,
            kind,
            characteristics,
            span: self.span_from(start),
        })
    }

    /// `REFERENCES table [(columns)]` after `FOREIGN KEY (columns)`.
    pub(super) fn foreign_key_ref(&mut self) -> PR<(ObjectName, Vec<Ident>)> {
        self.expect_kw("REFERENCES")?;
        let table = self.table_name()?;
        let columns = if self.sym_at(0, Sym::LParen) {
            self.paren_list(|p| p.name_part(true))?
        } else {
            Vec::new()
        };
        Ok((table, columns))
    }

    /// `ENFORCED`, `NOT ENFORCED`, `RELY` and `NORELY`, in any order.
    pub(super) fn constraint_characteristics(&mut self) -> PR<Vec<ConstraintCharacteristic>> {
        let mut out = Vec::new();
        loop {
            if self.eat_kw("ENFORCED") {
                out.push(ConstraintCharacteristic::Enforced(true));
            } else if self.kw_at(0, "NOT") && self.kw_at(1, "ENFORCED") {
                self.pos += 2;
                out.push(ConstraintCharacteristic::Enforced(false));
            } else if self.eat_kw("RELY") {
                out.push(ConstraintCharacteristic::Rely(true));
            } else if self.kw_at(0, "NORELY") {
                self.bump();
                out.push(ConstraintCharacteristic::Rely(false));
            } else {
                break;
            }
        }
        Ok(out)
    }

    /// `(key [=] value, ...)` for `OPTIONS` (full expressions) and
    /// `TBLPROPERTIES` / `DBPROPERTIES` (literals only): `expressions`
    /// selects which.
    pub(super) fn property_list(&mut self, expressions: bool) -> PR<Vec<TableOption>> {
        self.paren_list(|p| {
            let start = p.start();
            let key = p.option_key()?;
            let has_eq = p.eat_sym(Sym::Eq);
            let value = if has_eq || !(p.sym_at(0, Sym::Comma) || p.sym_at(0, Sym::RParen)) {
                if expressions {
                    Some(p.expr()?)
                } else {
                    Some(p.property_value()?)
                }
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

    /// A `TBLPROPERTIES` / `DBPROPERTIES` value: a number, string or
    /// boolean literal.
    pub(super) fn property_value(&mut self) -> PR<Expr> {
        let start = self.start();
        let kind = match self.kind(0) {
            Some(TokenKind::Number(kind @ (NumKind::Int | NumKind::Decimal))) => {
                let text = self.text(0).to_owned();
                self.bump();
                Literal::Number { text, kind }
            }
            Some(TokenKind::String) => Literal::String(self.string_lit().unwrap_or_default()),
            Some(TokenKind::DoubleQuoted) if !self.opts.double_quoted_identifiers => {
                Literal::String(self.string_lit().unwrap_or_default())
            }
            Some(TokenKind::Word) if self.kw_at(0, "TRUE") || self.kw_at(0, "FALSE") => {
                let value = self.kw_at(0, "TRUE");
                self.bump();
                Literal::Boolean(value)
            }
            _ => return self.fail(),
        };
        Ok(Expr {
            kind: ExprKind::Literal(kind),
            span: self.span_from(start),
        })
    }

    /// One `PARTITIONED BY` element: `name [(args)]`, or `name type [NOT
    /// NULL] [COMMENT '...']` for a single plain name. A
    /// `PARTITIONED BY` column takes no `COLLATE`: one reads as an
    /// identity transform instead, which fails at the `COLLATE`.
    /// (Applied and dotted dynamic names are not supported.)
    pub(super) fn partition_field(&mut self) -> PR<PartitionField> {
        let start = self.start();
        let name = if self.kw_at(0, "IDENTIFIER") && self.sym_at(1, Sym::LParen) {
            DdlName::Dynamic(self.identifier_name()?)
        } else {
            let first = self.name_part(true)?;
            if self.sym_at(0, Sym::LParen) {
                return self.partition_apply_transform(start, first);
            }
            let mut parts = vec![first];
            while self.eat_sym(Sym::Dot) {
                parts.push(self.name_part(true)?);
            }
            let span = self.span_from(start);
            DdlName::Parts(ObjectName { parts, span })
        };
        // An identity transform, or a column when a single name takes a
        // type.
        if let DdlName::Parts(name) = &name
            && name.parts.len() == 1
            && matches!(
                self.kind(0),
                Some(TokenKind::Word | TokenKind::QuotedIdent | TokenKind::DoubleQuoted)
                if self.kind(0) != Some(TokenKind::DoubleQuoted)
                    || self.opts.double_quoted_identifiers
            )
        {
            let save = self.pos;
            let (far, extra, missing, matched) = (self.far, self.extra, self.missing, self.matched);
            match self.data_type() {
                Ok(data_type) => {
                    if data_type.collation.is_none() {
                        let mut not_null = false;
                        if self.eat_kw("NOT") {
                            not_null = true;
                            if !self.eat_kw("NULL") {
                                if self.kind(0).is_none()
                                    || self.sym_at(0, Sym::Comma)
                                    || self.sym_at(0, Sym::RParen)
                                    || self.kw_at(0, "COMMENT")
                                {
                                    self.missing = Some((self.pos, "'NULL'"));
                                }
                                return self.fail();
                            }
                        }
                        let comment = if self.eat_kw("COMMENT") {
                            match self.string_lit() {
                                Some(comment) => Some(comment),
                                None => return self.fail(),
                            }
                        } else {
                            None
                        };
                        let span = self.span_from(start);
                        return Ok(PartitionField {
                            kind: PartitionFieldKind::Column(PartitionColumn {
                                name: ColumnName::Ident(name.parts[0].clone()),
                                data_type,
                                not_null,
                                comment,
                                span,
                            }),
                            span,
                        });
                    }
                    self.pos = save;
                    (self.far, self.extra, self.missing, self.matched) =
                        (far, extra, missing, matched);
                }
                Err(Fail::Syntax) => {
                    self.pos = save;
                    (self.far, self.extra, self.missing, self.matched) =
                        (far, extra, missing, matched);
                }
                Err(fatal) => return Err(fatal),
            }
        }
        // An identity transform.
        let span = self.span_from(start);
        Ok(PartitionField {
            kind: PartitionFieldKind::Transform(PartitionTransform {
                name,
                args: Vec::new(),
                span,
            }),
            span,
        })
    }

    /// `name(args)` in `PARTITIONED BY`.
    pub(super) fn partition_apply_transform(
        &mut self,
        start: usize,
        first: Ident,
    ) -> PR<PartitionField> {
        let name_span = first.span;
        self.expect_sym(Sym::LParen)?;
        let args = self.comma_sep(Self::primary_expr)?;
        self.expect_closing(Sym::RParen, Close::List(args.len()))?;
        let span = self.span_from(start);
        Ok(PartitionField {
            kind: PartitionFieldKind::Transform(PartitionTransform {
                name: DdlName::Parts(ObjectName {
                    parts: vec![first],
                    span: name_span,
                }),
                args,
                span,
            }),
            span,
        })
    }

    /// `CLUSTERED BY (...) [SORTED BY (...)] INTO n BUCKETS`.
    pub(super) fn bucket_spec(&mut self) -> PR<BucketSpec> {
        let start = self.start();
        self.expect_kw("CLUSTERED")?;
        self.expect_kw("BY")?;
        let clustered_by = self.paren_list(|p| p.name_part(true))?;
        let mut sorted_by = Vec::new();
        if self.kw_at(0, "SORTED") && self.kw_at(1, "BY") {
            self.pos += 2;
            sorted_by = self.paren_list(Self::ordered_ident)?;
        }
        self.expect_kw("INTO")?;
        let buckets = self.bucket_count()?;
        self.expect_kw("BUCKETS")?;
        Ok(BucketSpec {
            clustered_by,
            sorted_by,
            buckets,
            span: self.span_from(start),
        })
    }

    /// `name [ASC | DESC]`.
    pub(super) fn ordered_ident(&mut self) -> PR<OrderedIdent> {
        let start = self.start();
        let name = self.name_part(true)?;
        let asc = if self.eat_kw("ASC") {
            Some(true)
        } else if self.eat_kw("DESC") {
            Some(false)
        } else {
            None
        };
        Ok(OrderedIdent {
            name,
            asc,
            span: self.span_from(start),
        })
    }

    /// `INTO n`: a plain integer token.
    pub(super) fn bucket_count(&mut self) -> PR<u64> {
        let text = match self.kind(0) {
            Some(TokenKind::Number(NumKind::Int)) => self.text(0).to_owned(),
            _ => return self.fail(),
        };
        match text.parse() {
            Ok(n) => {
                self.bump();
                Ok(n)
            }
            Err(_) => self.fail(),
        }
    }

    /// `SKEWED BY (...) ON (...) [STORED AS DIRECTORIES]`.
    pub(super) fn skew_spec(&mut self) -> PR<SkewSpec> {
        let start = self.start();
        self.expect_kw("SKEWED")?;
        self.expect_kw("BY")?;
        let skewed_by = self.paren_list(|p| p.name_part(true))?;
        self.expect_kw("ON")?;
        // `(...)` holds one value group, `((...), ...)` several.
        let values = if self.sym_at(0, Sym::LParen) && self.sym_at(1, Sym::LParen) {
            self.paren_list(|p| p.paren_list(Self::primary_expr))?
        } else {
            vec![self.paren_list(Self::primary_expr)?]
        };
        let mut stored_as_directories = false;
        if self.eat_kw("STORED") {
            self.expect_kw("AS")?;
            self.expect_kw("DIRECTORIES")?;
            stored_as_directories = true;
        }
        Ok(SkewSpec {
            skewed_by,
            values,
            stored_as_directories,
            span: self.span_from(start),
        })
    }

    /// `ROW FORMAT SERDE ...` or `ROW FORMAT DELIMITED ...`.
    pub(super) fn row_format(&mut self) -> PR<RowFormat> {
        let start = self.start();
        self.expect_kw("ROW")?;
        self.expect_kw("FORMAT")?;
        let kind = if self.eat_kw("SERDE") {
            let serde = self.start();
            let class = self.string_after()?;
            let properties = if self.kw_at(0, "WITH") && self.kw_at(1, "SERDEPROPERTIES") {
                self.pos += 2;
                self.property_list(false)?
            } else {
                Vec::new()
            };
            RowFormatKind::Serde(RowSerde {
                class,
                properties,
                span: self.span_from(serde),
            })
        } else if self.kw_at(0, "DELIMITED") {
            let delimited_start = self.start();
            self.bump();
            // `ROW FORMAT DELIMITED` on its own: every clause is empty.
            let mut delimited = RowDelimited {
                fields_terminated_by: None,
                escaped_by: None,
                collection_items_terminated_by: None,
                map_keys_terminated_by: None,
                lines_terminated_by: None,
                null_defined_as: None,
                span: Span::new(delimited_start, delimited_start),
            };
            if self.kw_at(0, "FIELDS") && self.kw_at(1, "TERMINATED") && self.kw_at(2, "BY") {
                self.pos += 3;
                delimited.fields_terminated_by = Some(self.string_after()?);
                if self.kw_at(0, "ESCAPED") && self.kw_at(1, "BY") {
                    self.pos += 2;
                    delimited.escaped_by = Some(self.string_after()?);
                }
            }
            if self.kw_at(0, "COLLECTION")
                && self.kw_at(1, "ITEMS")
                && self.kw_at(2, "TERMINATED")
                && self.kw_at(3, "BY")
            {
                self.pos += 4;
                delimited.collection_items_terminated_by = Some(self.string_after()?);
            }
            if self.kw_at(0, "MAP")
                && self.kw_at(1, "KEYS")
                && self.kw_at(2, "TERMINATED")
                && self.kw_at(3, "BY")
            {
                self.pos += 4;
                delimited.map_keys_terminated_by = Some(self.string_after()?);
            }
            if self.kw_at(0, "LINES") && self.kw_at(1, "TERMINATED") && self.kw_at(2, "BY") {
                self.pos += 3;
                delimited.lines_terminated_by = Some(self.string_after()?);
            }
            if self.kw_at(0, "NULL") && self.kw_at(1, "DEFINED") && self.kw_at(2, "AS") {
                self.pos += 3;
                delimited.null_defined_as = Some(self.string_after()?);
            }
            delimited.span = self.span_from(delimited_start);
            RowFormatKind::Delimited(delimited)
        } else {
            return self.fail();
        };
        Ok(RowFormat {
            kind,
            span: self.span_from(start),
        })
    }

    /// `STORED AS name` or `STORED BY 'handler'`.
    pub(super) fn file_format(&mut self) -> PR<FileFormat> {
        let start = self.start();
        self.expect_kw("STORED")?;
        let kind = if self.eat_kw("AS") {
            FileFormatKind::StoredAs(self.table_name()?)
        } else if self.eat_kw("BY") {
            FileFormatKind::StoredBy(self.string_after()?)
        } else {
            return self.fail();
        };
        Ok(FileFormat {
            kind,
            span: self.span_from(start),
        })
    }

    /// `CREATE DATABASE|SCHEMA|NAMESPACE [IF NOT EXISTS] name
    /// [COMMENT ...] [LOCATION ...] [DEFAULT COLLATION ...]
    /// [WITH DBPROPERTIES|PROPERTIES (...)]`, after the `CREATE` at
    /// `start`.
    pub(super) fn create_namespace_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            let kind = p.namespace_kind()?;
            let if_not_exists = if p.kw_at(0, "IF") && p.kw_at(1, "NOT") && p.kw_at(2, "EXISTS") {
                p.pos += 3;
                true
            } else {
                false
            };
            let name = p.ddl_name()?;
            let mut comment = None;
            let mut location = None;
            let mut collation = None;
            let mut properties = Vec::new();
            // Occurrences per clause, in Spark's duplicate-check order.
            let mut seen = [0_usize; CREATE_NAMESPACE_CLAUSES.len()];
            loop {
                let clause = if p.eat_kw("COMMENT") {
                    comment = Some(p.string_after()?);
                    0
                } else if p.kw_at(0, "DEFAULT") && p.kw_at(1, "COLLATION") {
                    p.pos += 2;
                    collation = Some(p.table_name()?);
                    1
                } else if p.eat_kw("LOCATION") {
                    location = Some(p.string_after()?);
                    2
                } else if p.eat_kw("WITH") {
                    let clause = if p.eat_kw("PROPERTIES") {
                        3
                    } else if p.eat_kw("DBPROPERTIES") {
                        4
                    } else {
                        return p.fail();
                    };
                    properties.extend(p.property_list(false)?);
                    clause
                } else {
                    break;
                };
                seen[clause] += 1;
            }
            let duplicate_clause = CREATE_NAMESPACE_CLAUSES
                .iter()
                .zip(seen)
                .find(|(_, n)| *n > 1)
                .map(|(c, _)| *c);
            let properties_and_dbproperties = seen[3] > 0 && seen[4] > 0;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::CreateNamespace(CreateNamespace {
                kind,
                if_not_exists,
                name,
                comment,
                location,
                collation,
                properties,
                duplicate_clause,
                properties_and_dbproperties,
                span,
            }))
        })
    }

    /// `DROP TABLE [IF EXISTS] name [PURGE]`, after the `DROP` at `start`.
    pub(super) fn drop_table_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            p.expect_kw("TABLE")?;
            let if_exists = if p.kw_at(0, "IF") && p.kw_at(1, "EXISTS") {
                p.pos += 2;
                true
            } else {
                false
            };
            let name = p.ddl_name()?;
            let purge = p.eat_kw("PURGE");
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::DropTable(DropTable {
                if_exists,
                name,
                purge,
                span,
            }))
        })
    }

    /// `DROP DATABASE|SCHEMA|NAMESPACE [IF EXISTS] name
    /// [RESTRICT|CASCADE]`, after the `DROP` at `start`.
    pub(super) fn drop_namespace_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            let kind = p.namespace_kind()?;
            let if_exists = if p.kw_at(0, "IF") && p.kw_at(1, "EXISTS") {
                p.pos += 2;
                true
            } else {
                false
            };
            let name = p.ddl_name()?;
            let behavior = if p.eat_kw("RESTRICT") {
                Some(DropBehavior::Restrict)
            } else if p.eat_kw("CASCADE") {
                Some(DropBehavior::Cascade)
            } else {
                None
            };
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::DropNamespace(DropNamespace {
                kind,
                if_exists,
                name,
                behavior,
                span,
            }))
        })
    }

    /// `USE name` or `USE DATABASE|SCHEMA|NAMESPACE name`.
    pub(super) fn use_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("USE")?;
            let kind = if p.any_kw_at(0, &["DATABASE", "SCHEMA", "NAMESPACE"]) {
                Some(p.namespace_kind()?)
            } else {
                None
            };
            let name = p.ddl_name()?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Use(Use { kind, name, span }))
        })
    }
}
