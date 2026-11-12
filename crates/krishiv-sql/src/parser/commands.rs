// SPDX-License-Identifier: Apache-2.0
//! Parsing the command statements: SET, RESET, DECLARE, EXECUTE IMMEDIATE and the statement entry.

use super::*;

impl Parser<'_> {
    pub(super) fn statement(&mut self) -> PR<Query> {
        let q = self.query()?;
        self.command_end()?;
        Ok(q)
    }

    /// Trailing `;`s and the end of the input after a complete statement.
    /// Anything else is a syntax error at the token parsing stopped at;
    /// one extraneous token before the end (or any after `;`, read by
    /// ANTLR's `SEMICOLON*` loop) is reported as "extra input", as in
    /// Spark.
    pub(super) fn command_end(&mut self) -> PR<()> {
        // Inside a script a statement ends at its `;`, which the script reads.
        if self.in_script > 0 {
            return if self.sym_at(0, Sym::Semi) {
                Ok(())
            } else {
                self.fail()
            };
        }
        let mut semi = false;
        while self.eat_sym(Sym::Semi) {
            semi = true;
        }
        if self.pos < self.toks.len() {
            if semi || self.trailing_is_single_token() {
                self.extra = Some(self.pos);
            }
            return self.fail();
        }
        Ok(())
    }

    /// `(EXPLAIN [mode])* query ;* EOF`. The `EXPLAIN` prefixes are read
    /// in a loop and the tree is built from the inside out, so a long
    /// chain neither recurses nor nests deeper than [`MAX_DEPTH`].
    pub(super) fn explainable(&mut self) -> PR<Statement> {
        let mut explains = Vec::new();
        while self.kw_at(0, "EXPLAIN") {
            let start = self.start();
            self.bump();
            if explains.len() == MAX_DEPTH {
                return Self::fatal(error::too_complex(statement_span(self.src, &self.toks)));
            }
            let word = if self.kind(0) == Some(TokenKind::Word) {
                self.text(0).to_ascii_uppercase()
            } else {
                String::new()
            };
            let mode = match word.as_str() {
                "EXTENDED" => Some(ExplainMode::Extended),
                "FORMATTED" => Some(ExplainMode::Formatted),
                "CODEGEN" => Some(ExplainMode::Codegen),
                "COST" => Some(ExplainMode::Cost),
                // Spark parses `LOGICAL`, then rejects it after parsing.
                "LOGICAL" => Some(ExplainMode::Simple),
                // Spark reads `EXPLAIN ANALYZE` as the start of `ANALYZE
                // TABLE` (unsupported here: its syntax error is at the next
                // token). Before anything else it is Krishiv's
                // `EXPLAIN ANALYZE <query>`, a syntax error in Spark.
                "ANALYZE" if self.kw_at(1, "TABLE") || self.kw_at(1, "TABLES") => {
                    self.bump();
                    return self.fail();
                }
                "ANALYZE" => Some(ExplainMode::Analyze),
                // A syntax error in Spark; Krishiv's `EXPLAIN VERBOSE`.
                "VERBOSE" => Some(ExplainMode::Verbose),
                _ => None,
            };
            let logical = word == "LOGICAL";
            if mode.is_some() {
                self.bump();
            }
            explains.push((start, mode.unwrap_or(ExplainMode::Simple), logical));
        }
        // `SET`, `RESET`, `CREATE`, `DROP` and `USE` start no query:
        // without an `EXPLAIN` the statement is that command, and under one
        // it is the explained statement. Anything else is a query, which
        // fails under `EXPLAIN` as in Spark when it is a command (a command
        // explains; it never reads as a query).
        let statement = if self.kw_at(0, "SET") {
            // `SET VAR | VARIABLE ...` assigns session variables only when
            // a variable form follows (a target list or a name); anything
            // else is the configuration command (a backquoted `` `VAR` ``
            // is a key, not the keyword, as `kw_at` only matches words),
            // so bare keys like `var.foo`, `VAR` or `VAR = 1` still read
            // as configuration.
            if (self.kw_at(1, "VAR") || self.kw_at(1, "VARIABLE"))
                && (matches!(self.kind(2), Some(TokenKind::Word | TokenKind::QuotedIdent))
                    || self.sym_at(2, Sym::LParen))
            {
                self.set_variable_command(true)?
            } else {
                self.set_command()?
            }
        } else if explains.is_empty() && self.kw_at(0, "BEGIN") && self.in_script == 0 {
            self.script_command()?
        } else if self.kw_at(0, "EXECUTE") && self.kw_at(1, "IMMEDIATE") {
            self.execute_immediate_command()?
        } else if self.kw_at(0, "DECLARE") {
            self.declare_command()?
        } else if self.kw_at(0, "RESET") {
            self.reset_command()?
        } else if self.kw_at(0, "CREATE") {
            self.create_command()?
        } else if self.kw_at(0, "REPLACE") && self.kw_at(1, "TABLE") {
            // `REPLACE TABLE` is `CREATE OR REPLACE TABLE`.
            let start = self.start();
            self.bump();
            self.create_table_command(start, true, false, false)?
        } else if self.kw_at(0, "DROP") {
            self.drop_command()?
        } else if self.kw_at(0, "USE") {
            self.use_command()?
        } else if self.any_kw_at(
            0,
            &[
                "ALTER", "SHOW", "DESC", "DESCRIBE", "INSERT", "DELETE", "UPDATE", "MERGE",
                "TRUNCATE", "CACHE", "UNCACHE", "CLEAR", "REFRESH", "ANALYZE", "MSCK", "REPAIR",
                "ADD", "LIST", "COMMENT",
            ],
        ) {
            self.command_statement()?
        } else if self.kw_at(0, "WITH")
            && let Some(insert) = self.with_insert_command()?
        {
            insert
        } else if self.kw_at(0, "FROM") {
            // A multi-table insert, or a `FROM ... SELECT` query. The
            // query runs first; only its syntax failure reads as an
            // insert (a fatal error is the query's own). When both fail,
            // the farther failure is reported.
            let save = self.checkpoint();
            match self.statement() {
                Ok(query) => {
                    self.commit(save);
                    Statement::Query(Box::new(query))
                }
                Err(Fail::Fatal(e)) => return Err(Fail::Fatal(e)),
                Err(Fail::Syntax) => {
                    let (far, extra, missing, matched) =
                        (self.far, self.extra, self.missing, self.matched);
                    self.restore(save);
                    match self.multi_insert_command() {
                        Ok(statement) => statement,
                        Err(Fail::Fatal(e)) => return Err(Fail::Fatal(e)),
                        Err(Fail::Syntax) if self.far >= far => return self.fail(),
                        Err(Fail::Syntax) => {
                            self.far = far;
                            self.extra = extra;
                            self.missing = missing;
                            self.matched = matched;
                            return self.fail();
                        }
                    }
                }
            }
        } else {
            Statement::Query(Box::new(self.statement()?))
        };
        let end = statement.span().end;
        // The outermost `EXPLAIN LOGICAL` is reported first, before any
        // error Spark raises while building the explained query.
        if let Some(&(start, ..)) = explains.iter().find(|e| e.2) {
            self.deferred = Some(error::invalid_statement(
                "EXPLAIN LOGICAL",
                Span::new(start, end),
            ));
        }
        let mut statement = statement;
        for (start, mode, _) in explains.into_iter().rev() {
            statement = Statement::Explain(Explain {
                mode,
                statement: Box::new(statement),
                span: Span::new(start, end),
            });
        }
        Ok(statement)
    }

    /// `WITH ctes INSERT INTO|OVERWRITE ... query`: the common table expressions scope over the
    /// rows written, so they become the `WITH` of the insert's query. `None` (with nothing
    /// consumed) when no `INSERT` follows the `WITH` clause, so the statement is a query.
    pub(super) fn with_insert_command(&mut self) -> PR<Option<Statement>> {
        let save = self.checkpoint();
        let start = self.start();
        let with = match self.with_clause() {
            Ok(with) if self.kw_at(0, "INSERT") || self.kw_at(0, "FROM") => with,
            Ok(_) | Err(Fail::Syntax) => {
                self.restore(save);
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        // `WITH ctes FROM source INSERT ...`: a multi-table insert whose clauses all see the
        // common table expressions; `WITH ctes FROM source SELECT ...` stays a query (a syntax
        // failure here gives the statement back to the query parser).
        let mut insert = if self.kw_at(0, "FROM") {
            let from = self.checkpoint();
            match self.multi_insert_command() {
                Ok(Statement::Insert(insert)) => {
                    self.commit(from);
                    self.commit(save);
                    insert
                }
                Ok(_) | Err(Fail::Syntax) => {
                    self.restore(from);
                    self.restore(save);
                    return Ok(None);
                }
                Err(e) => return Err(e),
            }
        } else {
            self.commit(save);
            let Statement::Insert(insert) = self.insert_command()? else {
                return self.fail();
            };
            insert
        };
        for clause in &mut insert.clauses {
            // Spark refuses a `WITH` in both places.
            if clause.query.with.is_some() {
                return self.fail();
            }
            clause.query.with = Some(with.clone());
        }
        insert.span = self.span_from(start);
        Ok(Some(Statement::Insert(insert)))
    }

    /// `SET`, `SET -v`, `SET key`, `SET key = value` or `SET TIME ZONE
    /// ...`. A bare key holds no whitespace and matches Spark's
    /// `[a-zA-Z_\d\.:]+`; anything else is `INVALID_SET_SYNTAX`. Only
    /// backquotes quote a key or value: a double-quoted key is a syntax
    /// error, a single-quoted one fails the key check. The value is the
    /// raw rest of the statement, trimmed (decoded only when it is one
    /// backquoted token); anything after its `;` is extra input, as for
    /// queries.
    pub(super) fn set_command(&mut self) -> PR<Statement> {
        let start = self.start();
        self.expect_kw("SET")?;
        // `SET` with nothing after it lists every explicitly set key.
        if self.pos >= self.toks.len() || self.sym_at(0, Sym::Semi) {
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Set(Set {
                kind: SetKind::All,
                span,
            }));
        }
        // `SET -v`: only `-v` with nothing after it.
        if self.sym_at(0, Sym::Minus) {
            return self.set_verbose_command(start);
        }
        // `SET TIME ZONE ...`.
        if self.kw_at(0, "TIME") && self.kw_at(1, "ZONE") {
            return self.set_timezone_command(start);
        }
        // `SET PATH = ...`.
        if self.kw_at(0, "PATH") && self.sym_at(1, Sym::Eq) {
            self.pos += 2;
            return self.set_path_command(start);
        }
        // A backquoted key.
        if let Some((key, key_span)) = self.quoted_key()? {
            if self.eat_sym(Sym::Eq) {
                let (value, value_span) = self.set_value()?;
                let span = self.span_from(start);
                self.command_end()?;
                return Ok(Statement::Set(Set {
                    kind: SetKind::Assign {
                        key,
                        key_span,
                        value,
                        value_span,
                    },
                    span,
                }));
            }
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Set(Set {
                kind: SetKind::Show { key, key_span },
                span,
            }));
        }
        // A bare key runs to the first `=` (an assignment) or to the `;`
        // or end of input (a lookup). A double-quoted token there is a
        // syntax error, as in Spark; the text is the key as written, which
        // must match `[a-zA-Z_\d\.:]+`.
        let mut eq = None;
        let mut semi = self.toks.len();
        for (i, t) in self.toks.iter().enumerate().skip(self.pos) {
            match t.kind {
                TokenKind::Sym(Sym::Eq) => {
                    eq = Some(i);
                    break;
                }
                TokenKind::Sym(Sym::Semi) => {
                    semi = i;
                    break;
                }
                TokenKind::Unterminated | TokenKind::DoubleQuoted => {
                    self.pos = i;
                    return self.fail();
                }
                _ => {}
            }
        }
        if let Some(e) = eq {
            let from = self.start();
            let (key, key_span) = self.trimmed_span(from, self.toks[e].span.start);
            if !valid_bare_key(&key) {
                return Self::fatal(error::invalid_set_syntax(self.span_from(start)));
            }
            self.pos = e + 1;
            let (value, value_span) = self.set_value()?;
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Set(Set {
                kind: SetKind::Assign {
                    key,
                    key_span,
                    value,
                    value_span,
                },
                span,
            }));
        }
        let from = self.start();
        let to = (self.pos..semi)
            .filter_map(|i| self.toks.get(i))
            .map(|t| t.span.end)
            .max()
            .unwrap_or(from);
        let (key, key_span) = self.trimmed_span(from, to);
        if !valid_bare_key(&key) {
            return Self::fatal(error::invalid_set_syntax(self.span_from(start)));
        }
        self.pos = semi;
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::Set(Set {
            kind: SetKind::Show { key, key_span },
            span,
        }))
    }

    /// `SET -...` after the `SET` at `start`: only `-v` with nothing after
    /// it; anything else is `INVALID_SET_SYNTAX`.
    pub(super) fn set_verbose_command(&mut self, start: usize) -> PR<Statement> {
        self.bump();
        if self.kind(0) == Some(TokenKind::Word) && self.text(0).eq_ignore_ascii_case("v") {
            self.bump();
            if self.pos >= self.toks.len() || self.sym_at(0, Sym::Semi) {
                let span = self.span_from(start);
                self.command_end()?;
                return Ok(Statement::Set(Set {
                    kind: SetKind::Verbose,
                    span,
                }));
            }
        }
        Self::fatal(error::invalid_set_syntax(self.span_from(start)))
    }

    /// `SET TIME ZONE LOCAL | '<zone>'` after the `SET` at `start`. An
    /// `INTERVAL` zone is a supported grammar form the engine does not
    /// implement, so it fails with `KRISHIV_UNSUPPORTED_FEATURE` naming
    /// the form.
    pub(super) fn set_timezone_command(&mut self, start: usize) -> PR<Statement> {
        self.pos += 2;
        let (zone, zone_span) = match self.kind(0) {
            Some(TokenKind::Word) if self.text(0).eq_ignore_ascii_case("LOCAL") => {
                let span = self.tok(0).map_or(self.span_from(start), |t| t.span);
                self.bump();
                (TimeZone::Local, span)
            }
            Some(TokenKind::String) => {
                let Some(t) = self.tok(0) else {
                    return self.fail();
                };
                let zone = unescape_string(
                    t.span.text(self.src),
                    !self.opts.escaped_string_literals,
                    false,
                );
                self.bump();
                (TimeZone::Named(zone), t.span)
            }
            Some(TokenKind::DoubleQuoted) if !self.opts.double_quoted_identifiers => {
                // A string when `"` does not quote identifiers.
                let Some(t) = self.tok(0) else {
                    return self.fail();
                };
                let zone = unescape_string(
                    t.span.text(self.src),
                    !self.opts.escaped_string_literals,
                    false,
                );
                self.bump();
                (TimeZone::Named(zone), t.span)
            }
            _ if self.kw_at(0, "INTERVAL") => {
                let from = self.pos;
                let mut end = self.src.len();
                for t in self.toks.iter().skip(self.pos) {
                    if t.kind == TokenKind::Sym(Sym::Semi) {
                        end = t.span.start;
                        break;
                    }
                }
                let whole = Span::new(start, end);
                // An interval that does not reach the end of the statement
                // is the grammar's catch-all alternative.
                let parsed = self
                    .interval()
                    .ok()
                    .filter(|_| self.pos >= self.toks.len() || self.sym_at(0, Sym::Semi));
                let offset = match parsed {
                    Some(Literal::Interval(iv)) => validate::interval::time_zone_offset(&iv, whole),
                    Some(Literal::Typed { value, .. }) => {
                        validate::interval::time_zone_offset_typed(&value, whole)
                    }
                    _ => {
                        self.pos = from;
                        return Self::fatal(error::invalid_time_zone_displacement(whole));
                    }
                };
                match offset {
                    Ok(zone) => {
                        // `from` is a token index; the span starts at the byte.
                        let begin = self.toks.get(from).map_or(start, |t| t.span.start);
                        (TimeZone::Named(zone), self.span_from(begin))
                    }
                    Err(e) => return Self::fatal(e),
                }
            }
            // The grammar's catch-all alternative (`SET TIME ZONE .*?`).
            _ => {
                let mut end = self.src.len();
                for t in self.toks.iter().skip(self.pos) {
                    if t.kind == TokenKind::Sym(Sym::Semi) {
                        end = t.span.start;
                        break;
                    }
                }
                let end = end.max(self.span_from(start).end);
                return Self::fatal(error::invalid_time_zone_displacement(Span::new(start, end)));
            }
        };
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::Set(Set {
            kind: SetKind::TimeZone { zone, zone_span },
            span,
        }))
    }

    /// `DECLARE [OR REPLACE] [VARIABLE | VAR] name [, ...] [type]
    /// [{DEFAULT | =} expr]`. Names are dotted (`v`, `session.v`,
    /// `system.session.v`); whether they are well-formed is execution's
    /// check, so `DECLARE builtin.var1 INT` parses and fails with
    /// `UNRESOLVED_VARIABLE`. `IF NOT EXISTS` is not supported: `IF NOT`
    /// is consumed so the error is at `EXISTS`, as Spark reports it.
    pub(super) fn declare_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("DECLARE")?;
            let or_replace = if p.kw_at(0, "OR") {
                p.pos += 1;
                p.expect_kw("REPLACE")?;
                true
            } else {
                false
            };
            // One keyword at most: `DECLARE VAR variable INT` declares
            // a variable named `variable`.
            // A `VAR` or `VARIABLE` followed by `=` or `,` is the variable's name
            // (`DECLARE var = 'x'`), as ANTLR's prediction reads it.
            let named_var = p.sym_at(1, Sym::Eq) || p.sym_at(1, Sym::Comma);
            if !named_var && !p.eat_kw("VARIABLE") {
                p.eat_kw("VAR");
            }
            if p.kw_at(0, "IF") && p.kw_at(1, "NOT") && p.kw_at(2, "EXISTS") {
                p.pos += 2;
                return p.fail();
            }
            let mut names = vec![p.variable_reference()?];
            while p.eat_sym(Sym::Comma) {
                names.push(p.variable_reference()?);
            }
            // A type follows unless the names end here: `DEFAULT`, `=`,
            // `;` or the end of input (anything else fails after the
            // statement parsed, or when the type is built).
            let data_type = match p.kind(0) {
                Some(TokenKind::Word) if !p.kw_at(0, "DEFAULT") => Some(p.data_type()?),
                Some(TokenKind::QuotedIdent) => Some(p.data_type()?),
                _ => None,
            };
            let default = if p.eat_kw("DEFAULT") || p.eat_sym(Sym::Eq) {
                Some(p.expr()?)
            } else {
                None
            };
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::DeclareVariable(DeclareVariable {
                or_replace,
                names,
                data_type,
                default,
                span,
            }))
        })
    }

    /// `EXECUTE IMMEDIATE query [INTO v, ...] [USING [(]arg [AS alias], ...[)]]`.
    /// The query is any expression; the parenthesized argument list is
    /// tried first and read as one parenthesized argument when it does not
    /// end the statement (`USING (SELECT 1) AS a`).
    #[inline(never)]
    pub(super) fn execute_immediate_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("EXECUTE")?;
            p.expect_kw("IMMEDIATE")?;
            let query = p.expr()?;
            let into = if p.eat_kw("INTO") {
                p.comma_sep(Self::variable_name)?
            } else {
                Vec::new()
            };
            let mut using = Vec::new();
            if p.eat_kw("USING") {
                let mut parsed = false;
                if p.sym_at(0, Sym::LParen) {
                    let save = p.checkpoint();
                    p.bump();
                    let args = p.comma_sep(Self::using_argument);
                    if let Ok(args) = args
                        && p.eat_sym(Sym::RParen)
                        && (p.pos >= p.toks.len() || p.sym_at(0, Sym::Semi))
                    {
                        using = args;
                        parsed = true;
                        p.commit(save);
                    } else {
                        p.restore(save);
                    }
                }
                if !parsed {
                    using = p.comma_sep(Self::using_argument)?;
                }
            }
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::ExecuteImmediate(ExecuteImmediate {
                query,
                into,
                using,
                span,
            }))
        })
    }

    /// One `USING` argument: `expr [[AS] alias]`.
    #[inline(never)]
    pub(super) fn using_argument(&mut self) -> PR<UsingArgument> {
        let start = self.start();
        let value = self.expr()?;
        let alias = match self.select_alias()? {
            Some(Alias::Name(id)) => Some(id),
            Some(Alias::List(_)) => return self.fail(),
            None => None,
        };
        Ok(UsingArgument {
            value,
            alias,
            span: self.span_from(start),
        })
    }

    /// `SET VAR[IABLE] v = expr [, ...]` or `SET VAR[IABLE] (v, ...) =
    /// (query)` (`VAR` for `VARIABLE`), dispatched from [`Self::explainable`]
    /// when `VAR` or `VARIABLE` follows `SET`.
    pub(super) fn set_variable_command(&mut self, keyword: bool) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("SET")?;
            // Dispatched on `VAR` or `VARIABLE` following `SET`; only that
            // one keyword is the form (`SET VAR variable = 1` assigns a
            // variable named `variable`). In a script `SET` is always an
            // assignment and the keyword is not part of the form.
            if keyword && !p.eat_kw("VAR") {
                p.eat_kw("VARIABLE");
            }
            if p.sym_at(0, Sym::LParen) {
                // The row form: `(v, ...) = (query)`. An empty target
                // list parses and fails after the statement parsed, with
                // `INVALID_SET_SYNTAX`.
                p.bump();
                let mut targets = Vec::new();
                if !p.sym_at(0, Sym::RParen) {
                    targets = p.comma_sep(Self::variable_name)?;
                }
                p.expect_sym(Sym::RParen)?;
                p.expect_sym(Sym::Eq)?;
                p.expect_sym(Sym::LParen)?;
                let query = Box::new(p.query()?);
                p.expect_sym(Sym::RParen)?;
                let span = p.span_from(start);
                p.command_end()?;
                return Ok(Statement::SetVariable(SetVariable {
                    kind: SetVariableKind::Row { targets, query },
                    span,
                }));
            }
            let assignments = p.comma_sep(|q| {
                let element = q.start();
                let name = q.variable_name()?;
                q.expect_sym(Sym::Eq)?;
                let value = q.expr()?;
                Ok(SetAssignment {
                    name,
                    value,
                    span: q.span_from(element),
                })
            })?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::SetVariable(SetVariable {
                kind: SetVariableKind::Single { assignments },
                span,
            }))
        })
    }

    /// The raw rest of a `SET key =` assignment, trimmed: decoded only
    /// when it is one backquoted token, which is how Spark lets a value
    /// hold a semicolon. An unterminated quote fails where the quote
    /// is, as in Spark. The cursor ends at the `;` or end of input.
    /// The elements of `SET PATH =`: `DEFAULT_PATH`, `SYSTEM_PATH`, `PATH`,
    /// `CURRENT_SCHEMA`, `CURRENT_DATABASE` or a namespace of at least two
    /// parts.
    pub(super) fn set_path_command(&mut self, start: usize) -> PR<Statement> {
        let mut elements = Vec::new();
        loop {
            let one = self.name_part(true)?;
            let mut parts = vec![one];
            while self.sym_at(0, Sym::Dot) && self.is_ident_at(1, true) {
                self.bump();
                parts.push(self.name_part(true)?);
            }
            let element = if parts.len() == 1 && !parts[0].quoted {
                let w = parts[0].value.to_ascii_uppercase();
                match w.as_str() {
                    "DEFAULT_PATH" => Some(PathEl::DefaultPath),
                    "SYSTEM_PATH" => Some(PathEl::SystemPath),
                    "PATH" => Some(PathEl::Path),
                    "CURRENT_SCHEMA" => Some(PathEl::CurrentSchema { database: false }),
                    "CURRENT_DATABASE" => Some(PathEl::CurrentSchema { database: true }),
                    _ => None,
                }
            } else {
                None
            };
            let element = match element {
                Some(e) => e,
                None if parts.len() == 1 => {
                    return Self::fatal(error::invalid_sql_path_schema_reference(
                        &parts[0].value,
                        parts[0].span,
                    ));
                }
                None => PathEl::Schema(parts.into_iter().map(|p| p.value).collect()),
            };
            elements.push(element);
            if !self.eat_sym(Sym::Comma) {
                break;
            }
        }
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::Set(Set {
            kind: SetKind::Path { elements },
            span,
        }))
    }

    pub(super) fn set_value(&mut self) -> PR<(String, Span)> {
        let from_pos = self.pos;
        let mut end = self.toks.len();
        for (i, t) in self.toks.iter().enumerate().skip(self.pos) {
            if t.kind == TokenKind::Sym(Sym::Semi) {
                end = i;
                break;
            }
            if t.kind == TokenKind::Unterminated {
                self.pos = i;
                return self.fail();
            }
        }
        let from = self.tok(0).map_or(self.src.len(), |t| t.span.start);
        let to = (self.pos..end)
            .filter_map(|i| self.toks.get(i))
            .map(|t| t.span.end)
            .max()
            .unwrap_or(from);
        let (value, value_span) = self.trimmed_span(from, to);
        self.pos = end;
        // One backquoted token for the whole value is decoded, as Spark
        // allows backquoted keys and values.
        if from_pos + 1 == end
            && let Some(t) = self.toks.get(from_pos)
            && t.kind == TokenKind::QuotedIdent
            && t.span.range() == value_span.range()
        {
            return Ok((unquote_ident(t.span.text(self.src)), value_span));
        }
        Ok((value, value_span))
    }

    /// The backquoted key at the cursor, decoded; `None` when the token
    /// is no backquoted key. An unterminated quote fails where the quote
    /// is, as in Spark.
    pub(super) fn quoted_key(&mut self) -> PR<Option<(String, Span)>> {
        let Some(t) = self.tok(0) else {
            return Ok(None);
        };
        if t.kind == TokenKind::Unterminated {
            return self.fail();
        }
        if t.kind != TokenKind::QuotedIdent {
            return Ok(None);
        }
        self.bump();
        Ok(Some((unquote_ident(t.span.text(self.src)), t.span)))
    }

    /// `text[from..to]` trimmed the way Spark trims statement text, with
    /// its span.
    pub(super) fn trimmed_span(&self, from: usize, to: usize) -> (String, Span) {
        let raw = self.src.get(from..to).unwrap_or_default();
        let trimmed = raw.trim_matches(|c: char| c <= ' ');
        let leading = raw.len() - raw.trim_start_matches(|c: char| c <= ' ').len();
        (
            trimmed.to_owned(),
            Span::new(from + leading, from + leading + trimmed.len()),
        )
    }

    /// `RESET` or `RESET key`: the key is one backquoted key or a bare
    /// key matching `[a-zA-Z_\d\.:]+`; anything else is
    /// `INVALID_RESET_COMMAND_FORMAT`.
    pub(super) fn reset_command(&mut self) -> PR<Statement> {
        let start = self.start();
        self.expect_kw("RESET")?;
        if self.pos >= self.toks.len() || self.sym_at(0, Sym::Semi) {
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Reset(Reset {
                key: None,
                key_span: None,
                span,
            }));
        }
        if let Some((key, key_span)) = self.quoted_key()? {
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Reset(Reset {
                key: Some(key),
                key_span: Some(key_span),
                span,
            }));
        }
        let mut semi = self.toks.len();
        for (i, t) in self.toks.iter().enumerate().skip(self.pos) {
            if t.kind == TokenKind::Sym(Sym::Semi) {
                semi = i;
                break;
            }
            if t.kind == TokenKind::Unterminated {
                self.pos = i;
                return self.fail();
            }
        }
        let from = self.start();
        let to = (self.pos..semi)
            .filter_map(|i| self.toks.get(i))
            .map(|t| t.span.end)
            .max()
            .unwrap_or(from);
        let (key, key_span) = self.trimmed_span(from, to);
        if !valid_bare_key(&key) {
            return Self::fatal(error::invalid_reset_command_format(self.span_from(start)));
        }
        self.pos = semi;
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::Reset(Reset {
            key: Some(key),
            key_span: Some(key_span),
            span,
        }))
    }
}
