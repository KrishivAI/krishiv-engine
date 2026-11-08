// SPDX-License-Identifier: Apache-2.0
//! SQL scripts: `[NOT ATOMIC] BEGIN ... END` with variables, control flow, conditions and
//! handlers (Spark's `compoundOrSingleStatement` and the `sqlScriptingStatement` rules).
//!
//! The grammar here is wider than Spark's placement rules: declarations and handlers are
//! accepted among statements and the checks after parsing place them, so each misplacement
//! gets its own error class. Refusals Spark makes while it builds the tree (`CONTINUE`
//! handlers, cursors) are kept in the tree and raised by the checks, in `AstBuilder` order.

use super::{Fail, PR, Parser};
use crate::ast::{
    CaseStatement, Compound, ConditionalBranch, DeclareCondition, ForStatement, Handler,
    HandlerBody, HandlerCondition, Ident, IfStatement, LabelJump, LoopStatement, RepeatStatement,
    ScriptStatement, Statement, WhileStatement,
};
use crate::error;
use crate::lexer::{Sym, TokenKind};

/// How deep script statements may nest (a compound in a branch in a loop ...). Spark has no limit;
/// this one keeps the parser and the interpreter, which recurse once per level, within any stack.
const MAX_SCRIPT_NESTING: usize = 32;

/// Words that end a statement list: the statement after them belongs to the enclosing construct.
const LIST_END: &[&str] = &["END", "ELSE", "ELSEIF", "UNTIL", "WHEN"];

impl Parser<'_> {
    /// A whole script: `[NOT ATOMIC] BEGIN ... END` and the end of the input.
    pub(super) fn script_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.in_script += 1;
            let compound = p.compound(None, start, true);
            p.in_script -= 1;
            let compound = compound?;
            p.command_end()?;
            Ok(Statement::Script(Box::new(compound)))
        })
    }

    /// `BEGIN ... END [label]` at the cursor; `label` is the one written before it, `start` where
    /// the construct starts.
    fn compound(&mut self, label: Option<Ident>, start: usize, top_level: bool) -> PR<Compound> {
        self.nested(|p| {
            p.expect_kw("BEGIN")?;
            // `BEGIN NOT ATOMIC` starts the same script.
            if top_level && p.kw_at(0, "NOT") && p.kw_at(1, "ATOMIC") {
                p.pos += 2;
            }
            let body = p.statement_list()?;
            p.expect_kw("END")?;
            // Nothing after the `END` of the script itself is its label.
            let end_label = if top_level { None } else { p.end_label()? };
            Ok(Compound {
                label,
                end_label,
                body,
                span: p.span_from(start),
            })
        })
    }

    /// The label after `END` / `END <kind>`, if one is written.
    fn end_label(&mut self) -> PR<Option<Ident>> {
        if self.kind(0) == Some(TokenKind::Word) || self.kind(0) == Some(TokenKind::QuotedIdent) {
            if self.any_kw_at(0, LIST_END) {
                return Ok(None);
            }
            return Ok(Some(self.ident()?));
        }
        Ok(None)
    }

    /// Statements, each followed by `;`, up to (not including) a word that ends the list.
    fn statement_list(&mut self) -> PR<Vec<ScriptStatement>> {
        let mut out = Vec::new();
        while self.pos < self.toks.len() {
            if self.any_kw_at(0, LIST_END) && !self.sym_at(1, Sym::Colon) {
                break;
            }
            out.push(self.script_statement()?);
            self.expect_sym(Sym::Semi)?;
        }
        Ok(out)
    }

    /// One statement of a script body, without its `;`.
    fn script_statement(&mut self) -> PR<ScriptStatement> {
        if self.script_depth >= MAX_SCRIPT_NESTING {
            return Self::fatal(error::too_complex(super::statement_span(
                self.src, &self.toks,
            )));
        }
        self.script_depth += 1;
        let statement = self.script_statement_inner();
        self.script_depth -= 1;
        statement
    }

    fn script_statement_inner(&mut self) -> PR<ScriptStatement> {
        self.nested(|p| {
            let start = p.start();
            // `label: BEGIN | WHILE | REPEAT | LOOP | FOR`.
            if p.is_ident_at(0, false)
                && p.sym_at(1, Sym::Colon)
                && p.any_kw_at(2, &["BEGIN", "WHILE", "REPEAT", "LOOP", "FOR"])
            {
                let label = p.ident()?;
                p.expect_sym(Sym::Colon)?;
                return p.labelled(Some(label), start);
            }
            if p.any_kw_at(0, &["BEGIN", "WHILE", "REPEAT", "LOOP", "FOR"]) {
                return p.labelled(None, start);
            }
            if p.kw_at(0, "IF") {
                return p.if_statement(start);
            }
            if p.kw_at(0, "CASE") {
                return p.case_statement(start);
            }
            if p.any_kw_at(0, &["LEAVE", "ITERATE"]) {
                let leave = p.kw_at(0, "LEAVE");
                p.bump();
                let label = p.ident()?;
                let jump = LabelJump {
                    label,
                    span: p.span_from(start),
                };
                return Ok(if leave {
                    ScriptStatement::Leave(jump)
                } else {
                    ScriptStatement::Iterate(jump)
                });
            }
            if p.kw_at(0, "DECLARE") {
                return p.script_declaration(start);
            }
            if p.kw_at(0, "SET") {
                return match p.script_set()? {
                    Statement::SetVariable(set) => Ok(ScriptStatement::Assign(set)),
                    _ => p.fail(),
                };
            }
            // Cursors parse so that the checks can refuse them.
            if p.any_kw_at(0, &["OPEN", "FETCH", "CLOSE"]) && p.is_ident_at(1, true) {
                p.skip_to_statement_end();
                return Ok(ScriptStatement::Cursor(p.span_from(start)));
            }
            // `RESET` is no script statement (Spark fails at the token after it).
            if p.kw_at(0, "RESET") {
                p.bump();
                return p.fail();
            }
            let statement = p.statement_or_label_start()?;
            Ok(ScriptStatement::Sql(Box::new(statement)))
        })
    }

    /// The SQL statement at the cursor. A statement Spark does not know (`GET DIAGNOSTICS`, `SIGNAL`,
    /// `RESIGNAL`, a qualified label) starts with a word its script grammar could still take as a
    /// label, so ANTLR's prediction goes on to the next token and the error is there.
    fn statement_or_label_start(&mut self) -> PR<Statement> {
        let first = self.pos;
        let result = self.explainable();
        if matches!(result, Err(Fail::Syntax)) && self.far == first && self.is_ident_at(0, false) {
            self.pos = first + 1;
            // The error is recorded at the second token, as if the first had matched.
            return self.fail();
        }
        result
    }

    /// `SET` in a script, always an assignment. `SET VARIABLE x = 1` and `SET TIME ZONE 'UTC'` read
    /// their second word as the variable and fail at the next token, which ANTLR then recovers from:
    /// by dropping the token when the one after it is the `=` ("extra input"), else by inserting
    /// the `=` ("missing EQ").
    fn script_set(&mut self) -> PR<Statement> {
        let start = self.pos;
        let word = self.is_ident_at(1, true);
        let result = self.set_variable_command(false);
        if matches!(result, Err(Fail::Syntax))
            && self.far == start + 2
            && self.extra.is_none()
            && self.missing.is_none()
            && word
        {
            if self.toks.get(self.far + 1).map(|t| t.kind) == Some(TokenKind::Sym(Sym::Eq)) {
                self.extra = Some(self.far);
            } else {
                self.missing = Some((self.far, "EQ"));
            }
        }
        result
    }

    /// A compound or a loop, after its optional label.
    fn labelled(&mut self, label: Option<Ident>, start: usize) -> PR<ScriptStatement> {
        if self.kw_at(0, "BEGIN") {
            return Ok(ScriptStatement::Compound(Box::new(
                self.compound(label, start, false)?,
            )));
        }
        if self.kw_at(0, "WHILE") {
            self.bump();
            let condition = self.expr()?;
            self.expect_kw("DO")?;
            let body = self.statement_list()?;
            self.expect_kw("END")?;
            self.expect_kw("WHILE")?;
            let end_label = self.end_label()?;
            return Ok(ScriptStatement::While(Box::new(WhileStatement {
                label,
                end_label,
                condition,
                body,
                span: self.span_from(start),
            })));
        }
        if self.kw_at(0, "REPEAT") {
            self.bump();
            let body = self.statement_list()?;
            self.expect_kw("UNTIL")?;
            let condition = self.expr()?;
            self.expect_kw("END")?;
            self.expect_kw("REPEAT")?;
            let end_label = self.end_label()?;
            return Ok(ScriptStatement::Repeat(Box::new(RepeatStatement {
                label,
                end_label,
                body,
                condition,
                span: self.span_from(start),
            })));
        }
        if self.kw_at(0, "LOOP") {
            self.bump();
            let body = self.statement_list()?;
            self.expect_kw("END")?;
            self.expect_kw("LOOP")?;
            let end_label = self.end_label()?;
            return Ok(ScriptStatement::Loop(Box::new(LoopStatement {
                label,
                end_label,
                body,
                span: self.span_from(start),
            })));
        }
        // FOR [variable AS] query DO ... END FOR
        self.expect_kw("FOR")?;
        let variable = if self.is_ident_at(0, false) && self.kw_at(1, "AS") {
            let name = self.ident()?;
            self.expect_kw("AS")?;
            Some(name)
        } else {
            None
        };
        let query = self.query()?;
        self.expect_kw("DO")?;
        let body = self.statement_list()?;
        self.expect_kw("END")?;
        self.expect_kw("FOR")?;
        let end_label = self.end_label()?;
        Ok(ScriptStatement::For(Box::new(ForStatement {
            label,
            end_label,
            variable,
            query,
            body,
            span: self.span_from(start),
        })))
    }

    /// `IF cond THEN ... [ELSEIF cond THEN ...]... [ELSE ...] END IF`.
    fn if_statement(&mut self, start: usize) -> PR<ScriptStatement> {
        self.expect_kw("IF")?;
        let mut branches = Vec::new();
        let mut else_body = None;
        let condition = self.expr()?;
        self.expect_kw("THEN")?;
        branches.push(ConditionalBranch {
            condition,
            body: self.statement_list()?,
        });
        loop {
            if self.eat_kw("ELSEIF") {
                let condition = self.expr()?;
                self.expect_kw("THEN")?;
                branches.push(ConditionalBranch {
                    condition,
                    body: self.statement_list()?,
                });
            } else if self.eat_kw("ELSE") {
                else_body = Some(self.statement_list()?);
                break;
            } else {
                break;
            }
        }
        self.expect_kw("END")?;
        self.expect_kw("IF")?;
        Ok(ScriptStatement::If(Box::new(IfStatement {
            branches,
            else_body,
            span: self.span_from(start),
        })))
    }

    /// `CASE [subject] WHEN ... THEN ... [ELSE ...] END CASE`.
    fn case_statement(&mut self, start: usize) -> PR<ScriptStatement> {
        self.expect_kw("CASE")?;
        let subject = if self.kw_at(0, "WHEN") {
            None
        } else {
            Some(self.expr()?)
        };
        let mut branches = Vec::new();
        self.expect_kw("WHEN")?;
        loop {
            let condition = self.expr()?;
            self.expect_kw("THEN")?;
            branches.push(ConditionalBranch {
                condition,
                body: self.statement_list()?,
            });
            if !self.eat_kw("WHEN") {
                break;
            }
        }
        let else_body = if self.eat_kw("ELSE") {
            Some(self.statement_list()?)
        } else {
            None
        };
        self.expect_kw("END")?;
        self.expect_kw("CASE")?;
        Ok(ScriptStatement::Case(Box::new(CaseStatement {
            subject,
            branches,
            else_body,
            span: self.span_from(start),
        })))
    }

    /// `DECLARE ...`: a local variable, a condition, a handler or a cursor.
    fn script_declaration(&mut self, start: usize) -> PR<ScriptStatement> {
        if self.any_kw_at(1, &["EXIT", "CONTINUE"]) && self.kw_at(2, "HANDLER") {
            return self.handler(start);
        }
        // `DECLARE name CONDITION ...` and `DECLARE name CURSOR ...`.
        if self.is_ident_at(1, false) && self.kw_at(2, "CONDITION") {
            self.bump();
            let name = self.ident()?;
            self.expect_kw("CONDITION")?;
            let sqlstate = if self.eat_kw("FOR") {
                self.expect_kw("SQLSTATE")?;
                self.eat_kw("VALUE");
                Some(self.sqlstate_literal()?)
            } else {
                None
            };
            return Ok(ScriptStatement::Condition(DeclareCondition {
                name,
                sqlstate,
                span: self.span_from(start),
            }));
        }
        if self.is_ident_at(1, false)
            && (self.kw_at(2, "CURSOR")
                || (self.any_kw_at(2, &["ASENSITIVE", "INSENSITIVE"]) && self.kw_at(3, "CURSOR")))
        {
            self.skip_to_statement_end();
            return Ok(ScriptStatement::Cursor(self.span_from(start)));
        }
        match self.declare_command()? {
            Statement::DeclareVariable(d) => Ok(ScriptStatement::Declare(Box::new(d))),
            _ => self.fail(),
        }
    }

    /// `DECLARE {EXIT | CONTINUE} HANDLER FOR conditions body`.
    fn handler(&mut self, start: usize) -> PR<ScriptStatement> {
        self.expect_kw("DECLARE")?;
        let continue_handler = self.kw_at(0, "CONTINUE");
        self.bump();
        self.expect_kw("HANDLER")?;
        self.expect_kw("FOR")?;
        let conditions = self.comma_sep(Self::handler_condition)?;
        let body_start = self.start();
        let body = if self.kw_at(0, "BEGIN") {
            HandlerBody::Compound(Box::new(self.compound(None, body_start, false)?))
        } else if self.kw_at(0, "SET") {
            match self.set_variable_command(false)? {
                Statement::SetVariable(set) => HandlerBody::Assign(set),
                _ => return self.fail(),
            }
        } else if self.kw_at(0, "IF") && !self.sym_at(1, Sym::LParen) {
            // Spark has no `IF` statement as a handler body: it fails at the token after `IF`.
            self.bump();
            return self.fail();
        } else {
            HandlerBody::Sql(Box::new(self.statement_or_label_start()?))
        };
        Ok(ScriptStatement::Handler(Box::new(Handler {
            continue_handler,
            conditions,
            body,
            span: self.span_from(start),
        })))
    }

    /// `SQLEXCEPTION`, `NOT FOUND`, `SQLSTATE [VALUE] 's'` or a condition or error class name.
    fn handler_condition(&mut self) -> PR<HandlerCondition> {
        let start = self.start();
        if self.eat_kw("SQLEXCEPTION") {
            return Ok(HandlerCondition::SqlException(self.span_from(start)));
        }
        if self.kw_at(0, "NOT") && self.kw_at(1, "FOUND") {
            self.pos += 2;
            return Ok(HandlerCondition::NotFound(self.span_from(start)));
        }
        if self.kw_at(0, "SQLSTATE") {
            self.bump();
            self.eat_kw("VALUE");
            let (text, span) = self.sqlstate_literal()?;
            return Ok(HandlerCondition::SqlState(text, span));
        }
        Ok(HandlerCondition::Name(self.ident()?))
    }

    /// A string literal naming a SQLSTATE, with its position.
    fn sqlstate_literal(&mut self) -> PR<(String, crate::ast::Span)> {
        let Some(token) = self.tok(0) else {
            return self.fail();
        };
        if token.kind != TokenKind::String {
            return self.fail();
        }
        match self.string_lit() {
            Some(text) => Ok((text, token.span)),
            None => self.fail(),
        }
    }

    /// Moves the cursor to the `;` that ends the statement (or the end of the input).
    fn skip_to_statement_end(&mut self) {
        while self.pos < self.toks.len() && !self.sym_at(0, Sym::Semi) {
            self.bump();
        }
    }
}
