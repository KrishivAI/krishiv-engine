// SPDX-License-Identifier: Apache-2.0
//! `CREATE [OR REPLACE] [TEMPORARY] FUNCTION` and `DROP [TEMPORARY]
//! FUNCTION`: SQL-bodied user functions (Spark's
//! `createUserDefinedFunction` and `dropFunction` rules).

use super::{Close, PR, Parser, QUERY_START};
use crate::ast::{
    CreateFunction, DropFunction, FunctionBody, FunctionBodyKind, FunctionReturns,
    FunctionReturnsKind, ReturnColumn, RoutineCharacteristics, Span, SqlDataAccess, SqlSecurity,
    Statement,
};
use crate::error;
use crate::lexer::Sym;

impl Parser<'_> {
    /// `FUNCTION [IF NOT EXISTS] name ([parameters]) [RETURNS ...]
    /// [characteristics] RETURN body`, after the `CREATE [OR REPLACE]
    /// [TEMPORARY]` at `start`. Checks Spark makes on the parsed statement
    /// run here: a temporary function takes no database and no `IF NOT
    /// EXISTS`, and `IF NOT EXISTS` excludes `OR REPLACE`; the checks on
    /// the parameters, types and body are in `validate`.
    pub(super) fn create_function_command(
        &mut self,
        start: usize,
        or_replace: bool,
        temporary: bool,
    ) -> PR<Statement> {
        self.nested(|p| {
            p.expect_kw("FUNCTION")?;
            let if_not_exists = if p.kw_at(0, "IF") && p.kw_at(1, "NOT") && p.kw_at(2, "EXISTS") {
                p.pos += 3;
                true
            } else {
                false
            };
            let name = p.object_name()?;
            p.expect_sym(Sym::LParen)?;
            let params_start = p.start();
            let params = if p.sym_at(0, Sym::RParen) {
                Vec::new()
            } else {
                p.comma_sep(Self::table_column)?
            };
            let params_span = p.span_from(params_start);
            p.expect_closing(Sym::RParen, Close::List(params.len()))?;
            let returns = if p.kw_at(0, "RETURNS") {
                Some(p.function_returns()?)
            } else {
                None
            };
            let characteristics = p.routine_characteristics()?;
            p.expect_kw("RETURN")?;
            let body = p.function_body()?;
            let span = p.span_from(start);
            p.command_end()?;
            if temporary && name.parts.len() > 1 {
                let database: Vec<String> = name.parts[..name.parts.len() - 1]
                    .iter()
                    .map(|part| part.value.clone())
                    .collect();
                return Self::fatal(error::create_temp_func_with_database(
                    database.join("."),
                    span,
                ));
            }
            if temporary && if_not_exists {
                return Self::fatal(error::create_temp_func_with_if_not_exists(span));
            }
            if if_not_exists && or_replace {
                return Self::fatal(error::create_routine_with_if_not_exists_and_replace(span));
            }
            Ok(Statement::CreateFunction(Box::new(CreateFunction {
                or_replace,
                temporary,
                if_not_exists,
                name,
                params,
                params_span,
                returns,
                characteristics,
                body,
                span,
            })))
        })
    }

    /// `RETURNS type` or `RETURNS TABLE [(name type [NOT NULL] [COMMENT
    /// '...'], ...)]`.
    fn function_returns(&mut self) -> PR<FunctionReturns> {
        let start = self.start();
        self.expect_kw("RETURNS")?;
        let kind = if self.eat_kw("TABLE") {
            let columns = if self.sym_at(0, Sym::LParen) {
                Some(self.paren_list(Self::return_column)?)
            } else {
                None
            };
            FunctionReturnsKind::Table(columns)
        } else {
            FunctionReturnsKind::Type(self.data_type()?)
        };
        Ok(FunctionReturns {
            kind,
            span: self.span_from(start),
        })
    }

    fn return_column(&mut self) -> PR<ReturnColumn> {
        let start = self.start();
        let name = self.name_part(true)?;
        let data_type = self.data_type()?;
        let not_null = if self.eat_kw("NOT") {
            self.expect_kw("NULL")?;
            true
        } else {
            false
        };
        let comment = if self.eat_kw("COMMENT") {
            match self.string_lit() {
                Some(comment) => Some(comment),
                None => return self.fail(),
            }
        } else {
            None
        };
        Ok(ReturnColumn {
            name,
            data_type,
            not_null,
            comment,
            span: self.span_from(start),
        })
    }

    /// Zero or more of `COMMENT '...'`, `LANGUAGE name`, `[NOT]
    /// DETERMINISTIC`, `CONTAINS SQL`, `READS SQL DATA` and `SQL SECURITY
    /// {INVOKER | DEFINER}`, in any order. A clause written twice keeps its
    /// later value and the first repeated clause's name, which `validate`
    /// reports.
    fn routine_characteristics(&mut self) -> PR<RoutineCharacteristics> {
        let start = self.start();
        let mut c = RoutineCharacteristics::default();
        let mut seen: Vec<&'static str> = Vec::new();
        loop {
            let clause = if self.kw_at(0, "COMMENT") {
                "COMMENT"
            } else if self.kw_at(0, "LANGUAGE") {
                "LANGUAGE"
            } else if self.kw_at(0, "DETERMINISTIC")
                || (self.kw_at(0, "NOT") && self.kw_at(1, "DETERMINISTIC"))
            {
                "DETERMINISTIC"
            } else if self.kw_at(0, "CONTAINS") || self.kw_at(0, "READS") {
                "SQL DATA ACCESS"
            } else if self.kw_at(0, "SQL") && self.kw_at(1, "SECURITY") {
                "SQL SECURITY RIGHTS"
            } else {
                break;
            };
            if seen.contains(&clause) {
                c.duplicate = c.duplicate.or(Some(clause));
            } else {
                seen.push(clause);
            }
            match clause {
                "COMMENT" => {
                    self.bump();
                    c.comment = Some(match self.string_lit() {
                        Some(comment) => comment,
                        None => return self.fail(),
                    });
                }
                "LANGUAGE" => {
                    self.bump();
                    c.language = Some(self.any_ident()?);
                }
                "DETERMINISTIC" => {
                    c.deterministic = Some(!self.eat_kw("NOT"));
                    self.expect_kw("DETERMINISTIC")?;
                }
                "SQL DATA ACCESS" => {
                    if self.eat_kw("CONTAINS") {
                        self.expect_kw("SQL")?;
                        c.data_access = Some(SqlDataAccess::ContainsSql);
                    } else {
                        self.expect_kw("READS")?;
                        self.expect_kw("SQL")?;
                        self.expect_kw("DATA")?;
                        c.data_access = Some(SqlDataAccess::ReadsSqlData);
                    }
                }
                _ => {
                    self.pos += 2;
                    c.security = Some(if self.eat_kw("INVOKER") {
                        SqlSecurity::Invoker
                    } else {
                        self.expect_kw("DEFINER")?;
                        SqlSecurity::Definer
                    });
                }
            }
        }
        c.span = if seen.is_empty() {
            Span::new(start, start)
        } else {
            self.span_from(start)
        };
        Ok(c)
    }

    /// What follows `RETURN`: a query when it starts like one (`SELECT`,
    /// `WITH`, `VALUES`, `TABLE`, `FROM`), else an expression, which
    /// includes a parenthesised subquery.
    fn function_body(&mut self) -> PR<FunctionBody> {
        let start = self.start();
        let kind = if QUERY_START.iter().any(|kw| self.kw_at(0, kw)) {
            FunctionBodyKind::Query(Box::new(self.query()?))
        } else {
            FunctionBodyKind::Expr(Box::new(self.expr()?))
        };
        Ok(FunctionBody {
            kind,
            span: self.span_from(start),
        })
    }

    /// `[TEMPORARY] FUNCTION [IF EXISTS] name`, after the `DROP` at `start`.
    pub(super) fn drop_function_command(&mut self, start: usize) -> PR<Statement> {
        let temporary = self.eat_kw("TEMP") || self.eat_kw("TEMPORARY");
        self.expect_kw("FUNCTION")?;
        let if_exists = if self.kw_at(0, "IF") && self.kw_at(1, "EXISTS") {
            self.pos += 2;
            true
        } else {
            false
        };
        let name = self.object_name()?;
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::DropFunction(DropFunction {
            temporary,
            if_exists,
            name,
            span,
        }))
    }
}
