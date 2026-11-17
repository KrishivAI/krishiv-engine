// SPDX-License-Identifier: Apache-2.0
//! The access-control statements (P3-T062, a Krishiv extension; open Spark has no such grammar):
//!
//! ```text
//! GRANT <priv>[, ...] ON <securable> TO <grantee>[, ...] [WITH GRANT OPTION]
//! REVOKE [GRANT OPTION FOR] <priv>[, ...] ON <securable> FROM <grantee>[, ...]
//! SHOW GRANTS [ON <securable>] [TO <grantee>]
//! CREATE ROLE r;  DROP ROLE r;  GRANT ROLE r TO <grantee>[, ...];  REVOKE ROLE r FROM <grantee>[, ...]
//! ```
//!
//! `<priv>` is `SELECT [(col, ...)] | MODIFY | CREATE | DROP | MONITOR | USAGE | BROWSE |
//! ALL [PRIVILEGES]`; `<securable>` is `CATALOG c | SCHEMA|NAMESPACE|DATABASE n | [TABLE] t |
//! VIEW v | FUNCTION f | VOLUME|PATH 'url' | CONNECTION c | SECRET SCOPE s`; `<grantee>` is
//! `USER:name | GROUP:name | ROLE:name` (a name with other characters is backquoted or a string).
//! These statements start only where no Spark statement can (`GRANT`, `REVOKE`, `CREATE ROLE`,
//! `DROP ROLE`, `SHOW GRANTS` before the end, `ON` or `TO`), so nothing Spark parses changes.

use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::access_control::{
    AccessCommand, GrantObject, GrantPrivilege, GrantPrivileges, Grantee, grantable_privilege,
};

use super::{PR, Parser};
use crate::ast::{AccessControl, Statement};
use crate::error;
use crate::lexer::{Sym, TokenKind};

impl Parser<'_> {
    /// Whether the statement at the cursor is an access-control statement.
    pub(super) fn at_access_control(&self) -> bool {
        self.kw_at(0, "GRANT")
            || self.kw_at(0, "REVOKE")
            || ((self.kw_at(0, "CREATE") || self.kw_at(0, "DROP")) && self.kw_at(1, "ROLE"))
            || (self.kw_at(0, "SHOW")
                && self.kw_at(1, "GRANTS")
                && (self.tok(2).is_none()
                    || self.sym_at(2, Sym::Semi)
                    || self.kw_at(2, "ON")
                    || self.kw_at(2, "TO")))
    }

    /// An access-control statement (see the module docs).
    pub(super) fn access_control_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            let command = p.access_command()?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::AccessControl(Box::new(AccessControl {
                command,
                span,
            })))
        })
    }

    fn access_command(&mut self) -> PR<AccessCommand> {
        if self.eat_kw("CREATE") {
            self.expect_kw("ROLE")?;
            return Ok(AccessCommand::CreateRole {
                name: self.any_ident()?.value,
            });
        }
        if self.eat_kw("DROP") {
            self.expect_kw("ROLE")?;
            return Ok(AccessCommand::DropRole {
                name: self.any_ident()?.value,
            });
        }
        if self.eat_kw("SHOW") {
            self.expect_kw("GRANTS")?;
            let object = if self.eat_kw("ON") {
                Some(self.grant_object()?)
            } else {
                None
            };
            let grantee = if self.eat_kw("TO") {
                Some(self.grantee()?)
            } else {
                None
            };
            return Ok(AccessCommand::ShowGrants { object, grantee });
        }
        if self.eat_kw("GRANT") {
            if self.eat_kw("ROLE") {
                let role = self.any_ident()?.value;
                self.expect_kw("TO")?;
                let grantees = self.comma_sep(Self::grantee)?;
                return Ok(AccessCommand::GrantRole { role, grantees });
            }
            let privileges = self.grant_privileges()?;
            self.expect_kw("ON")?;
            let object = self.grant_object()?;
            check_columns(&privileges, &object, self)?;
            self.expect_kw("TO")?;
            let grantees = self.comma_sep(Self::grantee)?;
            let with_grant_option = if self.eat_kw("WITH") {
                self.expect_kw("GRANT")?;
                self.expect_kw("OPTION")?;
                true
            } else {
                false
            };
            return Ok(AccessCommand::Grant {
                privileges,
                object,
                grantees,
                with_grant_option,
            });
        }
        self.expect_kw("REVOKE")?;
        if self.eat_kw("ROLE") {
            let role = self.any_ident()?.value;
            self.expect_kw("FROM")?;
            let grantees = self.comma_sep(Self::grantee)?;
            return Ok(AccessCommand::RevokeRole { role, grantees });
        }
        let grant_option_only = if self.kw_at(0, "GRANT") {
            self.bump();
            self.expect_kw("OPTION")?;
            self.expect_kw("FOR")?;
            true
        } else {
            false
        };
        let privileges = self.grant_privileges()?;
        self.expect_kw("ON")?;
        let object = self.grant_object()?;
        check_columns(&privileges, &object, self)?;
        self.expect_kw("FROM")?;
        let grantees = self.comma_sep(Self::grantee)?;
        Ok(AccessCommand::Revoke {
            grant_option_only,
            privileges,
            object,
            grantees,
        })
    }

    /// `ALL [PRIVILEGES]` or a list of privileges, each once (a repeated one merges its columns).
    fn grant_privileges(&mut self) -> PR<GrantPrivileges> {
        if self.eat_kw("ALL") {
            self.eat_kw("PRIVILEGES");
            return Ok(GrantPrivileges::All);
        }
        let mut out: Vec<GrantPrivilege> = Vec::new();
        for item in self.comma_sep(Self::grant_privilege)? {
            match out.iter_mut().find(|p| p.privilege == item.privilege) {
                // A whole-object grant covers any column list.
                Some(seen) if seen.columns.is_empty() || item.columns.is_empty() => {
                    seen.columns.clear();
                }
                Some(seen) => {
                    for c in item.columns {
                        if !seen.columns.contains(&c) {
                            seen.columns.push(c);
                        }
                    }
                }
                None => out.push(item),
            }
        }
        Ok(GrantPrivileges::List(out))
    }

    fn grant_privilege(&mut self) -> PR<GrantPrivilege> {
        if self.kind(0) != Some(TokenKind::Word) {
            return self.fail();
        }
        let Some(privilege) = grantable_privilege(self.text(0)) else {
            return self.fail();
        };
        self.bump();
        let columns = if self.sym_at(0, Sym::LParen) {
            if privilege != krishiv_plan::analyzer::Privilege::Select {
                return self.fail();
            }
            self.paren_list(|p| Ok(p.any_ident()?.value))?
        } else {
            Vec::new()
        };
        Ok(GrantPrivilege { privilege, columns })
    }

    /// `<securable>` (see the module docs).
    fn grant_object(&mut self) -> PR<GrantObject> {
        let parts = |p: &mut Self| -> PR<Vec<String>> {
            Ok(p.object_name()?
                .parts
                .into_iter()
                .map(|i| i.value)
                .collect())
        };
        if self.eat_kw("CATALOG") {
            return Ok(GrantObject::Catalog(self.any_ident()?.value));
        }
        if self.eat_kw("SCHEMA") || self.eat_kw("NAMESPACE") || self.eat_kw("DATABASE") {
            return Ok(GrantObject::Namespace(parts(self)?));
        }
        if self.eat_kw("VIEW") {
            return Ok(GrantObject::View(parts(self)?));
        }
        if self.eat_kw("FUNCTION") {
            return Ok(GrantObject::Function(parts(self)?));
        }
        if self.eat_kw("VOLUME") || self.eat_kw("PATH") {
            let Some(url) = self.string_lit() else {
                return self.fail();
            };
            return Ok(GrantObject::Path(OptionValue::from_user("path", url)));
        }
        if self.eat_kw("CONNECTION") {
            return Ok(GrantObject::Connection(self.any_ident()?.value));
        }
        if self.kw_at(0, "SECRET") && self.kw_at(1, "SCOPE") {
            self.pos += 2;
            return Ok(GrantObject::SecretScope(self.any_ident()?.value));
        }
        // `TABLE` is optional; a table named `table` is backquoted.
        self.eat_kw("TABLE");
        Ok(GrantObject::Table(parts(self)?))
    }

    /// `USER:name | GROUP:name | ROLE:name`.
    fn grantee(&mut self) -> PR<Grantee> {
        let make: fn(String) -> Grantee = if self.kw_at(0, "USER") {
            Grantee::User
        } else if self.kw_at(0, "GROUP") {
            Grantee::Group
        } else if self.kw_at(0, "ROLE") {
            Grantee::Role
        } else {
            return self.fail();
        };
        self.bump();
        self.expect_sym(Sym::Colon)?;
        let name = match self.string_lit() {
            Some(s) => s,
            None => self.any_ident()?.value,
        };
        if name.is_empty() {
            return self.fail();
        }
        Ok(make(name))
    }
}

/// Column privileges name only a table or view.
fn check_columns(privileges: &GrantPrivileges, object: &GrantObject, p: &Parser<'_>) -> PR<()> {
    let columns = match privileges {
        GrantPrivileges::List(l) => l.iter().any(|g| !g.columns.is_empty()),
        _ => false,
    };
    if columns && !object.has_columns() {
        return Parser::fatal(error::invalid_statement(
            "column privileges on a securable without columns",
            p.span_from(p.start()),
        ));
    }
    Ok(())
}
