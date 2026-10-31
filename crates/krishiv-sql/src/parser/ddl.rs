// SPDX-License-Identifier: Apache-2.0
//! Command statement parsers: `ALTER`, `SHOW`, `DESCRIBE`, `INSERT`,
//! `DELETE`, `UPDATE`, `MERGE`, `TRUNCATE`, `CACHE`, `REFRESH`, `ANALYZE`,
//! `MSCK`/`REPAIR`, `ADD`/`LIST`, and the `CREATE`/`DROP VIEW` shapes the
//! legacy temporary-view parsers cannot express.
//!
//! Parsing only: every tree here is syntax (names resolved and literals
//! checked in [`crate::validate`], like the rest); there is no lowering
//! for these statements.
//! Errors Spark raises while parsing (`PARSE_SYNTAX_ERROR`, the
//! `DESCRIBE` classes, `_LEGACY_ERROR_TEMP_0035`, ...) are raised here or
//! in validation with Spark's class and SQLSTATE.

use super::{Close, Fail, PR, Parser};
use crate::ast::{
    AddResource, AlterColumnAction, AlterNamespace, AlterNamespaceAction, AlterTable,
    AlterTableAction, AlterView, AlterViewAction, Analyze, AnalyzeColumns, Cache, CacheKind,
    ChangeColumnAction, CreatePermanentView, DdlName, Delete, Describe, DescribeKind, DescribeMode,
    DropPermanentView, FunctionScope, Ident, Insert, InsertClause, ListResource, Merge,
    MergeAction, MergeClause, MergeInsert, MergeMatched, MergeSource, NamespaceKind, PartitionSpec,
    PartitionSpecPart, PartitionWithLocation, PermanentViewColumn, Query, Refresh, RefreshKind,
    Relation, RelationExt, Repair, RepairAction, ResourceType, SchemaBinding, SchemaBindingClause,
    SetExprKind, Show, ShowKind, ShowPattern, Span, Statement, Truncate, Update, UpdateAssignment,
};
use krishiv_common::error::{LegacyErrorTemp0035, LegacyErrorTemp0064, MergeWithoutWhen};

use crate::error::{self, ParseError};
use crate::lexer::{Sym, TokenKind};

impl<'s> Parser<'s> {
    /// An `ALTER`, `SHOW`, `DESCRIBE`, `INSERT`, `DELETE`, `UPDATE`,
    /// `MERGE`, `TRUNCATE`, `CACHE`, `UNCACHE`, `CLEAR`, `REFRESH`,
    /// `ANALYZE`, `MSCK`, `REPAIR`, `ADD` or `LIST` statement.
    pub(crate) fn command_statement(&mut self) -> PR<Statement> {
        if self.kw_at(0, "ALTER") {
            self.alter_command()
        } else if self.kw_at(0, "SHOW") {
            self.show_command()
        } else if self.kw_at(0, "DESC") || self.kw_at(0, "DESCRIBE") {
            self.describe_command()
        } else if self.kw_at(0, "INSERT") {
            self.insert_command()
        } else if self.kw_at(0, "DELETE") {
            self.delete_command()
        } else if self.kw_at(0, "UPDATE") {
            self.update_command()
        } else if self.kw_at(0, "MERGE") {
            self.merge_command()
        } else if self.kw_at(0, "TRUNCATE") {
            self.truncate_command()
        } else if self.any_kw_at(0, &["CACHE", "UNCACHE", "CLEAR"]) {
            self.cache_command()
        } else if self.kw_at(0, "REFRESH") {
            self.refresh_command()
        } else if self.kw_at(0, "ANALYZE") {
            self.analyze_command()
        } else if self.kw_at(0, "MSCK") || self.kw_at(0, "REPAIR") {
            self.repair_command()
        } else if self.kw_at(0, "ADD") || self.kw_at(0, "LIST") {
            self.resource_command()
        } else {
            self.fail()
        }
    }

    /// `IF NOT EXISTS`, if written.
    fn eat_if_not_exists(&mut self) -> bool {
        if self.kw_at(0, "IF") && self.kw_at(1, "NOT") && self.kw_at(2, "EXISTS") {
            self.pos += 3;
            true
        } else {
            false
        }
    }

    /// `IF EXISTS`, if written.
    fn eat_if_exists(&mut self) -> bool {
        if self.kw_at(0, "IF") && self.kw_at(1, "EXISTS") {
            self.pos += 2;
            true
        } else {
            false
        }
    }

    /// Whether a query starts here (`SELECT`, `WITH`, `VALUES`, `FROM` or a
    /// parenthesized query).
    fn query_starts(&self) -> bool {
        self.any_kw_at(0, &["SELECT", "WITH", "VALUES", "FROM"])
            || (self.sym_at(0, Sym::LParen) && self.query_follows(1))
    }

    /// A rename target: a table name, except that the legacy identifier
    /// mode rejects a computed name there (Spark reports the `(`), as it
    /// does for column names.
    fn rename_target(&mut self) -> PR<DdlName> {
        if self.opts.legacy_identifier_clause {
            self.table_name().map(DdlName::Parts)
        } else {
            self.ddl_name()
        }
    }

    /// `kw`, or a syntax error here with ANTLR's "missing" hint when the
    /// token here could follow it (`follows`).
    fn expect_kw_or_missing(
        &mut self,
        kw: &'static str,
        display: &'static str,
        follows: bool,
    ) -> PR<()> {
        if self.eat_kw(kw) {
            return Ok(());
        }
        if follows {
            self.missing = Some((self.pos, display));
        }
        self.fail()
    }

    /// Whether a name starts here (the token after a missing keyword that
    /// a name follows).
    fn name_here(&self) -> bool {
        matches!(self.kind(0), Some(TokenKind::Word | TokenKind::QuotedIdent))
    }

    /// `[AS] alias`: any identifier but `terminator` (`UPDATE t x SET ...`
    /// names `x`, while `UPDATE t SET ...` has no alias).
    fn stmt_alias(&mut self, terminator: &str) -> PR<Option<crate::ast::Ident>> {
        if self.eat_kw("AS") {
            return self.any_ident().map(Some);
        }
        // Spark's `tableAlias` takes any identifier but the clause keyword
        // that follows (`UPDATE t c1 = 2` names `t` `c1`, then wants
        // `SET`).
        if self.is_ident_at(0, true) && !self.kw_at(0, terminator) {
            return self.any_ident().map(Some);
        }
        Ok(None)
    }

    /// `PARTITION (key [= value], ...)` with the `PARTITION` keyword.
    fn partition_spec(&mut self) -> PR<PartitionSpec> {
        let start = self.start();
        self.expect_kw("PARTITION")?;
        let spec = self.partition_spec_body(start)?;
        Ok(spec)
    }

    /// `(key [= value], ...)` after `PARTITION`.
    fn partition_spec_body(&mut self, start: usize) -> PR<PartitionSpec> {
        self.expect_sym(Sym::LParen)?;
        let parts = self.comma_sep(|p| {
            let start = p.start();
            let key = p.name_part(true)?;
            // `partitionVal: identifier (EQ constant)? | identifier EQ DEFAULT`.
            let value = if p.eat_sym(Sym::Eq) {
                Some(p.constant_or_default()?)
            } else {
                None
            };
            Ok(PartitionSpecPart {
                key,
                value,
                span: p.span_from(start),
            })
        })?;
        self.expect_closing(Sym::RParen, Close::List(parts.len()))?;
        Ok(PartitionSpec {
            parts,
            span: self.span_from(start),
        })
    }

    // ---- ALTER --------------------------------------------------------

    /// `ALTER TABLE ...`, `ALTER VIEW ...` or `ALTER
    /// DATABASE|SCHEMA|NAMESPACE ...`.
    fn alter_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("ALTER")?;
            if p.kw_at(0, "TABLE") {
                p.bump();
                let table = p.ddl_name()?;
                let action = p.alter_table_action(start)?;
                let span = p.span_from(start);
                p.command_end()?;
                Ok(Statement::AlterTable(Box::new(AlterTable {
                    table,
                    action,
                    span,
                })))
            } else if p.kw_at(0, "VIEW") {
                p.bump();
                let name = p.ddl_name()?;
                let action = p.alter_view_action()?;
                let span = p.span_from(start);
                p.command_end()?;
                Ok(Statement::AlterView(Box::new(AlterView {
                    name,
                    action,
                    span,
                })))
            } else if p.any_kw_at(0, &["DATABASE", "SCHEMA", "NAMESPACE"]) {
                let kind = p.namespace_kind()?;
                let name = p.ddl_name()?;
                let action = p.alter_namespace_action()?;
                let span = p.span_from(start);
                p.command_end()?;
                Ok(Statement::AlterNamespace(AlterNamespace {
                    kind,
                    name,
                    action,
                    span,
                }))
            } else {
                p.fail()
            }
        })
    }

    /// The action of an `ALTER TABLE`, after the table name
    /// (`statement_start` is the statement's start, where Spark reports a
    /// bare `CHANGE`).
    fn alter_table_action(&mut self, statement_start: usize) -> PR<AlterTableAction> {
        let start = self.start();
        if self.eat_kw("ADD") {
            // `ADD [COLUMN|COLUMNS] ...` or `ADD [IF NOT EXISTS]
            // PARTITION ...`.
            if self.eat_kw("COLUMN") || self.eat_kw("COLUMNS") {
                return self.alter_add_columns();
            }
            let if_not_exists = self.eat_if_not_exists();
            let mut parts = Vec::new();
            loop {
                let start = self.start();
                self.expect_kw("PARTITION")?;
                let spec = self.partition_spec_body(start)?;
                let location = if self.eat_kw("LOCATION") {
                    Some(self.string_after()?)
                } else {
                    None
                };
                parts.push(PartitionWithLocation {
                    spec,
                    location,
                    span: self.span_from(start),
                });
                if !self.kw_at(0, "PARTITION") {
                    break;
                }
            }
            return Ok(AlterTableAction::AddPartition {
                if_not_exists,
                parts,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("DROP") {
            // `DROP [COLUMN|COLUMNS] ...` or `DROP [IF EXISTS] PARTITION
            // ..., PARTITION ... [PURGE]`.
            if self.eat_kw("COLUMN") || self.eat_kw("COLUMNS") {
                return self.alter_drop_columns();
            }
            let if_exists = self.eat_if_exists();
            let specs = self.comma_sep(Self::partition_spec)?;
            let purge = self.eat_kw("PURGE");
            return Ok(AlterTableAction::DropPartition {
                if_exists,
                specs,
                purge,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("RENAME") {
            // `RENAME TO name` or `RENAME COLUMN from TO to` (`COLUMN` is
            // required, as in Spark's grammar).
            if self.eat_kw("TO") {
                return Ok(AlterTableAction::RenameTo {
                    name: self.rename_target()?,
                    span: self.span_from(start),
                });
            }
            self.expect_kw("COLUMN")?;
            let from = self.column_name()?;
            self.expect_kw("TO")?;
            let to = self.column_name()?;
            return Ok(AlterTableAction::RenameColumn {
                from,
                to,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("CHANGE") {
            // `CHANGE [COLUMN] name ...`: a bare name is
            // `_LEGACY_ERROR_TEMP_0035`, as in Spark.
            self.eat_kw("COLUMN");
            let name = self.column_name()?;
            if self.eat_kw("TYPE") {
                let data_type = self.data_type()?;
                return Ok(AlterTableAction::ChangeColumn {
                    name,
                    action: ChangeColumnAction::Type(data_type),
                    span: self.span_from(start),
                });
            }
            if self.eat_kw("COMMENT") {
                let comment = self.string_after()?;
                return Ok(AlterTableAction::ChangeColumn {
                    name,
                    action: ChangeColumnAction::Comment(comment),
                    span: self.span_from(start),
                });
            }
            if self.eat_kw("FIRST") {
                return Ok(AlterTableAction::ChangeColumn {
                    name,
                    action: ChangeColumnAction::First,
                    span: self.span_from(start),
                });
            }
            if self.eat_kw("AFTER") {
                let column = self.column_name()?;
                return Ok(AlterTableAction::ChangeColumn {
                    name,
                    action: ChangeColumnAction::After(column),
                    span: self.span_from(start),
                });
            }
            if self.kw_at(0, "SET") && self.kw_at(1, "NOT") && self.kw_at(2, "NULL") {
                self.pos += 3;
                return Ok(AlterTableAction::ChangeColumn {
                    name,
                    action: ChangeColumnAction::SetNotNull,
                    span: self.span_from(start),
                });
            }
            if self.kw_at(0, "DROP") && self.kw_at(1, "NOT") && self.kw_at(2, "NULL") {
                self.pos += 3;
                return Ok(AlterTableAction::ChangeColumn {
                    name,
                    action: ChangeColumnAction::DropNotNull,
                    span: self.span_from(start),
                });
            }
            return Self::fatal(error::legacy_error_temp_0035(
                "ALTER TABLE table CHANGE COLUMN requires a TYPE, a SET/DROP, a COMMENT, or a FIRST/AFTER"
                    .to_owned(),
                self.span_from(statement_start),
            ));
        }
        if self.eat_kw("ALTER") {
            // `ALTER [COLUMN] name ...`.
            self.eat_kw("COLUMN");
            let name = self.column_name()?;
            if self.eat_kw("COMMENT") {
                let comment = self.string_after()?;
                return Ok(AlterTableAction::AlterColumn {
                    name,
                    action: AlterColumnAction::Comment(comment),
                    span: self.span_from(start),
                });
            }
            if self.kw_at(0, "SET") && self.kw_at(1, "NOT") && self.kw_at(2, "NULL") {
                self.pos += 3;
                return Ok(AlterTableAction::AlterColumn {
                    name,
                    action: AlterColumnAction::SetNotNull,
                    span: self.span_from(start),
                });
            }
            if self.kw_at(0, "DROP") && self.kw_at(1, "NOT") && self.kw_at(2, "NULL") {
                self.pos += 3;
                return Ok(AlterTableAction::AlterColumn {
                    name,
                    action: AlterColumnAction::DropNotNull,
                    span: self.span_from(start),
                });
            }
            if self.kw_at(0, "SET") && self.kw_at(1, "DEFAULT") {
                self.pos += 2;
                let default = self.expr()?;
                return Ok(AlterTableAction::AlterColumn {
                    name,
                    action: AlterColumnAction::SetDefault(default),
                    span: self.span_from(start),
                });
            }
            if self.kw_at(0, "DROP") && self.kw_at(1, "DEFAULT") {
                self.pos += 2;
                return Ok(AlterTableAction::AlterColumn {
                    name,
                    action: AlterColumnAction::DropDefault,
                    span: self.span_from(start),
                });
            }
            return self.fail();
        }
        if self.eat_kw("SET") {
            if self.eat_kw("TBLPROPERTIES") {
                return Ok(AlterTableAction::SetTblProperties(
                    self.property_list(false)?,
                ));
            }
            if self.eat_kw("SERDEPROPERTIES") {
                return Ok(AlterTableAction::SetSerdeProperties(
                    self.property_list(false)?,
                ));
            }
            if self.eat_kw("LOCATION") {
                return Ok(AlterTableAction::SetLocation(self.string_after()?));
            }
            return self.fail();
        }
        if self.eat_kw("UNSET") {
            self.expect_kw("TBLPROPERTIES")?;
            let if_exists = self.eat_if_exists();
            let keys = self.property_list(false)?;
            return Ok(AlterTableAction::UnsetTblProperties {
                if_exists,
                keys,
                span: self.span_from(start),
            });
        }
        if self.kw_at(0, "PARTITION") {
            // `PARTITION ... RENAME TO PARTITION ...` or `PARTITION ...
            // SET LOCATION '...'`.
            let from = self.partition_spec()?;
            if self.kw_at(0, "RENAME") && self.kw_at(1, "TO") {
                self.pos += 2;
                let to = self.partition_spec()?;
                return Ok(AlterTableAction::RenamePartition {
                    from,
                    to,
                    span: self.span_from(start),
                });
            }
            if !self.eat_kw("SET") {
                return self.fail();
            }
            if self.eat_kw("SERDEPROPERTIES") {
                let properties = self.property_list(false)?;
                return Ok(AlterTableAction::SetPartitionSerdeProperties {
                    partition: from,
                    properties,
                    span: self.span_from(start),
                });
            }
            if self.eat_kw("LOCATION") {
                let location = self.string_after()?;
                return Ok(AlterTableAction::SetPartitionLocation {
                    partition: from,
                    location,
                    span: self.span_from(start),
                });
            }
            return self.fail();
        }
        if self.kw_at(0, "RECOVER") && self.kw_at(1, "PARTITIONS") {
            self.pos += 2;
            return Ok(AlterTableAction::RecoverPartitions);
        }
        self.fail()
    }

    /// `ADD [COLUMN|COLUMNS]` columns: one column or a parenthesized list.
    fn alter_add_columns(&mut self) -> PR<AlterTableAction> {
        if self.sym_at(0, Sym::LParen) {
            Ok(AlterTableAction::AddColumns(
                self.paren_list(Self::table_column)?,
            ))
        } else {
            Ok(AlterTableAction::AddColumns(vec![self.table_column()?]))
        }
    }

    /// `DROP [COLUMN|COLUMNS]` columns: one column or a parenthesized
    /// list.
    fn alter_drop_columns(&mut self) -> PR<AlterTableAction> {
        if self.sym_at(0, Sym::LParen) {
            Ok(AlterTableAction::DropColumns(
                self.paren_list(Self::column_name)?,
            ))
        } else {
            Ok(AlterTableAction::DropColumns(vec![self.column_name()?]))
        }
    }

    /// The action of an `ALTER VIEW`, after the view name.
    fn alter_view_action(&mut self) -> PR<AlterViewAction> {
        let start = self.start();
        if self.eat_kw("AS") {
            let query = Box::new(self.query()?);
            return Ok(AlterViewAction::SetQuery {
                query,
                schema_binding: None,
                span: self.span_from(start),
            });
        }
        if self.kw_at(0, "WITH") && self.kw_at(1, "SCHEMA") {
            let binding = self.schema_binding()?;
            return Ok(AlterViewAction::SetSchemaBinding(binding));
        }
        if self.eat_kw("SET") {
            self.expect_kw("TBLPROPERTIES")?;
            return Ok(AlterViewAction::SetTblProperties(
                self.property_list(false)?,
            ));
        }
        if self.eat_kw("UNSET") {
            self.expect_kw("TBLPROPERTIES")?;
            let if_exists = self.eat_if_exists();
            let keys = self.property_list(false)?;
            return Ok(AlterViewAction::UnsetTblProperties {
                if_exists,
                keys,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("RENAME") {
            self.expect_kw("TO")?;
            return Ok(AlterViewAction::RenameTo {
                name: self.rename_target()?,
                span: self.span_from(start),
            });
        }
        self.fail()
    }

    /// The action of an `ALTER DATABASE|SCHEMA|NAMESPACE`, after the name.
    fn alter_namespace_action(&mut self) -> PR<AlterNamespaceAction> {
        if self.eat_kw("SET") {
            if self.eat_kw("PROPERTIES") {
                return Ok(AlterNamespaceAction::SetProperties(
                    self.property_list(false)?,
                ));
            }
            if self.eat_kw("LOCATION") {
                return Ok(AlterNamespaceAction::SetLocation(self.string_after()?));
            }
            return self.fail();
        }
        if self.kw_at(0, "DEFAULT") && self.kw_at(1, "COLLATION") {
            self.pos += 2;
            let collation = self.table_name()?;
            return Ok(AlterNamespaceAction::DefaultCollation(collation));
        }
        self.fail()
    }

    /// `WITH SCHEMA ...` on a view, at its `WITH`: `BINDING`, `EVOLUTION`,
    /// `COMPENSATION` or `TYPE EVOLUTION`.
    fn schema_binding(&mut self) -> PR<SchemaBindingClause> {
        let start = self.start();
        self.pos += 2;
        let mode = if self.eat_kw("BINDING") {
            SchemaBinding::Binding
        } else if self.eat_kw("EVOLUTION") {
            SchemaBinding::Evolution
        } else if self.eat_kw("COMPENSATION") {
            SchemaBinding::Compensation
        } else if self.kw_at(0, "TYPE") && self.kw_at(1, "EVOLUTION") {
            self.pos += 2;
            SchemaBinding::TypeEvolution
        } else {
            return self.fail();
        };
        Ok(SchemaBindingClause {
            mode,
            span: self.span_from(start),
        })
    }

    // ---- CREATE/DROP VIEW (full grammar) ---------------------------------

    /// `CREATE [OR REPLACE] [GLOBAL] [TEMPORARY] VIEW ...`: every view
    /// shape the legacy temporary-view parser cannot express (a permanent
    /// view, or a temporary view with a qualified or computed name, a
    /// datasource provider, or the `GLOBAL` qualifier). The legacy parser
    /// runs first; this is its fallback.
    pub(crate) fn create_full_view_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            p.expect_kw("CREATE")?;
            let or_replace = if p.kw_at(0, "OR") && p.kw_at(1, "REPLACE") {
                p.pos += 2;
                true
            } else {
                false
            };
            let global = p.eat_kw("GLOBAL");
            let temporary = p.eat_kw("TEMP") || p.eat_kw("TEMPORARY");
            if global && !temporary {
                return p.fail();
            }
            p.expect_kw("VIEW")?;
            let if_not_exists = p.eat_if_not_exists();
            let name = p.ddl_name()?;
            // Spark has two rules: `createTempViewUsing` (temporary, no
            // `IF NOT EXISTS`, typed columns, `USING`) and `createView`
            // (untyped columns, `AS query`). ANTLR picks the first by the
            // first column: a typed one reads a typed list.
            // `createTempViewUsing` names the view with `tableIdentifier`
            // (at most two parts).
            let using_name = match &name {
                DdlName::Parts(n) => n.parts.len() <= 2,
                DdlName::Dynamic(_) => true,
            };
            let typed = temporary && !if_not_exists && using_name && p.view_column_typed();
            let columns = if p.sym_at(0, Sym::LParen) {
                p.paren_list(|p| p.full_view_column(typed, temporary && !if_not_exists))?
            } else {
                Vec::new()
            };
            let using_rule = typed
                || (temporary
                    && !if_not_exists
                    && using_name
                    && columns.is_empty()
                    && p.kw_at(0, "USING"));
            let mut view = CreatePermanentView {
                or_replace,
                global,
                temporary,
                if_not_exists,
                name,
                columns,
                collation: None,
                using: None,
                options: Vec::new(),
                comment: None,
                tblproperties: Vec::new(),
                schema_binding: None,
                partitioned_on: false,
                duplicate_clause: None,
                query: None,
                span: Span::new(start, start),
            };
            if using_rule {
                if !p.kw_at(0, "USING") {
                    p.missing = Some((p.pos, "'USING'"));
                    return p.fail();
                }
                p.bump();
                view.using = Some(p.table_name()?);
                if p.eat_kw("OPTIONS") {
                    view.options = p.property_list(true)?;
                }
            } else {
                p.view_clauses(&mut view)?;
                p.expect_kw("AS")?;
                view.query = Some(Box::new(p.query()?));
            }
            view.span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::CreatePermanentView(Box::new(view)))
        })
    }

    /// Whether the column list here (if any) starts with a typed column:
    /// a name followed by anything but `,`, `)` or `COMMENT`.
    fn view_column_typed(&self) -> bool {
        if !self.sym_at(0, Sym::LParen) {
            return false;
        }
        let after = if self.kw_at(1, "IDENTIFIER") && self.sym_at(2, Sym::LParen) {
            match self.closers.get(self.pos + 2) {
                Some(&close) if close != usize::MAX => close + 1 - self.pos,
                _ => return false,
            }
        } else {
            2
        };
        !(self.sym_at(after, Sym::Comma)
            || self.sym_at(after, Sym::RParen)
            || self.kw_at(after, "COMMENT"))
    }

    /// `createView`'s clauses, in any order: `COMMENT`, `WITH SCHEMA`,
    /// `DEFAULT COLLATION`, `PARTITIONED ON (...)`, `TBLPROPERTIES`. A
    /// clause written twice is recorded for validation (Spark checks after
    /// parsing).
    fn view_clauses(&mut self, view: &mut CreatePermanentView) -> PR<()> {
        // In Spark's check order, with how often each was written.
        let mut seen = [
            ("COMMENT", 0u8),
            ("DEFAULT COLLATION", 0),
            ("WITH SCHEMA", 0),
            ("PARTITIONED ON", 0),
            ("TBLPROPERTIES", 0),
        ];
        loop {
            let clause = if self.eat_kw("COMMENT") {
                view.comment = Some(self.string_after()?);
                0
            } else if self.kw_at(0, "DEFAULT") && self.kw_at(1, "COLLATION") {
                self.pos += 2;
                view.collation = Some(self.table_name()?);
                1
            } else if self.kw_at(0, "WITH") && self.kw_at(1, "SCHEMA") {
                view.schema_binding = Some(self.schema_binding()?);
                2
            } else if self.kw_at(0, "PARTITIONED") && self.kw_at(1, "ON") {
                self.pos += 2;
                self.paren_list(Self::any_ident)?;
                view.partitioned_on = true;
                3
            } else if self.eat_kw("TBLPROPERTIES") {
                view.tblproperties = self.property_list(false)?;
                4
            } else {
                break;
            };
            if let Some(entry) = seen.get_mut(clause) {
                entry.1 = entry.1.saturating_add(1);
            }
        }
        view.duplicate_clause = seen.iter().find(|(_, n)| *n > 1).map(|(c, _)| *c);
        Ok(())
    }

    /// A column of a [`CreatePermanentView`] list: `name type [NOT NULL]
    /// [COMMENT '...']` in a `typed` list (`colTypeList`), else `name
    /// [COMMENT '...']`. A type in an untyped list fails where it stands
    /// (as "extra input" when `extra_hint`: ANTLR's recovery in a
    /// temporary view's list); an untyped column in a typed list fails
    /// after its name.
    fn full_view_column(&mut self, typed: bool, extra_hint: bool) -> PR<PermanentViewColumn> {
        let start = self.start();
        let name = self.column_name()?;
        let ends = |p: &Self| p.sym_at(0, Sym::Comma) || p.sym_at(0, Sym::RParen);
        let (data_type, not_null) = if typed {
            if ends(self) {
                return self.fail();
            }
            let t = self.data_type()?;
            let not_null = if self.kw_at(0, "NOT") && self.kw_at(1, "NULL") {
                self.pos += 2;
                true
            } else {
                false
            };
            (Some(t), not_null)
        } else {
            if !(ends(self) || self.kw_at(0, "COMMENT")) {
                if extra_hint {
                    self.extra = Some(self.pos);
                }
                return self.fail();
            }
            (None, false)
        };
        let comment = if self.eat_kw("COMMENT") {
            Some(self.string_after()?)
        } else {
            None
        };
        Ok(PermanentViewColumn {
            name,
            data_type,
            not_null,
            comment,
            span: self.span_from(start),
        })
    }

    /// `DROP VIEW [IF EXISTS] name` with a qualified or computed name (the
    /// legacy parser takes one plain part; this is its fallback).
    pub(crate) fn drop_full_view_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            p.expect_kw("VIEW")?;
            let if_exists = p.eat_if_exists();
            let name = p.ddl_name()?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::DropPermanentView(DropPermanentView {
                if_exists,
                name,
                span,
            }))
        })
    }

    // ---- SHOW ------------------------------------------------------------

    /// `SHOW ...`.
    fn show_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("SHOW")?;
            let kind = p.show_kind()?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Show(Show { kind, span }))
        })
    }

    /// What a `SHOW` shows, after the `SHOW` keyword.
    fn show_kind(&mut self) -> PR<ShowKind> {
        let start = self.start();
        if self.eat_kw("TABLES") {
            let namespace = self.show_namespace()?;
            let pattern = self.show_pattern_opt()?;
            return Ok(ShowKind::Tables {
                namespace,
                pattern,
                span: self.span_from(start),
            });
        }
        if self.kw_at(0, "TABLE") && self.kw_at(1, "EXTENDED") {
            // `SHOW TABLE EXTENDED` names one table: the pattern is
            // required, as in Spark.
            self.pos += 2;
            let namespace = self.show_namespace()?;
            let pattern = self.show_pattern_req()?;
            let partition = if self.kw_at(0, "PARTITION") {
                Some(self.partition_spec()?)
            } else {
                None
            };
            return Ok(ShowKind::TableExtended {
                namespace,
                pattern,
                partition,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("VIEWS") {
            let namespace = self.show_namespace()?;
            let pattern = self.show_pattern_opt()?;
            return Ok(ShowKind::Views {
                namespace,
                pattern,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("COLUMNS") {
            // `SHOW COLUMNS {FROM | IN} table [{FROM | IN} namespace]`.
            if !(self.eat_kw("FROM") || self.eat_kw("IN")) {
                return self.fail();
            }
            let table = self.ddl_name()?;
            let namespace = self.show_namespace()?;
            return Ok(ShowKind::Columns {
                table,
                namespace,
                span: self.span_from(start),
            });
        }
        if self.kw_at(0, "DATABASES") || self.kw_at(0, "SCHEMAS") || self.kw_at(0, "NAMESPACES") {
            let kind = if self.eat_kw("DATABASES") {
                NamespaceKind::Database
            } else if self.eat_kw("SCHEMAS") {
                NamespaceKind::Schema
            } else {
                self.expect_kw("NAMESPACES")?;
                NamespaceKind::Namespace
            };
            let pattern = self.show_pattern_opt()?;
            return Ok(ShowKind::Namespaces {
                kind,
                pattern,
                span: self.span_from(start),
            });
        }
        if self.kw_at(0, "CREATE") && self.kw_at(1, "TABLE") {
            self.pos += 2;
            let name = self.ddl_name()?;
            let serde = if self.kw_at(0, "AS") && self.kw_at(1, "SERDE") {
                self.pos += 2;
                true
            } else {
                false
            };
            return Ok(ShowKind::CreateTable {
                name,
                serde,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("TBLPROPERTIES") {
            let name = self.ddl_name()?;
            let (key, key_span) = if self.eat_sym(Sym::LParen) {
                let start = self.start();
                let key = if let Some(lit) = self.string_lit() {
                    lit
                } else {
                    self.any_ident()?.value
                };
                let span = self.span_from(start);
                self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
                (Some(key), Some(span))
            } else {
                (None, None)
            };
            return Ok(ShowKind::TblProperties {
                name,
                key,
                key_span,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("PARTITIONS") {
            let name = self.ddl_name()?;
            let partition = if self.kw_at(0, "PARTITION") {
                Some(self.partition_spec()?)
            } else {
                None
            };
            return Ok(ShowKind::Partitions {
                name,
                partition,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("CURRENT") {
            let kind = self.namespace_kind()?;
            return Ok(ShowKind::CurrentNamespace {
                kind,
                span: self.span_from(start),
            });
        }
        self.show_functions()
    }

    /// `SHOW [scope.] [USER|SYSTEM|ALL] FUNCTIONS [LIKE pattern]
    /// [{FROM | IN} namespace]`.
    fn show_functions(&mut self) -> PR<ShowKind> {
        let start = self.start();
        // A computed scope is rejected where it stands, as in Spark.
        if self.kw_at(0, "IDENTIFIER") && self.sym_at(1, Sym::LParen) {
            self.pos += 1;
            return self.fail();
        }
        let mut scope = None;
        if !self.kw_at(0, "FUNCTIONS")
            && !(self.any_kw_at(0, &["USER", "SYSTEM", "ALL"]) && self.kw_at(1, "FUNCTIONS"))
        {
            scope = Some(self.table_name()?);
        }
        // A one-part scope naming a function scope reads as the scope.
        let mut system = None;
        if let Some(name) = &scope
            && name.parts.len() == 1
        {
            let word = name.parts[0].value.to_ascii_uppercase();
            system = match word.as_str() {
                "USER" => Some(FunctionScope::User),
                "SYSTEM" => Some(FunctionScope::System),
                "ALL" => Some(FunctionScope::All),
                _ => None,
            };
            if system.is_some() {
                scope = None;
            }
        }
        if system.is_none() {
            for (word, kind) in [
                ("USER", FunctionScope::User),
                ("SYSTEM", FunctionScope::System),
                ("ALL", FunctionScope::All),
            ] {
                if self.eat_kw(word) {
                    system = Some(kind);
                    break;
                }
            }
        }
        self.expect_kw("FUNCTIONS")?;
        // The namespace reads before the pattern; the pattern is a string
        // or (Spark's legacy form) a dotted name.
        let namespace = self.show_namespace()?;
        let pattern_start = self.start();
        let like = self.eat_kw("LIKE");
        let mut pattern = None;
        let mut legacy_pattern = None;
        if let Some(text) = self.string_lit() {
            pattern = Some(ShowPattern {
                like,
                pattern: text,
                span: self.span_from(pattern_start),
            });
        } else if like || !self.item_end_at(0) {
            legacy_pattern = Some(self.table_name()?);
        }
        Ok(ShowKind::Functions {
            scope,
            system,
            namespace,
            pattern,
            legacy_pattern,
            span: self.span_from(start),
        })
    }

    /// `[{FROM | IN} name]`: the namespace of a `SHOW`.
    fn show_namespace(&mut self) -> PR<Option<DdlName>> {
        if (self.kw_at(0, "FROM") || self.kw_at(0, "IN")) && !self.kw_at(1, "FUNCTIONS") {
            self.bump();
            self.ddl_name().map(Some)
        } else {
            Ok(None)
        }
    }

    /// `[LIKE] 'pattern'`, if one starts here.
    fn show_pattern_opt(&mut self) -> PR<Option<ShowPattern>> {
        let start = self.start();
        let like = self.eat_kw("LIKE");
        if let Some(pattern) = self.string_lit() {
            Ok(Some(ShowPattern {
                like,
                pattern,
                span: self.span_from(start),
            }))
        } else if like {
            self.fail()
        } else {
            Ok(None)
        }
    }

    /// `[LIKE] 'pattern'` (required, as for `SHOW TABLE EXTENDED`).
    fn show_pattern_req(&mut self) -> PR<ShowPattern> {
        let start = self.start();
        let like = self.eat_kw("LIKE");
        match self.string_lit() {
            Some(pattern) => Ok(ShowPattern {
                like,
                pattern,
                span: self.span_from(start),
            }),
            None => self.fail(),
        }
    }

    // ---- DESCRIBE --------------------------------------------------------

    /// `DESCRIBE ...` (`DESC` for `DESCRIBE`).
    fn describe_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            if !(p.eat_kw("DESC") || p.eat_kw("DESCRIBE")) {
                return p.fail();
            }
            let kind = p.describe_kind()?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Describe(Describe { kind, span }))
        })
    }

    /// What a `DESCRIBE` describes, after the `DESCRIBE` keyword.
    fn describe_kind(&mut self) -> PR<DescribeKind> {
        let start = self.start();
        if self.eat_kw("FUNCTION") {
            let extended = self.eat_kw("EXTENDED") || self.eat_kw("FORMATTED");
            let name = self.ddl_name()?;
            return Ok(DescribeKind::Function {
                extended,
                name,
                span: self.span_from(start),
            });
        }
        if self.any_kw_at(0, &["DATABASE", "SCHEMA", "NAMESPACE"]) {
            let kind = self.namespace_kind()?;
            let extended = self.eat_kw("EXTENDED") || self.eat_kw("FORMATTED");
            let name = self.ddl_name()?;
            return Ok(DescribeKind::Namespace {
                kind,
                extended,
                name,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("QUERY") {
            let query_start = self.start();
            let query = Box::new(self.query()?);
            return Ok(DescribeKind::Query {
                query,
                span: self.span_from(query_start),
            });
        }
        if self.eat_kw("TABLE") {
            return self.describe_table();
        }
        if self.query_starts() {
            let query_start = self.start();
            let query = Box::new(self.query()?);
            return Ok(DescribeKind::Query {
                query,
                span: self.span_from(query_start),
            });
        }
        self.describe_table()
    }

    /// `[EXTENDED|FORMATTED] table [PARTITION ...] [column] [AS JSON]`.
    fn describe_table(&mut self) -> PR<DescribeKind> {
        let start = self.start();
        let mode = if self.eat_kw("EXTENDED") {
            Some(DescribeMode::Extended)
        } else if self.eat_kw("FORMATTED") {
            Some(DescribeMode::Formatted)
        } else {
            None
        };
        let table = self.ddl_name()?;
        let partition = if self.kw_at(0, "PARTITION") {
            Some(self.partition_spec()?)
        } else {
            None
        };
        // `AS JSON` reads before any column.
        if self.kw_at(0, "AS") && self.kw_at(1, "JSON") {
            self.pos += 2;
            return Ok(DescribeKind::Table {
                mode,
                table,
                partition,
                column: Vec::new(),
                as_json: true,
                span: self.span_from(start),
            });
        }
        // `describeColName`: a dotted name (a nested field).
        let mut column = Vec::new();
        if !(self.kw_at(0, "JSON") || self.item_end_at(0)) {
            column.push(self.column_name()?);
            while self.eat_sym(Sym::Dot) {
                column.push(self.column_name()?);
            }
        }
        let as_json = if self.kw_at(0, "AS") && self.kw_at(1, "JSON") {
            self.pos += 2;
            true
        } else {
            false
        };
        Ok(DescribeKind::Table {
            mode,
            table,
            partition,
            column,
            as_json,
            span: self.span_from(start),
        })
    }

    // ---- INSERT ----------------------------------------------------------

    /// `INSERT INTO|OVERWRITE ...`.
    fn insert_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("INSERT")?;
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
    fn strip_insert_alias(&mut self, relation: &mut Relation) {
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
    fn strip_query_insert_alias(&mut self, query: &mut Query) {
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
    fn insert_clause(&mut self, clause_start: usize, multi: bool) -> PR<InsertClause> {
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
    fn insert_query(&mut self, multi: bool) -> PR<Query> {
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

    // ---- DELETE / UPDATE / MERGE / TRUNCATE ------------------------------

    /// `DELETE FROM name [AS alias] [WHERE ...]`.
    fn delete_command(&mut self) -> PR<Statement> {
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
    fn update_command(&mut self) -> PR<Statement> {
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
    fn update_assignment(&mut self) -> PR<UpdateAssignment> {
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
    fn merge_command(&mut self) -> PR<Statement> {
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
    fn merge_source(&mut self) -> PR<MergeSource> {
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
    fn merge_clause(&mut self) -> PR<MergeClause> {
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
    fn merge_action(&mut self, matched: MergeMatched) -> PR<MergeAction> {
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

    /// `TRUNCATE TABLE name [PARTITION ...]`.
    fn truncate_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("TRUNCATE")?;
            let follows = p.name_here();
            p.expect_kw_or_missing("TABLE", "'TABLE'", follows)?;
            let table = p.ddl_name()?;
            let partition = if p.kw_at(0, "PARTITION") {
                Some(p.partition_spec()?)
            } else {
                None
            };
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Truncate(Truncate {
                table,
                partition,
                span,
            }))
        })
    }

    // ---- CACHE / REFRESH / ANALYZE / MSCK / ADD / LIST -------------------

    /// `CACHE ...`, `UNCACHE ...` or `CLEAR CACHE`.
    fn cache_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            let kind = if p.eat_kw("CACHE") {
                let kind_start = p.start();
                let lazy = p.eat_kw("LAZY");
                p.expect_kw("TABLE")?;
                let name = p.ddl_name()?;
                let mut options = Vec::new();
                if p.eat_kw("OPTIONS") {
                    options = p.property_list(true)?;
                }
                let query = if p.eat_kw("AS") || p.query_starts() {
                    Some(Box::new(p.query()?))
                } else {
                    None
                };
                CacheKind::Cache {
                    lazy,
                    name,
                    options,
                    query,
                    span: p.span_from(kind_start),
                }
            } else if p.eat_kw("UNCACHE") {
                let kind_start = p.start();
                let follows = p.name_here();
                p.expect_kw_or_missing("TABLE", "'TABLE'", follows)?;
                let if_exists = p.eat_if_exists();
                let name = p.ddl_name()?;
                CacheKind::Uncache {
                    if_exists,
                    name,
                    span: p.span_from(kind_start),
                }
            } else if p.eat_kw("CLEAR") {
                p.expect_kw("CACHE")?;
                CacheKind::Clear
            } else {
                return p.fail();
            };
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Cache(Cache { kind, span }))
        })
    }

    /// `REFRESH ...`.
    fn refresh_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("REFRESH")?;
            let kind_start = p.start();
            // `REFRESH TABLE|FUNCTION name` only when the name ends the
            // statement; anything else is `refreshResource`, which takes
            // the rest of the text (`REFRESH TABLE t ASYNC` is a path).
            let save = p.checkpoint();
            let table = p.eat_kw("TABLE");
            if table || p.eat_kw("FUNCTION") {
                match p.ddl_name() {
                    Ok(name)
                        if (0..)
                            .map_while(|n| p.kind(n))
                            .all(|k| k == TokenKind::Sym(Sym::Semi)) =>
                    {
                        p.commit(save);
                        let span = p.span_from(kind_start);
                        let kind = if table {
                            RefreshKind::Table { name, span }
                        } else {
                            RefreshKind::Function { name, span }
                        };
                        let span = p.span_from(start);
                        p.command_end()?;
                        return Ok(Statement::Refresh(Refresh { kind, span }));
                    }
                    Err(Fail::Fatal(e)) => return Err(Fail::Fatal(e)),
                    _ => p.restore(save),
                }
            } else {
                p.restore(save);
            }
            let kind = {
                // One string literal is the path; anything else is the
                // rest of the statement as an unquoted path, which must be
                // non-empty and hold no whitespace (Spark's
                // `extractUnquotedResourcePath`).
                let string_alone =
                    matches!(p.kind(0), Some(TokenKind::String | TokenKind::DoubleQuoted))
                        && (1..)
                            .map_while(|n| p.kind(n))
                            .all(|k| k == TokenKind::Sym(Sym::Semi));
                let path = if string_alone {
                    p.string_after()?
                } else {
                    let after = p
                        .toks
                        .get(p.pos.wrapping_sub(1))
                        .map_or(0, |t| t.span.range().end);
                    let (end, end_byte) = p.raw_end();
                    let path = p
                        .src
                        .get(after..end_byte)
                        .unwrap_or_default()
                        .trim_matches(|c: char| c <= ' ')
                        .to_owned();
                    p.pos = end;
                    let message = if path.is_empty() {
                        Some(
                            "Resource paths cannot be empty in REFRESH statements. \
                             Use / to match everything",
                        )
                    } else if path.contains([' ', '\n', '\r', '\t']) {
                        Some(
                            "REFRESH statements cannot contain ' ', '\\n', '\\r', '\\t' \
                             inside unquoted resource paths",
                        )
                    } else {
                        None
                    };
                    if let Some(msg) = message {
                        return Self::fatal(ParseError {
                            error: LegacyErrorTemp0064 {
                                msg: msg.to_owned(),
                            }
                            .into(),
                            span: p.span_from(start),
                        });
                    }
                    path
                };
                RefreshKind::Path {
                    path,
                    span: p.span_from(kind_start),
                }
            };
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Refresh(Refresh { kind, span }))
        })
    }

    /// `ANALYZE TABLE name [PARTITION ...] COMPUTE STATISTICS ...`.
    fn analyze_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("ANALYZE")?;
            p.expect_kw("TABLE")?;
            let table = p.ddl_name()?;
            let partition = if p.kw_at(0, "PARTITION") {
                Some(p.partition_spec()?)
            } else {
                None
            };
            p.expect_kw("COMPUTE")?;
            let follows = matches!(p.kind(0), None | Some(TokenKind::Sym(Sym::Semi)))
                || p.any_kw_at(0, &["NOSCAN", "FOR"]);
            p.expect_kw_or_missing("STATISTICS", "'STATISTICS'", follows)?;
            let mut noscan = false;
            let mut columns = AnalyzeColumns::All;
            if p.eat_kw("NOSCAN") {
                noscan = true;
            } else if p.eat_kw("FOR") {
                if p.kw_at(0, "ALL") && p.kw_at(1, "COLUMNS") {
                    p.pos += 2;
                    columns = AnalyzeColumns::ForAll;
                } else {
                    p.expect_kw("COLUMNS")?;
                    let mut names = vec![p.name_part(true)?];
                    while p.eat_sym(Sym::Comma) {
                        names.push(p.name_part(true)?);
                    }
                    columns = AnalyzeColumns::ForColumns(names);
                }
            }
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Analyze(Analyze {
                table,
                partition,
                noscan,
                columns,
                span,
            }))
        })
    }

    /// `MSCK REPAIR TABLE ...` or `REPAIR TABLE ...`.
    fn repair_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            let msck = p.eat_kw("MSCK");
            p.expect_kw("REPAIR")?;
            let follows = p.name_here();
            p.expect_kw_or_missing("TABLE", "'TABLE'", follows)?;
            let table = p.ddl_name()?;
            let action = if p.eat_kw("ADD") {
                p.expect_kw("PARTITIONS")?;
                Some(RepairAction::Add)
            } else if p.eat_kw("DROP") {
                p.expect_kw("PARTITIONS")?;
                Some(RepairAction::Drop)
            } else if p.eat_kw("SYNC") {
                p.expect_kw("PARTITIONS")?;
                Some(RepairAction::Sync)
            } else {
                None
            };
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Repair(Repair {
                msck,
                table,
                action,
                span,
            }))
        })
    }

    /// `ADD|LIST kind paths`: Spark reads the kind as any identifier and
    /// the rest of the statement as raw text, split into quoted and
    /// unquoted paths (`SparkSqlAstBuilder.visitManageResource`).
    fn resource_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            let add = p.eat_kw("ADD");
            if !add && !p.eat_kw("LIST") {
                return p.fail();
            }
            let word = p.any_ident()?;
            let (end, end_byte) = p.raw_end();
            let rest = p
                .src
                .get(word.span.range().end..end_byte)
                .unwrap_or_default();
            let paths = resource_paths(rest);
            p.pos = end;
            let span = p.span_from(start);
            let kind = match word.value.to_ascii_lowercase().as_str() {
                "jar" | "jars" => ResourceType::Jar,
                "file" | "files" => ResourceType::File,
                "archive" | "archives" => ResourceType::Archive,
                other => {
                    let op = if add { "ADD" } else { "LIST" };
                    return Self::fatal(ParseError {
                        error: LegacyErrorTemp0035 {
                            message: format!("{op} with resource type '{other}'"),
                        }
                        .into(),
                        span,
                    });
                }
            };
            if add {
                Ok(Statement::AddResource(AddResource { kind, paths, span }))
            } else {
                Ok(Statement::ListResource(ListResource { kind, paths, span }))
            }
        })
    }
}

/// The paths of an `ADD`/`LIST` resource command's raw argument text, as
/// Spark's `"(".*?[^\\]"|'.*?[^\\]'|[^ \n\r\t"']+)"` finds them in the
/// trimmed text: a quoted path is unescaped, an unquoted one kept as is.
fn resource_paths(raw: &str) -> Vec<String> {
    let s: Vec<char> = raw.trim_matches(|c: char| c <= ' ').chars().collect();
    let is_space = |c: char| matches!(c, ' ' | '\n' | '\r' | '\t');
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(&c) = s.get(i) {
        if c == '"' || c == '\'' {
            // `q.*?[^\\]q`: the first closing quote after at least one
            // character that is not preceded by a backslash; `.` stops at
            // a line break.
            let mut k = i + 1;
            let mut end = None;
            while let Some(&ck) = s.get(k) {
                if ck != '\\' && s.get(k + 1) == Some(&c) {
                    end = Some(k + 1);
                    break;
                }
                if matches!(ck, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}') {
                    break;
                }
                k += 1;
            }
            if let Some(end) = end {
                let quoted: String = s.get(i..=end).unwrap_or_default().iter().collect();
                out.push(crate::lexer::unescape_string(&quoted, true, false));
                i = end + 1;
            } else {
                i += 1;
            }
        } else if is_space(c) {
            i += 1;
        } else {
            let run: String = s
                .get(i..)
                .unwrap_or_default()
                .iter()
                .take_while(|&&ch| !is_space(ch) && ch != '"' && ch != '\'')
                .collect();
            i += run.chars().count();
            out.push(run);
        }
    }
    out
}
