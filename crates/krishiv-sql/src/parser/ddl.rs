// SPDX-License-Identifier: Apache-2.0
//! Command statement parsers: `ALTER`, `SHOW`, `DESCRIBE`, `INSERT`,
//! `DELETE`, `UPDATE`, `MERGE`, `TRUNCATE`, `CACHE`, `REFRESH`, `ANALYZE`,
//! `MSCK`/`REPAIR`, `ADD`/`LIST`, and the `CREATE`/`DROP VIEW` shapes the
//! legacy temporary-view parsers cannot express.
//!
//! Parsing only: every tree here is syntax (names resolved and literals
//! checked in [`crate::validate`], like the rest), and lowering or
//! execution answers `KRISHIV_UNSUPPORTED_FEATURE` naming the statement.
//! Errors Spark raises while parsing (`PARSE_SYNTAX_ERROR`, the
//! `DESCRIBE` classes, `_LEGACY_ERROR_TEMP_0035`, ...) are raised here or
//! in validation with Spark's class and SQLSTATE.

use super::{Close, Fail, PR, Parser};
use crate::ast::{
    AddResource, AlterColumnAction, AlterNamespace, AlterNamespaceAction, AlterTable,
    AlterTableAction, AlterView, AlterViewAction, Analyze, AnalyzeColumns, Cache, CacheKind,
    ChangeColumnAction, CreatePermanentView, DdlName, Delete, Describe, DescribeKind, DescribeMode,
    DropPermanentView, FunctionScope, Insert, InsertClause, ListResource, Merge, MergeAction,
    MergeClause, MergeInsert, MergeMatched, MergeSource, NamespaceKind, PartitionSpec,
    PartitionSpecPart, PartitionWithLocation, PermanentViewColumn, Query, Refresh, RefreshKind,
    Relation, RelationExt, Repair, RepairAction, ResourceType, SchemaBinding, SetExprKind, Show,
    ShowKind, ShowPattern, Statement, Truncate, Update, UpdateAssignment,
};
use crate::error;
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
        } else if self.kw_at(0, "CACHE")
            || self.kw_at(0, "UNCACHE")
            || (self.kw_at(0, "CLEAR") && self.kw_at(1, "CACHE"))
        {
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

    /// `[AS] alias` where a bare alias reads only before `terminator`
    /// (`UPDATE t x SET ...` names `x`, while `UPDATE t SET ...` has no
    /// alias, as `SET` alone is no identifier position).
    fn stmt_alias(&mut self, terminator: &str) -> PR<Option<crate::ast::Ident>> {
        if self.eat_kw("AS") {
            return self.any_ident().map(Some);
        }
        if self.is_ident_at(0, true)
            && (self.kw_at(1, terminator)
                || matches!(self.kind(1), None | Some(TokenKind::Sym(Sym::Semi))))
        {
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
            let value = if p.eat_sym(Sym::Eq) {
                Some(p.expr()?)
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
            // ... [PURGE]`.
            if self.eat_kw("COLUMN") || self.eat_kw("COLUMNS") {
                return self.alter_drop_columns();
            }
            let if_exists = self.eat_if_exists();
            let mut specs = Vec::new();
            loop {
                specs.push(self.partition_spec()?);
                if !self.kw_at(0, "PARTITION") {
                    break;
                }
            }
            let purge = self.eat_kw("PURGE");
            return Ok(AlterTableAction::DropPartition {
                if_exists,
                specs,
                purge,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("RENAME") {
            // `RENAME TO name` or `RENAME [COLUMN] from TO to`.
            if self.eat_kw("TO") {
                return Ok(AlterTableAction::RenameTo {
                    name: self.rename_target()?,
                    span: self.span_from(start),
                });
            }
            self.eat_kw("COLUMN");
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
            if self.kw_at(0, "SET") && self.kw_at(1, "LOCATION") {
                self.pos += 2;
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
            self.pos += 2;
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

    /// `WITH SCHEMA ...` on a view: `BINDING`, `EVOLUTION`,
    /// `COMPENSATION` or `TYPE EVOLUTION`.
    fn schema_binding(&mut self) -> PR<SchemaBinding> {
        if self.eat_kw("BINDING") {
            Ok(SchemaBinding::Binding)
        } else if self.eat_kw("EVOLUTION") {
            Ok(SchemaBinding::Evolution)
        } else if self.eat_kw("COMPENSATION") {
            Ok(SchemaBinding::Compensation)
        } else if self.kw_at(0, "TYPE") && self.kw_at(1, "EVOLUTION") {
            self.pos += 2;
            Ok(SchemaBinding::TypeEvolution)
        } else {
            self.fail()
        }
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
            let columns = if p.sym_at(0, Sym::LParen) {
                p.paren_list(Self::full_view_column)?
            } else {
                Vec::new()
            };
            let collation = if p.kw_at(0, "DEFAULT") && p.kw_at(1, "COLLATION") {
                p.pos += 2;
                Some(p.table_name()?)
            } else {
                None
            };
            let using = if p.eat_kw("USING") {
                Some(p.table_name()?)
            } else {
                None
            };
            let mut options = Vec::new();
            if p.eat_kw("OPTIONS") {
                options = p.property_list(true)?;
            }
            let comment = if p.eat_kw("COMMENT") {
                Some(p.string_after()?)
            } else {
                None
            };
            let mut tblproperties = Vec::new();
            if p.eat_kw("TBLPROPERTIES") {
                tblproperties = p.property_list(false)?;
            }
            let schema_binding = if p.kw_at(0, "WITH") && p.kw_at(1, "SCHEMA") {
                p.pos += 2;
                Some(p.schema_binding()?)
            } else {
                None
            };
            let query = if p.eat_kw("AS") {
                Some(Box::new(p.query()?))
            } else {
                None
            };
            if query.is_none() && using.is_none() {
                return p.fail();
            }
            if !temporary && using.is_some() {
                // A datasource view is always temporary, as in Spark.
                return p.fail();
            }
            let span = p.span_from(start);
            if temporary {
                if let DdlName::Parts(name) = &name
                    && name.parts.len() > 1
                {
                    let object_name: Vec<String> = name
                        .parts
                        .iter()
                        .map(|q| super::quote_ident(&q.value))
                        .collect();
                    let qualifier: Vec<String> = name.parts[..name.parts.len() - 1]
                        .iter()
                        .map(|q| super::quote_ident(&q.value))
                        .collect();
                    return Self::fatal(error::invalid_temp_obj_qualifier(
                        "view",
                        object_name.join("."),
                        qualifier.join("."),
                        span,
                    ));
                }
                if if_not_exists && or_replace {
                    return Self::fatal(error::create_view_with_if_not_exists_and_replace(
                        super::quote_ident(&view_name_label(&name, p.src)),
                        span,
                    ));
                }
                if if_not_exists {
                    return Self::fatal(error::temp_view_if_not_exists(span));
                }
            }
            p.command_end()?;
            Ok(Statement::CreatePermanentView(Box::new(
                CreatePermanentView {
                    or_replace,
                    global,
                    temporary,
                    if_not_exists,
                    name,
                    columns,
                    collation,
                    using,
                    options,
                    comment,
                    tblproperties,
                    schema_binding,
                    query,
                    span,
                },
            )))
        })
    }

    /// `name [type] [COMMENT '...']` in a [`CreatePermanentView`]
    /// column list.
    fn full_view_column(&mut self) -> PR<PermanentViewColumn> {
        let start = self.start();
        let name = self.column_name()?;
        // A type reads only for datasource (`USING`) views; the check
        // runs in validation, where the whole statement is known.
        let data_type = if self.kw_at(0, "COMMENT")
            || self.sym_at(0, Sym::Comma)
            || self.sym_at(0, Sym::RParen)
        {
            None
        } else {
            Some(self.data_type()?)
        };
        let comment = if self.eat_kw("COMMENT") {
            Some(self.string_after()?)
        } else {
            None
        };
        Ok(PermanentViewColumn {
            name,
            data_type,
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
            // `SHOW COLUMNS [{FROM | IN} table] [{FROM | IN} namespace]`.
            let table = self.show_namespace()?;
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
        let pattern = self.show_pattern_opt()?;
        let namespace = self.show_namespace()?;
        Ok(ShowKind::Functions {
            scope,
            system,
            pattern,
            namespace,
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
                column: None,
                as_json: true,
                span: self.span_from(start),
            });
        }
        let column = if self.kw_at(0, "JSON") || self.item_end_at(0) {
            None
        } else {
            Some(self.column_name()?)
        };
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
            let clause = p.insert_clause(start)?;
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
            loop {
                let clause_start = p.start();
                p.expect_kw("INSERT")?;
                clauses.push(p.insert_clause(clause_start)?);
                if !p.kw_at(0, "INSERT") {
                    break;
                }
            }
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

    /// `INTO|OVERWRITE [TABLE] name [(columns)] [PARTITION ...] [REPLACE
    /// WHERE ...] [BY NAME] query`, after the `INSERT` at `clause_start`.
    /// A `(` opening a query reads as the query (it consumes its own
    /// parens), anything else as a column list.
    fn insert_clause(&mut self, clause_start: usize) -> PR<InsertClause> {
        let overwrite = if self.eat_kw("OVERWRITE") {
            true
        } else {
            self.expect_kw("INTO")?;
            false
        };
        self.eat_kw("TABLE");
        let name = self.ddl_name()?;
        let (columns, paren_query) = if self.sym_at(0, Sym::LParen) && self.query_follows(1) {
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
        let partition = if self.kw_at(0, "PARTITION") {
            Some(self.partition_spec()?)
        } else {
            None
        };
        // `REPLACE WHERE ...` and `BY NAME`, in either order.
        let mut replace_where = None;
        let mut by_name = false;
        loop {
            if self.kw_at(0, "REPLACE") && self.kw_at(1, "WHERE") {
                self.pos += 2;
                replace_where = Some(self.expr()?);
            } else if self.kw_at(0, "BY") && self.kw_at(1, "NAME") {
                self.pos += 2;
                by_name = true;
            } else {
                break;
            }
        }
        let query = match paren_query {
            Some(query) => query,
            None => {
                let mut query = self.query()?;
                self.strip_query_insert_alias(&mut query);
                query
            }
        };
        Ok(InsertClause {
            overwrite,
            name,
            columns,
            partition,
            replace_where,
            by_name,
            query: Box::new(query),
            span: self.span_from(clause_start),
        })
    }

    // ---- DELETE / UPDATE / MERGE / TRUNCATE ------------------------------

    /// `DELETE FROM name [AS alias] [WHERE ...]`.
    fn delete_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("DELETE")?;
            p.expect_kw("FROM")?;
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
            if clauses.is_empty() {
                return p.fail();
            }
            let span = p.span_from(start);
            p.command_end()?;
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
        let action = self.merge_action()?;
        Ok(MergeClause {
            matched,
            condition,
            action,
            span: self.span_from(start),
        })
    }

    /// The action of a `WHEN ... THEN ...`. Any action reads after any
    /// `WHEN` here.
    fn merge_action(&mut self) -> PR<MergeAction> {
        let start = self.start();
        if self.kw_at(0, "UPDATE") && self.kw_at(1, "SET") {
            self.pos += 2;
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
            // `INSERT *` or `INSERT [(columns)] VALUES (...)`.
            if self.eat_sym(Sym::Star) {
                return Ok(MergeAction::Insert(MergeInsert::Star));
            }
            let columns = if self.sym_at(0, Sym::LParen) {
                self.paren_list(Self::any_ident)?
            } else {
                Vec::new()
            };
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
            p.expect_kw("TABLE")?;
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
                p.expect_kw("TABLE")?;
                let name = p.ddl_name()?;
                CacheKind::Uncache {
                    name,
                    span: p.span_from(kind_start),
                }
            } else if p.kw_at(0, "CLEAR") && p.kw_at(1, "CACHE") {
                p.pos += 2;
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
            let kind = if p.eat_kw("TABLE") {
                let name = p.ddl_name()?;
                let async_refresh = p.eat_kw("ASYNC");
                RefreshKind::Table {
                    name,
                    async_refresh,
                    span: p.span_from(kind_start),
                }
            } else if p.eat_kw("FUNCTION") {
                let name = p.ddl_name()?;
                RefreshKind::Function {
                    name,
                    span: p.span_from(kind_start),
                }
            } else if p.eat_kw("RESOURCE") {
                let path = p.string_after()?;
                RefreshKind::Resource {
                    path,
                    span: p.span_from(kind_start),
                }
            } else {
                let path = p.string_after()?;
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
            p.expect_kw("STATISTICS")?;
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
            p.expect_kw("TABLE")?;
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

    /// `ADD JAR|FILE|ARCHIVE ...` or `LIST JAR|FILE|ARCHIVE [...]`.
    fn resource_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            let add = p.eat_kw("ADD");
            if !add && !p.eat_kw("LIST") {
                return p.fail();
            }
            let kind = if p.eat_kw("JAR") {
                ResourceType::Jar
            } else if p.eat_kw("FILE") {
                ResourceType::File
            } else if p.eat_kw("ARCHIVE") {
                ResourceType::Archive
            } else {
                return p.fail();
            };
            if add {
                let path = p.string_after()?;
                let span = p.span_from(start);
                p.command_end()?;
                Ok(Statement::AddResource(AddResource { kind, path, span }))
            } else {
                let path = p.string_lit();
                let span = p.span_from(start);
                p.command_end()?;
                Ok(Statement::ListResource(ListResource { kind, path, span }))
            }
        })
    }
}

/// The name of a [`DdlName`], for error parameters: the dotted parts, or
/// the text of a computed name.
fn view_name_label(name: &DdlName, src: &str) -> String {
    match name {
        DdlName::Parts(name) => name
            .parts
            .iter()
            .map(|p| p.value.clone())
            .collect::<Vec<_>>()
            .join("."),
        DdlName::Dynamic(name) => name.span.text(src).to_owned(),
    }
}
