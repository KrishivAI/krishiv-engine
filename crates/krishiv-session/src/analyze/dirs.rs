// SPDX-License-Identifier: Apache-2.0
//! The directory securables of a statement's run (T063 N1b item 1, review M1): a run of `DROP TABLE`,
//! `TRUNCATE TABLE`, `INSERT` and of the DDL below also asks the policy about directories, which analyze
//! lists after the statement's targets, in the run's order, only where the run would ask:
//!
//! - `DROP TABLE` and `TRUNCATE TABLE` of a managed table: `MODIFY` on its directory, when the directory
//!   exists (`table_commands/files.rs`, `deletable_location`); never for an external table, a view or a
//!   temporary view, and not when the run fails before it asks (a `TRUNCATE` of an external table or of a
//!   partition, a directory that is not inside the warehouse);
//! - `INSERT INTO|OVERWRITE` (each clause, a multi-table insert too): `MODIFY` and `CREATE` on the table's
//!   location, existing or not, as the writer asks them (`Session::authorize_write_path`); not for a
//!   clause the run refuses before it writes (`REPLACE`, `OVERWRITE ... PARTITION`). A
//!   multi-table insert with a table that cannot be found lists none: the run resolves every target before
//!   any clause writes, so it fails before it asks. A Delta table's clause lists nothing: its writer
//!   commits to the log the table's own `MODIFY` covers and asks about no directory (`insert_delta`);
//!   each statement finds its table with the lookup its run uses, so a `DROP TABLE` of a managed Delta
//!   table lists its directory and a `TRUNCATE TABLE` of one (refused by name) lists none;
//! - `CREATE TABLE` (`table_commands/questions.rs`): `MANAGE` on the cluster first when its `OPTIONS` hold
//!   a server-scoped store key or the session changed one (`Session::authorize_store_options`, review D2),
//!   then `CREATE` on the new table's location as `add_table` asks it (`location_asked`: a managed table's directory, a `LOCATION` or `path` option, a relative one
//!   qualified against the working directory, an object store's URL); a managed CTAS asks `MODIFY` and
//!   `CREATE` on its directory first, then its write asks both on the location as the writer locates it
//!   (a bucketed table's too); nothing when the table exists (`IF NOT EXISTS`
//!   too), when a managed table's directory is already there, or for a refusal before (`OR REPLACE` and
//!   `REPLACE TABLE`, a missing namespace, a managed table without columns, a clause Krishiv refuses);
//! - `ALTER TABLE ... RENAME TO` of a managed table: `MODIFY` on its directory (when it exists) and `CREATE`
//!   on the new one; nothing for an external table, a taken name or another namespace;
//! - `DROP NAMESPACE ... CASCADE`: `MODIFY` on the directory of each managed table in it that exists, in
//!   the run's order (up to one the run cannot delete, where it stops); nothing without `CASCADE`. The
//!   list names the namespace's tables, so it is given only to a caller who may list them: `BROWSE` on
//!   the namespace, as `SHOW TABLES IN` asks it, tried as quietly as the targets
//!   (`Session::listing_allowed_quietly`, review D1); without it nothing is listed, whatever the
//!   namespace holds. The run asks only `DROP`, so a policy that grants `DROP` without `BROWSE` sees a
//!   run's 42501 name at most the first directory it refuses.
//!
//! A partition's `SET LOCATION` changes the catalog only: its run asks no directory beyond its target.
//!
//! Only after the statement's targets pass (`Session::with_directories`, review B2): the run checks them
//! before it looks anything up, so analyze reads no catalog entry and no directory of a target the caller
//! may not write, and lists nothing more then, exactly as for a missing table. Past them, what is listed
//! tells the caller what the run would (whether a table exists in a namespace it may create in, the
//! managed tables of a namespace it may drop, as their directories' names).
//!
//! The list is what the run asks when it reaches its questions: analyze cannot foresee a later refusal (an
//! arity or assignment error, `UNSUPPORTED_OVERWRITE.TABLE`, partition values that do not cast, a format's
//! refusal of the write), nor what needs the files read (the schema of a `CREATE TABLE` without columns):
//! the questions are listed as if they pass.
//!
//! The path is the text the run's denial names (`locate_write_text`, `access::lexical_dir`): the location
//! as written, qualified (`file:` for a local one, the store URL for an object-store table), with `.` and
//! `..` folded in the text. The policy itself is asked about a local path with the links of its existing
//! part resolved; without a link on it, the two are the same text. Listing it takes file-system I/O
//! (whether the directory exists), so the list reflects the directory as it is at analysis. Analysis makes
//! no directory: the runtime's own warehouse directory (no `spark.sql.warehouse.dir` set) is looked up by
//! the name it will have (`Session::warehouse_path`, review D3), not made.
//!
//! A table that cannot be found lists nothing here: the run fails before it asks.
//!
//! Not listed yet (a run asks the policy about a directory, analyze does not list it; the property test
//! `securables_equal_what_a_run_authorizes` pins each difference): the partition directories of `ADD`,
//! `DROP` and `RENAME PARTITION` and `MSCK REPAIR TABLE` (`partitions.rs`). Which directory a partition
//! spec names takes its values, which the run computes by running each value's cast as a query
//! (`written_spec`, `typed_partition_spec`); analyze runs no query.

use std::collections::{BTreeMap, BTreeSet};

use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{Privilege, Securable, SecurableUse};
use krishiv_sql::ast::{AlterTableAction, DdlName, DropBehavior, Statement};

use crate::session::{QueryEnv, Session};
use crate::store::CatalogTable;

impl Session {
    /// The directory securables a run of `statement` asks the policy about (see the module doc), in the
    /// order the run asks them. A table, path or directory this cannot place lists nothing: the run fails
    /// before it asks.
    pub(crate) fn directory_securables(
        &self,
        parts: &crate::access::StatementParts,
        env: &QueryEnv,
    ) -> Vec<SecurableUse> {
        let mut out = Vec::new();
        let asked = |out: &mut Vec<SecurableUse>, questions: Vec<(Securable, Privilege)>| {
            for (securable, privilege) in questions {
                push_securable(out, &securable, &[privilege]);
            }
        };
        match &parts.statement {
            Statement::CreateTable(c) => {
                let query = parts.analyzed.first().map(|a| &**a);
                asked(
                    &mut out,
                    self.create_table_questions(c, &parts.text, query, env),
                );
            }
            Statement::AlterTable(a) => {
                if let AlterTableAction::RenameTo { name, .. } = &a.action {
                    asked(&mut out, self.rename_questions(a, name, env));
                }
            }
            Statement::DropNamespace(n) => {
                asked(&mut out, self.drop_namespace_questions(n, env));
            }
            Statement::DropTable(d) => {
                if let Some(table) = self.stored_table(&d.name, env, Lookup::Drop) {
                    self.push_deletable(&table, &mut out);
                }
            }
            Statement::Truncate(t) if t.partition.is_none() => {
                if let Some(table) =
                    self.stored_table(&t.table, env, Lookup::Write("TRUNCATE TABLE"))
                {
                    self.push_deletable(&table, &mut out);
                }
            }
            Statement::Insert(i) => {
                // The run resolves every clause's table before any clause writes (review m4).
                let mut tables = Vec::with_capacity(i.clauses.len());
                for c in &i.clauses {
                    let Some(table) = self.stored_table(&c.name, env, Lookup::Write("INSERT"))
                    else {
                        return Vec::new();
                    };
                    tables.push(table);
                }
                for (c, table) in i.clauses.iter().zip(&tables) {
                    let refused = c.replace_where.is_some()
                        || c.replace_on.is_some()
                        || c.replace_using.is_some()
                        || (c.partition.is_some() && (c.overwrite || c.if_not_exists));
                    #[cfg(feature = "delta")]
                    let delta = crate::table_commands::is_delta_table(table);
                    #[cfg(not(feature = "delta"))]
                    let delta = false;
                    if !refused && !delta {
                        self.push_location(table, env, &mut out);
                    }
                }
            }
            _ => {}
        }
        out
    }

    /// The stored table `name` resolves to for `lookup` (not a temporary view, not a view), as the
    /// run finds it, or `None` where the run fails first.
    fn stored_table(
        &self,
        name: &DdlName,
        env: &QueryEnv,
        lookup: Lookup,
    ) -> Option<std::sync::Arc<CatalogTable>> {
        let name = self.table_name(name, env).ok()?;
        match lookup {
            // A bare name that a temporary view shadows is that view (`DROP TABLE` drops it); the run
            // drops any stored table, a Delta one too.
            Lookup::Drop if self.temp_entry_of(&name, env) => None,
            Lookup::Drop => self.table_for_read_command(&name, "DROP TABLE", env).ok(),
            Lookup::Write(operation) => self.table_for_write(&name, operation, env).ok(),
        }
    }

    /// `MODIFY` on the directory a `DROP TABLE` or `TRUNCATE TABLE` of `table` deletes, if the run asks it.
    fn push_deletable(&self, table: &CatalogTable, out: &mut Vec<SecurableUse>) {
        let Ok(Some((location, _))) = self.deletable_location(table) else {
            return;
        };
        let Some(text) = location.to_str() else {
            return;
        };
        if let Ok(at) = crate::access::lexical_dir(std::path::Path::new(text)) {
            push(out, &at.display, &[Privilege::Modify]);
        }
    }

    /// `MODIFY` and `CREATE` on the location an `INSERT` into `table` writes, as the writer locates it.
    fn push_location(&self, table: &CatalogTable, env: &QueryEnv, out: &mut Vec<SecurableUse>) {
        let options: BTreeMap<String, OptionValue> = table
            .options
            .iter()
            .map(|(k, v)| (k.clone(), OptionValue::from_user(k, v)))
            .collect();
        let text = table.location.display().to_string();
        if let Ok(at) = self.locate_write_text(&text, &options, env) {
            push(out, &at.display, &[Privilege::Modify, Privilege::Create]);
        }
    }
}

impl Session {
    /// Whether the caller may see what a listing shows, when `statement`'s directory list names it
    /// (review D1): `BROWSE` on the namespace of a `DROP NAMESPACE ... CASCADE` (whose list names the
    /// namespace's tables), asked as `SHOW TABLES IN` asks it, without an audit record, a denial
    /// count or a call of the policy's `denied` hook (review R2-1). Every other statement lists
    /// only what its targets already allow.
    pub(super) async fn listing_allowed_quietly(
        &self,
        statement: &Statement,
        env: &QueryEnv,
    ) -> bool {
        match statement {
            Statement::DropNamespace(n) => {
                self.allowed_quietly(
                    krishiv_sql::TargetObject::Namespace(&n.name),
                    Privilege::Browse,
                    env,
                )
                .await
            }
            _ => true,
        }
    }
}

/// Whether `statement` is one whose run may ask about a directory ([`Session::directory_securables`]):
/// only these get the quiet trial of their targets.
pub(super) const fn may_list(statement: &Statement) -> bool {
    match statement {
        Statement::DropTable(_) | Statement::Truncate(_) | Statement::Insert(_) => true,
        Statement::CreateTable(c) => !c.temporary,
        Statement::AlterTable(a) => matches!(a.action, AlterTableAction::RenameTo { .. }),
        Statement::DropNamespace(n) => matches!(n.behavior, Some(DropBehavior::Cascade)),
        _ => false,
    }
}

/// How a run finds the stored table it acts on.
#[derive(Clone, Copy)]
enum Lookup {
    /// `DROP TABLE`: any stored table.
    Drop,
    /// A write by this operation name (`Session::table_for_write`: a Delta table only for `INSERT`).
    Write(&'static str),
}

/// Adds each of `privileges` on the path `display` to `out`, once.
fn push(out: &mut Vec<SecurableUse>, display: &str, privileges: &[Privilege]) {
    let securable = Securable::Path(OptionValue::from_user("path", display.to_owned()));
    push_securable(out, &securable, privileges);
}

/// Adds each of `privileges` on `securable` to `out`, once.
fn push_securable(out: &mut Vec<SecurableUse>, securable: &Securable, privileges: &[Privilege]) {
    for privilege in privileges {
        if !out
            .iter()
            .any(|u| u.securable == *securable && u.privilege == *privilege)
        {
            out.push(SecurableUse {
                securable: securable.clone(),
                privilege: *privilege,
                columns: BTreeSet::new(),
            });
        }
    }
}
