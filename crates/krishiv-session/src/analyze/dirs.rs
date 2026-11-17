// SPDX-License-Identifier: Apache-2.0
//! The directory securables of a statement's run (T063 N1b item 1): a run of `DROP TABLE`, `TRUNCATE TABLE`
//! or `INSERT` on a stored table also asks the policy about the table's directory, which analyze lists
//! after the statement's targets, only where the run would ask:
//!
//! - `DROP TABLE` and `TRUNCATE TABLE` of a managed table: `MODIFY` on its directory, when the directory
//!   exists (`table_commands/files.rs`, `deletable_location`); never for an external table, a view or a
//!   temporary view, and not when the run fails before it asks (a `TRUNCATE` of an external table or of a
//!   partition, a directory that is not inside the warehouse);
//! - `INSERT INTO|OVERWRITE` (each clause, a multi-table insert too): `MODIFY` and `CREATE` on the table's
//!   location, existing or not, as the writer asks them (`Session::authorize_write_path`); not for a
//!   clause the run refuses before it writes (`REPLACE`, `OVERWRITE ... PARTITION`, a bucketed table). A
//!   multi-table insert with a table that cannot be found lists none: the run resolves every target before
//!   any clause writes, so it fails before it asks.
//!
//! Only after the statement's targets pass (`Session::with_directories`, review B2): the run checks them
//! before it looks anything up, so analyze reads no catalog entry and no directory of a target the caller
//! may not write, and lists nothing more then, exactly as for a missing table.
//!
//! The list is what the run asks when it reaches its write: analyze cannot foresee a later refusal (an
//! arity or assignment error, `UNSUPPORTED_OVERWRITE.TABLE`, partition values that do not cast), after
//! which the run asks nothing more.
//!
//! The path is the text the run's denial names (`locate_write_text`, `access::lexical_dir`): the location
//! as written, qualified (`file:` for a local one, the store URL for an object-store table), with `.` and
//! `..` folded in the text. The policy itself is asked about a local path with the links of its existing
//! part resolved; without a link on it, the two are the same text. Listing it takes file-system I/O (for
//! `DROP` and `TRUNCATE`, whether the directory exists), so the list reflects the directory as it is at
//! analysis.
//!
//! A table that cannot be found lists nothing here: the run fails before it asks.
//!
//! Not listed yet (a run asks the policy about a directory, analyze does not list it; the property test
//! `securables_equal_what_a_run_authorizes` pins each difference):
//! - `CREATE TABLE` of a managed table: `CREATE` on its new directory (`table_commands/create.rs`,
//!   `authorize_location`);
//! - `CREATE TABLE ... AS SELECT` of a managed table: `MODIFY` and `CREATE` on its new directory;
//! - `CREATE OR REPLACE TABLE` / `REPLACE TABLE` of a managed table: `MODIFY` on the directory it replaces
//!   (`deletable_dir`);
//! - `ALTER TABLE ... RENAME TO` of a managed table: `MODIFY` on its directory and `CREATE` on the new one;
//! - `DROP NAMESPACE ... CASCADE`: `MODIFY` on the directory of every managed table in it;
//! - partition DDL (`ADD`/`DROP`/`RENAME PARTITION`, `SET LOCATION`, `MSCK REPAIR`): the partition
//!   directories (`partitions.rs`).

use std::collections::{BTreeMap, BTreeSet};

use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{Privilege, Securable, SecurableUse};
use krishiv_sql::ast::{DdlName, Statement};

use crate::session::{QueryEnv, Session};
use crate::store::CatalogTable;

impl Session {
    /// The directory securables a run of `statement` asks the policy about (see the module doc), in the
    /// order the run asks them. A table, path or directory this cannot place lists nothing: the run fails
    /// before it asks.
    pub(crate) fn directory_securables(
        &self,
        statement: &Statement,
        env: &QueryEnv,
    ) -> Vec<SecurableUse> {
        let mut out = Vec::new();
        match statement {
            Statement::DropTable(d) => {
                if let Some(table) = self.stored_table(&d.name, env, true) {
                    self.push_deletable(&table, &mut out);
                }
            }
            Statement::Truncate(t) if t.partition.is_none() => {
                if let Some(table) = self.stored_table(&t.table, env, false) {
                    self.push_deletable(&table, &mut out);
                }
            }
            Statement::Insert(i) => {
                // The run resolves every clause's table before any clause writes (review m4).
                let mut tables = Vec::with_capacity(i.clauses.len());
                for c in &i.clauses {
                    let Some(table) = self.stored_table(&c.name, env, false) else {
                        return Vec::new();
                    };
                    tables.push(table);
                }
                for (c, table) in i.clauses.iter().zip(&tables) {
                    let refused = c.replace_where.is_some()
                        || c.replace_on.is_some()
                        || c.replace_using.is_some()
                        || (c.partition.is_some() && (c.overwrite || c.if_not_exists));
                    if !refused && table.layout.bucket.is_none() {
                        self.push_location(table, env, &mut out);
                    }
                }
            }
            _ => {}
        }
        out
    }

    /// The stored table `name` resolves to for a write (not a temporary view, not a view), or `None`.
    /// With `drop`, a bare name that a temporary view shadows is that view (`DROP TABLE` drops it).
    fn stored_table(
        &self,
        name: &DdlName,
        env: &QueryEnv,
        drop: bool,
    ) -> Option<std::sync::Arc<CatalogTable>> {
        let name = self.table_name(name, env).ok()?;
        if drop && self.temp_entry_of(&name, env) {
            return None;
        }
        self.table_for_write(&name, "ANALYZE", env).ok()
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

/// Whether `statement` is one whose run may ask about a directory ([`Session::directory_securables`]):
/// only these get the quiet trial of their targets.
pub(super) const fn may_list(statement: &Statement) -> bool {
    matches!(
        statement,
        Statement::DropTable(_) | Statement::Truncate(_) | Statement::Insert(_)
    )
}

/// Adds each of `privileges` on the path `display` to `out`, once.
fn push(out: &mut Vec<SecurableUse>, display: &str, privileges: &[Privilege]) {
    let securable = Securable::Path(OptionValue::from_user("path", display.to_owned()));
    for privilege in privileges {
        if !out
            .iter()
            .any(|u| u.securable == securable && u.privilege == *privilege)
        {
            out.push(SecurableUse {
                securable: securable.clone(),
                privilege: *privilege,
                columns: BTreeSet::new(),
            });
        }
    }
}
