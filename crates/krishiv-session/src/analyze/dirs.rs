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
//!   clause the run refuses before it writes (`REPLACE`, `OVERWRITE ... PARTITION`, a bucketed table).
//!
//! The path is the `file:` form the run's denial names: the location as written, qualified, with `.` and
//! `..` folded in the text. The policy itself is asked about the same path with the links of its existing
//! part resolved; without a link on it, the two are the same text. Listing it takes file-system I/O (for
//! `DROP` and `TRUNCATE`, whether the directory exists), so the list reflects the directory as it is at
//! analysis.
//!
//! A table that cannot be found lists nothing here: the run fails before it asks.

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
    pub(crate) fn directory_securables(&self, statement: &Statement) -> Vec<SecurableUse> {
        let env = self.query_env();
        let mut out = Vec::new();
        match statement {
            Statement::DropTable(d) => {
                if let Some(table) = self.stored_table(&d.name, &env, true) {
                    self.push_deletable(&table, &mut out);
                }
            }
            Statement::Truncate(t) if t.partition.is_none() => {
                if let Some(table) = self.stored_table(&t.table, &env, false) {
                    self.push_deletable(&table, &mut out);
                }
            }
            Statement::Insert(i) => {
                for c in &i.clauses {
                    let refused = c.replace_where.is_some()
                        || c.replace_on.is_some()
                        || c.replace_using.is_some()
                        || (c.partition.is_some() && (c.overwrite || c.if_not_exists));
                    if refused {
                        continue;
                    }
                    if let Some(table) = self.stored_table(&c.name, &env, false)
                        && table.layout.bucket.is_none()
                    {
                        self.push_location(&table, &env, &mut out);
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
