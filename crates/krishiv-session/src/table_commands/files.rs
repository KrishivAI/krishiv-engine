// SPDX-License-Identifier: Apache-2.0
//! The files of a managed table's directory: the policy check of a path and the deletes of DROP, TRUNCATE and
//! overwrite, which never reach outside the warehouse root the table was created under.

use std::path::{Path, PathBuf};

use krishiv_common::error::KrishivError;
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{Privilege, Securable, authorize};
use krishiv_sql::StatementClass;

use super::internal;
use crate::session::Session;
use crate::store::CatalogTable;

impl Session {
    /// Asks the session's policy about the local directory `path` for each privilege (the writer does
    /// the same for its files), in the one form [`dir_securable`] gives every check of a table's
    /// directories: `file:` and the canonical absolute path, the path a write to it goes to.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a path that cannot be made canonical (relative, unresolvable,
    /// non-UTF-8), before the policy is asked; `KRISHIV_INSUFFICIENT_PERMISSIONS` for a denial.
    pub(super) async fn authorize_path(
        &self,
        path: &Path,
        privileges: &[Privilege],
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        let securable = dir_securable(path)?;
        self.authorize_securable(&securable, privileges, class)
            .await
    }

    /// The securable of a table's `location` as CREATE TABLE asks it: a managed table's directory (and an
    /// external local location) by [`dir_securable`], a relative external location after qualifying it
    /// against the working directory as its writes do, and an object-store location in its qualified form.
    pub(super) fn location_securable(
        &self,
        location: &Path,
        managed: bool,
    ) -> Result<Securable, KrishivError> {
        match location.to_str() {
            Some(text) if !managed && !location.is_absolute() => {
                let qualified = self.qualify_path(text)?;
                match qualified.strip_prefix("file:") {
                    Some(local) => dir_securable(Path::new(local)),
                    None => Ok(Securable::Path(OptionValue::from_user("path", qualified))),
                }
            }
            _ => dir_securable(location),
        }
    }

    /// Asks the session's policy about `securable` for each privilege, auditing each answer.
    pub(super) async fn authorize_securable(
        &self,
        securable: &Securable,
        privileges: &[Privilege],
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        for privilege in privileges {
            let r = authorize(
                &*self.state.policy,
                &self.state.principal,
                securable,
                *privilege,
            )
            .await;
            crate::access::audit(self, class, r)?;
        }
        Ok(())
    }

    /// Deletes the files of the managed `table` (all of its directory, or with `keep_dir` what is in it).
    ///
    /// The directory must lie inside the warehouse, must not be the warehouse and must not be a link: a name
    /// can never reach outside it (names are checked at creation, and this is the second line). The policy is
    /// asked about the directory before anything is removed.
    pub(super) async fn delete_table_files(
        &self,
        table: &CatalogTable,
        keep_dir: bool,
    ) -> Result<(), KrishivError> {
        if let Some(dir) = self.deletable_dir(table).await? {
            if keep_dir {
                clear_dir(&dir)
            } else {
                remove_dir(&dir)
            }
        } else {
            Ok(())
        }
    }

    /// The directory whose files a delete of `table` would remove, after every check (containment in the root
    /// the table was created under, no links, the policy): `None` when there is nothing to delete. Run before
    /// the metadata changes, so a refused delete leaves the table in place.
    pub(super) async fn deletable_dir(
        &self,
        table: &CatalogTable,
    ) -> Result<Option<PathBuf>, KrishivError> {
        if !table.managed {
            return Ok(None);
        }
        let location = &table.location;
        let meta = match location.symlink_metadata() {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(internal(format!(
                    "cannot inspect {}: {e}",
                    location.display()
                )));
            }
        };
        let root = table
            .root
            .canonicalize()
            .map_err(|e| internal(format!("cannot resolve the warehouse: {e}")))?;
        let canonical = location
            .canonicalize()
            .map_err(|e| internal(format!("cannot resolve {}: {e}", location.display())))?;
        if meta.file_type().is_symlink() || !canonical.starts_with(&root) || canonical == root {
            return Err(internal(format!(
                "refusing to delete {}: it is not a directory inside the warehouse",
                location.display()
            )));
        }
        self.authorize_path(&canonical, &[Privilege::Modify], StatementClass::Write)
            .await?;
        Ok(Some(canonical))
    }
}

/// The path securable of the local directory `path`, in the one form every check of a table's
/// directories uses ([`Session::authorize_path`], the partition DDL of `partitions.rs`): `file:` and the
/// canonical absolute path, with `.`, `..` and the links of its existing part resolved and the part that
/// does not exist yet appended (`access::physical`, the rule the write path uses). So one rule answers
/// CREATE, INSERT and DROP of a directory alike, however the warehouse was written.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a relative path (also a `file:` or other scheme kept in the text: a
/// local directory is expected), one that cannot be resolved (an I/O error, a link loop, a dangling link)
/// and a non-UTF-8 one. It never reaches the policy as written (fail closed).
pub(crate) fn dir_securable(path: &Path) -> Result<Securable, KrishivError> {
    let text = crate::access::canonical_file_path(path)?;
    Ok(Securable::Path(OptionValue::from_user("path", text)))
}

/// Deletes a managed table's directory; a directory that is already gone
/// is fine.
pub(super) fn remove_dir(dir: &Path) -> Result<(), KrishivError> {
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(internal(format!("cannot delete {}: {e}", dir.display()))),
    }
}

/// Removes everything in `dir` and keeps `dir`.
fn clear_dir(dir: &Path) -> Result<(), KrishivError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(internal(format!("cannot list {}: {e}", dir.display()))),
    };
    for entry in entries {
        let path = entry
            .map_err(|e| internal(format!("cannot list {}: {e}", dir.display())))?
            .path();
        let removed = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        removed.map_err(|e| internal(format!("cannot delete {}: {e}", path.display())))?;
    }
    Ok(())
}
