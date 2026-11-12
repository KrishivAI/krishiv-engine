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
    /// Asks the session's policy about `path` for each privilege (the writer does the same for its files).
    pub(super) async fn authorize_path(
        &self,
        path: &Path,
        privileges: &[Privilege],
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        let securable = Securable::Path(OptionValue::from_user("path", path.display().to_string()));
        for privilege in privileges {
            let r = authorize(
                &*self.state.policy,
                &self.state.principal,
                &securable,
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
