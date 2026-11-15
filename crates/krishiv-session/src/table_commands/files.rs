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
    /// the same for its files), in the one form every check of a table's directories takes: `file:` and
    /// the canonical absolute path, with `.` and `..` folded in the text as reads and writes fold them,
    /// then the links of its existing part resolved, the part that does not exist yet appended
    /// (`access::canonical_dir`, the write path's rule). Returns that canonical path: the caller acts on
    /// it, not on `path`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` (audited) for a relative or non-UTF-8 path, before the policy is
    /// asked; `KRISHIV_INSUFFICIENT_PERMISSIONS` for a denial and for a path that cannot be resolved (the
    /// same answer, see `access::physical`).
    pub(crate) async fn authorize_path(
        &self,
        path: &Path,
        privileges: &[Privilege],
        class: StatementClass,
    ) -> Result<PathBuf, KrishivError> {
        let first = privileges.first().copied().unwrap_or(Privilege::Modify);
        let at = crate::access::canonical_dir(self, path, first, class)?;
        let securable = Securable::Path(OptionValue::from_user("path", at.display.clone()));
        self.authorize_securable(&securable, privileges, class)
            .await?;
        Ok(local_of(&at.display))
    }

    /// Asks `CREATE` on a table's `location` as CREATE TABLE asks it: a managed table's directory (and an
    /// external local location) as [`Self::authorize_path`] does, a relative external location after
    /// qualifying it against the working directory as its writes do, and an object-store location in its
    /// qualified form. Returns the canonical local path checked (`None` for an object store).
    pub(super) async fn authorize_location(
        &self,
        location: &Path,
        managed: bool,
    ) -> Result<Option<PathBuf>, KrishivError> {
        let (create, class) = (&[Privilege::Create], StatementClass::Ddl);
        match location.to_str() {
            Some(text) if !managed && !location.is_absolute() => {
                let qualified = self.qualify_path(text)?;
                match qualified.strip_prefix("file:") {
                    Some(local) => self
                        .authorize_path(Path::new(local), create, class)
                        .await
                        .map(Some),
                    None => {
                        let securable = Securable::Path(OptionValue::from_user("path", qualified));
                        self.authorize_securable(&securable, create, class).await?;
                        Ok(None)
                    }
                }
            }
            _ => self.authorize_path(location, create, class).await.map(Some),
        }
    }

    /// Makes the directory `checked` (a canonical path [`Self::authorize_path`] allowed) and the missing
    /// directories above it one level at a time, each verified to still resolve to itself once it exists.
    /// A link swapped in on the way (after the check) is refused with 42501 for `privilege`, and the
    /// directory just made through it is removed: only that directory, found by the identity (device and
    /// inode) it had right after the mkdir, so a second swap cannot point the removal elsewhere.
    ///
    /// Not closed: between the verification of one level and the mkdir of the next, a swap makes one
    /// empty directory outside, which the next verification finds and removes (best effort).
    pub(crate) fn create_checked_dir(
        &self,
        checked: &Path,
        privilege: Privilege,
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        let mut at = PathBuf::new();
        for part in checked.components() {
            at.push(part);
            let made = match std::fs::create_dir(&at) {
                Ok(()) => dir_identity(&at),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => return Err(internal(format!("cannot create {}: {e}", at.display()))),
            };
            match std::fs::canonicalize(&at) {
                Ok(now) if now == at => {}
                now => {
                    if let (Some(made), Ok(now)) = (made, now)
                        && dir_identity(&now) == Some(made)
                    {
                        let _ = std::fs::remove_dir(now);
                    }
                    return Err(crate::access::unresolved(
                        self,
                        class,
                        &format!("file:{}", checked.display()),
                        privilege,
                    ));
                }
            }
        }
        Ok(())
    }

    /// Moves the directory `source` to `target`, both canonical paths the policy allowed, making the
    /// target's parents with [`Self::create_checked_dir`]. After the move the target and the source's
    /// parent must still resolve to themselves: if a link was swapped in on either side after the checks,
    /// the directory is moved back where it came from and the move is refused with 42501 (`CREATE` on the
    /// target), so nothing is moved into or out of the checked paths through it.
    ///
    /// # Errors
    /// 42501 as above; `INTERNAL_ERROR` when the move fails, or when the move back fails (the server log
    /// names both paths at error level; the caller is not told where the directory is).
    ///
    /// Not closed: a swap during the `rename` itself, which the verification after it catches.
    pub(crate) fn move_checked_dir(
        &self,
        source: &Path,
        target: &Path,
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        if let Some(parent) = target.parent() {
            self.create_checked_dir(parent, Privilege::Create, class)?;
        }
        std::fs::rename(source, target).map_err(|e| {
            internal(format!(
                "cannot move {} to {}: {e}",
                source.display(),
                target.display()
            ))
        })?;
        let in_place = |p: &Path| std::fs::canonicalize(p).is_ok_and(|now| now == p);
        if in_place(target) && source.parent().is_none_or(in_place) {
            return Ok(());
        }
        // Where the directory went, and where it came from, as the links resolve now.
        let went = std::fs::canonicalize(target).ok();
        let came = source
            .parent()
            .and_then(|p| std::fs::canonicalize(p).ok())
            .zip(source.file_name())
            .map(|(p, name)| p.join(name));
        if let (Some(went), Some(came)) = (&went, &came)
            && std::fs::rename(went, came).is_ok()
        {
            return Err(crate::access::unresolved(
                self,
                class,
                &format!("file:{}", target.display()),
                Privilege::Create,
            ));
        }
        Err(move_back_failed(&went, &came))
    }

    /// Moves `from` back to `to` after a later step of a move failed (both canonical); a failure is
    /// [`move_back_failed`].
    pub(crate) fn move_back(&self, from: &Path, to: &Path) -> Result<(), KrishivError> {
        std::fs::rename(from, to)
            .map_err(|_| move_back_failed(&Some(from.to_path_buf()), &Some(to.to_path_buf())))
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
        // `..` folded in the text first, as every other check and the writes fold it.
        let location = &local_of(&crate::access::lexical_dir(&table.location)?.display);
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
        let root = local_of(&crate::access::lexical_dir(&table.root)?.display)
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
            .await
            .map(Some)
    }
}

/// The error-level server log of a directory that could not be moved back (`went` is where it is, `came`
/// where it belongs, `None` where a path no longer resolves), and the caller's `INTERNAL_ERROR`, which names
/// neither path.
fn move_back_failed(went: &Option<PathBuf>, came: &Option<PathBuf>) -> KrishivError {
    tracing::error!(
        target: "krishiv::session",
        directory = ?went,
        belongs_at = ?came,
        "a directory could not be moved back after a refused or failed move; move it by hand"
    );
    internal(
        "a directory could not be moved back after a refused or failed move; the server log names it"
            .to_owned(),
    )
}

/// The device and inode of the directory `path` resolves to (`None` if it is not one, or off Unix).
fn dir_identity(path: &Path) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = std::fs::metadata(path).ok()?;
        meta.is_dir().then(|| (meta.dev(), meta.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// The local path of a `file:/...` display.
fn local_of(display: &str) -> PathBuf {
    PathBuf::from(display.strip_prefix("file:").unwrap_or(display))
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
