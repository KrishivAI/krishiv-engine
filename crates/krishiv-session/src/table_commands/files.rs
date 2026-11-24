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
    /// (`access::canonical_dir`, the write path's rule). Returns that canonical path, which the caller acts
    /// on (not `path`), and the text a denial names (`path` as written, `..` folded from the text only).
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
    ) -> Result<CheckedDir, KrishivError> {
        let first = privileges.first().copied().unwrap_or(Privilege::Modify);
        let checked = crate::access::canonical_dir(self, path, first, class)?;
        crate::access::authorize_checked(self, &checked, privileges, class).await?;
        Ok(CheckedDir {
            path: local_of(&checked.at.display),
            shown: checked.shown,
        })
    }

    /// Asks `CREATE` on a table's `location` as CREATE TABLE asks it: where [`Self::location_asked`]
    /// places it. Returns the canonical local path checked (`None` for an object store).
    pub(super) async fn authorize_location(
        &self,
        location: &Path,
        managed: bool,
    ) -> Result<Option<CheckedDir>, KrishivError> {
        let (create, class) = (&[Privilege::Create], StatementClass::Ddl);
        match self.location_asked(location, managed)? {
            LocationAsked::Local(local) => {
                self.authorize_path(&local, create, class).await.map(Some)
            }
            LocationAsked::Store(qualified) => {
                let securable = Securable::Path(OptionValue::from_user("path", qualified));
                self.authorize_securable(&securable, create, class).await?;
                Ok(None)
            }
        }
    }

    /// Where CREATE TABLE asks `CREATE` on a table's `location` ([`Self::authorize_location`]): a managed
    /// table's directory (and an external local location) as [`Self::authorize_path`] asks it, a relative
    /// external location after qualifying it against the working directory as its writes do, and an
    /// object-store location in its qualified form. The analyze API lists the same (`questions.rs`).
    pub(super) fn location_asked(
        &self,
        location: &Path,
        managed: bool,
    ) -> Result<LocationAsked, KrishivError> {
        match location.to_str() {
            Some(text) if !managed && !location.is_absolute() => {
                let qualified = self.qualify_path(text)?;
                Ok(match qualified.strip_prefix("file:") {
                    Some(local) => LocationAsked::Local(krishiv_exec::files::native_path(local)),
                    None => LocationAsked::Store(qualified),
                })
            }
            _ => Ok(LocationAsked::Local(location.to_path_buf())),
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
        shown: &str,
        privilege: Privilege,
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        let mut at = PathBuf::new();
        for part in checked.components() {
            at.push(part);
            // A Windows prefix (`\\?\C:`) and the root are not directories to make: `create_dir` on a
            // bare prefix fails with neither success nor `AlreadyExists`.
            if matches!(
                part,
                std::path::Component::Prefix(_) | std::path::Component::RootDir
            ) {
                continue;
            }
            let made = match std::fs::create_dir(&at) {
                Ok(()) => dir_identity(&at),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => {
                    return Err(fs_failed(
                        "cannot create a directory of the statement",
                        &at,
                        None,
                        &e,
                    ));
                }
            };
            match std::fs::canonicalize(&at) {
                Ok(now) if now == at => {}
                now => {
                    if let Ok(now) = now {
                        remove_made(made, &now);
                    }
                    return Err(crate::access::unresolved(self, class, shown, privilege));
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
        target: &CheckedDir,
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        let (shown, target) = (&target.shown, &target.path);
        if let Some(parent) = target.parent() {
            self.create_checked_dir(parent, parent_text(shown), Privilege::Create, class)?;
        }
        std::fs::rename(source, target)
            .map_err(|e| fs_failed("cannot move a directory", source, Some(target), &e))?;
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
                shown,
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
        let Some((location, canonical)) = self.deletable_location(table)? else {
            return Ok(None);
        };
        let location = &location;
        // Asked about the location as written (`..` folded): the policy gets it resolved, the caller's
        // denial names it as written. It resolves to the directory checked above, or nothing is deleted.
        let checked = self
            .authorize_path(location, &[Privilege::Modify], StatementClass::Write)
            .await?;
        if checked.path != canonical {
            return Err(internal(format!(
                "refusing to delete {}: it changed while it was checked",
                location.display()
            )));
        }
        Ok(Some(checked.path))
    }

    /// The directory of the managed `table` that [`Self::deletable_dir`] asks the policy about, before it
    /// asks: the location as written (`..` folded) and the directory it resolves to, after the containment
    /// checks (inside the root the table was created under, not that root, not a link). `None` when there
    /// is nothing to delete (an external table, or a directory that does not exist). The analyze API lists
    /// `MODIFY` on it as a run's question (`analyze/dirs.rs`).
    pub(crate) fn deletable_location(
        &self,
        table: &CatalogTable,
    ) -> Result<Option<(PathBuf, PathBuf)>, KrishivError> {
        if !table.managed {
            return Ok(None);
        }
        // `..` folded in the text first, as every other check and the writes fold it.
        let location = local_of(&crate::access::lexical_dir(&table.location)?.display);
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
        Ok(Some((location, canonical)))
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

/// The caller's `INTERNAL_ERROR` for a failed file system step (`what`, naming no path and no I/O error,
/// which could describe a directory behind a swapped link), with the paths and the error in an error-level
/// server log.
fn fs_failed(what: &str, path: &Path, other: Option<&Path>, e: &std::io::Error) -> KrishivError {
    tracing::error!(
        target: "krishiv::session",
        path = %path.display(),
        other = ?other,
        error = %e,
        "{what}"
    );
    internal(format!("{what}; the server log names it"))
}

/// Removes the empty directory `now` only if it is the one made, `made` (its device and inode right after
/// the mkdir): a second swap cannot point the removal at another directory.
fn remove_made(made: Option<(u64, u64)>, now: &Path) {
    if made.is_some() && dir_identity(now) == made {
        let _ = std::fs::remove_dir(now);
    }
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

/// Where a CREATE TABLE asks about a table's location ([`Session::location_asked`]).
pub(super) enum LocationAsked {
    /// A local path, asked as [`Session::authorize_path`] asks it.
    Local(PathBuf),
    /// An object-store location, in its qualified form.
    Store(String),
}

/// A local directory as [`Session::authorize_path`] checked it: `path`, canonical, is the one to act on;
/// `shown` is the caller's text (`..` folded from the text only) that a denial names.
#[derive(Debug, Clone)]
pub(crate) struct CheckedDir {
    /// The canonical path the policy allowed.
    pub(crate) path: PathBuf,
    /// The text a denial names (`file:/...`).
    pub(crate) shown: String,
}

impl CheckedDir {
    /// The parent directory, with the parent of the shown text.
    pub(crate) fn parent(&self) -> Option<CheckedDir> {
        Some(CheckedDir {
            path: self.path.parent()?.to_path_buf(),
            shown: parent_text(&self.shown).to_owned(),
        })
    }
}

/// `shown` without its last `/segment` (`file:/` stays).
fn parent_text(shown: &str) -> &str {
    match shown.rfind('/') {
        Some(i) if shown[..i].ends_with(':') => &shown[..=i],
        Some(i) => &shown[..i],
        None => shown,
    }
}

/// The local path of a `file:/...` display.
fn local_of(display: &str) -> PathBuf {
    krishiv_exec::files::native_path(display)
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

#[cfg(test)]
mod tests {
    use super::{dir_identity, remove_made};

    #[test]
    fn only_the_directory_made_is_removed() {
        let tmp = tempfile::tempdir().unwrap();
        let made = tmp.path().join("made");
        let other = tmp.path().join("other");
        std::fs::create_dir(&made).unwrap();
        std::fs::create_dir(&other).unwrap();
        let identity = dir_identity(&made);
        assert!(identity.is_some());
        // A second swap points the removal at another empty directory: it stays.
        remove_made(identity, &other);
        assert!(other.is_dir());
        remove_made(None, &made);
        assert!(made.is_dir());
        remove_made(identity, &made);
        assert!(!made.exists());
    }

    #[test]
    fn a_failed_mkdir_or_move_is_an_internal_error_naming_no_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        std::fs::write(root.join("file-secret"), b"").unwrap();
        let session = crate::Session::new().unwrap();
        let mkdir = session
            .create_checked_dir(
                &root.join("file-secret/sub"),
                "file:/x",
                krishiv_plan::analyzer::Privilege::Create,
                krishiv_sql::StatementClass::Ddl,
            )
            .unwrap_err();
        let moved = session
            .move_checked_dir(
                &root.join("missing-secret"),
                &super::CheckedDir {
                    path: root.join("target-secret"),
                    shown: "file:/x".to_owned(),
                },
                krishiv_sql::StatementClass::Ddl,
            )
            .unwrap_err();
        for e in [mkdir, moved] {
            assert_eq!(
                e.error_class(),
                krishiv_common::error::ErrorClass::InternalError
            );
            let text = e.to_string();
            assert!(
                !text.contains("secret") && !text.contains("os error"),
                "{text}"
            );
        }
    }

    #[test]
    fn a_failed_move_back_is_an_internal_error_naming_no_path() {
        let tmp = tempfile::tempdir().unwrap();
        let from = tmp.path().join("went-secret");
        let to = tmp.path().join("came-secret");
        let session = crate::Session::new().unwrap();
        let e = session.move_back(&from, &to).unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::InternalError
        );
        let text = e.to_string();
        assert!(!text.contains("secret"), "{text}");
        assert!(!text.contains(&*tmp.path().to_string_lossy()), "{text}");
    }
}
