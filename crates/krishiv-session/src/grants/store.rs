// SPDX-License-Identifier: Apache-2.0
//! Where grants live (design P3-T062 §2.3, amended B1): a [`GrantStore`] serves an in-memory
//! snapshot to every check and swaps a whole new state in on change, its epoch bumped in the same
//! swap. [`MemoryGrantStore`] is the embedded and test store; [`FileGrantStore`] keeps one
//! versioned file (written to a temporary file, synced, then renamed over the old one, so a crash
//! leaves the old or the new state, never half of one).
//!
//! File format version 1 (JSON): `{"format": "krishiv-grants", "version": 1, "epoch": N,
//! "grants": [{"grantee", "privilege", "kind", "parts", "grant_option", "granted_by",
//! "granted_at"}], "roles": [{"name", "owner", "members"}], "owners": [{"kind", "parts",
//! "owner"}]}`. A reader takes versions 1 to [`FORMAT_VERSION`] (ADR 0026: N reads N-1) and
//! refuses a newer one or any malformed field: an unreadable store is a refusal, never an empty
//! (allow-nothing would be safe, but a silently emptied store would lose owners' grants).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use krishiv_common::error::{KrishivError, KrishivGrantsStoreFailure};
use krishiv_plan::analyzer::Privilege;
use krishiv_plan::analyzer::access_control::Grantee;
use serde_json::{Map, Value, json};

use super::model::{GrantRow, GrantState, KeyKind, ObjectKey, RoleEntry};

/// The current file format version.
pub(crate) const FORMAT_VERSION: u64 = 1;
const FORMAT_NAME: &str = "krishiv-grants";
/// The largest store file read (a store this big is not a grants table any more).
const MAX_FILE_BYTES: u64 = 64 << 20;

/// A grants store: the current state, and an atomic change to it.
pub trait GrantStore: Send + Sync {
    /// The current state.
    #[doc(hidden)]
    fn snapshot(&self) -> Arc<GrantStateHandle>;
    /// Applies `change` to the current state and swaps the result in, epoch bumped, as one step
    /// (changes are serialised). Nothing changes when `change` or the write fails.
    ///
    /// # Errors
    /// `change`'s error, or `KRISHIV_GRANTS_STORE_FAILURE` when the state cannot be persisted.
    #[doc(hidden)]
    fn update(
        &self,
        change: &mut dyn FnMut(&mut GrantStateHandle) -> Result<(), KrishivError>,
    ) -> Result<Arc<GrantStateHandle>, KrishivError>;
}

/// An opaque grants state (the store's snapshot type).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GrantStateHandle(pub(crate) GrantState);

/// Serialises changes and holds the snapshot.
#[derive(Debug, Default)]
struct Cell {
    current: RwLock<Arc<GrantStateHandle>>,
    writer: Mutex<()>,
}

impl Cell {
    fn snapshot(&self) -> Arc<GrantStateHandle> {
        Arc::clone(&self.current.read().unwrap_or_else(PoisonError::into_inner))
    }

    fn update(
        &self,
        change: &mut dyn FnMut(&mut GrantStateHandle) -> Result<(), KrishivError>,
        persist: impl FnOnce(&GrantState) -> Result<(), KrishivError>,
    ) -> Result<Arc<GrantStateHandle>, KrishivError> {
        let _writer = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        let mut next = (*self.snapshot()).clone();
        change(&mut next)?;
        next.0.epoch = next
            .0
            .epoch
            .checked_add(1)
            .ok_or_else(|| failure("updated", "the epoch overflowed"))?;
        persist(&next.0)?;
        let next = Arc::new(next);
        *self.current.write().unwrap_or_else(PoisonError::into_inner) = Arc::clone(&next);
        Ok(next)
    }
}

/// Grants in memory only (embedded use and tests): lost when the process ends.
#[derive(Debug, Default)]
pub struct MemoryGrantStore {
    cell: Cell,
}

impl MemoryGrantStore {
    /// An empty store at epoch 0.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl GrantStore for MemoryGrantStore {
    fn snapshot(&self) -> Arc<GrantStateHandle> {
        self.cell.snapshot()
    }

    fn update(
        &self,
        change: &mut dyn FnMut(&mut GrantStateHandle) -> Result<(), KrishivError>,
    ) -> Result<Arc<GrantStateHandle>, KrishivError> {
        self.cell.update(change, |_| Ok(()))
    }
}

/// Grants in one file (a server's data directory, `grants.json`), loaded at open and rewritten
/// whole on every change.
#[derive(Debug)]
pub struct FileGrantStore {
    path: PathBuf,
    cell: Cell,
    /// The exclusive lock on `<file>.lock`, held while the store is open (review m2): a second
    /// opener (another process) would keep a snapshot of its own and lose writes.
    _lock: std::fs::File,
}

impl FileGrantStore {
    /// Opens the store at `path`, creating an empty one when the file does not exist, and holds
    /// an exclusive lock on `<path>.lock` until it is dropped.
    ///
    /// # Errors
    /// `KRISHIV_GRANTS_STORE_FAILURE` for a file that cannot be read, is malformed or has a newer
    /// format version, and when the store is already open (in this or another process).
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, KrishivError> {
        let path = path.into();
        let mut lock_name = path.as_os_str().to_os_string();
        lock_name.push(".lock");
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let lock = options
            .open(PathBuf::from(lock_name))
            .map_err(|e| failure("open", &e.kind().to_string()))?;
        lock.try_lock().map_err(|e| match e {
            std::fs::TryLockError::WouldBlock => {
                failure("open", "the store is already open in another process")
            }
            std::fs::TryLockError::Error(e) => failure("open", &e.kind().to_string()),
        })?;
        let state = match std::fs::metadata(&path) {
            Ok(meta) if meta.len() > MAX_FILE_BYTES => {
                return Err(failure("read", "the file is larger than 64 MiB"));
            }
            Ok(_) => {
                let bytes =
                    std::fs::read(&path).map_err(|e| failure("read", &e.kind().to_string()))?;
                decode(&bytes)?
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => GrantState::default(),
            Err(e) => return Err(failure("read", &e.kind().to_string())),
        };
        Ok(Self {
            path,
            cell: Cell {
                current: RwLock::new(Arc::new(GrantStateHandle(state))),
                writer: Mutex::new(()),
            },
            _lock: lock,
        })
    }

    /// The file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl GrantStore for FileGrantStore {
    fn snapshot(&self) -> Arc<GrantStateHandle> {
        self.cell.snapshot()
    }

    fn update(
        &self,
        change: &mut dyn FnMut(&mut GrantStateHandle) -> Result<(), KrishivError>,
    ) -> Result<Arc<GrantStateHandle>, KrishivError> {
        let path = self.path.clone();
        self.cell
            .update(change, |state| write_atomically(&path, &encode(state)))
    }
}

fn failure(operation: &str, reason: &str) -> KrishivError {
    KrishivGrantsStoreFailure {
        operation: operation.to_owned(),
        reason: reason.to_owned(),
    }
    .into()
}

/// Writes `bytes` to a temporary file beside `path` (owner-only on Unix), syncs it, renames it
/// over `path` and syncs the directory.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), KrishivError> {
    use std::io::Write as _;
    let io = |e: std::io::Error| failure("written", &e.kind().to_string());
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| failure("written", "the path names no file"))?;
    // A fresh name, created exclusively (review m2): a file or link planted at a fixed name is
    // never followed or reused, and the new file gets its mode.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut tmp_name = name.to_os_string();
    tmp_name.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let tmp = dir.join(tmp_name);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(&tmp).map_err(io)?;
    let written = file
        .write_all(bytes)
        .and_then(|()| file.sync_all())
        .and_then(|()| {
            drop(file);
            std::fs::rename(&tmp, path)
        });
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(io(e));
    }
    #[cfg(unix)]
    std::fs::File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(io)?;
    Ok(())
}

fn grantee_json(g: &Grantee) -> Value {
    Value::String(g.to_string())
}

fn encode(state: &GrantState) -> Vec<u8> {
    let grants: Vec<Value> = state
        .grants
        .iter()
        .map(|g| {
            json!({
                "grantee": grantee_json(&g.grantee),
                "privilege": g.privilege.to_string(),
                "kind": g.object.kind.name(),
                "parts": g.object.parts,
                "grant_option": g.with_grant_option,
                "granted_by": g.granted_by,
                "granted_at": g.granted_at,
            })
        })
        .collect();
    let roles: Vec<Value> = state
        .roles
        .iter()
        .map(|(name, r)| {
            json!({
                "name": name,
                "owner": r.owner,
                "members": r.members.iter().map(grantee_json).collect::<Vec<_>>(),
            })
        })
        .collect();
    let owners: Vec<Value> = state
        .owners
        .iter()
        .map(|(k, owner)| json!({"kind": k.kind.name(), "parts": k.parts, "owner": owner}))
        .collect();
    let doc = json!({
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "epoch": state.epoch,
        "grants": grants,
        "roles": roles,
        "owners": owners,
    });
    // Serialising a `Value` cannot fail.
    serde_json::to_vec_pretty(&doc).unwrap_or_default()
}

fn malformed(what: &str) -> KrishivError {
    failure("read", &format!("the file is malformed ({what})"))
}

fn field<'a>(o: &'a Map<String, Value>, name: &str) -> Result<&'a Value, KrishivError> {
    o.get(name).ok_or_else(|| malformed(name))
}

fn string(o: &Map<String, Value>, name: &str) -> Result<String, KrishivError> {
    field(o, name)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| malformed(name))
}

fn array<'a>(o: &'a Map<String, Value>, name: &str) -> Result<&'a Vec<Value>, KrishivError> {
    field(o, name)?.as_array().ok_or_else(|| malformed(name))
}

fn object(v: &Value) -> Result<&Map<String, Value>, KrishivError> {
    v.as_object().ok_or_else(|| malformed("an entry"))
}

/// `user:<name>` / `group:<name>` / `role:<name>`.
pub(crate) fn parse_grantee(text: &str) -> Option<Grantee> {
    let (kind, name) = text.split_once(':')?;
    if name.is_empty() {
        return None;
    }
    let name = name.to_owned();
    Some(match kind {
        "user" => Grantee::User(name),
        "group" => Grantee::Group(name),
        "role" => Grantee::Role(name),
        _ => return None,
    })
}

fn key(o: &Map<String, Value>) -> Result<ObjectKey, KrishivError> {
    let kind = KeyKind::from_name(&string(o, "kind")?).ok_or_else(|| malformed("kind"))?;
    let parts = array(o, "parts")?
        .iter()
        .map(|p| {
            p.as_str()
                .map(str::to_owned)
                .ok_or_else(|| malformed("parts"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parts.is_empty() {
        return Err(malformed("parts"));
    }
    Ok(ObjectKey::new(kind, parts))
}

fn decode(bytes: &[u8]) -> Result<GrantState, KrishivError> {
    let doc: Value = serde_json::from_slice(bytes).map_err(|_| malformed("not JSON"))?;
    let doc = object(&doc)?;
    if string(doc, "format")? != FORMAT_NAME {
        return Err(malformed("format"));
    }
    let version = field(doc, "version")?
        .as_u64()
        .ok_or_else(|| malformed("version"))?;
    if version == 0 || version > FORMAT_VERSION {
        return Err(failure(
            "read",
            &format!(
                "format version {version} is not one this build reads (1 to {FORMAT_VERSION})"
            ),
        ));
    }
    let epoch = field(doc, "epoch")?
        .as_u64()
        .ok_or_else(|| malformed("epoch"))?;
    let mut state = GrantState {
        epoch,
        ..GrantState::default()
    };
    for g in array(doc, "grants")? {
        let g = object(g)?;
        let privilege = string(g, "privilege")?;
        let privilege = krishiv_plan::analyzer::access_control::grantable_privilege(&privilege)
            .filter(|p| *p != Privilege::Impersonate)
            .ok_or_else(|| malformed("privilege"))?;
        state.grants.push(GrantRow {
            grantee: parse_grantee(&string(g, "grantee")?).ok_or_else(|| malformed("grantee"))?,
            privilege,
            object: key(g)?,
            with_grant_option: field(g, "grant_option")?
                .as_bool()
                .ok_or_else(|| malformed("grant_option"))?,
            granted_by: string(g, "granted_by")?,
            granted_at: field(g, "granted_at")?
                .as_i64()
                .ok_or_else(|| malformed("granted_at"))?,
        });
    }
    for r in array(doc, "roles")? {
        let r = object(r)?;
        let members = array(r, "members")?
            .iter()
            .map(|m| {
                m.as_str()
                    .and_then(parse_grantee)
                    .ok_or_else(|| malformed("members"))
            })
            .collect::<Result<_, _>>()?;
        state.roles.insert(
            string(r, "name")?,
            RoleEntry {
                owner: string(r, "owner")?,
                members,
            },
        );
    }
    for o in array(doc, "owners")? {
        let o = object(o)?;
        state.owners.insert(key(o)?, string(o, "owner")?);
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GrantState {
        let mut s = GrantState {
            epoch: 4,
            ..GrantState::default()
        };
        s.grants.push(GrantRow {
            grantee: Grantee::Group("eng".to_owned()),
            privilege: Privilege::Select,
            object: ObjectKey::new(KeyKind::Relation, vec!["c".into(), "s".into(), "t".into()]),
            with_grant_option: true,
            granted_by: "admin".to_owned(),
            granted_at: 1_700_000_000_000,
        });
        s.roles.insert(
            "r".to_owned(),
            RoleEntry {
                owner: "admin".to_owned(),
                members: [Grantee::User("ana".to_owned())].into(),
            },
        );
        s.owners.insert(
            ObjectKey::new(KeyKind::Path, vec!["s3://b/p".into()]),
            "ana".to_owned(),
        );
        s
    }

    #[test]
    fn the_file_round_trips_and_refuses_what_it_cannot_read() {
        let state = sample();
        assert_eq!(decode(&encode(&state)).unwrap(), state);
        let mut doc: Value = serde_json::from_slice(&encode(&state)).unwrap();
        doc["version"] = json!(2);
        let e = decode(&serde_json::to_vec(&doc).unwrap()).unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_GRANTS_STORE_FAILURE");
        assert!(e.to_string().contains("format version 2"), "{e}");
        for bad in [
            b"".as_slice(),
            b"[]",
            br#"{"format":"krishiv-grants","version":1,"epoch":0,"grants":[{}],"roles":[],"owners":[]}"#,
            br#"{"format":"other","version":1,"epoch":0,"grants":[],"roles":[],"owners":[]}"#,
        ] {
            assert_eq!(decode(bad).unwrap_err().error_class().name(), "KRISHIV_GRANTS_STORE_FAILURE");
        }
        // A role-matrix privilege never comes back from a file.
        let mut doc: Value = serde_json::from_slice(&encode(&state)).unwrap();
        doc["grants"][0]["privilege"] = json!("MANAGE");
        assert!(decode(&serde_json::to_vec(&doc).unwrap()).is_err());
    }

    #[test]
    fn file_store_persists_and_bumps_the_epoch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grants.json");
        let store = FileGrantStore::open(&path).unwrap();
        assert_eq!(store.snapshot().0.epoch, 0);
        let s = store
            .update(&mut |s| {
                s.0.roles.insert("r".to_owned(), RoleEntry::default());
                Ok(())
            })
            .unwrap();
        assert_eq!(s.0.epoch, 1);
        // A failed change leaves the state and the file as they were.
        let e = store.update(&mut |s| {
            s.0.roles.clear();
            Err(failure("updated", "test"))
        });
        assert!(e.is_err());
        // One opener at a time (review m2).
        assert!(FileGrantStore::open(&path).is_err());
        drop(store);
        let reopened = FileGrantStore::open(&path).unwrap();
        assert_eq!(*reopened.snapshot(), *s);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "{mode:o}");
        }
        drop(reopened);
        std::fs::write(&path, b"{").unwrap();
        let e = FileGrantStore::open(&path).unwrap_err();
        assert!(e.to_string().contains("malformed"), "{e}");
    }
}
