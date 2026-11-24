// SPDX-License-Identifier: Apache-2.0
//! [`PathPrefixPolicy`]: local paths only under allowed roots (links resolved), object-store paths only under
//! allowed prefixes, every other decision delegated; and the `krishiv.io.*` allowlist settings it reads.

use std::path::{Component, Path, PathBuf, Prefix};
use std::sync::Arc;

use krishiv_common::config::{
    Config, ConfigValue, IO_LOCAL_ALLOWED_ROOTS, IO_OBJECT_STORE_ALLOWED_PREFIXES,
};
use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue, PathNotFound};
use krishiv_common::secret::OptionValue;
use krishiv_io::{StoreId, parse_object_url};
use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, BoxFuture, PolicyVersion, Principal, Privilege, Securable,
};

/// Allows paths only under the given root directories, and delegates every
/// other decision to `inner`: the policy behind a server's data-directory
/// restriction.
///
/// A path is compared in the qualified form reads and writes use
/// (`file:/a/b`, `.` and `..` already resolved), after resolving symbolic
/// links in the part of it that exists, against the roots with their links
/// resolved. Object-store paths are allowed only under the prefixes of
/// [`Self::with_object_prefixes`] (none by default). Anything else is
/// denied: other schemes, paths with credentials, any path containing `%`
/// (percent-encoded text would not be decoded the way the comparison reads
/// it, so even a literal `%` in a file name is refused) and relative or
/// unresolved paths. A glob or directory read is checked as written; the
/// runtime's local store refuses a listing that follows a link out of
/// `krishiv.io.local.allowedRoots` (set it on the runtime too). Session
/// writes go to the resolved directory, which is opened once and then used
/// by descriptor (no name is resolved through a link while the write runs),
/// and fail if a link is swapped into it before the write begins or commits.
/// Resolving links reads file metadata on the calling thread, once per
/// path and query.
///
/// ```
/// use std::sync::Arc;
/// use krishiv_session::{AllowAll, PathPrefixPolicy, Session};
///
/// let dir = tempfile::tempdir().unwrap();
/// let policy = PathPrefixPolicy::new([dir.path()], Arc::new(AllowAll))?;
/// let s = Session::builder().access_policy(Arc::new(policy)).build()?;
/// let e = futures::executor::block_on(s.sql("SELECT * FROM parquet.`/etc`")).unwrap_err();
/// assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[derive(Clone)]
pub struct PathPrefixPolicy {
    roots: Vec<PathBuf>,
    prefixes: Vec<ObjectPrefix>,
    inner: Arc<dyn AccessPolicy>,
}

/// One allowed object-store prefix: a store and the whole key segments under it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectPrefix {
    store: StoreId,
    segments: Vec<String>,
}

impl std::fmt::Debug for PathPrefixPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathPrefixPolicy")
            .field("roots", &self.roots)
            .field("prefixes", &self.prefixes)
            .finish_non_exhaustive()
    }
}

impl PathPrefixPolicy {
    /// A policy allowing paths under `roots` (existing directories), with
    /// `inner` deciding everything else.
    ///
    /// # Errors
    /// `PATH_NOT_FOUND` for a root that does not exist.
    pub fn new<P: AsRef<Path>>(
        roots: impl IntoIterator<Item = P>,
        inner: Arc<dyn AccessPolicy>,
    ) -> Result<Self, KrishivError> {
        let roots = roots
            .into_iter()
            .map(|r| {
                std::fs::canonicalize(r.as_ref()).map_err(|_| {
                    KrishivError::from(PathNotFound {
                        path: format!("file:{}", r.as_ref().display()),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            roots,
            prefixes: Vec::new(),
            inner,
        })
    }

    /// Also allows object-store paths under `prefixes` (`s3://bucket/prefix`, `gs://bucket`,
    /// `abfss://container@account.dfs.core.windows.net/prefix`): a path is allowed when it
    /// names the same store (`s3` and `s3a` are one; for Azure the same container in the same
    /// account) and its key starts with the prefix's whole segments, so `s3://b/data` allows
    /// `s3://b/data/x` but not `s3://b/data2` (ADR 0046 §3a). An Azure path without an
    /// account host is never allowed, since its account would come from the client's settings.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for a prefix that is not an object-store URL with a
    /// bucket, an Azure prefix without an `account.<host>`, or one that holds credentials,
    /// `%`, a query or fragment, a glob character, or a `.`, `..` or empty segment.
    pub fn with_object_prefixes<S: AsRef<str>>(
        mut self,
        prefixes: impl IntoIterator<Item = S>,
    ) -> Result<Self, KrishivError> {
        for prefix in prefixes {
            let prefix = prefix.as_ref();
            let parsed = parse_object_prefix(prefix)
                .filter(|_| !prefix.contains(['*', '?', '[', ']', '{', '}']))
                // An Azure prefix names its account: without one, the store would take the
                // account from the client's `fs.azure.account.name`.
                .filter(|_| {
                    parse_object_url(prefix).is_ok_and(|u| {
                        u.scheme != krishiv_io::ObjectScheme::Azure || u.account.is_some()
                    })
                })
                .ok_or_else(|| {
                    KrishivError::from(KrishivInvalidConfigValue {
                        value: format!("'{prefix}'"),
                        key: format!("\"{}\"", IO_OBJECT_STORE_ALLOWED_PREFIXES.key),
                        reason:
                            "expected an object-store prefix such as s3://bucket/prefix (Azure: \
                             abfss://container@account.dfs.core.windows.net/prefix), without \
                             credentials, `%`, a query or fragment, glob characters, or `.`, `..` \
                             or empty segments"
                                .to_owned(),
                    })
                })?;
            self.prefixes.push(parsed);
        }
        Ok(self)
    }

    /// The server-mode confinement of ADR 0046 §3a around `inner`: local paths under
    /// `krishiv.io.local.allowedRoots`, object-store paths under
    /// `krishiv.io.objectStore.allowedPrefixes` (both comma separated, default empty, so
    /// nothing by path), and nothing else, for every role.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for a root that is not an absolute path to an existing
    /// directory, or a prefix [`Self::with_object_prefixes`] refuses.
    pub fn from_config(
        config: &Config,
        inner: Arc<dyn AccessPolicy>,
    ) -> Result<Self, KrishivError> {
        Self {
            roots: allowed_roots(config)?,
            prefixes: Vec::new(),
            inner,
        }
        .with_object_prefixes(config_list(config, &IO_OBJECT_STORE_ALLOWED_PREFIXES))
    }

    /// Whether the qualified path `path` lies under a root or an object-store prefix.
    pub(crate) fn allows(&self, path: &OptionValue) -> bool {
        let text = match path {
            OptionValue::Plain(text) => text.as_str(),
            // `container@account.host` reads as user info, so an Azure path arrives typed as a
            // URL with credentials; Azure parsing takes it as container and account and refuses
            // `user:secret@` forms, so only Azure URLs are compared, and the text never leaves
            // this check.
            OptionValue::Url(url) => {
                let exposed = url.expose_secret();
                let text = exposed.expose_secret();
                return parse_object_url(text)
                    .is_ok_and(|u| u.scheme == krishiv_io::ObjectScheme::Azure)
                    && self.allows_object(text);
            }
            OptionValue::Secret(_) => return false,
        };
        let Some(local) = text.strip_prefix("file:") else {
            return self.allows_object(text);
        };
        let local = &krishiv_exec::files::native_path(local);
        if !plain_absolute(local) || text.contains('%') {
            return false;
        }
        resolve_links(local).is_some_and(|p| self.roots.iter().any(|r| p.starts_with(r)))
    }
}

impl PathPrefixPolicy {
    /// Whether the object-store URL `text` lies under one of the prefixes.
    fn allows_object(&self, text: &str) -> bool {
        parse_object_prefix(text).is_some_and(|p| {
            self.prefixes
                .iter()
                .any(|a| a.store == p.store && p.segments.starts_with(&a.segments))
        })
    }
}

/// Whether `local` is absolute and holds only plain names after its root: no `.`, no `..`. On
/// Windows it may start with a drive (`C:\`, or verbatim `\\?\C:\`), and nothing else: a UNC share
/// or device path is refused (fails closed).
fn plain_absolute(local: &Path) -> bool {
    let mut parts = local.components().peekable();
    if let Some(Component::Prefix(p)) = parts.peek() {
        if !matches!(p.kind(), Prefix::Disk(_) | Prefix::VerbatimDisk(_)) {
            return false;
        }
        parts.next();
    }
    local.is_absolute()
        && parts.next() == Some(Component::RootDir)
        && parts.all(|c| matches!(c, Component::Normal(_)))
}

/// The comma-separated entries of the string setting `entry` (blank entries dropped).
fn config_list(config: &Config, entry: &krishiv_common::config::ConfigEntry) -> Vec<String> {
    match config.get(entry) {
        Some(ConfigValue::String(raw)) => raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

/// `krishiv.io.local.allowedRoots`, canonical (links resolved); empty when unset. The policy
/// checks paths against them and the runtime's local store refuses listings that follow a
/// link out of them (ADR 0046 §3a).
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` naming the key for a root that is not an absolute path to
/// an existing directory.
pub(crate) fn allowed_roots(config: &Config) -> Result<Vec<PathBuf>, KrishivError> {
    config_list(config, &IO_LOCAL_ALLOWED_ROOTS)
        .into_iter()
        .map(|root| {
            let path = Path::new(&root);
            std::fs::canonicalize(path)
                .ok()
                .filter(|c| path.is_absolute() && c.is_dir())
                .ok_or_else(|| {
                    KrishivError::from(KrishivInvalidConfigValue {
                        value: format!("'{root}'"),
                        key: format!("\"{}\"", IO_LOCAL_ALLOWED_ROOTS.key),
                        reason: "expected absolute paths of existing directories, comma \
                                 separated"
                            .to_owned(),
                    })
                })
        })
        .collect()
}

/// `text` as an object-store store and key segments, if it is a plain object-store URL: a
/// bucket, no credentials, no `%` (percent-encoded text would not be compared the way the
/// store reads it), no query or fragment, and no `.`, `..` or empty key segment (the URL
/// parser would fold them, so the text written would not be the key compared). A trailing
/// `/` is allowed.
fn parse_object_prefix(text: &str) -> Option<ObjectPrefix> {
    if text.contains(['%', '?', '#']) {
        return None;
    }
    let (_, rest) = text.split_once("://")?;
    let key = rest.split_once('/').map_or("", |(_, key)| key);
    let key = key.strip_suffix('/').unwrap_or(key);
    if !key.is_empty() && key.split('/').any(|s| matches!(s, "" | "." | "..")) {
        return None;
    }
    let url = parse_object_url(text).ok()?;
    let segments = url
        .key
        .as_ref()
        .split('/')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    Some(ObjectPrefix {
        store: url.store_id,
        segments,
    })
}

/// `path` with the symbolic links of its longest existing prefix resolved
/// (the rest does not exist yet, so it holds no links).
pub(crate) fn resolve_links(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    loop {
        match std::fs::canonicalize(&existing) {
            Ok(resolved) => return Some(rest.iter().rev().fold(resolved, |p, part| p.join(part))),
            // Only a name that is not there at all is kept as written (below a file it cannot be there
            // either; the write then fails as a storage error). A dangling link (its own entry exists), a
            // link loop or an unreadable directory fails closed.
            Err(e)
                if absent(&e)
                    && std::fs::symlink_metadata(&existing).is_err_and(|m| absent(&m)) => {}
            Err(_) => return None,
        }
        // `None` for a missing part ending in `..`: it is not resolved against a directory that is not there.
        rest.push(existing.file_name()?.to_owned());
        if !existing.pop() {
            return None;
        }
    }
}

/// Whether `a` and `b` name the same place, a drive written plain (`C:\x`) or verbatim (`\\?\C:\x`) alike:
/// `std::fs::canonicalize` gives the verbatim form on Windows, while a path made from a `file:/C:/x`
/// location is plain. Any other prefix (a UNC share, a device) must match as written.
pub(crate) fn same_place(a: &Path, b: &Path) -> bool {
    fn drive(c: Component<'_>) -> Option<u8> {
        match c {
            Component::Prefix(p) => match p.kind() {
                Prefix::Disk(d) | Prefix::VerbatimDisk(d) => Some(d.to_ascii_uppercase()),
                _ => None,
            },
            _ => None,
        }
    }
    let (mut x, mut y) = (a.components(), b.components());
    loop {
        match (x.next(), y.next()) {
            (None, None) => return true,
            (Some(p), Some(q)) if p == q => {}
            (Some(p), Some(q)) => match (drive(p), drive(q)) {
                (Some(d), Some(e)) if d == e => {}
                _ => return false,
            },
            _ => return false,
        }
    }
}

/// Whether `e` says the entry is not there: missing, or below a file.
fn absent(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
    )
}

impl AccessPolicy for PathPrefixPolicy {
    fn version(&self) -> PolicyVersion {
        self.inner.version()
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        if let Securable::Path(path) = securable
            && !self.allows(path)
        {
            return Box::pin(std::future::ready(Ok(AccessDecision::Deny {
                reason: "path outside the allowed roots".to_owned(),
            })));
        }
        self.inner.check(principal, securable, privilege)
    }

    fn column_level(&self) -> bool {
        self.inner.column_level()
    }

    fn denied(&self, principal: &Principal, securable: &Securable, privilege: Privilege) {
        self.inner.denied(principal, securable, privilege);
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use krishiv_common::secret::OptionValue;
    use krishiv_plan::analyzer::AllowAll;

    use super::{PathPrefixPolicy, plain_absolute, same_place};

    #[test]
    fn plain_absolute_paths() {
        assert!(!plain_absolute(Path::new("data/x")));
        // `/data/x` has no drive, so it is not absolute on Windows.
        if cfg!(unix) {
            assert!(plain_absolute(Path::new("/data/x")));
            assert!(!plain_absolute(Path::new("/data/../x")));
        }
        if cfg!(windows) {
            assert!(plain_absolute(Path::new(r"C:\data\x")));
            assert!(plain_absolute(Path::new("C:/data/x")));
            assert!(plain_absolute(Path::new(r"\\?\C:\data\x")));
            // Drive-relative, a share and a device path are refused.
            assert!(!plain_absolute(Path::new("C:data")));
            assert!(!plain_absolute(Path::new(r"\\srv\share\x")));
            assert!(!plain_absolute(Path::new(r"\\.\COM1")));
            assert!(!plain_absolute(Path::new(r"C:\data\..\x")));
        }
    }

    /// An allowed root admits the files under it in every spelling a local path takes on this
    /// platform, and refuses a sibling and a `..` escape.
    #[test]
    fn an_allowed_root_admits_its_files_and_nothing_beside_it() {
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().join("allowed");
        let sibling = tmp.path().join("other");
        std::fs::create_dir(&root).expect("root");
        std::fs::create_dir(&sibling).expect("sibling");
        std::fs::write(root.join("a.csv"), "x").expect("file");
        let policy = PathPrefixPolicy {
            roots: vec![std::fs::canonicalize(&root).expect("canonical")],
            prefixes: Vec::new(),
            inner: Arc::new(AllowAll),
        };
        let allows = |text: String| policy.allows(&OptionValue::Plain(text));
        let slashed = |p: &Path| p.display().to_string().replace('\\', "/");
        let file = root.join("a.csv");
        let mut admitted = vec![format!("file:{}", slashed(&file))];
        if cfg!(windows) {
            // `LocalPaths` writes a drive path as `file:/C:/x`; `C:\x` text is accepted too.
            admitted.push(format!("file:/{}", slashed(&file)));
            admitted.push(format!("file:{}", file.display()));
        }
        for text in admitted {
            assert!(allows(text.clone()), "{text}");
        }
        assert!(!allows(format!("file:{}", slashed(&sibling.join("b.csv")))));
        assert!(!allows(format!("file:{}/../other/b.csv", slashed(&root))));
    }

    /// A canonical path is the same place as the path it came from, so the directory checks hold where
    /// `canonicalize` writes a drive verbatim (Windows).
    #[test]
    fn a_canonical_path_is_the_same_place() {
        let tmp = tempfile::tempdir().expect("tmp");
        let canonical = std::fs::canonicalize(tmp.path()).expect("canonical");
        assert!(same_place(&canonical, &canonical));
        assert!(!same_place(&canonical, &canonical.join("x")));
        assert!(!same_place(&canonical.join("x"), &canonical.join("y")));
        if cfg!(windows) {
            assert!(same_place(Path::new(r"\\?\C:\a\b"), Path::new(r"C:\a\b")));
            assert!(same_place(Path::new(r"\\?\c:\a"), Path::new(r"C:\a")));
            assert!(!same_place(Path::new(r"\\?\D:\a"), Path::new(r"C:\a")));
            assert!(!same_place(Path::new(r"\\srv\share\a"), Path::new(r"C:\a")));
        }
    }
}
