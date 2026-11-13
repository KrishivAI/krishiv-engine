// SPDX-License-Identifier: Apache-2.0
//! Table-root confinement of every path the kernel asks for (design note P3-T021 section 3.2,
//! ADR 0046 section 3a).
//!
//! The session confines the table root itself; here every URL a handler is given (commit files,
//! checkpoints, sidecars, deletion vectors, data files) must lie under that root: the same scheme
//! and authority, and a path inside the root's path. The `url` crate removes `.` and `..`
//! segments when a URL is parsed, so a log entry such as `../other/x.parquet` resolved against
//! the root lands outside it and is refused here; a percent-encoded `%2E%2E` segment is refused
//! by the object path's own parse. Absolute paths to another bucket or host fail the authority
//! check.
//!
//! On a local (`file://`) table the check is repeated on the canonical path, symlinks resolved,
//! so a link under the root that points outside it (a commit file, a `_sidecars` directory) is
//! refused too. A path that does not exist yet (a commit about to be written) resolves through
//! its deepest existing ancestor. The check runs before the I/O, so a link swapped in between
//! is not caught; object stores have no links.

use std::path::{Path as FsPath, PathBuf};

use krishiv_common::error::{KrishivError, UnsupportedFeatureTableOperation};
use object_store::path::Path;
use url::Url;

/// The table root and the object paths under it.
#[derive(Debug, Clone)]
pub struct TableRoot {
    root: Url,
    /// The canonical directory of a local root, symlinks resolved.
    local: Option<PathBuf>,
}

impl TableRoot {
    /// The root `url`, made directory-like (a trailing `/`).
    #[must_use]
    pub fn new(mut url: Url) -> Self {
        if !url.path().ends_with('/') {
            let path = format!("{}/", url.path());
            url.set_path(&path);
        }
        let local = local_path(&url).map(|p| resolve(&p));
        Self { root: url, local }
    }

    /// The root URL (ends with `/`).
    #[must_use]
    pub fn url(&self) -> &Url {
        &self.root
    }

    /// The object path of `url`, which must lie under the root.
    ///
    /// # Errors
    /// `UNSUPPORTED_FEATURE.TABLE_OPERATION` naming the path, for a URL outside the root.
    pub fn confine(&self, url: &Url) -> Result<Path, KrishivError> {
        let inside = url.scheme() == self.root.scheme()
            && url.username() == self.root.username()
            && url.host_str() == self.root.host_str()
            && url.port_or_known_default() == self.root.port_or_known_default()
            && (url.path().starts_with(self.root.path())
                || format!("{}/", url.path()) == self.root.path())
            && url.query().is_none()
            && url.fragment().is_none();
        if !inside {
            return Err(outside(url.as_str(), self.root.as_str()));
        }
        let path = Path::from_url_path(url.path())
            .map_err(|_| outside(url.as_str(), self.root.as_str()))?;
        if let Some(root) = &self.local {
            let resolved = local_path(url).map(|p| resolve(&p));
            if !resolved.is_some_and(|p| p.starts_with(root)) {
                return Err(outside(url.as_str(), self.root.as_str()));
            }
        }
        Ok(path)
    }

    /// Whether `url`, which does not end with `/`, names the root itself.
    #[must_use]
    pub fn is_root_without_slash(&self, url: &Url) -> bool {
        let mut with_slash = url.clone();
        with_slash.set_path(&format!("{}/", url.path()));
        with_slash == self.root
    }

    /// The URL of the object `path` of this root's store.
    ///
    /// # Errors
    /// As [`Self::confine`], for a path outside the root.
    pub fn url_of(&self, path: &Path) -> Result<Url, KrishivError> {
        // Each part is percent-encoded on its own, so a literal `%` in a key (a Spark-escaped
        // partition directory such as `p=a%3Ab`) survives the round trip through `confine`.
        let mut url = self.root.clone();
        url.path_segments_mut()
            .map_err(|()| outside(path.as_ref(), self.root.as_str()))?
            .clear()
            .extend(path.parts().map(|part| part.as_ref().to_owned()));
        self.confine(&url)?;
        Ok(url)
    }
}

/// The local file path of a `file://` URL.
fn local_path(url: &Url) -> Option<PathBuf> {
    if url.scheme() == "file" {
        url.to_file_path().ok()
    } else {
        None
    }
}

/// `path` with symlinks resolved: its deepest existing ancestor made canonical, then the parts
/// that do not exist yet appended as they are (the URL parser has removed `.` and `..`).
fn resolve(path: &FsPath) -> PathBuf {
    let mut rest = Vec::new();
    let mut at = path;
    loop {
        if let Ok(canonical) = at.canonicalize() {
            return rest
                .into_iter()
                .rev()
                .fold(canonical, |p, part| p.join(part));
        }
        match (at.parent(), at.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_owned());
                at = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

fn outside(path: &str, root: &str) -> KrishivError {
    UnsupportedFeatureTableOperation {
        table_name: format!("`{root}`"),
        operation: format!("reading or writing {path}, which is outside the table root"),
    }
    .into()
}
