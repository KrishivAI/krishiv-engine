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

use krishiv_common::error::{KrishivError, UnsupportedFeatureTableOperation};
use object_store::path::Path;
use url::Url;

/// The table root and the object paths under it.
#[derive(Debug, Clone)]
pub struct TableRoot {
    root: Url,
}

impl TableRoot {
    /// The root `url`, made directory-like (a trailing `/`).
    #[must_use]
    pub fn new(mut url: Url) -> Self {
        if !url.path().ends_with('/') {
            let path = format!("{}/", url.path());
            url.set_path(&path);
        }
        Self { root: url }
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
        Path::from_url_path(url.path()).map_err(|_| outside(url.as_str(), self.root.as_str()))
    }

    /// The URL of the object `path` of this root's store.
    ///
    /// # Errors
    /// As [`Self::confine`], for a path outside the root.
    pub fn url_of(&self, path: &Path) -> Result<Url, KrishivError> {
        let mut url = self.root.clone();
        url.set_path(&format!("/{path}"));
        self.confine(&url)?;
        Ok(url)
    }
}

fn outside(path: &str, root: &str) -> KrishivError {
    UnsupportedFeatureTableOperation {
        table_name: format!("`{root}`"),
        operation: format!("reading or writing {path}, which is outside the table root"),
    }
    .into()
}
