// SPDX-License-Identifier: Apache-2.0
//! Names of Spark Connect artifacts (`AddArtifacts`, design P3-T011a §3.9),
//! validated before any file exists.
//!
//! Spark (`SparkConnectAddArtifactsHandler.StagedArtifact`, v4.3.0 lines
//! 200-225; `ArtifactUtils.concatenatePaths`, lines 26-40) accepts any
//! relative name whose normalized form stays below the staging directory,
//! splits one `#fragment` off an `archives/` name, and treats an unknown
//! prefix as a plain file. Krishiv is stricter, so a name means one file
//! below the session's directory on every file system:
//!
//! - one of the prefixes `cache/`, `pyfiles/`, `archives/`, `files/`;
//!   `jars/`, `classes/` (no JVM) and `forward_to_fs/` (no Hadoop file
//!   system; Spark refuses a local destination by default) are
//!   `KRISHIV_UNSUPPORTED_FEATURE`; any other is `INVALID_ARTIFACT_PATH`;
//! - relative, `/`-separated, no empty, `.` or `..` segment, no `\` (Spark
//!   turns it into a separator), no control character and no invisible
//!   formatting character (bidirectional overrides, zero-width
//!   characters), at most [`MAX_SEGMENTS`] segments of at most
//!   [`MAX_SEGMENT_BYTES`] bytes and [`MAX_NAME_BYTES`] bytes in all;
//! - `cache/` takes exactly one segment of ASCII letters and digits (the
//!   client sends a SHA-256 in hex), at most 128 characters;
//! - an `archives/` name may end in one `#fragment`, itself one valid
//!   segment; elsewhere `#` is part of the file name.
//!
//! Every refusal other than the unsupported prefixes is
//! `INVALID_ARTIFACT_PATH` (22023), Spark's class.

use krishiv_common::error::{InvalidArtifactPath, KrishivError, KrishivUnsupportedFeature};

/// The most bytes of a name (Linux `PATH_MAX`).
pub const MAX_NAME_BYTES: usize = 4096;
/// The most bytes of one segment (Linux `NAME_MAX`).
pub const MAX_SEGMENT_BYTES: usize = 255;
/// The most segments of a name, the prefix included.
pub const MAX_SEGMENTS: usize = 32;
/// The longest `cache/` hash.
const MAX_HASH_CHARS: usize = 128;

/// The kind of an artifact: the first segment of its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArtifactKind {
    /// `cache/<hash>`: a local relation the client uploads above its inline
    /// threshold, kept in memory (T012's `CachedLocalRelation` reads it).
    Cache,
    /// `pyfiles/...`: Python modules and archives for UDF workers (T014).
    PyFiles,
    /// `archives/...`: archives, unpacked by T014, never here.
    Archives,
    /// `files/...`: plain files.
    Files,
}

/// A validated artifact name: its kind, the relative path it is stored
/// under (without the fragment) and an archive's fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactName {
    kind: ArtifactKind,
    path: String,
    fragment: Option<String>,
}

impl ArtifactName {
    /// The kind (the first segment).
    #[must_use]
    pub fn kind(&self) -> ArtifactKind {
        self.kind
    }

    /// The relative path, `/`-separated, the prefix included and the
    /// fragment left out.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The segments of [`Self::path`]: each one non-empty, never `.` or
    /// `..`, without `/`, `\` or a control character.
    pub fn segments(&self) -> impl Iterator<Item = &str> {
        self.path.split('/')
    }

    /// An archive's `#fragment`, if it has one.
    #[must_use]
    pub fn fragment(&self) -> Option<&str> {
        self.fragment.as_deref()
    }

    /// The hash of a `cache/<hash>` name.
    #[must_use]
    pub fn cache_hash(&self) -> Option<&str> {
        match self.kind {
            ArtifactKind::Cache => self.path.strip_prefix("cache/"),
            _ => None,
        }
    }
}

/// Validates the artifact name `name` (module docs).
///
/// # Errors
/// `INVALID_ARTIFACT_PATH` for a name that is not a valid relative path
/// under a known prefix; `KRISHIV_UNSUPPORTED_FEATURE` for `jars/`,
/// `classes/` and `forward_to_fs/`.
pub fn parse_artifact_name(name: &str) -> Result<ArtifactName, KrishivError> {
    let invalid = || -> KrishivError {
        InvalidArtifactPath {
            name: name.to_owned(),
        }
        .into()
    };
    if name.len() > MAX_NAME_BYTES {
        return Err(invalid());
    }
    let (path, fragment) = match name.split_once('#') {
        Some((path, fragment)) if path.starts_with("archives/") => {
            if fragment.contains('#') {
                return Err(invalid());
            }
            let fragment = (!fragment.is_empty()).then_some(fragment);
            if fragment.is_some_and(|f| !valid_segment(f)) {
                return Err(invalid());
            }
            (path, fragment)
        }
        _ => (name, None),
    };
    let segments: Vec<&str> = path.split('/').collect();
    if segments.len() < 2
        || segments.len() > MAX_SEGMENTS
        || !segments.iter().all(|s| valid_segment(s))
    {
        return Err(invalid());
    }
    let kind = match segments[0] {
        "cache" => {
            let hash = segments[1];
            if segments.len() != 2
                || hash.len() > MAX_HASH_CHARS
                || !hash.bytes().all(|b| b.is_ascii_alphanumeric())
            {
                return Err(invalid());
            }
            ArtifactKind::Cache
        }
        "pyfiles" => ArtifactKind::PyFiles,
        "archives" => ArtifactKind::Archives,
        "files" => ArtifactKind::Files,
        prefix @ ("jars" | "classes" | "forward_to_fs") => {
            let why = if prefix == "forward_to_fs" {
                "copying to a file system (no Hadoop file system)"
            } else {
                "JVM classes and jars (Krishiv runs no JVM)"
            };
            return Err(KrishivUnsupportedFeature {
                feature: format!("AddArtifacts `{prefix}/`: {why}"),
            }
            .into());
        }
        _ => return Err(invalid()),
    };
    Ok(ArtifactName {
        kind,
        path: path.to_owned(),
        fragment: fragment.map(str::to_owned),
    })
}

/// One path segment: non-empty, not `.` or `..`, at most
/// [`MAX_SEGMENT_BYTES`] bytes, and no separator, control or invisible
/// formatting character.
fn valid_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment != "."
        && segment != ".."
        && segment.len() <= MAX_SEGMENT_BYTES
        && !segment.chars().any(forbidden)
}

/// A character no segment may hold: separators (`/`, `\`), control
/// characters (C0, DEL, C1) and the invisible formatting characters
/// (Unicode general category `Cf` in the ranges a name can use to look like
/// another: soft hyphen, Arabic letter mark, Mongolian vowel separator,
/// zero-width and directional marks, embeddings and overrides, invisible
/// operators, isolates, the byte order mark and interlinear annotations).
fn forbidden(c: char) -> bool {
    matches!(
        c,
        '/' | '\\'
            | '\u{0}'..='\u{1f}'
            | '\u{7f}'..='\u{9f}'
            | '\u{ad}'
            | '\u{61c}'
            | '\u{180e}'
            | '\u{200b}'..='\u{200f}'
            | '\u{202a}'..='\u{202e}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206f}'
            | '\u{feff}'
            | '\u{fff9}'..='\u{fffb}'
    )
}

#[cfg(test)]
mod tests;
