// SPDX-License-Identifier: Apache-2.0
//! The golden-file exclusion manifest: files the engine will not run, each with a reason class
//! from a fixed list. An unknown reason class, an unknown file or a duplicate is an error.
//!
//! ```json
//! {"format_version": 1, "frozen": false,
//!  "exclusions": [{"file": "udf/udf-union", "reason": "jvm_udf", "note": "optional"}]}
//! ```
//!
//! Once `frozen` is true, every entry added afterwards must carry `"added_after_freeze": true`;
//! the report counts those separately.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::GoldenSet;

/// Manifest format version this loader reads.
pub const MANIFEST_FORMAT_VERSION: u64 = 1;

/// Why a golden file is excluded; the fixed list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReasonClass {
    /// Exercises JVM (Scala/Java) user-defined functions.
    JvmUdf,
    /// Exercises Python or pandas user-defined functions.
    PythonUdf,
    /// Needs a JVM-only API, class or catalog extension.
    JvmOnlyApi,
    /// Needs an external system (metastore, remote service).
    ExternalSystem,
    /// Output differs by JDK or platform.
    JdkSpecific,
    /// Result depends on time, randomness or the environment.
    NonDeterministicEnvironment,
    /// A Spark feature Krishiv deliberately does not provide.
    UnsupportedByDesign,
    /// The upstream golden is itself wrong or flaky.
    UpstreamDefect,
}

impl ReasonClass {
    /// Every reason class.
    pub const ALL: [Self; 8] = [
        Self::JvmUdf,
        Self::PythonUdf,
        Self::JvmOnlyApi,
        Self::ExternalSystem,
        Self::JdkSpecific,
        Self::NonDeterministicEnvironment,
        Self::UnsupportedByDesign,
        Self::UpstreamDefect,
    ];

    /// The manifest key of the class.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::JvmUdf => "jvm_udf",
            Self::PythonUdf => "python_udf",
            Self::JvmOnlyApi => "jvm_only_api",
            Self::ExternalSystem => "external_system",
            Self::JdkSpecific => "jdk_specific",
            Self::NonDeterministicEnvironment => "nondeterministic_environment",
            Self::UnsupportedByDesign => "unsupported_by_design",
            Self::UpstreamDefect => "upstream_defect",
        }
    }

    /// Look a class up by its manifest key.
    #[must_use]
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.key() == key)
    }
}

/// A manifest problem.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// Reading the file failed.
    #[error("reading {path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// Not valid JSON, or the wrong shape.
    #[error("invalid exclusion manifest: {0}")]
    Invalid(String),
    /// A reason class outside the fixed list.
    #[error("exclusion `{file}`: unknown reason class `{reason}`")]
    UnknownReason {
        /// Excluded file.
        file: String,
        /// The unknown class.
        reason: String,
    },
    /// The file is listed twice.
    #[error("exclusion `{0}` is listed twice")]
    Duplicate(String),
    /// The file is not a golden file of the set.
    #[error("exclusion `{0}` is not a golden file")]
    UnknownFile(String),
}

/// One excluded file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exclusion {
    /// Golden name relative to `inputs/`, without `.sql`.
    pub file: String,
    /// Why it is excluded.
    pub reason: ReasonClass,
    /// Free-text explanation.
    pub note: Option<String>,
    /// Added after the manifest was frozen.
    pub added_after_freeze: bool,
}

/// The exclusion manifest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExclusionManifest {
    /// Whether the manifest has been frozen.
    pub frozen: bool,
    /// Entries, in file order.
    pub entries: Vec<Exclusion>,
}

impl ExclusionManifest {
    /// Default manifest path in the repo.
    #[must_use]
    pub fn default_path() -> PathBuf {
        crate::tests_dir()
            .join("spark-golden")
            .join("exclusions.json")
    }

    /// Load a manifest from `path`.
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let text = std::fs::read_to_string(path).map_err(|source| ManifestError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text)
    }

    /// Parse manifest JSON text.
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        let bad = |m: &str| ManifestError::Invalid(m.to_owned());
        let root: Value =
            serde_json::from_str(text).map_err(|e| ManifestError::Invalid(e.to_string()))?;
        if root.get("format_version").and_then(Value::as_u64) != Some(MANIFEST_FORMAT_VERSION) {
            return Err(bad("unsupported or missing `format_version`"));
        }
        let frozen = root
            .get("frozen")
            .and_then(Value::as_bool)
            .ok_or_else(|| bad("missing `frozen`"))?;
        let raw = root
            .get("exclusions")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("missing `exclusions` array"))?;
        let mut seen = BTreeSet::new();
        let mut entries = Vec::with_capacity(raw.len());
        for e in raw {
            let text = |k: &str| e.get(k).and_then(Value::as_str);
            let file = text("file")
                .ok_or_else(|| bad("exclusion without `file`"))?
                .to_owned();
            let key = text("reason").ok_or_else(|| bad("exclusion without `reason`"))?;
            let reason =
                ReasonClass::from_key(key).ok_or_else(|| ManifestError::UnknownReason {
                    file: file.clone(),
                    reason: key.to_owned(),
                })?;
            let added_after_freeze = e
                .get("added_after_freeze")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if added_after_freeze && !frozen {
                return Err(bad("`added_after_freeze` on a manifest that is not frozen"));
            }
            if !seen.insert(file.clone()) {
                return Err(ManifestError::Duplicate(file));
            }
            entries.push(Exclusion {
                file,
                reason,
                note: text("note").map(str::to_owned),
                added_after_freeze,
            });
        }
        Ok(Self { frozen, entries })
    }

    /// Check every entry names a golden file of `set`.
    pub fn validate(&self, set: &GoldenSet) -> Result<(), ManifestError> {
        let known: BTreeSet<&str> = set
            .pairs
            .iter()
            .map(|p| p.name.as_str())
            .chain(set.unpaired_inputs.iter().map(String::as_str))
            .collect();
        match self
            .entries
            .iter()
            .find(|e| !known.contains(e.file.as_str()))
        {
            Some(e) => Err(ManifestError::UnknownFile(e.file.clone())),
            None => Ok(()),
        }
    }

    /// The entry excluding `file`, if any.
    #[must_use]
    pub fn get(&self, file: &str) -> Option<&Exclusion> {
        self.entries.iter().find(|e| e.file == file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn parses_entries_and_rejects_unknown_reason() -> R {
        let m = ExclusionManifest::parse(
            r#"{"format_version":1,"frozen":true,"exclusions":[
              {"file":"a","reason":"jvm_udf"},
              {"file":"b","reason":"jdk_specific","note":"n","added_after_freeze":true}]}"#,
        )?;
        assert!(m.frozen);
        assert_eq!(m.get("b").map(|e| e.added_after_freeze), Some(true));
        assert_eq!(m.get("a").map(|e| e.reason), Some(ReasonClass::JvmUdf));
        let err = ExclusionManifest::parse(
            r#"{"format_version":1,"frozen":false,"exclusions":[{"file":"a","reason":"because"}]}"#,
        );
        assert!(matches!(err, Err(ManifestError::UnknownReason { .. })));
        Ok(())
    }

    #[test]
    fn rejects_duplicates_stale_files_and_premature_freeze_flag() -> R {
        let dup = r#"{"format_version":1,"frozen":false,"exclusions":[
          {"file":"a","reason":"jvm_udf"},{"file":"a","reason":"jvm_udf"}]}"#;
        assert!(matches!(
            ExclusionManifest::parse(dup),
            Err(ManifestError::Duplicate(_))
        ));
        let early = r#"{"format_version":1,"frozen":false,"exclusions":[
          {"file":"a","reason":"jvm_udf","added_after_freeze":true}]}"#;
        assert!(matches!(
            ExclusionManifest::parse(early),
            Err(ManifestError::Invalid(_))
        ));
        let m = ExclusionManifest::parse(
            r#"{"format_version":1,"frozen":false,"exclusions":[{"file":"zzz","reason":"jvm_udf"}]}"#,
        )?;
        assert!(matches!(
            m.validate(&GoldenSet::default()),
            Err(ManifestError::UnknownFile(_))
        ));
        Ok(())
    }

    #[test]
    fn reason_keys_round_trip() {
        for r in ReasonClass::ALL {
            assert_eq!(ReasonClass::from_key(r.key()), Some(r));
        }
    }
}
