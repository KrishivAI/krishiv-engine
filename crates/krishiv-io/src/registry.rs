// SPDX-License-Identifier: Apache-2.0
//! The file-format registry (P2-T182): name-based resolution of
//! [`FileFormat`] implementations.
//!
//! Plans and catalog entries carry the format NAME and
//! [`FormatOptions`](crate::FormatOptions) as data; the worker resolves the
//! name against its own registry when it instantiates the scan. The registry
//! is an engine-instance object (built once per runtime, shared with
//! sessions), never a `static`, so connectors can add formats without
//! touching production code.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::error::{DataSourceNotFound, KrishivError};

use crate::FileFormat;

/// Name-based resolution of [`FileFormat`] implementations.
#[derive(Default)]
pub struct FormatRegistry {
    /// Formats by lower-case name.
    by_name: BTreeMap<String, Arc<dyn FileFormat>>,
}

impl std::fmt::Debug for FormatRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FormatRegistry")
            .field("formats", &self.names())
            .finish()
    }
}

impl FormatRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            by_name: BTreeMap::new(),
        }
    }

    /// The built-in formats (Parquet, CSV, JSON, text).
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        // Production code outside `crate::parquet` never names a format
        // type directly; this constructor is the one exemption the
        // `check-format-neutral` lint allows.
        let _ = r.register(Arc::new(crate::parquet::ParquetFormat));
        let _ = r.register(Arc::new(crate::csv::CsvFormat));
        let _ = r.register(Arc::new(crate::json::JsonFormat));
        let _ = r.register(Arc::new(crate::text_format::TextFormat));
        r
    }

    /// Adds `format` under its [`FileFormat::name`]; names compare
    /// case-insensitively.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when a format of that name is already registered.
    pub fn register(&mut self, format: Arc<dyn FileFormat>) -> crate::error::Result<()> {
        let key = format.name().to_ascii_lowercase();
        if self.by_name.contains_key(&key) {
            return Err(crate::error::internal(format!(
                "a file format named {} is already registered",
                format.name()
            )));
        }
        self.by_name.insert(key, format);
        Ok(())
    }

    /// The format `name` (case-insensitive).
    ///
    /// # Errors
    /// Spark's `DATA_SOURCE_NOT_FOUND` for an unknown name.
    pub fn get(&self, name: &str) -> Result<Arc<dyn FileFormat>, KrishivError> {
        let key = name.to_ascii_lowercase();
        self.by_name.get(&key).cloned().ok_or_else(|| {
            DataSourceNotFound {
                provider: name.to_owned(),
            }
            .into()
        })
    }

    /// Whether a format `name` is registered (case-insensitive).
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(&name.to_ascii_lowercase())
    }

    /// The registered names, in order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.by_name.values().map(|f| f.name().to_owned()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_names_fail_with_data_source_not_found() {
        let r = FormatRegistry::with_defaults();
        for name in ["parquet", "PARQUET", "csv", "json", "text"] {
            assert!(r.get(name).is_ok(), "{name}");
            assert!(r.contains(name), "{name}");
        }
        let Err(e) = r.get("delta") else {
            panic!("unknown format resolved")
        };
        assert_eq!(e.error_class().name(), "DATA_SOURCE_NOT_FOUND");
        assert_eq!(e.sql_state(), Some("42K02"));
        assert!(e.to_string().contains("delta"), "{e}");
        // A registry that only knows one format still resolves it by case.
        let mut single = FormatRegistry::new();
        single.register(Arc::new(crate::csv::CsvFormat)).unwrap();
        assert!(single.get("CSV").is_ok());
        assert!(!single.contains("parquet"));
    }

    #[test]
    fn duplicates_are_rejected_case_insensitively() {
        let mut r = FormatRegistry::new();
        r.register(Arc::new(crate::csv::CsvFormat)).unwrap();
        assert!(r.register(Arc::new(crate::csv::CsvFormat)).is_err());
    }

    #[test]
    fn defaults_cover_the_built_in_file_sources() {
        let r = FormatRegistry::with_defaults();
        assert_eq!(r.names(), ["csv", "json", "parquet", "text"]);
    }
}
