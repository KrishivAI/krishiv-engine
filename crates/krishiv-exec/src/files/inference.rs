// SPDX-License-Identifier: Apache-2.0
//! Which listed files a read scans, which its schema inference opens, and how it merges them.

use std::collections::BTreeMap;

use krishiv_common::config::{ConfigValue, IO_PARQUET_INFER_EMPTY_FILES};
use krishiv_common::secret::OptionValue;
use krishiv_io::FileRef;

use super::FileReads;

/// Takes the zero-byte files out of `files` (no Spark line scans one) and returns the files schema
/// inference opens when they are not the scanned ones: `all` (every listed file, where the partition
/// layout left some unread), else `files` as they were. With `skip`, inference leaves them out too.
pub(super) fn scanned_files(
    files: &mut Vec<FileRef>,
    all: Option<Vec<FileRef>>,
    skip: bool,
) -> Option<Vec<FileRef>> {
    let mut inferred = all;
    if files.iter().any(|f| f.size == 0) {
        inferred.get_or_insert_with(|| files.clone());
        files.retain(|f| f.size > 0);
    }
    if skip && let Some(inferred) = inferred.as_mut() {
        inferred.retain(|f| f.size > 0);
    }
    inferred
}

impl FileReads {
    /// Whether `format`'s schema inference leaves zero-byte files out: Parquet's under the Spark 4.3
    /// profile (`krishiv.io.parquet.inferFromEmptyFiles` off). Up to 4.2 it opens them and fails; CSV,
    /// JSON and text infer from them what an empty file gives.
    pub(super) fn skips_empty_files(&self, format: &str) -> bool {
        format == "parquet"
            && !matches!(
                self.session.get(&IO_PARQUET_INFER_EMPTY_FILES),
                Some(ConfigValue::Bool(true))
            )
    }

    /// Whether the read asks to merge the files' schemas: the
    /// `mergeSchema` option, else `spark.sql.<format>.mergeSchema`.
    pub(super) fn merges_schemas(
        &self,
        format: &str,
        options: &BTreeMap<String, OptionValue>,
    ) -> bool {
        let on = |v: &str| v.trim().eq_ignore_ascii_case("true");
        let option = options
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("mergeSchema"))
            .map(|(_, v)| v.as_plain().is_some_and(on));
        option.unwrap_or_else(|| {
            self.session
                .get_unregistered(&format!("spark.sql.{format}.mergeSchema"))
                .is_some_and(on)
        })
    }
}
