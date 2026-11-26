// SPDX-License-Identifier: Apache-2.0
//! The `options` map of `rewrite_data_files` (design P3-T027 PR-36e §4.3): Java's names and
//! defaults (`RewriteDataFiles`, `SizeBasedFileRewritePlanner`, `BinPackRewriteFilePlanner`,
//! apache-iceberg 1.12.0). Planning ([`super::rewrite_plan`]) reads the resolved values only.
//!
//! Only the unknown-option text was confirmed by a real Spark 4.1.0 run (design §8, Q4); the
//! texts of malformed values are Krishiv's until the golden run pins Java's.

#![expect(
    dead_code,
    reason = "the planner and the procedure's call arm (next commits of 36e-2a) read these"
)]

use std::collections::BTreeMap;

use krishiv_common::error::{KrishivError, KrishivIcebergProcedureArgument};

/// 512 MiB: `write.target-file-size-bytes`'s default, the planner's target when the option and
/// the table property are both absent.
pub(crate) const DEFAULT_TARGET_FILE_SIZE: i64 = 512 * 1024 * 1024;

const MIN_RATIO: f64 = 0.75;
const MAX_RATIO: f64 = 1.8;

/// Every name the bin-pack action accepts (`cache-delete-files` is accepted and ignored).
const KNOWN: &[&str] = &[
    "target-file-size-bytes",
    "min-file-size-bytes",
    "max-file-size-bytes",
    "min-input-files",
    "rewrite-all",
    "max-file-group-size-bytes",
    "max-file-group-input-files",
    "delete-file-threshold",
    "delete-ratio-threshold",
    "max-files-to-rewrite",
    "rewrite-job-order",
    "partial-progress.enabled",
    "partial-progress.max-commits",
    "partial-progress.max-failed-commits",
    "max-concurrent-file-group-rewrites",
    "use-starting-sequence-number",
    "remove-dangling-deletes",
    "output-spec-id",
    "cache-delete-files",
];

/// `rewrite-job-order`: the order groups are rewritten in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobOrder {
    None,
    BytesAsc,
    BytesDesc,
    FilesAsc,
    FilesDesc,
}

/// The resolved options of one `rewrite_data_files` call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RewriteOptions {
    pub target_file_size: i64,
    pub min_file_size: i64,
    pub max_file_size: i64,
    pub min_input_files: usize,
    pub rewrite_all: bool,
    pub max_group_size: i64,
    pub max_group_input_files: usize,
    pub delete_file_threshold: usize,
    pub delete_ratio_threshold: f64,
    pub max_files_to_rewrite: Option<usize>,
    pub job_order: JobOrder,
    pub partial_progress: bool,
    pub max_commits: usize,
    pub max_failed_commits: usize,
    pub max_concurrent_groups: usize,
    pub use_starting_sequence_number: bool,
    pub remove_dangling_deletes: bool,
    pub output_spec_id: Option<i32>,
}

fn bad(message: String) -> KrishivError {
    KrishivIcebergProcedureArgument { message }.into()
}

struct Reader<'a> {
    map: &'a BTreeMap<String, String>,
}

impl Reader<'_> {
    fn raw(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(String::as_str)
    }

    fn long(&self, name: &str) -> Result<Option<i64>, KrishivError> {
        self.raw(name)
            .map(|v| {
                v.trim()
                    .parse::<i64>()
                    .map_err(|_| bad(format!("Cannot parse {name}: '{v}' is not a long")))
            })
            .transpose()
    }

    fn count(&self, name: &str) -> Result<Option<usize>, KrishivError> {
        match self.long(name)? {
            None => Ok(None),
            Some(n) => usize::try_from(n)
                .map(Some)
                .map_err(|_| bad(format!("{name} should have value >= 0, value: {n}"))),
        }
    }

    fn flag(&self, name: &str) -> Result<Option<bool>, KrishivError> {
        // Java's `Boolean.parseBoolean`-style properties: only "true"/"false", any case.
        match self.raw(name).map(str::to_ascii_lowercase).as_deref() {
            None => Ok(None),
            Some("true") => Ok(Some(true)),
            Some("false") => Ok(Some(false)),
            Some(_) => Err(bad(format!("Cannot parse {name}: not a boolean"))),
        }
    }
}

impl RewriteOptions {
    /// The options of `map` over `table_target`, the table's `write.target-file-size-bytes`
    /// when it sets one.
    ///
    /// # Errors
    /// `KRISHIV_ICEBERG_PROCEDURE_ARGUMENT` for an unknown option (Java's text, confirmed by a
    /// Spark run) or a malformed or out-of-range value.
    pub(crate) fn parse(
        map: &BTreeMap<String, String>,
        table_target: Option<i64>,
    ) -> Result<Self, KrishivError> {
        let mut unknown: Vec<&str> = map
            .keys()
            .map(String::as_str)
            .filter(|k| !KNOWN.contains(k))
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(bad(format!(
                "Cannot use options [{}], they are not supported by the action or the rewriter BIN-PACK",
                unknown.join(", ")
            )));
        }
        let r = Reader { map };
        let target = r
            .long("target-file-size-bytes")?
            .or(table_target)
            .unwrap_or(DEFAULT_TARGET_FILE_SIZE);
        let min_file_size = r
            .long("min-file-size-bytes")?
            .unwrap_or_else(|| ratio(target, MIN_RATIO));
        let max_file_size = r
            .long("max-file-size-bytes")?
            .unwrap_or_else(|| ratio(target, MAX_RATIO));
        let max_group_size = r
            .long("max-file-group-size-bytes")?
            .unwrap_or(100 * 1024 * 1024 * 1024);
        let min_input_files = r.count("min-input-files")?.unwrap_or(5);
        let max_commits = r.count("partial-progress.max-commits")?.unwrap_or(10);
        if r.flag("partial-progress.enabled")?.unwrap_or(false) && max_commits == 0 {
            return Err(bad(format!(
                "partial-progress.max-commits should have value > 0, value: {max_commits}"
            )));
        }
        let max_concurrent_groups = r.count("max-concurrent-file-group-rewrites")?.unwrap_or(5);
        if max_concurrent_groups == 0 {
            return Err(bad(
                "max-concurrent-file-group-rewrites should have value > 0, value: 0".to_owned(),
            ));
        }
        let ratio_threshold = match r.raw("delete-ratio-threshold") {
            None => 0.3,
            Some(v) => v.trim().parse::<f64>().map_err(|_| {
                bad(format!(
                    "Cannot parse delete-ratio-threshold: '{v}' is not a double"
                ))
            })?,
        };
        let output_spec_id = match r.long("output-spec-id")? {
            None => None,
            Some(n) => Some(
                i32::try_from(n)
                    .map_err(|_| bad(format!("output-spec-id {n} is not a partition spec id")))?,
            ),
        };
        Ok(Self {
            target_file_size: target,
            min_file_size,
            max_file_size,
            min_input_files,
            rewrite_all: r.flag("rewrite-all")?.unwrap_or(false),
            max_group_size,
            max_group_input_files: r.count("max-file-group-input-files")?.unwrap_or(usize::MAX),
            delete_file_threshold: r.count("delete-file-threshold")?.unwrap_or(usize::MAX),
            delete_ratio_threshold: ratio_threshold,
            max_files_to_rewrite: r.count("max-files-to-rewrite")?,
            job_order: job_order(r.raw("rewrite-job-order"))?,
            partial_progress: r.flag("partial-progress.enabled")?.unwrap_or(false),
            max_commits,
            max_failed_commits: r
                .count("partial-progress.max-failed-commits")?
                .unwrap_or(max_commits),
            max_concurrent_groups,
            use_starting_sequence_number: r.flag("use-starting-sequence-number")?.unwrap_or(true),
            remove_dangling_deletes: r.flag("remove-dangling-deletes")?.unwrap_or(false),
            output_spec_id,
        })
    }
}

/// `(long) (target * ratio)`, Java's cast: truncation.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Java computes these as (long) (target * ratio) in double arithmetic"
)]
fn ratio(target: i64, ratio: f64) -> i64 {
    (target as f64 * ratio) as i64
}

fn job_order(v: Option<&str>) -> Result<JobOrder, KrishivError> {
    Ok(match v.map(str::to_ascii_lowercase).as_deref() {
        None | Some("none") => JobOrder::None,
        Some("bytes-asc") => JobOrder::BytesAsc,
        Some("bytes-desc") => JobOrder::BytesDesc,
        Some("files-asc") => JobOrder::FilesAsc,
        Some("files-desc") => JobOrder::FilesDesc,
        Some(other) => return Err(bad(format!("Invalid rewrite job order name: {other}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, &str)]) -> Result<RewriteOptions, KrishivError> {
        let m = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        RewriteOptions::parse(&m, None)
    }

    #[test]
    fn defaults_follow_java() {
        let o = opts(&[]).unwrap();
        assert_eq!(o.target_file_size, 536_870_912);
        assert_eq!(o.min_file_size, 402_653_184);
        assert_eq!(o.max_file_size, 966_367_641);
        assert_eq!(o.min_input_files, 5);
        assert_eq!(o.max_commits, 10);
        assert_eq!(o.max_failed_commits, 10);
        assert_eq!(o.max_concurrent_groups, 5);
        assert!(o.use_starting_sequence_number);
        assert!(!o.rewrite_all && !o.partial_progress && !o.remove_dangling_deletes);
        assert_eq!(o.job_order, JobOrder::None);
        assert!((o.delete_ratio_threshold - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn min_and_max_follow_an_explicit_target_and_the_table_property() {
        let o = opts(&[("target-file-size-bytes", "1000")]).unwrap();
        assert_eq!((o.min_file_size, o.max_file_size), (750, 1800));
        let m = BTreeMap::new();
        let o = RewriteOptions::parse(&m, Some(2000)).unwrap();
        assert_eq!(o.target_file_size, 2000);
        let o = opts(&[("min-file-size-bytes", "1"), ("max-file-size-bytes", "9")]).unwrap();
        assert_eq!((o.min_file_size, o.max_file_size), (1, 9));
    }

    #[test]
    fn an_unknown_option_is_refused_with_javas_text() {
        let e = opts(&[("nope", "1"), ("alsonope", "2")])
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(
                "Cannot use options [alsonope, nope], they are not supported by the action or the rewriter BIN-PACK"
            ),
            "{e}"
        );
        assert!(opts(&[("cache-delete-files", "true")]).is_ok());
    }

    #[test]
    fn partial_progress_failed_commits_default_to_max_commits() {
        let o = opts(&[
            ("partial-progress.enabled", "TRUE"),
            ("partial-progress.max-commits", "3"),
        ])
        .unwrap();
        assert!(o.partial_progress);
        assert_eq!(o.max_failed_commits, 3);
        let o = opts(&[("partial-progress.max-failed-commits", "1")]).unwrap();
        assert_eq!(o.max_failed_commits, 1);
    }

    #[test]
    fn malformed_values_are_refused() {
        assert!(opts(&[("min-input-files", "x")]).is_err());
        assert!(opts(&[("min-input-files", "-1")]).is_err());
        assert!(opts(&[("rewrite-all", "yes")]).is_err());
        assert!(opts(&[("rewrite-job-order", "sideways")]).is_err());
        assert!(opts(&[("max-concurrent-file-group-rewrites", "0")]).is_err());
        assert_eq!(
            opts(&[("rewrite-job-order", "Bytes-Desc")])
                .unwrap()
                .job_order,
            JobOrder::BytesDesc
        );
    }
}
