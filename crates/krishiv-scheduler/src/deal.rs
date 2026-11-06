// SPDX-License-Identifier: Apache-2.0
//! [`TableDealer`]: a [`StageSplits`] that deals the files of named tables
//! to a stage's tasks (P2-T178): the coordinator's split assignment over
//! files it listed once, with their pinned versions.
//!
//! Only the scans of a fragment's streaming pipeline are dealt
//! ([`streaming_scans`]); every task reads the others whole. A fragment
//! with a scan of an unknown table, or of anything but a table, is not
//! dealt at all (one task reads it whole). A file its format cannot split
//! (gzip CSV, `multiLine` CSV or JSON) is dealt whole, as a local scan
//! keeps it ([`TableDealer::with_unsplittable`]).

use std::collections::{BTreeMap, BTreeSet};

use krishiv_common::error::KrishivError;
use krishiv_exec::physical::{
    PhysicalPlan, ScanId, ScanSplits, SplitFileRef, assign_file_splits, plan_file_partitions,
};
use krishiv_io::split::SplitConfig;
use krishiv_plan::ReadSource;

use crate::fragment::streaming_scans;
use crate::stages::StageSplits;

/// Deals the listed files of tables, by name, to tasks (see the module
/// docs).
#[derive(Debug, Clone)]
pub struct TableDealer {
    /// Each table's files in listing order, by its name parts joined
    /// with `.` as the plan names it (`spark_catalog.default.t`).
    files: BTreeMap<String, Vec<SplitFileRef>>,
    config: SplitConfig,
    /// The paths of files dealt whole.
    unsplittable: BTreeSet<String>,
}

impl TableDealer {
    /// A dealer of `files` (by table name, see [`TableDealer`]), planning
    /// splits with `config`.
    #[must_use]
    pub fn new(files: BTreeMap<String, Vec<SplitFileRef>>, config: SplitConfig) -> Self {
        Self {
            files,
            config,
            unsplittable: BTreeSet::new(),
        }
    }

    /// The same dealer keeping the files at `paths` whole: those its
    /// format's capabilities do not split (`is_splittable` false), so
    /// every byte of one is read by one task and `_metadata` reports the
    /// whole file, as a local scan does. Every other file is splittable.
    #[must_use]
    pub fn with_unsplittable(mut self, paths: impl IntoIterator<Item = String>) -> Self {
        self.unsplittable.extend(paths);
        self
    }
}

impl StageSplits for TableDealer {
    fn splits(
        &self,
        fragment: &PhysicalPlan,
        tasks: u32,
    ) -> Result<Option<Vec<BTreeMap<ScanId, ScanSplits>>>, KrishivError> {
        let scans = streaming_scans(fragment.root());
        if scans.is_empty() {
            return Ok(None);
        }
        let tasks = usize::try_from(tasks).unwrap_or(usize::MAX);
        let mut out = vec![BTreeMap::new(); tasks];
        for scan in scans {
            let ReadSource::Table { name, .. } = &scan.scan.source else {
                return Ok(None);
            };
            let Some(files) = self.files.get(&name.join(".")) else {
                return Ok(None);
            };
            let partitions = plan_file_partitions(&self.config, files, |f| {
                !self.unsplittable.contains(f.path.as_ref())
            });
            for (t, splits) in assign_file_splits(files, &partitions, tasks)
                .into_iter()
                .enumerate()
            {
                if let Some(task) = out.get_mut(t) {
                    task.insert(scan.id, splits);
                }
            }
        }
        Ok(Some(out))
    }
}
