// SPDX-License-Identifier: Apache-2.0
//! Opening the file tables an analyzed plan scans: listing their files
//! (reads by path, file views, file-backed tables), reading the footers on the
//! executor's I/O runtime, and the plan-time distinct-value samples that
//! follow ([`super::ndv_sampling`]).

use std::sync::{Arc, PoisonError};

use arrow_schema::{Schema, SchemaRef};
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::FileTable;
use krishiv_plan::{LogicalPlan, ReadSource};

use super::{FileScans, FilesKey, QueryCatalog, TempTable, collect_scans, reference_key, run_on};

/// A scan of files in an analyzed plan: what it reads, the table's columns
/// and the scan's column ids with their names.
pub(super) type FileScanOf = (
    FilesKey,
    ReadSource,
    SchemaRef,
    Vec<(krishiv_plan::AttrId, String)>,
);

impl QueryCatalog {
    /// Lists the files of every read by path and file view that the
    /// analyzed `plan` scans (subqueries included), and opens them as file
    /// tables for the planner. A read the analyzer listed keeps that
    /// listing; a view lists its files now.
    ///
    /// # Errors
    /// Listing errors (`PATH_NOT_FOUND` once a view's files are gone, ...)
    /// and errors reading the files' footers.
    pub(crate) async fn open_files(&self, plan: &LogicalPlan) -> Result<(), KrishivError> {
        self.check_reference_options(plan)?;
        let scans = self.file_scans(plan);
        for (key, source, table, _) in &scans {
            if self.opened(key).is_some() {
                continue;
            }
            let ReadSource::Files {
                format,
                paths,
                options,
            } = source
            else {
                continue;
            };
            let listing = match key {
                FilesKey::Table(_) | FilesKey::Reference { known: true, .. } => {
                    self.files
                        .listing_known(format, paths, options, Arc::clone(table))
                        .await?
                }
                _ => self.files.listing(format, paths, options).await?,
            };
            let opened = open_table(
                Arc::clone(&listing.format),
                self.scans.clone(),
                Arc::clone(table),
                listing.files.to_vec(),
                listing.options.clone(),
            )
            .await?
            .with_partition_columns(listing.partition_columns.to_vec());
            self.opened
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((key.clone(), Arc::new(opened)));
        }
        // Boxed: the sample's future would otherwise nest inside every
        // caller's state machine (the MCP server's overflows the layout depth).
        Box::pin(self.sample_distinct(plan, &scans)).await;
        Ok(())
    }

    /// Refuses a reference whose option has the key (any case) of one the catalog table stores
    /// with another value, as Spark's `DataSourceUtils.generateDatasourceOptions` does
    /// (`_LEGACY_ERROR_TEMP_1151`); the same value is accepted and the table's own is kept.
    fn check_reference_options(&self, plan: &LogicalPlan) -> Result<(), KrishivError> {
        let mut scans = Vec::new();
        collect_scans(plan, &mut scans);
        for scan in scans {
            let ReadSource::Table { name, options } = &scan.source else {
                continue;
            };
            let Some((_, table)) = self.entry_of(name) else {
                continue;
            };
            let TempTable::Files {
                source:
                    ReadSource::Files {
                        options: stored, ..
                    },
                permanent: true,
                ..
            } = &*table
            else {
                continue;
            };
            for (k, v) in options {
                let clash = stored
                    .iter()
                    .find(|(b, bv)| b.eq_ignore_ascii_case(k) && *bv != v);
                // Spark names the table's spelling of the key.
                if let Some((stored_key, _)) = clash {
                    return Err(krishiv_common::error::LegacyErrorTemp1151 {
                        table: format!(
                            "`{}`",
                            name.iter()
                                .map(String::as_str)
                                .collect::<Vec<_>>()
                                .join("`.`")
                        ),
                        key: stored_key.clone(),
                        config: "spark.sql.legacy.extraOptionsBehavior.enabled".to_owned(),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    /// The scans of files in `plan`: what each reads, and the columns the
    /// analyzer bound (the view's, for a view), which the files map to by
    /// name.
    pub(super) fn file_scans(&self, plan: &LogicalPlan) -> Vec<FileScanOf> {
        let mut scans = Vec::new();
        collect_scans(plan, &mut scans);
        scans
            .into_iter()
            .filter_map(|scan| {
                let (key, source) = match &scan.source {
                    ReadSource::Table { name, options } => {
                        let (key, table) = self.entry_of(name)?;
                        let TempTable::Files {
                            source,
                            known,
                            permanent,
                            ..
                        } = &*table
                        else {
                            return None;
                        };
                        reference_key(key, *known, *permanent, source, options)
                    }
                    source @ ReadSource::Files { .. } => {
                        (FilesKey::Read(source.clone()), source.clone())
                    }
                    _ => return None,
                };
                // The hidden `_metadata` column is the scan's, not the table's.
                let columns = scan.output.len() - usize::from(scan.metadata_column().is_some());
                let table = Arc::new(Schema::new(
                    scan.output[..columns]
                        .iter()
                        .map(|c| c.data_type.to_arrow_field(c.name.clone(), c.nullable))
                        .collect::<Vec<_>>(),
                ));
                let ids = scan.output[..columns]
                    .iter()
                    .map(|c| (c.id, c.name.clone()))
                    .collect();
                Some((key, source, table, ids))
            })
            .collect()
    }
}

/// [`FileTable::open`] on the executor's I/O runtime: footer reads are
/// object-store requests, which belong there and not on the compute
/// runtime. Dropping the returned future aborts the open.
async fn open_table(
    format: Arc<dyn krishiv_io::FileFormat>,
    scans: FileScans,
    table: SchemaRef,
    files: Vec<krishiv_io::FileRef>,
    options: krishiv_io::FormatOptions,
) -> Result<FileTable, KrishivError> {
    let runtime = scans.runtime.clone();
    run_on(&runtime, async move {
        FileTable::open(
            format,
            scans.io,
            table,
            files,
            options,
            &scans.config,
            scans.slots,
        )
        .await
    })
    .await
}
