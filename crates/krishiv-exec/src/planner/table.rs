// SPDX-License-Identifier: Apache-2.0
//! The execution half of tables (spec 13 §7): what the planner opens for a
//! resolved scan.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_common::error::{KrishivUnsupportedFeature, TableOrViewNotFound};
use krishiv_memory::RelationLease;
use krishiv_optimizer::{Statistics, TableStats};
use krishiv_plan::{ReadSource, Scan};
use krishiv_types::batch::Batch;

use crate::error::Result;
use crate::operators::{MemoryScan, ScanPushdown};

/// A table the executor can read: the scan factory the planner pushes
/// projections and filters into, and the table's statistics.
pub trait TableProvider: Send + Sync + fmt::Debug {
    /// The scan factory. Its [`table_schema`](ScanPushdown::table_schema)
    /// has a column of each name the analyzer resolved, of the Arrow form
    /// of its type.
    ///
    /// # Errors
    /// Errors opening the table.
    fn scan(&self) -> Result<Arc<dyn ScanPushdown>>;

    /// Row count and column statistics, when known (Parquet footers,
    /// table-format metadata); the optimizer orders joins by them.
    fn statistics(&self) -> Option<TableStats> {
        None
    }
}

/// Finds the provider of the table a resolved scan reads, at the version
/// the analyzer pinned (ADR 0037).
pub trait TableProviderResolver: Send + Sync + fmt::Debug {
    /// The provider of `scan`'s table.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND` when the table is gone;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a source the resolver cannot open.
    fn provider(&self, scan: &Scan) -> Result<Arc<dyn TableProvider>>;
}

/// The optimizer's [`Statistics`] read from table providers.
#[derive(Debug, Clone)]
pub struct ProviderStatistics(pub Arc<dyn TableProviderResolver>);

impl Statistics for ProviderStatistics {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        self.0.provider(scan).ok()?.statistics()
    }
}

/// A table held in memory.
#[derive(Debug, Clone)]
pub struct MemoryTable {
    schema: SchemaRef,
    /// The batches, when the table owns them. Tables registered with a
    /// relation store hold none: their scans resolve the lease's batches
    /// through the store, so the batches live once, as content-addressed
    /// IPC, instead of twice.
    batches: Vec<Batch>,
    statistics: Option<TableStats>,
    /// The relation store lease: the batches as content-addressed IPC, so
    /// workers resolve a batch range through the store instead of reading
    /// rows from the plan. Tables built without a store (tests) have none.
    relation: Option<RelationLease>,
}

impl MemoryTable {
    /// A table of `batches` (of `schema`). Its statistics are its row count.
    pub fn new(schema: SchemaRef, batches: Vec<Batch>) -> Self {
        let rows = batches.iter().map(|b| b.num_rows() as u64).sum();
        Self {
            schema,
            batches,
            statistics: Some(TableStats {
                rows,
                ..TableStats::default()
            }),
            relation: None,
        }
    }

    /// A table of `rows` rows (of `schema`) whose batches live only in the
    /// relation store: scans resolve them through `lease`. The session
    /// registers memory tables this way, so the batches live once,
    /// memory-accounted, instead of both in the table and in the store.
    pub fn from_lease(schema: SchemaRef, rows: u64, lease: RelationLease) -> Self {
        Self {
            schema,
            batches: Vec::new(),
            statistics: Some(TableStats {
                rows,
                ..TableStats::default()
            }),
            relation: Some(lease),
        }
    }

    /// The same table reporting `statistics`.
    #[must_use]
    pub fn with_statistics(mut self, statistics: Option<TableStats>) -> Self {
        self.statistics = statistics;
        self
    }

    /// The same table resolving its batch splits through `lease`: the
    /// session's relation store entry of its batches.
    #[must_use]
    pub fn with_relation(mut self, lease: RelationLease) -> Self {
        self.relation = Some(lease);
        self
    }

    /// The relation store lease, if the table was registered with one.
    #[must_use]
    pub fn relation_lease(&self) -> Option<&RelationLease> {
        self.relation.as_ref()
    }
}

impl TableProvider for MemoryTable {
    fn scan(&self) -> Result<Arc<dyn ScanPushdown>> {
        // A leased table's batches live in the relation store: resolve
        // them through it, so scans read the stored copy. Anything else
        // reads the held batches.
        if let Some(lease) = &self.relation {
            let store = lease.store();
            let batches = store.resolve(&lease.hash())?;
            return Ok(Arc::new(MemoryScan::try_new(
                Arc::clone(&self.schema),
                batches,
            )?));
        }
        Ok(Arc::new(MemoryScan::try_new(
            Arc::clone(&self.schema),
            self.batches.clone(),
        )?))
    }

    fn statistics(&self) -> Option<TableStats> {
        self.statistics.clone()
    }
}

/// Providers by table name: the fully qualified name parts the analyzer
/// resolves to (`["spark_catalog", "default", "t"]`), compared ASCII
/// case-insensitively.
#[derive(Debug, Clone, Default)]
pub struct MemoryTables {
    tables: BTreeMap<Vec<String>, Arc<dyn TableProvider>>,
}

impl MemoryTables {
    /// No tables.
    pub fn new() -> Self {
        Self::default()
    }

    /// The same set with `provider` as the table `name`.
    #[must_use]
    pub fn with_table(mut self, name: &[&str], provider: Arc<dyn TableProvider>) -> Self {
        self.tables.insert(key(name.iter().copied()), provider);
        self
    }
}

fn key<'a>(parts: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    parts.into_iter().map(str::to_ascii_lowercase).collect()
}

impl TableProviderResolver for MemoryTables {
    fn provider(&self, scan: &Scan) -> Result<Arc<dyn TableProvider>> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return Err(KrishivUnsupportedFeature {
                feature: "reading files by path from in-memory tables".to_owned(),
            }
            .into());
        };
        self.tables
            .get(&key(name.iter().map(String::as_str)))
            .cloned()
            .ok_or_else(|| {
                let quoted = |parts: &[String]| {
                    parts
                        .iter()
                        .map(|p| format!("`{p}`"))
                        .collect::<Vec<_>>()
                        .join(".")
                };
                let namespace = &name[..name.len().saturating_sub(1)];
                TableOrViewNotFound {
                    relation_name: quoted(name),
                    search_path: format!("[{}]", quoted(namespace)),
                }
                .into()
            })
    }
}
