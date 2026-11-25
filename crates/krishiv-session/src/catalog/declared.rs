// SPDX-License-Identifier: Apache-2.0
//! Reading the tables of declared catalogs (design P3-T067 PR-58b): the analysis recorded each
//! table a declared catalog loaded (`DeclaredLoads`); before planning, every scanned one is opened
//! once on the executor's I/O runtime, and its scans read that provider.
//!
//! In server mode every file such a table reads (metadata, manifests, data and delete files) is
//! confined to `krishiv.io.local.allowedRoots` and `krishiv.io.objectStore.allowedPrefixes`
//! (ADR 0046 §3a), with the server's own store configuration: a location outside them is the
//! catalog's fault, `KRISHIV_CATALOG_LOCATION_REFUSED` naming the catalog and the table, never the
//! location. Embedded use is not confined, as for reads by path.

use std::sync::Arc;

use krishiv_common::error::{KrishivCatalogLocationRefused, KrishivError};
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::{LocatedPath, PathLocator};
use krishiv_exec::planner::TableProvider;
use krishiv_plan::{LogicalPlan, ReadSource, TableVersion};

use super::scans::collect_scans;
use super::{CATALOG, QueryCatalog, run_on, unsupported};
use crate::access::PathPrefixPolicy;
use crate::catalog_registry::{DeclaredKey, ScanInputs};

/// The rule a confined location breaks, as the error names it.
const OUTSIDE: &str = "outside the allowed roots and prefixes";

impl QueryCatalog {
    /// The key of a scan of `name` at `version`.
    fn declared_key(name: &[String], version: Option<TableVersion>) -> DeclaredKey {
        (name.to_vec(), version)
    }

    /// Whether a scan of `name` at `version` reads a declared catalog's table.
    pub(super) fn is_declared_scan(&self, name: &[String], version: Option<TableVersion>) -> bool {
        self.system
            .catalogs
            .loads()
            .loaded(&Self::declared_key(name, version))
            .is_some()
    }

    /// The provider opened for a scan of `name` at `version`.
    pub(super) fn declared_provider(
        &self,
        name: &[String],
        version: Option<TableVersion>,
    ) -> Option<Arc<dyn TableProvider>> {
        self.system
            .catalogs
            .loads()
            .opened(&Self::declared_key(name, version))
    }

    /// Opens every declared table `plan` scans (subqueries included) that is not open yet.
    ///
    /// # Errors
    /// The table's refusal or failure to open; `KRISHIV_INVALID_CONFIG_VALUE` for a confinement
    /// setting that does not parse (server mode).
    pub(super) async fn open_declared(&self, plan: &LogicalPlan) -> Result<(), KrishivError> {
        let loads = Arc::clone(self.system.catalogs.loads());
        let mut scans = Vec::new();
        collect_scans(plan, &mut scans);
        for scan in scans {
            let ReadSource::Table { name, options } = &scan.source else {
                continue;
            };
            let key = Self::declared_key(name, scan.version);
            if loads.loaded(&key).is_some() {
                refuse_snapshot_options(name, options)?;
            }
            if loads.opened(&key).is_some() {
                continue;
            }
            let (Some(table), Some(query)) = (loads.loaded(&key), loads.inputs()) else {
                continue;
            };
            let mut inputs: ScanInputs = query.inputs.clone();
            if query.confine {
                inputs.paths = Arc::new(ConfinedPaths::new(
                    Arc::clone(&inputs.paths),
                    PathPrefixPolicy::from_config(
                        &inputs.config,
                        Arc::new(krishiv_plan::analyzer::AllowAll),
                    )?,
                    name,
                ));
            }
            let provider = run_on(&self.scans.runtime, table.open(inputs)).await?;
            loads.put_opened(key, provider);
        }
        Ok(())
    }
}

/// Spark's time-travel read options (`DataFrameReader.option`, `WITH (...)`), refused as Spark's
/// `V2SessionCatalog.failTimeTravel` refuses the clause on a table that cannot travel.
const TIME_TRAVEL_OPTIONS: &[&str] = &["versionasof", "timestampasof"];

/// Iceberg's read options that choose a snapshot other than the current one or an incremental
/// range, or an internal task set that chooses the files (`SparkReadOptions`; keys are case-insensitive there). A declared table reads only its
/// current snapshot until time travel lands (design `0.4-iceberg-branches-tags` I1), so each is
/// refused rather than ignored.
const SNAPSHOT_OPTIONS: &[&str] = &[
    "snapshot-id",
    "as-of-timestamp",
    "branch",
    "tag",
    "start-snapshot-id",
    "end-snapshot-id",
    "start-timestamp",
    "end-timestamp",
    "scan-task-set-id",
];

/// Refuses a read of the declared table `name` with an option that would choose another snapshot
/// (above): reading the current one instead would return the wrong rows.
///
/// # Errors
/// `UNSUPPORTED_FEATURE.TIME_TRAVEL` naming the table for `versionAsOf` / `timestampAsOf`;
/// `KRISHIV_UNSUPPORTED_FEATURE` naming the option for Iceberg's own.
pub(super) fn refuse_snapshot_options(
    name: &[String],
    options: &std::collections::BTreeMap<String, OptionValue>,
) -> Result<(), KrishivError> {
    for key in options.keys() {
        let lower = key.to_ascii_lowercase();
        if TIME_TRAVEL_OPTIONS.contains(&lower.as_str()) {
            return Err(krishiv_common::error::UnsupportedFeatureTimeTravel {
                relation_id: relation_id(name),
            }
            .into());
        }
        if SNAPSHOT_OPTIONS.contains(&lower.as_str()) {
            return Err(unsupported(&format!(
                "the Iceberg read option {key} on {} (only the current snapshot is read)",
                relation_id(name)
            )));
        }
    }
    Ok(())
}

/// `name` as Spark prints a relation: each part back-quoted, joined by dots.
pub(crate) fn relation_id(name: &[String]) -> String {
    name.iter()
        .map(|p| format!("`{}`", p.replace('`', "``")))
        .collect::<Vec<_>>()
        .join(".")
}

/// `inputs` with their path placement confined by the server's roots and prefixes
/// (`krishiv.io.local.allowedRoots`, `krishiv.io.objectStore.allowedPrefixes`), as a declared
/// table's scan is: what a `CALL` procedure reads and deletes through (`name` is the procedure, for
/// the refusal).
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for a confinement setting that does not parse.
pub(crate) fn confined_inputs(
    mut inputs: ScanInputs,
    name: &[String],
) -> Result<ScanInputs, KrishivError> {
    inputs.paths = Arc::new(ConfinedPaths::new(
        Arc::clone(&inputs.paths),
        PathPrefixPolicy::from_config(&inputs.config, Arc::new(krishiv_plan::analyzer::AllowAll))?,
        name,
    ));
    Ok(inputs)
}

/// A path placement that refuses every location outside the server's confinement (module docs).
pub(crate) struct ConfinedPaths {
    inner: Arc<dyn PathLocator>,
    policy: PathPrefixPolicy,
    catalog: String,
    relation: String,
}

impl std::fmt::Debug for ConfinedPaths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfinedPaths")
            .field("inner", &self.inner)
            .field("relation", &self.relation)
            .finish_non_exhaustive()
    }
}

impl ConfinedPaths {
    /// `inner`, confined by `policy`'s roots and prefixes, for the table the scan names `name`.
    pub(crate) fn new(
        inner: Arc<dyn PathLocator>,
        policy: PathPrefixPolicy,
        name: &[String],
    ) -> Self {
        let quote = |p: &String| format!("`{}`", p.replace('`', "``"));
        Self {
            inner,
            policy,
            catalog: name.first().map(quote).unwrap_or_default(),
            relation: name.iter().map(quote).collect::<Vec<_>>().join("."),
        }
    }

    fn refused(&self) -> KrishivError {
        tracing::warn!(
            target: "krishiv::catalog",
            catalog = self.catalog.as_str(),
            table = self.relation.as_str(),
            rule = OUTSIDE,
            "a catalog's table reads a location outside the server's confinement"
        );
        KrishivCatalogLocationRefused {
            catalog: self.catalog.clone(),
            relation_name: self.relation.clone(),
            rule: OUTSIDE.to_owned(),
        }
        .into()
    }
}

impl PathLocator for ConfinedPaths {
    fn locate(&self, path: &str) -> Result<LocatedPath, KrishivError> {
        // An object-store URL is checked as written, before any store is built for it; anything
        // else (a local path) by the form the store reads it as.
        let object = path
            .split_once("://")
            .is_some_and(|(scheme, _)| !scheme.eq_ignore_ascii_case("file"));
        if object && !self.policy.allows(&OptionValue::Plain(path.to_owned())) {
            return Err(self.refused());
        }
        let at = self.inner.locate(path)?;
        if !object && !self.policy.allows(&OptionValue::Plain(at.display.clone())) {
            return Err(self.refused());
        }
        Ok(at)
    }
}

impl QueryCatalog {
    /// The refusal of a scan of `name` in a declared catalog whose table this query has no
    /// provider for (a catalog that resolves names only), not failed as an internal error. `None`
    /// when `name` is no declared catalog's.
    pub(super) fn declared_read(&self, name: &[String]) -> Option<KrishivError> {
        let catalog = match name {
            [c, _, _, ..] if self.system.catalogs.catalog(c).is_some() => c.clone(),
            [_] | [_, _]
                if !self.catalog.eq_ignore_ascii_case(CATALOG)
                    && !crate::system::is_system(&self.catalog) =>
            {
                self.catalog.clone()
            }
            _ => return None,
        };
        Some(self.declared_refusal(&[catalog], ""))
    }

    /// `KRISHIV_UNSUPPORTED_FEATURE` for reading the tables of the catalog `name` names (its
    /// first part), `place` appended (` in a distributed stage`).
    pub(super) fn declared_refusal(&self, name: &[String], place: &str) -> KrishivError {
        let catalog = name.first().map_or("", String::as_str);
        unsupported(&format!(
            "reading the tables of the declared catalog `{}`{place}",
            catalog.replace('`', "``")
        ))
    }
}
