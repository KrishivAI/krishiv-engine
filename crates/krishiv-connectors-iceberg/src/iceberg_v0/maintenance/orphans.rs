// SPDX-License-Identifier: Apache-2.0
//! `remove_orphan_files` (design P3-T027 PR-36e §4.4): the files under a table's location that no
//! snapshot, metadata file or statistics file names and that are older than `older_than`.
//!
//! The decision ([`classify`]) is a pure function of the valid set and the listing, so it is
//! tested without a store; [`remove`] gathers both from the table and deletes.

use std::collections::{BTreeMap, HashMap, HashSet};

use iceberg::spec::TableMetadata;
use krishiv_common::error::{
    KrishivError, KrishivIcebergOrphanPathUnreadable, KrishivIcebergProcedureArgument,
    KrishivIcebergProcedureGcDisabled,
};

use super::super::commit::{CommitClock, CommitTable};
use super::expire::{ManifestCache, checked_load_located, reachable};
use super::paths::{Equivalences, FileIdentity, Relation};

/// Iceberg's shortest `older_than` interval, in milliseconds (24 hours).
const MIN_INTERVAL_MS: i64 = 24 * 3_600_000;
/// The default `older_than` interval (3 days).
const DEFAULT_INTERVAL_MS: i64 = 3 * 24 * 3_600_000;
/// `max-orphan-file-sample-size`: the rows `stream_results` returns.
const SAMPLE: usize = 20_000;

/// What a listed file that matches a valid one but for scheme or authority becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Fail naming the conflicts (the default).
    Error,
    /// Keep the file.
    Ignore,
    /// Treat it as an orphan.
    Delete,
}

impl Mode {
    /// Reads `prefix_mismatch_mode`, case-insensitively.
    ///
    /// # Errors
    /// `KRISHIV_ICEBERG_PROCEDURE_ARGUMENT` with Iceberg's `Invalid mode: %s`.
    pub(crate) fn parse(s: Option<&str>) -> Result<Self, KrishivError> {
        match s.map(str::to_ascii_uppercase).as_deref() {
            None | Some("ERROR") => Ok(Self::Error),
            Some("IGNORE") => Ok(Self::Ignore),
            Some("DELETE") => Ok(Self::Delete),
            Some(_) => Err(argument(format!("Invalid mode: {}", s.unwrap_or_default()))),
        }
    }
}

/// The call's arguments.
#[derive(Debug, Clone)]
pub(crate) struct OrphanArgs {
    pub(crate) older_than_ms: Option<i64>,
    pub(crate) location: Option<String>,
    pub(crate) dry_run: bool,
    pub(crate) max_concurrent_deletes: Option<usize>,
    pub(crate) equivalences: Equivalences,
    pub(crate) mode: Mode,
    pub(crate) stream_results: bool,
}

/// One listed file.
#[derive(Debug, Clone)]
pub(crate) struct Listed {
    /// The URI in the form the table was registered with.
    pub(crate) uri: String,
    pub(crate) modified_ms: i64,
    /// The key relative to the listed location, for the hidden-path rule.
    pub(crate) relative: String,
}

fn argument(message: String) -> KrishivError {
    KrishivIcebergProcedureArgument { message }.into()
}

/// A `Boolean.parseBoolean` of `gc.enabled`: absent means on.
pub(crate) fn check_gc(meta: &TableMetadata) -> Result<(), KrishivError> {
    if meta
        .properties()
        .get("gc.enabled")
        .is_some_and(|v| !v.eq_ignore_ascii_case("true"))
    {
        return Err(KrishivIcebergProcedureGcDisabled {
            message: "Cannot delete orphan files: GC is disabled (deleting files may corrupt other tables)"
                .to_owned(),
        }
        .into());
    }
    Ok(())
}

/// `older_than` resolved: the default is three days back; under 24 hours is refused.
///
/// # Errors
/// `KRISHIV_ICEBERG_PROCEDURE_ARGUMENT` with Iceberg's text.
pub(crate) fn resolve_older_than(given: Option<i64>, now_ms: i64) -> Result<i64, KrishivError> {
    let t = given.unwrap_or_else(|| now_ms.saturating_sub(DEFAULT_INTERVAL_MS));
    if t > now_ms.saturating_sub(MIN_INTERVAL_MS) {
        return Err(argument(
            "Cannot remove orphan files with an interval less than 24 hours. Executing this \
             procedure with a short interval may corrupt the table if other operations are \
             happening at the same time. If you are absolutely confident that no concurrent \
             operations will be affected by removing orphan files with such a short interval, \
             you can use the Action API to remove orphan files with an arbitrary interval."
                .to_owned(),
        ));
    }
    Ok(t)
}

/// The location to list: the call's, which must be the table location, its data or metadata path
/// or below one of them (a declared divergence toward refusal, design §4.4 F5).
///
/// # Errors
/// `KRISHIV_ICEBERG_PROCEDURE_ARGUMENT` with the Krishiv text.
pub(crate) fn resolve_location(
    meta: &TableMetadata,
    table: &str,
    given: Option<&str>,
    eq: &Equivalences,
) -> Result<String, KrishivError> {
    let Some(given) = given else {
        return Ok(meta.location().to_owned());
    };
    let props = meta.properties();
    let roots = [
        Some(meta.location()),
        props.get("write.data.path").map(String::as_str),
        props.get("write.metadata.path").map(String::as_str),
    ];
    if let Some(l) = FileIdentity::parse(given) {
        for root in roots.into_iter().flatten().filter_map(FileIdentity::parse) {
            let below = l.path == root.path
                || root.path == "/"
                || l.path.starts_with(&format!("{}/", root.path));
            let same_store = FileIdentity {
                path: l.path.clone(),
                ..root.clone()
            }
            .relation(&l, eq)
                == Relation::Same;
            if below && same_store {
                return Ok(given.to_owned());
            }
        }
    }
    Err(argument(format!(
        "Cannot remove orphan files under `{given}`: it is not the location, data path or metadata \
         path of table `{table}`, nor under one of them."
    )))
}

/// Whether a key under the listed location is hidden: a segment starting `_` or `.`, unless it is
/// a partition directory of a field whose name starts so (`FileSystemWalker`'s partition-aware
/// filter). The file name counts as a segment.
fn hidden(relative: &str, partitions: &[String]) -> bool {
    relative.split('/').any(|seg| {
        (seg.starts_with('_') || seg.starts_with('.'))
            && !partitions.iter().any(|p| seg.starts_with(p.as_str()))
    })
}

/// The partition directory prefixes (`name=`) of every spec of the table.
pub(crate) fn partition_prefixes(meta: &TableMetadata) -> Vec<String> {
    let mut v: Vec<String> = meta
        .partition_specs_iter()
        .flat_map(|s| s.fields().iter().map(|f| format!("{}=", f.name)))
        .collect();
    v.sort();
    v.dedup();
    v
}

/// The orphans among `listed`, sorted. `valid` is `(path, source)`: every string the table
/// names, with the file that named it.
///
/// A listed file that `FileIdentity::parse` cannot read is kept; a valid string it cannot read
/// fails everything, before any delete.
///
/// # Errors
/// `KRISHIV_ICEBERG_ORPHAN_PATH_UNREADABLE`; with [`Mode::Error`] the conflicts of scheme or
/// authority.
pub(crate) fn classify(
    table: &str,
    valid: &[(String, String)],
    listed: &[Listed],
    older_than_ms: i64,
    mode: Mode,
    eq: &Equivalences,
    partitions: &[String],
) -> Result<Vec<String>, KrishivError> {
    let mut by_path: HashMap<String, Vec<FileIdentity>> = HashMap::new();
    for (path, source) in valid {
        let Some(id) = FileIdentity::parse(path) else {
            return Err(KrishivIcebergOrphanPathUnreadable {
                table: table.to_owned(),
                path: path.clone(),
                source: source.clone(),
            }
            .into());
        };
        by_path.entry(id.path.clone()).or_default().push(id);
    }
    let mut orphans = Vec::new();
    let mut conflicts: BTreeMap<String, ()> = BTreeMap::new();
    for l in listed {
        if l.modified_ms >= older_than_ms || hidden(&l.relative, partitions) {
            continue;
        }
        let Some(id) = FileIdentity::parse(&l.uri) else {
            tracing::warn!(path = %l.uri, "a listed file's path cannot be read; kept");
            continue;
        };
        let relations: Vec<Relation> = by_path
            .get(&id.path)
            .map(|vs| vs.iter().map(|v| v.relation(&id, eq)).collect())
            .unwrap_or_default();
        if relations.contains(&Relation::Same) {
            continue;
        }
        if relations.contains(&Relation::PrefixMismatch) {
            match mode {
                Mode::Ignore => continue,
                Mode::Error => {
                    conflicts.insert(format!("({}://{})", id.scheme, id.authority), ());
                    continue;
                }
                Mode::Delete => {}
            }
        }
        orphans.push(l.uri.clone());
    }
    if !conflicts.is_empty() {
        let names: Vec<String> = conflicts.into_keys().collect();
        return Err(argument(format!(
            "Unable to determine whether certain files are orphan. Metadata references files \
             that match listed files except for authority/scheme. Please, inspect the conflicting \
             authorities/schemes and provide which of them are equal by further configuring the \
             action via equal_schemes() and equal_authorities() options. Conflicting \
             authorities/schemes: {}.",
            names.join(", ")
        )));
    }
    orphans.sort();
    Ok(orphans)
}

/// Every path the table's metadata names, with its source: all snapshots' manifest lists,
/// manifests and files, statistics files, the metadata files and `version-hint.text`.
async fn valid_set(
    io: &iceberg::io::FileIO,
    meta: &TableMetadata,
    metadata_location: &str,
    max_bytes: u64,
    memory: Option<&std::sync::Arc<dyn krishiv_memory::MemoryPool>>,
) -> Result<Vec<(String, String)>, KrishivError> {
    let mut charge = super::expire::Charge::new(memory);
    let mut cache = ManifestCache::new();
    let mut out: HashMap<String, super::expire::Kind> = HashMap::new();
    for s in meta.snapshots() {
        reachable(
            io,
            meta,
            s.snapshot_id(),
            max_bytes,
            &mut cache,
            &mut out,
            &mut charge,
        )
        .await?;
    }
    let mut valid: Vec<(String, String)> = out
        .into_keys()
        .map(|p| (p, format!("{metadata_location} (snapshots)")))
        .collect();
    let src = metadata_location.to_owned();
    valid.push((metadata_location.to_owned(), src.clone()));
    for e in meta.metadata_log() {
        valid.push((e.metadata_file.clone(), src.clone()));
    }
    for s in meta.statistics_iter() {
        valid.push((s.statistics_path.clone(), src.clone()));
    }
    for s in meta.partition_statistics_iter() {
        valid.push((s.statistics_path.clone(), src.clone()));
    }
    let base = meta.location().trim_end_matches('/');
    valid.push((format!("{base}/metadata/version-hint.text"), src));
    let mut seen = HashSet::new();
    valid.retain(|(p, _)| seen.insert(p.clone()));
    Ok(valid)
}

/// What a `remove_orphan_files` came to.
#[derive(Debug)]
pub(crate) struct Removed {
    /// The orphan files, every one unless `stream_results` (then a sample).
    pub(crate) orphans: Vec<String>,
}

/// Runs the procedure: refusals first, then the valid set, the listing, the decision, the deletes.
///
/// # Errors
/// As the pieces above; a delete that fails is logged, not an error.
pub(crate) async fn remove<F, Fut>(
    table: &dyn CommitTable,
    io: &iceberg::io::FileIO,
    clock: &dyn CommitClock,
    list: &F,
    max_bytes: u64,
    memory: Option<&std::sync::Arc<dyn krishiv_memory::MemoryPool>>,
    args: &OrphanArgs,
) -> Result<Removed, KrishivError>
where
    // The listing of a location: `(uri, last-modified ms, key relative to the location)` per file.
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<(String, i64, String)>, KrishivError>>,
{
    use futures::StreamExt as _;
    let name = table.name();
    let (meta, metadata_location) = checked_load_located(table, io, max_bytes).await?;
    check_gc(&meta)?;
    let older = resolve_older_than(args.older_than_ms, clock.now_ms())?;
    let location = resolve_location(&meta, &name, args.location.as_deref(), &args.equivalences)?;
    let valid = valid_set(io, &meta, &metadata_location, max_bytes, memory).await?;
    let listed: Vec<Listed> = list(location)
        .await?
        .into_iter()
        .map(|(uri, modified_ms, relative)| Listed {
            uri,
            modified_ms,
            relative,
        })
        .collect();
    let orphans = classify(
        &name,
        &valid,
        &listed,
        older,
        args.mode,
        &args.equivalences,
        &partition_prefixes(&meta),
    )?;
    if !args.dry_run {
        let width = args
            .max_concurrent_deletes
            .unwrap_or(1)
            .clamp(1, super::expire::MAX_CONCURRENT_DELETES);
        futures::stream::iter(orphans.iter().cloned())
            .map(|path| async move {
                // Not retried (Java's `noRetry()`): a failed delete is logged, its row returned.
                if let Err(e) = io.delete(&path).await {
                    tracing::warn!(path, error = %e, "an orphan file was not deleted");
                }
            })
            .buffer_unordered(width)
            .collect::<Vec<()>>()
            .await;
    }
    let orphans = if args.stream_results {
        orphans.into_iter().take(SAMPLE).collect()
    } else {
        orphans
    };
    Ok(Removed { orphans })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listed(uri: &str, modified_ms: i64) -> Listed {
        let relative = uri.rsplit_once("/t/").map_or(uri, |(_, r)| r).to_owned();
        Listed {
            uri: uri.to_owned(),
            modified_ms,
            relative,
        }
    }

    fn valid(paths: &[&str]) -> Vec<(String, String)> {
        paths
            .iter()
            .map(|p| ((*p).to_owned(), "v1.metadata.json".to_owned()))
            .collect()
    }

    fn run(
        v: &[&str],
        l: &[Listed],
        mode: Mode,
        eq: &Equivalences,
    ) -> Result<Vec<String>, KrishivError> {
        classify("c.db.t", &valid(v), l, 100, mode, eq, &[])
    }

    #[test]
    fn only_old_unreferenced_files_are_orphans() {
        let l = [
            listed("s3://b/t/data/a.parquet", 1),
            listed("s3://b/t/data/stray.parquet", 1),
            listed("s3://b/t/data/young.parquet", 100),
        ];
        let got = run(
            &["s3://b/t/data/a.parquet"],
            &l,
            Mode::Error,
            &Equivalences::defaults(),
        )
        .unwrap();
        assert_eq!(got, ["s3://b/t/data/stray.parquet"]);
    }

    #[test]
    fn spellings_of_one_file_are_not_orphans() {
        let l = [
            listed("s3://b/t//data/a%20b.parquet", 1),
            listed("s3://b/t/data/p=a+b/x.parquet", 1),
            listed("s3://b/t/data/dir/", 1),
        ];
        let got = run(
            &[
                "s3a://b/t/data/a%20b.parquet",
                "s3://b/t/data/p=a+b/x.parquet",
                "s3://b/t/data/dir",
            ],
            &l,
            Mode::Error,
            &Equivalences::defaults(),
        )
        .unwrap();
        assert!(got.is_empty(), "{got:?}");
    }

    #[test]
    fn hidden_paths_are_skipped_but_partition_directories_are_not() {
        let l = [
            listed("s3://b/t/data/_tmp/x.parquet", 1),
            listed("s3://b/t/data/.hidden", 1),
            listed("s3://b/t/data/_p=1/y.parquet", 1),
        ];
        let got = classify(
            "c.db.t",
            &[],
            &l,
            100,
            Mode::Error,
            &Equivalences::defaults(),
            &["_p=".to_owned()],
        )
        .unwrap();
        assert_eq!(got, ["s3://b/t/data/_p=1/y.parquet"]);
    }

    #[test]
    fn a_scheme_or_authority_mismatch_follows_the_mode() {
        let l = [listed("gs://b/t/data/a.parquet", 1)];
        let v = ["s3://b/t/data/a.parquet"];
        let eq = Equivalences::defaults();
        let e = run(&v, &l, Mode::Error, &eq).unwrap_err();
        assert!(
            e.to_string()
                .contains("Unable to determine whether certain files are orphan"),
            "{e}"
        );
        assert!(run(&v, &l, Mode::Ignore, &eq).unwrap().is_empty());
        assert_eq!(run(&v, &l, Mode::Delete, &eq).unwrap().len(), 1);
        // Declared equal, it is the same file.
        let eq = Equivalences::with(
            &[("gs".to_owned(), "s3".to_owned())].into_iter().collect(),
            &BTreeMap::new(),
        );
        assert!(run(&v, &l, Mode::Error, &eq).unwrap().is_empty());
    }

    #[test]
    fn a_valid_path_that_cannot_be_read_fails_before_any_decision() {
        let l = [listed("s3://b/t/data/a.parquet", 1)];
        let e = run(
            &["relative/a.parquet"],
            &l,
            Mode::Error,
            &Equivalences::defaults(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("relative/a.parquet"), "{e}");
        assert!(e.to_string().contains("v1.metadata.json"), "{e}");
    }

    #[test]
    fn mode_and_interval_follow_iceberg() {
        assert_eq!(Mode::parse(Some("delete")).unwrap(), Mode::Delete);
        assert_eq!(Mode::parse(None).unwrap(), Mode::Error);
        assert!(
            Mode::parse(Some("x"))
                .unwrap_err()
                .to_string()
                .contains("Invalid mode: x")
        );
        let now = 10 * DEFAULT_INTERVAL_MS;
        assert_eq!(
            resolve_older_than(None, now).unwrap(),
            now - DEFAULT_INTERVAL_MS
        );
        assert!(resolve_older_than(Some(now - MIN_INTERVAL_MS), now).is_ok());
        let e = resolve_older_than(Some(now - MIN_INTERVAL_MS + 1), now).unwrap_err();
        assert!(e.to_string().contains("less than 24 hours"), "{e}");
    }
}
