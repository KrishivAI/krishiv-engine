// SPDX-License-Identifier: Apache-2.0
//! Deterministic fault scenarios of Iceberg reads (design note §10, PR-25d):
//! a store wrapper whose faults are a pure function of a seed, the request
//! (path and range) and its attempt number, so a failing case replays
//! whatever order the requests run in. Transient failures, before a response
//! or in the middle of its body (a manifest list, a manifest, a data or
//! delete file), are retried by the scheduler and the rows equal the model of
//! `delete_tests`; a request that runs out of retries fails the query with
//! the store-request class. A file that vanishes after the scan is planned
//! fails the query naming it, unless the scan never reads it; rows are never
//! wrong.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::ops::Range;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_io::IoConfig;
use krishiv_io::io::TransientError;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as StoreResult,
};
use proptest::prelude::*;

use super::delete_tests::{Table, build_in, data, equality, history, ops, positions};
use super::provider_tests::{env_on, provider_session, render};
use crate::plan::VersionSpec;

/// The faults of an armed [`Flaky`].
#[derive(Default)]
struct Faults {
    seed: u64,
    /// The first `fail_first` attempts of every GET fail.
    fail_first: u32,
    /// Chance in 1,000 that any other attempt fails.
    permille: u64,
    /// File names whose GETs answer "not found".
    vanished: HashSet<String>,
    /// Attempts so far, by path and range.
    attempts: HashMap<(String, String, bool), u32>,
    /// Failures injected.
    injected: u64,
    /// The file names of the GETs failed in the middle of their body.
    mid_body: HashSet<String>,
}

/// What one GET attempt does.
enum Fault {
    None,
    /// Fails before any byte.
    Before,
    /// Returns half the body, then fails.
    MidBody,
    Vanished,
}

/// The local file system with seeded faults on reads (writes, listings and
/// deletes pass through). Disarmed until [`Flaky::arm`], so a table is
/// written without faults.
pub(super) struct Flaky {
    inner: Arc<dyn ObjectStore>,
    faults: Mutex<Option<Faults>>,
}

impl fmt::Debug for Flaky {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Flaky({})", self.inner)
    }
}

impl fmt::Display for Flaky {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Flaky({})", self.inner)
    }
}

/// FNV-1a of `bytes`: stable across toolchains (`DefaultHasher` is not), so
/// a seed replays the same faults after an upgrade.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn transient() -> object_store::Error {
    object_store::Error::Generic {
        store: "Flaky",
        source: Box::new(TransientError { retry_after: None }),
    }
}

impl Flaky {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(object_store::local::LocalFileSystem::new()),
            faults: Mutex::new(None),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Option<Faults>> {
        self.faults.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// From now on, the first `fail_first` attempts of every GET fail, and
    /// any other attempt fails with chance `permille` in 1,000.
    pub(super) fn arm(&self, seed: u64, fail_first: u32, permille: u64) {
        let mut g = self.lock();
        let f = g.get_or_insert_with(Faults::default);
        f.seed = seed;
        f.fail_first = fail_first;
        f.permille = permille;
    }

    /// GETs of the file named `name` answer "not found" from now on.
    pub(super) fn vanish(&self, name: &str) {
        self.lock()
            .get_or_insert_with(Faults::default)
            .vanished
            .insert(name.to_owned());
    }

    /// Failures injected so far.
    pub(super) fn injected(&self) -> u64 {
        self.lock().as_ref().map_or(0, |f| f.injected)
    }

    /// Whether a GET of the file named `name` failed mid-body.
    pub(super) fn failed_mid_body(&self, name: &str) -> bool {
        self.lock()
            .as_ref()
            .is_some_and(|f| f.mid_body.contains(name))
    }

    fn fault(&self, location: &Path, options: &GetOptions) -> Fault {
        let mut g = self.lock();
        let Some(f) = g.as_mut() else {
            return Fault::None;
        };
        if location.filename().is_some_and(|n| f.vanished.contains(n)) {
            return Fault::Vanished;
        }
        let key = (
            location.to_string(),
            format!("{:?}", options.range),
            options.head,
        );
        let n = {
            let a = f.attempts.entry(key.clone()).or_default();
            *a += 1;
            *a - 1
        };
        let text = format!("{}|{}|{}|{n}", key.0, key.1, key.2);
        let r = splitmix(f.seed ^ fnv(text.as_bytes()));
        if n >= f.fail_first && r % 1000 >= f.permille {
            return Fault::None;
        }
        f.injected += 1;
        if options.head || (r >> 32) & 1 == 0 && n >= f.fail_first {
            Fault::Before
        } else {
            if let Some(name) = location.filename() {
                f.mid_body.insert(name.to_owned());
            }
            Fault::MidBody
        }
    }
}

#[async_trait]
impl ObjectStore for Flaky {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> StoreResult<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
        match self.fault(location, &options) {
            Fault::None => self.inner.get_opts(location, options).await,
            Fault::Before => Err(transient()),
            Fault::Vanished => Err(object_store::Error::NotFound {
                path: location.to_string(),
                source: "vanished".into(),
            }),
            Fault::MidBody => {
                let mut head = self.inner.get_opts(location, options.clone()).await?;
                let body = self
                    .inner
                    .get_opts(location, options)
                    .await?
                    .bytes()
                    .await?;
                let half = body.slice(..body.len() / 2);
                head.payload = GetResultPayload::Stream(Box::pin(futures::stream::iter([
                    Ok(half),
                    Err(transient()),
                ])));
                Ok(head)
            }
        }
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> StoreResult<Vec<bytes::Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> StoreResult<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// A table of `files` over a [`Flaky`] store whose scheduler retries fast
/// (the default five retries, backoff capped at 4 ms).
fn flaky_table(files: &[super::delete_tests::File], partitioned: bool) -> (Table, Arc<Flaky>) {
    let flaky = Flaky::new();
    let config = IoConfig {
        retry_base: Duration::from_millis(1),
        retry_max: Duration::from_millis(4),
        ..IoConfig::default()
    };
    let env = env_on(Arc::clone(&flaky) as Arc<dyn ObjectStore>, config);
    (build_in(env, files, partitioned), flaky)
}

/// Opens the provider (plans the snapshot), calls `between`, then runs
/// `SELECT id FROM t`; the sorted ids.
fn ids_after(t: &Table, between: impl FnOnce()) -> Result<Vec<i64>, KrishivError> {
    let p = t.env.provider(&t.table, &VersionSpec::Current)?;
    between();
    let batches = provider_session(p).run("SELECT id FROM t")?;
    let mut out: Vec<i64> = render(&batches)
        .iter()
        .map(|r| r.parse().unwrap())
        .collect();
    out.sort_unstable();
    Ok(out)
}

fn retries(t: &Table) -> u64 {
    t.env._sched.totals().retries.load(Ordering::Relaxed)
}

/// The first attempt of every read fails, in the middle of its body where
/// it has one: the manifest list, both manifests, the data files and the
/// position delete file are each read again, and the rows are right.
#[test]
fn a_transient_failure_mid_manifest_is_retried() {
    let (t, flaky) = flaky_table(
        &[
            data("a.parquet", 0, 1, 0, 3),
            data("b.parquet", 0, 1, 3, 2),
            positions(
                "d.parquet",
                0,
                2,
                &[("a.parquet", 1), ("b.parquet", 0)],
                None,
            ),
        ],
        false,
    );
    let before = retries(&t);
    flaky.arm(1, 1, 0);
    assert_eq!(ids_after(&t, || ()).unwrap(), [0, 2, 4]);
    for name in [
        "snap.avro",
        "data-m.avro",
        "delete-m.avro",
        "a.parquet",
        "b.parquet",
        "d.parquet",
    ] {
        assert!(flaky.failed_mid_body(name), "{name}");
    }
    assert!(retries(&t) - before >= flaky.injected(), "{}", retries(&t));
}

/// Seeded faults on 30% of attempts: nearly every seed reads the right rows
/// after retries (the property test below allows a failed query, so this
/// shows its success path is taken).
#[test]
fn random_faults_are_retried_to_the_right_rows() {
    let mut retried = 0;
    for seed in 0..16 {
        let (t, flaky) = flaky_table(
            &[
                data("a.parquet", 0, 1, 0, 3),
                positions("d.parquet", 0, 2, &[("a.parquet", 1)], None),
                equality("e.parquet", 0, 2, &[Some(2)]),
            ],
            false,
        );
        flaky.arm(seed, 0, 300);
        if ids_after(&t, || ()).as_deref() == Ok(&[0][..]) && flaky.injected() > 0 {
            retried += 1;
        }
    }
    assert!(retried >= 12, "{retried} of 16 seeds");
}

/// An equality delete file whose key bounds miss the data file's is not
/// assigned to it (Java's `canContainEqDeletesForFile`), so the scan never
/// reads it: removing it after planning changes nothing.
#[test]
fn a_disjoint_equality_delete_file_is_never_read() {
    let (t, flaky) = flaky_table(
        &[
            data("a.parquet", 0, 1, 0, 3),
            equality("far.parquet", 0, 2, &[Some(50)]),
            equality("near.parquet", 0, 2, &[Some(1)]),
        ],
        false,
    );
    assert_eq!(
        ids_after(&t, || flaky.vanish("far.parquet")).unwrap(),
        [0, 2]
    );
}

/// A read that fails more often than the scheduler retries fails the query
/// with the store-request class naming the file; no partial result.
#[test]
fn a_read_out_of_retries_fails_the_query() {
    let (t, flaky) = flaky_table(&[data("a.parquet", 0, 1, 0, 3)], false);
    flaky.arm(2, 100, 0);
    let e = ids_after(&t, || ()).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivObjectStoreRequestFailed,
        "{e}"
    );
}

/// A data file deleted after the scan was planned fails the query naming it.
#[test]
fn a_data_file_vanished_after_planning_fails() {
    let (t, flaky) = flaky_table(
        &[data("a.parquet", 0, 1, 0, 3), data("b.parquet", 0, 1, 3, 2)],
        false,
    );
    let e = ids_after(&t, || flaky.vanish("b.parquet")).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::FailedReadFileFileNotExist,
        "{e}"
    );
    assert!(e.message().contains("b.parquet"), "{e}");
}

/// A delete file deleted after the scan was planned (its deletes load on the
/// first batch) fails the query naming it: its rows are never returned.
#[test]
fn a_delete_file_vanished_after_planning_fails() {
    let (t, flaky) = flaky_table(
        &[
            data("a.parquet", 0, 1, 0, 3),
            positions("d.parquet", 0, 2, &[("a.parquet", 0)], None),
        ],
        false,
    );
    let e = ids_after(&t, || flaky.vanish("d.parquet")).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::FailedReadFileFileNotExist,
        "{e}"
    );
    assert!(e.message().contains("d.parquet"), "{e}");
}

fn model_ids(survivors: &BTreeMap<i64, i32>) -> Vec<i64> {
    survivors.keys().copied().collect()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

    /// Random histories of `delete_tests` read under seeded transient
    /// failures (up to 40% of attempts, before or in the middle of a body):
    /// the rows are the model's, or a read ran out of retries and the query
    /// failed with the store-request class.
    #[test]
    fn reads_under_transient_faults_match_model(
        ops in ops(),
        partitioned in any::<bool>(),
        seed in any::<u64>(),
        permille in 0..400u64,
    ) {
        let (files, survivors) = history(&ops, partitioned);
        let (t, flaky) = flaky_table(&files, partitioned);
        flaky.arm(seed, 0, permille);
        match ids_after(&t, || ()) {
            Ok(got) => prop_assert_eq!(got, model_ids(&survivors)),
            Err(e) => prop_assert_eq!(
                e.error_class(),
                ErrorClass::KrishivObjectStoreRequestFailed,
                "seed {} permille {}: {}", seed, permille, e
            ),
        }
    }

    /// One file of a random history vanishes after the scan is planned: the
    /// query fails naming it, or (a delete file the scan never reads) the
    /// rows are the model's. A vanished data file always fails.
    #[test]
    fn a_file_vanished_after_planning_never_gives_wrong_rows(
        ops in ops(),
        partitioned in any::<bool>(),
        pick in any::<usize>(),
    ) {
        let (files, survivors) = history(&ops, partitioned);
        prop_assume!(!files.is_empty());
        let gone = &files[pick % files.len()];
        let (t, flaky) = flaky_table(&files, partitioned);
        match ids_after(&t, || flaky.vanish(&gone.name)) {
            Ok(got) => {
                prop_assert!(!gone.is_data(), "{} vanished, yet {:?}", gone.name, got);
                prop_assert_eq!(got, model_ids(&survivors));
            }
            Err(e) => {
                prop_assert_eq!(e.error_class(), ErrorClass::FailedReadFileFileNotExist, "{}", e);
                prop_assert!(e.message().contains(&gone.name), "{}: {}", gone.name, e);
            }
        }
    }
}
