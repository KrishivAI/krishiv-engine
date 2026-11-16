// SPDX-License-Identifier: Apache-2.0
//! Plan-time key counts over Parquet files (design P2b A5 (d), unit 2):
//! `FileTable::discover_unique` proves a key by reading the whole table, the
//! strictly increasing fast path and the hash path, NULL keys ignored, an
//! early stop at a repeat, partition columns refused, and the answers cached
//! by file version.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_exec::planner::{
    FileTable, KeyOptions, KeyOutcome, KeyResult, KeyUnknown, ProviderStatistics, TableProvider,
};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, StoreId};
use object_store::ObjectStore;

use super::partial_aggregate::write;

/// A scheduler over the local directory `dir`, and a runtime.
struct Files {
    rt: tokio::runtime::Runtime,
    sched: Arc<IoScheduler>,
    store: Arc<object_store::local::LocalFileSystem>,
    id: StoreId,
}

impl Files {
    fn new(dir: &Path) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let store = Arc::new(object_store::local::LocalFileSystem::new_with_prefix(dir).unwrap());
        let sched = IoScheduler::new(IoConfig::default());
        let id = StoreId::new("file://");
        sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
        Self {
            rt,
            sched,
            store,
            id,
        }
    }

    /// The table of `names` (in order) with columns `schema`; the last
    /// `partitions` columns of `schema` come from the directories.
    fn open(&self, names: &[&str], schema: &SchemaRef, partitions: usize) -> FileTable {
        let files: Vec<FileRef> = names
            .iter()
            .map(|n| {
                let meta = self
                    .rt
                    .block_on(object_store::ObjectStoreExt::head(
                        self.store.as_ref(),
                        &object_store::path::Path::from(*n),
                    ))
                    .unwrap();
                FileRef::from_meta(self.id.clone(), &meta)
            })
            .collect();
        let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
        let options = FormatOptions::defaults(ParquetFormat.options());
        let io = self.sched.context();
        let table_schema = Arc::clone(schema);
        let table = self
            .rt
            .block_on(self.rt.spawn(async move {
                let config = krishiv_common::config::Config::new();
                FileTable::open(format, io, table_schema, files, options, &config, 2).await
            }))
            .unwrap()
            .unwrap();
        if partitions == 0 {
            return table;
        }
        let fields = schema.fields();
        table.with_partition_columns(fields[fields.len() - partitions..].to_vec())
    }

    fn discover(&self, table: &FileTable, columns: &[&str]) -> KeyOutcome {
        self.discover_within(table, columns, Duration::from_secs(60))
    }

    fn discover_within(
        &self,
        table: &FileTable,
        columns: &[&str],
        timeout: Duration,
    ) -> KeyOutcome {
        let columns: Vec<String> = columns.iter().map(|c| (*c).to_owned()).collect();
        self.rt
            .block_on(table.discover_unique(&columns, KeyOptions { timeout }))
            .unwrap()
    }
}

fn keys_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
    ]))
}

/// A file of `(k, s)` rows.
fn file(dir: &Path, name: &str, rows: &[(Option<i64>, Option<&str>)]) {
    let k: Int64Array = rows.iter().map(|r| r.0).collect();
    let s: StringArray = rows.iter().map(|r| r.1).collect();
    let batch =
        RecordBatch::try_new(keys_schema(), vec![Arc::new(k) as ArrayRef, Arc::new(s)]).unwrap();
    write(dir, name, &batch);
}

fn ints(range: impl Iterator<Item = i64>) -> Vec<(Option<i64>, Option<&'static str>)> {
    range.map(|i| (Some(i), Some("x"))).collect()
}

fn discovered(o: &KeyOutcome) -> (bool, u64, bool) {
    match o.result {
        KeyResult::Discovered {
            unique,
            rows,
            sorted,
            ..
        } => (unique, rows, sorted),
        other => panic!("not counted: {other:?}"),
    }
}

fn cols(c: &[&str]) -> Vec<String> {
    c.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn a_key_increasing_across_files_is_unique_by_the_fast_path() {
    let dir = tempfile::tempdir().unwrap();
    file(dir.path(), "a.parquet", &ints(1..=100));
    file(dir.path(), "b.parquet", &ints(101..=250));
    let f = Files::new(dir.path());
    let t = f.open(&["a.parquet", "b.parquet"], &keys_schema(), 0);
    assert!(t.unique_keys().is_empty());
    let o = f.discover(&t, &["k"]);
    assert_eq!(discovered(&o), (true, 250, true), "{o}");
    assert_eq!(t.unique_keys(), vec![cols(&["k"])]);
    assert_eq!(t.key_outcomes(), vec![o]);
}

#[test]
fn an_unsorted_unique_key_is_unique_by_the_hash_path() {
    let dir = tempfile::tempdir().unwrap();
    // The second file's keys are below the first's.
    file(dir.path(), "a.parquet", &ints(101..=200));
    file(dir.path(), "b.parquet", &ints((1..=100).rev()));
    let f = Files::new(dir.path());
    let t = f.open(&["a.parquet", "b.parquet"], &keys_schema(), 0);
    let o = f.discover(&t, &["k"]);
    assert_eq!(discovered(&o), (true, 200, false), "{o}");
    assert_eq!(t.unique_keys(), vec![cols(&["k"])]);
}

#[test]
fn a_repeated_key_is_not_unique_and_stops_the_read() {
    let dir = tempfile::tempdir().unwrap();
    let mut first = ints(1..=10);
    first.push((Some(3), Some("y")));
    file(dir.path(), "a.parquet", &first);
    file(dir.path(), "b.parquet", &ints(1_000..=50_000));
    let f = Files::new(dir.path());
    let t = f.open(&["a.parquet", "b.parquet"], &keys_schema(), 0);
    let o = f.discover(&t, &["k"]);
    let (unique, rows, _) = discovered(&o);
    assert!(!unique, "{o}");
    // The repeat is the eleventh row: the second file is never read.
    assert_eq!(rows, 11, "{o}");
    assert!(t.unique_keys().is_empty());
}

#[test]
fn rows_with_a_null_key_are_not_counted() {
    let dir = tempfile::tempdir().unwrap();
    let mut rows = vec![(None, Some("n")), (Some(5), Some("a")), (None, Some("n"))];
    rows.extend([(Some(7), Some("b")), (None, None), (Some(2), Some("c"))]);
    file(dir.path(), "a.parquet", &rows);
    let f = Files::new(dir.path());
    let t = f.open(&["a.parquet"], &keys_schema(), 0);
    assert_eq!(discovered(&f.discover(&t, &["k"])), (true, 3, false));
    // `(k, s)`: a NULL in either column leaves the row out.
    assert!(discovered(&f.discover(&t, &["k", "s"])).0);
    // `s` alone repeats "n".
    assert!(!discovered(&f.discover(&t, &["s"])).0);
}

#[test]
fn a_key_of_two_columns_is_unique_as_a_pair() {
    let dir = tempfile::tempdir().unwrap();
    let rows = [
        (Some(1), Some("a")),
        (Some(1), Some("b")),
        (Some(2), Some("a")),
        (Some(2), Some("b")),
    ];
    file(dir.path(), "a.parquet", &rows);
    let f = Files::new(dir.path());
    let t = f.open(&["a.parquet"], &keys_schema(), 0);
    assert!(!discovered(&f.discover(&t, &["k"])).0);
    // Asked in any order, kept sorted.
    let o = f.discover(&t, &["s", "k"]);
    assert!(discovered(&o).0, "{o}");
    assert_eq!(o.columns, cols(&["k", "s"]));
    assert_eq!(t.unique_keys(), vec![cols(&["k", "s"])]);
}

#[test]
fn a_partition_or_unknown_column_is_unknown_and_nothing_is_read() {
    let dir = tempfile::tempdir().unwrap();
    file(dir.path(), "a.parquet", &ints(1..=10));
    let f = Files::new(dir.path());
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
        Field::new("p", DataType::Int64, true),
    ]));
    let t = f.open(&["a.parquet"], &schema, 1);
    for c in [&["p"][..], &["k", "p"], &["nope"]] {
        let o = f.discover(&t, c);
        assert_eq!(
            o.result,
            KeyResult::Unknown(KeyUnknown::NotFileColumns),
            "{o}"
        );
    }
    assert!(t.unique_keys().is_empty());
    assert_eq!(f.sched.key_facts().len(), 0, "nothing cached");
}

#[test]
fn a_second_count_of_the_same_files_is_cached_and_a_changed_file_counts_again() {
    let dir = tempfile::tempdir().unwrap();
    file(dir.path(), "a.parquet", &ints(1..=20));
    file(dir.path(), "b.parquet", &ints(21..=40));
    let f = Files::new(dir.path());
    let names = ["a.parquet", "b.parquet"];
    let first = f.open(&names, &keys_schema(), 0);
    assert!(discovered(&f.discover(&first, &["k"])).0);
    // A later query's table over the same files: the cached fact.
    let again = f.open(&names, &keys_schema(), 0);
    assert_eq!(
        f.discover(&again, &["k"]).result,
        KeyResult::Cached { unique: true }
    );
    assert_eq!(again.unique_keys(), vec![cols(&["k"])]);
    // `b` rewritten with a repeat of `a`'s key: another version, counted.
    let mut b = ints(21..=40);
    b.push((Some(1), Some("z")));
    file(dir.path(), "b.parquet", &b);
    let changed = f.open(&names, &keys_schema(), 0);
    let o = f.discover(&changed, &["k"]);
    assert!(!discovered(&o).0, "{o}");
    assert!(changed.unique_keys().is_empty());
    // Not unique is cached too.
    let after = f.open(&names, &keys_schema(), 0);
    assert_eq!(
        f.discover(&after, &["k"]).result,
        KeyResult::Cached { unique: false }
    );
}

#[test]
fn no_time_reads_nothing_and_caches_nothing() {
    let dir = tempfile::tempdir().unwrap();
    file(dir.path(), "a.parquet", &ints(1..=10));
    let f = Files::new(dir.path());
    let t = f.open(&["a.parquet"], &keys_schema(), 0);
    let o = f.discover_within(&t, &["k"], Duration::ZERO);
    assert_eq!(o.result, KeyResult::Unknown(KeyUnknown::NoTime));
    assert_eq!(f.sched.key_facts().len(), 0);
}

/// The optimizer's requests reach the table through the statistics, and a
/// proven key reaches the optimizer the same way.
#[test]
fn requests_and_facts_go_through_provider_statistics() {
    use krishiv_exec::planner::MemoryTables;
    use krishiv_optimizer::Statistics as _;
    use krishiv_plan::{BoundColumn, ReadSource, Scan};

    let dir = tempfile::tempdir().unwrap();
    file(dir.path(), "a.parquet", &ints(1..=10));
    let f = Files::new(dir.path());
    let t = Arc::new(f.open(&["a.parquet"], &keys_schema(), 0));
    let tables = MemoryTables::new().with_table(
        &["spark_catalog", "default", "t"],
        Arc::clone(&t) as Arc<dyn TableProvider>,
    );
    let stats = ProviderStatistics(Arc::new(tables));
    let scan = Scan {
        source: ReadSource::Table {
            name: vec!["spark_catalog".into(), "default".into(), "t".into()],
            options: Default::default(),
        },
        output: Vec::<BoundColumn>::new(),
        version: None,
    };
    stats.want_unique(&scan, &cols(&["k"]));
    stats.want_unique(&scan, &cols(&["k"]));
    assert_eq!(t.take_wanted(), vec![cols(&["k"])], "asked once");
    assert!(t.take_wanted().is_empty(), "taken");
    assert!(stats.unique_keys(&scan).is_empty());
    assert!(f.discover(&t, &["k"]).is_unique());
    assert_eq!(stats.unique_keys(&scan), vec![cols(&["k"])]);
}
