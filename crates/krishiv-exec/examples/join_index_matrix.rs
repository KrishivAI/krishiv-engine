// SPDX-License-Identifier: Apache-2.0
//! The P2-T158 join index sweep: the flat (`hashbrown`) and the salted
//! [`JoinIndex`] over the same join rows, the salted probe with and
//! without its touch pass, and today's arrow-row `KeyTable`, single
//! thread, interleaved.
//!
//! ```text
//! cargo run --release -p krishiv-exec --example join_index_matrix
//! JOIN_INDEX_SIZES=1000,1000000 JOIN_INDEX_REPS=5 JOIN_INDEX_KEYS=i64,str24 cargo run ...
//! JOIN_INDEX_PERF=salted,15000000,4,1 cargo run ...   # one probe run for `perf stat`
//! ```
//!
//! For each key type and number of distinct build keys `N` it builds each
//! index once over the same rows (the shape of a join partition's arena:
//! fixed-width rows back to back, per-row heaps for long strings) and
//! times the probe of `2N` pre-encoded keys in 4,096-row batches:
//! `hit` (uniform over the `N` keys) and `half` (uniform over `2N` values,
//! half present). Each repetition runs the variants in the order
//! `flat, salted, touch, keytable` and then reversed (ABBA), so drift
//! hits every variant alike; the report gives the median, min and max ns
//! per probe of `2 * reps` samples and the flat / salted ratio of medians
//! (above 1: the salted directory is faster). Build times (ns per row,
//! untimed by ABBA) and index bytes per key are reported once.
//!
//! `JOIN_OPERATOR=<threads>` instead runs whole hash joins (scan, build,
//! probe, output) with `krishiv.exec.join.rowLayout` off and on, ABBA:
//! `N` unique `Int64` build keys with a payload column, `4N` probe rows of
//! which half match, as an inner join (build columns gathered) and a left
//! semi join (key-only build), for `N` in `JOIN_INDEX_SIZES`.
//!
//! `JOIN_INDEX_PERF=<flat|salted|touch|keytable|none>,<N>,<probes per key>,<spread>[,str24]`
//! builds one index of `N` keys and probes it `probes * N` times with
//! keys uniform over `spread * N` values (1 in `spread` hits); `none`
//! builds and encodes the same keys without probing, so the difference of
//! two `perf stat -e LLC-load-misses` runs is the probes' misses.

#![expect(
    clippy::expect_used,
    reason = "a measurement tool: a broken fixture must stop it loudly"
)]

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_row::Rows;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_exec::hash::{HashSeed, KeyEncoder, KeyTable};
use krishiv_exec::rows::{
    EncodedKeys, GatheredRows, IndexChoice, JoinIndex, NO_MATCH, RowLayout, RowSource,
};
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, ExecutorConfig, HashJoin, HashJoinNode, JoinSpec, JoinType, QueryContext,
    QueryId, QueryRunner, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_types::batch::Batch;

const BATCH: usize = 4096;
const SEED: u64 = 0x5EED;

#[derive(Clone, Copy, Debug)]
enum KeyType {
    I64,
    Str24,
}

impl KeyType {
    fn name(self) -> &'static str {
        match self {
            Self::I64 => "i64",
            Self::Str24 => "str24",
        }
    }

    fn data_type(self) -> DataType {
        match self {
            Self::I64 => DataType::Int64,
            Self::Str24 => DataType::Utf8View,
        }
    }

    /// Key values for ranks `ranks` (rank `r` is the same key everywhere).
    fn column(self, ranks: &[u64]) -> ArrayRef {
        // A bijective scramble, so keys are not dense or ordered.
        let scramble = |r: u64| r.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5555;
        match self {
            Self::I64 => Arc::new(Int64Array::from_iter_values(
                ranks.iter().map(|&r| scramble(r).cast_signed()),
            )),
            Self::Str24 => Arc::new(StringViewArray::from_iter_values(
                ranks
                    .iter()
                    .map(|&r| format!("key-{:019}", scramble(r) >> 1)),
            )),
        }
    }
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Join rows: fixed-width rows back to back plus per-row heaps (what a
/// join partition's arena holds).
struct Arena {
    rows: GatheredRows,
    hashes: Vec<u64>,
    stride: usize,
}

impl RowSource for Arena {
    #[inline]
    fn row(&self, id: u32) -> (&[u8], &[u8]) {
        let at = id as usize * self.stride;
        let row = self
            .rows
            .fixed
            .get(at..at + self.stride)
            .unwrap_or_default();
        let heap = self.rows.heap.as_ref().map_or(&[][..], |(o, b)| {
            let i = id as usize;
            let s = usize::try_from(o[i]).unwrap_or(0);
            let e = usize::try_from(o[i + 1]).unwrap_or(0);
            b.get(s..e).unwrap_or_default()
        });
        (row, heap)
    }

    #[inline]
    fn hash(&self, id: u32) -> u64 {
        self.hashes[id as usize]
    }
}

/// One size's fixture: the build rows, the probe batches in both formats.
struct Fixture {
    layout: RowLayout,
    arena: Arena,
    /// Arrow-row build keys and their seeded hashes (today's path).
    build_arrow: (Rows, Vec<u64>),
    probes: Vec<(EncodedKeys, Rows, Vec<u64>)>,
    probe_rows: usize,
}

/// `spread`: probe keys are uniform over `spread * n` values, so `1/spread`
/// of them hit.
fn fixture(kt: KeyType, n: usize, probes_per_key: usize, spread: u64) -> Fixture {
    let types = [kt.data_type()];
    let layout = RowLayout::for_join(&types, &[false]).expect("layout");
    let encoder = KeyEncoder::new(&types).expect("encoder");
    let seed = HashSeed::new(Some(SEED));
    let build_ranks: Vec<u64> = (0..n as u64).collect();
    let col = kt.column(&build_ranks);
    let keys = layout.encode(&[Arc::clone(&col)]).expect("encode");
    let all: Vec<usize> = (0..n).collect();
    let arena = Arena {
        rows: keys.gather(&layout, &all).expect("gather"),
        hashes: keys.hashes().to_vec(),
        stride: layout.stride(),
    };
    let arrow = encoder.encode(&[col]).expect("arrow rows");
    let arrow_hashes = arrow.iter().map(|r| seed.hash(r.as_ref())).collect();
    let mut rng = Rng(n as u64);
    let domain = spread.max(1) * n as u64;
    let total = probes_per_key * n;
    let mut probes = Vec::new();
    for start in (0..total).step_by(BATCH) {
        let len = BATCH.min(total - start);
        let ranks: Vec<u64> = (0..len).map(|_| rng.next() % domain).collect();
        let col = kt.column(&ranks);
        let enc = layout.encode(&[Arc::clone(&col)]).expect("encode");
        let arrow = encoder.encode(&[col]).expect("arrow rows");
        let hashes = arrow.iter().map(|r| seed.hash(r.as_ref())).collect();
        probes.push((enc, arrow, hashes));
    }
    Fixture {
        layout,
        arena,
        build_arrow: (arrow, arrow_hashes),
        probes,
        probe_rows: total,
    }
}

/// Today's index: `KeyTable` over arrow-row keys with chain heads.
struct Legacy {
    table: KeyTable,
    heads: Vec<u32>,
}

fn legacy(f: &Fixture) -> Legacy {
    let (rows, hashes) = &f.build_arrow;
    let mut table = KeyTable::new();
    table.reserve(rows.num_rows(), rows.size());
    let mut heads = Vec::with_capacity(rows.num_rows());
    for (i, r) in rows.iter().enumerate() {
        let (_, new) = table.insert_hashed(r.as_ref(), hashes[i]).expect("insert");
        if new {
            heads.push(u32::try_from(i).expect("row id"));
        }
    }
    Legacy { table, heads }
}

fn index(f: &Fixture, choice: IndexChoice) -> JoinIndex {
    let n = f.arena.hashes.len();
    let mut next = vec![NO_MATCH; n];
    JoinIndex::build(&f.layout, &f.arena, n, SEED, choice, &mut next).expect("index")
}

/// Probes every batch; returns the matches (so nothing is optimized out).
fn probe_rows(f: &Fixture, ix: &JoinIndex, touch: bool, out: &mut Vec<u32>) -> usize {
    let mut found = 0;
    // Row ids of a full batch, built once: no allocation in the timed loop.
    let ids: Vec<u32> = (0..u32::try_from(BATCH).expect("batch")).collect();
    for (keys, _, _) in &f.probes {
        out.clear();
        ix.find_with(&f.layout, keys, &ids[..keys.len()], &f.arena, out, touch);
        found += out.iter().filter(|&&h| h != NO_MATCH).count();
    }
    found
}

fn probe_legacy(f: &Fixture, l: &Legacy) -> usize {
    let mut found = 0;
    for (_, rows, hashes) in &f.probes {
        for (i, r) in rows.iter().enumerate() {
            let head = l
                .table
                .find(r.as_ref(), hashes[i])
                .and_then(|id| l.heads.get(id as usize).copied());
            found += usize::from(head.is_some());
        }
    }
    found
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    Flat,
    Salted,
    Touch,
    KeyTable,
}

const VARIANTS: [Variant; 4] = [
    Variant::Flat,
    Variant::Salted,
    Variant::Touch,
    Variant::KeyTable,
];

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len().is_multiple_of(2) {
        f64::midpoint(v[m - 1], v[m])
    } else {
        v[m]
    }
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

fn key_type(name: &str) -> KeyType {
    match name {
        "str24" => KeyType::Str24,
        _ => KeyType::I64,
    }
}

/// `JOIN_INDEX_PERF`: one build and one probe run for `perf stat`.
fn perf_run(spec: &str) {
    let p: Vec<&str> = spec.split(',').collect();
    let kind = p.first().copied().unwrap_or("salted");
    let n: usize = p.get(1).and_then(|s| s.parse().ok()).unwrap_or(15_000_000);
    let per: usize = p.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let spread: u64 = p.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let kt = key_type(p.get(4).copied().unwrap_or("i64"));
    let f = fixture(kt, n, per, spread);
    let mut out = Vec::new();
    let t = Instant::now();
    let found = match kind {
        "flat" => probe_rows(&f, &index(&f, IndexChoice::Flat), false, &mut out),
        "salted" => probe_rows(&f, &index(&f, IndexChoice::Salted), false, &mut out),
        "touch" => probe_rows(&f, &index(&f, IndexChoice::Salted), true, &mut out),
        "keytable" => probe_legacy(&f, &legacy(&f)),
        // The baseline: the same fixture and a built index, no probes.
        _ => index(&f, IndexChoice::Salted).len(),
    };
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "{kind} n={n} probes={} found={found} {:.1} ns/probe (build included)",
        f.probe_rows,
        t.elapsed().as_nanos() as f64 / f.probe_rows.max(1) as f64
    )
    .expect("stdout");
}

fn join_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

/// Batches of `(scramble(rank), i)` for the given key ranks.
fn join_batches(ranks: &[u64]) -> Vec<Batch> {
    ranks
        .chunks(BATCH)
        .enumerate()
        .map(|(c, r)| {
            let k = Int64Array::from_iter_values(
                r.iter()
                    .map(|&x| x.wrapping_mul(0x9E37_79B9_7F4A_7C15).cast_signed()),
            );
            let start = i64::try_from(c * BATCH).expect("offset");
            let v = Int64Array::from_iter_values(start..start + r.len() as i64);
            let b = RecordBatch::try_new(join_schema(), vec![Arc::new(k), Arc::new(v)]);
            Batch::new(b.expect("batch"))
        })
        .collect()
}

fn run_join(
    runner: &QueryRunner,
    rt: &tokio::runtime::Runtime,
    join_type: JoinType,
    build: &[Batch],
    probe: &[Batch],
    rows: bool,
) -> usize {
    let spec = JoinSpec {
        join_type,
        probe_schema: join_schema(),
        build_schema: join_schema(),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        spill: None,
    };
    let scan = |b: &[Batch]| -> Arc<dyn ExecNode> {
        Arc::new(Scan(Batches::new(join_schema(), b.to_vec())))
    };
    let plan: Arc<dyn ExecNode> = Arc::new(HashJoinNode {
        join: HashJoin::try_new(spec).expect("join"),
        build: scan(build),
        probe: scan(probe),
    });
    let root = TreePool::new(&MemoryConfig::default());
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig {
            join_row_layout: rows,
            ..ExecConfig::default()
        },
        root.child("join", None),
    );
    rt.block_on(async {
        let mut s = runner.execute(plan, Arc::clone(&q)).await.expect("plan");
        let mut n = 0;
        while let Some(b) = s.next().await {
            n += b.expect("batch").num_rows();
        }
        q.finished().await;
        n
    })
}

/// `JOIN_OPERATOR`: whole joins, row layout off against on.
fn operator_abba(threads: usize, sizes: &[usize], reps: usize) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let runner = {
        let _in = rt.enter();
        QueryRunner::new(
            TokioExecutor::new(&ExecutorConfig {
                compute_threads: threads,
                io_threads: 1,
                blocking_threads: threads,
                io_blocking_threads: 2,
            })
            .expect("executor"),
        )
    };
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "join	N	threads	off_ms	on_ms	off/on	off_range	on_range"
    )
    .expect("stdout");
    for &n in sizes {
        let build = join_batches(&(0..n as u64).collect::<Vec<_>>());
        let mut rng = Rng(7);
        let probe_ranks: Vec<u64> = (0..4 * n).map(|_| rng.next() % (2 * n as u64)).collect();
        let probe = join_batches(&probe_ranks);
        for jt in [JoinType::Inner, JoinType::LeftSemi] {
            let expect = run_join(&runner, &rt, jt, &build, &probe, false);
            let mut samples = [Vec::new(), Vec::new()];
            for _ in 0..reps {
                for rows in [false, true, true, false] {
                    let t = Instant::now();
                    let got = run_join(&runner, &rt, jt, &build, &probe, rows);
                    samples[usize::from(rows)].push(t.elapsed().as_secs_f64() * 1e3);
                    assert_eq!(got, expect, "{jt:?} row layout {rows}");
                }
            }
            let range = |s: &[f64]| {
                let lo = s.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = s.iter().copied().fold(0.0, f64::max);
                format!("{lo:.1}-{hi:.1}")
            };
            let (r0, r1) = (range(&samples[0]), range(&samples[1]));
            let (off, on) = (median(&mut samples[0]), median(&mut samples[1]));
            writeln!(
                stdout,
                "{jt:?}	{n}	{threads}	{off:.2}	{on:.2}	{:.3}	{r0}	{r1}",
                off / on
            )
            .expect("stdout");
            stdout.flush().expect("flush");
        }
    }
}

fn main() {
    if let Ok(spec) = std::env::var("JOIN_INDEX_PERF") {
        perf_run(&spec);
        return;
    }
    let sizes: Vec<usize> = env_list(
        "JOIN_INDEX_SIZES",
        "1000,4000,16000,32000,64000,128000,256000,512000,1000000,4000000,15000000",
    )
    .iter()
    .map(|s| s.parse().expect("size"))
    .collect();
    let reps: usize = std::env::var("JOIN_INDEX_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    if let Some(threads) = std::env::var("JOIN_OPERATOR")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        operator_abba(threads, &sizes, reps);
        return;
    }
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "key\tN\tprobe\tflat_ns\tsalted_ns\ttouch_ns\tkeytable_ns\tflat/salted\tsalted/touch\tkeytable/best\tflat_B/key\tsalted_B/key\tbuild_flat_ns\tbuild_salted_ns\tranges(min-max)"
    )
    .expect("stdout");
    for kt in env_list("JOIN_INDEX_KEYS", "i64,str24")
        .iter()
        .map(|k| key_type(k))
    {
        for &n in &sizes {
            for hit in [true, false] {
                let f = fixture(kt, n, 2, if hit { 1 } else { 2 });
                let t = Instant::now();
                let flat = index(&f, IndexChoice::Flat);
                let build_flat = t.elapsed().as_nanos() as f64 / n as f64;
                let t = Instant::now();
                let salted = index(&f, IndexChoice::Salted);
                let build_salted = t.elapsed().as_nanos() as f64 / n as f64;
                let old = legacy(&f);
                let mut out = Vec::with_capacity(BATCH);
                let mut samples: Vec<Vec<f64>> = vec![Vec::new(); VARIANTS.len()];
                let expect = probe_rows(&f, &flat, false, &mut out);
                for _ in 0..reps {
                    for v in VARIANTS.iter().chain(VARIANTS.iter().rev()) {
                        let t = Instant::now();
                        let found = match v {
                            Variant::Flat => probe_rows(&f, &flat, false, &mut out),
                            Variant::Salted => probe_rows(&f, &salted, false, &mut out),
                            Variant::Touch => probe_rows(&f, &salted, true, &mut out),
                            Variant::KeyTable => probe_legacy(&f, &old),
                        };
                        let ns = t.elapsed().as_nanos() as f64 / f.probe_rows as f64;
                        assert_eq!(found, expect, "{v:?} disagrees");
                        samples[*v as usize].push(ns);
                    }
                }
                let ranges: Vec<String> = samples
                    .iter()
                    .map(|s| {
                        let lo = s.iter().copied().fold(f64::INFINITY, f64::min);
                        let hi = s.iter().copied().fold(0.0, f64::max);
                        format!("{lo:.1}-{hi:.1}")
                    })
                    .collect();
                let m: Vec<f64> = samples.iter_mut().map(|s| median(s)).collect();
                let best = m[0].min(m[1]).min(m[2]);
                writeln!(
                    stdout,
                    "{}\t{n}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{}",
                    kt.name(),
                    if hit { "hit" } else { "half" },
                    m[0],
                    m[1],
                    m[2],
                    m[3],
                    m[0] / m[1],
                    m[1] / m[2],
                    m[3] / best,
                    flat.memory_size() as f64 / n as f64,
                    salted.memory_size() as f64 / n as f64,
                    build_flat,
                    build_salted,
                    ranges.join(" "),
                )
                .expect("stdout");
                stdout.flush().expect("flush");
            }
        }
    }
}
