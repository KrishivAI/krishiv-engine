// SPDX-License-Identifier: Apache-2.0
//! The P2-T157 go/no-go matrix: the salted slot table against
//! `hashbrown::HashTable<u32>` over the same row pages, and both against
//! today's arrow-row `KeyTable`, single thread.
//!
//! ```text
//! cargo run --release -p krishiv-exec --example row_table_matrix
//! ROW_TABLE_SIZES=10000,1000000 ROW_TABLE_REPS=5 cargo run ...
//! ```
//!
//! For each key type and number of distinct keys `N` it times:
//! - **build** (aggregate-shaped find-or-insert) of a `2N`-row stream:
//!   `uniform` = every key twice, the second pass permuted; `zipf` = `2N`
//!   draws of a Zipf(1) rank;
//! - **probe** (join-shaped find) of `2N` keys into a table of the `N`
//!   keys: `hit` (uniform, all present), `half` (uniform over `2N`, half
//!   present), `zipf` (Zipf(1) ranks, all present).
//!
//! The batches are generated and encoded (both formats) before any index
//! is timed; encoding is timed once and reported apart. Index times are in
//! ns per row, the median of `ROW_TABLE_REPS` runs. The `hb/salted` column
//! is the go/no-go ratio (the salted table is adopted only at ≥ 1.15×).
//! `keytable` is today's path end to end (arrow-row encoding + `KeyTable`)
//! against the salted path end to end (row encoding + salted table).

#![expect(
    clippy::expect_used,
    reason = "a measurement tool: a broken fixture must stop it loudly"
)]

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Decimal128Array, Int64Array, StringViewArray};
use arrow_row::Rows;
use arrow_schema::DataType;
use krishiv_exec::hash::{HashSeed, KeyEncoder, KeyTable};
use krishiv_exec::rows::{EncodedKeys, HashbrownTable, RowIndex, RowLayout, RowPages, SaltedTable};

const BATCH: usize = 4096;
const SEED: u64 = 0x5EED;

#[derive(Clone, Copy, Debug)]
enum KeyType {
    I64,
    I64x2,
    Str8,
    Str24,
    Dec,
}

impl KeyType {
    fn name(self) -> &'static str {
        match self {
            Self::I64 => "i64",
            Self::I64x2 => "i64,i64",
            Self::Str8 => "str8",
            Self::Str24 => "str24",
            Self::Dec => "decimal(38,2)",
        }
    }

    fn types(self) -> Vec<DataType> {
        match self {
            Self::I64 => vec![DataType::Int64],
            Self::I64x2 => vec![DataType::Int64, DataType::Int64],
            Self::Str8 | Self::Str24 => vec![DataType::Utf8View],
            Self::Dec => vec![DataType::Decimal128(38, 2)],
        }
    }

    /// The key columns for ranks `r`.
    fn columns(self, r: &[u64]) -> Vec<ArrayRef> {
        let s = |x: u64| x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        match self {
            Self::I64 => vec![Arc::new(Int64Array::from_iter_values(
                r.iter().map(|&x| s(x).cast_signed()),
            ))],
            Self::I64x2 => vec![
                Arc::new(Int64Array::from_iter_values(
                    r.iter().map(|&x| (x >> 3).cast_signed()),
                )),
                Arc::new(Int64Array::from_iter_values(
                    r.iter().map(|&x| (x & 7).cast_signed()),
                )),
            ],
            Self::Str8 => vec![Arc::new(StringViewArray::from_iter_values(
                r.iter().map(|&x| format!("{:08x}", s(x) >> 32)),
            ))],
            Self::Str24 => vec![Arc::new(StringViewArray::from_iter_values(
                r.iter().map(|&x| format!("{:024x}", s(x))),
            ))],
            Self::Dec => vec![Arc::new(
                Decimal128Array::from_iter_values(
                    r.iter().map(|&x| i128::from(s(x)) * 1_000_000_007),
                )
                .with_precision_and_scale(38, 2)
                .expect("decimal"),
            )],
        }
    }
}

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "53 random bits")]
        let v = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        v
    }
}

#[derive(Clone, Copy, Debug)]
enum Stream {
    Uniform,
    Zipf,
    Hit,
    Half,
}

impl Stream {
    fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Zipf => "zipf",
            Self::Hit => "hit",
            Self::Half => "half",
        }
    }

    /// Rank `i` of `2n` for this stream.
    fn rank(self, i: u64, n: u64, rng: &mut Rng) -> u64 {
        match self {
            Self::Uniform if i < n => i,
            Self::Uniform => ((i - n) * 1_000_003) % n,
            Self::Zipf => {
                #[expect(clippy::cast_precision_loss, reason = "n is far below 2^53")]
                let nf = n as f64;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "n^u is in [1, n]"
                )]
                let r = nf.powf(rng.unit()) as u64;
                r.saturating_sub(1).min(n - 1)
            }
            Self::Hit => rng.next() % n,
            Self::Half => rng.next() % (2 * n),
        }
    }
}

/// The input of one cell, generated and encoded before anything is timed.
struct Input {
    /// Row-format keys, one per batch.
    rows: Vec<EncodedKeys>,
    /// arrow-row keys (today's path), one per batch.
    arrow: Vec<Rows>,
    /// Seconds to encode `rows` / `arrow`.
    encode: f64,
    encode_arrow: f64,
}

impl Input {
    fn new(kind: KeyType, layout: &RowLayout, enc: &KeyEncoder, stream: Stream, n: u64) -> Self {
        let mut rng = Rng(SEED);
        let total = 2 * n;
        let mut batches = Vec::new();
        let mut i = 0;
        while i < total {
            let end = (i + BATCH as u64).min(total);
            let ranks: Vec<u64> = (i..end).map(|j| stream.rank(j, n, &mut rng)).collect();
            batches.push(kind.columns(&ranks));
            i = end;
        }
        let start = Instant::now();
        let rows = batches
            .iter()
            .map(|c| layout.encode(c, SEED).expect("encode"))
            .collect();
        let encode = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let arrow = batches
            .iter()
            .map(|c| enc.encode(c).expect("encode"))
            .collect();
        let encode_arrow = start.elapsed().as_secs_f64();
        Self {
            rows,
            arrow,
            encode,
            encode_arrow,
        }
    }
}

/// `load_eighths`: 5 (≈ 2/3) for an aggregate build, 4 (1/2) for a join
/// build (P2-T157 §5.1); hashbrown keeps its own 7/8.
fn index(salted: bool, load_eighths: usize) -> Box<dyn RowIndex> {
    if salted {
        Box::new(SaltedTable::new(load_eighths))
    } else {
        Box::new(HashbrownTable::new())
    }
}

/// A built table: seconds to build, index + page bytes, the index, its rows.
type Built = (f64, usize, Box<dyn RowIndex>, RowPages);

fn build_rows(layout: &RowLayout, input: &[EncodedKeys], salted: bool, load: usize) -> Built {
    let mut t = index(salted, load);
    let mut pages = RowPages::new(layout);
    let mut ids = Vec::new();
    let start = Instant::now();
    for k in input {
        t.find_or_insert(layout, k, &mut pages, &mut ids)
            .expect("insert");
    }
    let secs = start.elapsed().as_secs_f64();
    let bytes = t.memory_size() + pages.memory_size();
    (secs, bytes, t, pages)
}

fn build_arrow(input: &[Rows]) -> (f64, usize, KeyTable) {
    let seed = HashSeed::new(Some(SEED));
    let mut t = KeyTable::new();
    let mut ids = Vec::new();
    let start = Instant::now();
    for r in input {
        t.find_or_insert(r, &seed, &mut ids).expect("insert");
    }
    (start.elapsed().as_secs_f64(), t.memory_size(), t)
}

fn probe_rows(
    layout: &RowLayout,
    t: &dyn RowIndex,
    pages: &RowPages,
    input: &[EncodedKeys],
) -> f64 {
    let mut out = Vec::new();
    let start = Instant::now();
    for k in input {
        t.find(layout, k, pages, &mut out);
        std::hint::black_box(&out);
    }
    start.elapsed().as_secs_f64()
}

fn probe_arrow(t: &KeyTable, input: &[Rows]) -> f64 {
    let seed = HashSeed::new(Some(SEED));
    let start = Instant::now();
    for rows in input {
        let mut hits = 0usize;
        for r in rows {
            hits += usize::from(t.find(r.data(), seed.hash(r.data())).is_some());
        }
        std::hint::black_box(hits);
    }
    start.elapsed().as_secs_f64()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

fn env_list(name: &str, default: &[u64]) -> Vec<u64> {
    std::env::var(name).map_or_else(
        |_| default.to_vec(),
        |s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
    )
}

/// One result line, all times in ns per row.
struct Line<'a> {
    kind: KeyType,
    n: u64,
    op: &'a str,
    stream: Stream,
    salted: f64,
    hb: f64,
    encode: f64,
    kt: f64,
    encode_arrow: f64,
    salted_bytes: String,
    kt_bytes: String,
}

impl Line<'_> {
    fn write(&self, out: &mut impl Write) -> std::io::Result<()> {
        let ours = self.salted + self.encode;
        let today = self.kt + self.encode_arrow;
        writeln!(
            out,
            "| {} | {} | {} | {} | {:.1} | {:.1} | **{:.2}** | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} | {} | {} |",
            self.kind.name(),
            self.n,
            self.op,
            self.stream.name(),
            self.salted,
            self.hb,
            self.hb / self.salted,
            self.encode,
            self.encode_arrow,
            ours,
            today,
            today / ours,
            self.salted_bytes,
            self.kt_bytes,
        )?;
        out.flush()
    }
}

fn main() -> std::io::Result<()> {
    let sizes = env_list("ROW_TABLE_SIZES", &[10_000, 1_000_000, 15_000_000]);
    let reps = env_list("ROW_TABLE_REPS", &[3])
        .first()
        .copied()
        .unwrap_or(3);
    let large_types = [KeyType::I64, KeyType::Str24];
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "| keys | N | op | stream | index ns/row: salted | hashbrown | **hb/salted** | encode ns/row: rows | arrow-row | e2e ns/row: salted | keytable | keytable/salted | B/key: salted+pages | keytable |"
    )?;
    writeln!(
        out,
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )?;
    let kinds = [
        KeyType::I64,
        KeyType::I64x2,
        KeyType::Str8,
        KeyType::Str24,
        KeyType::Dec,
    ];
    // ROW_TABLE_TYPES=i64,str24 limits the key types (names as printed).
    let only = std::env::var("ROW_TABLE_TYPES").ok();
    for kind in kinds {
        if only
            .as_deref()
            .is_some_and(|o| !o.split(',').any(|t| t == kind.name()))
        {
            continue;
        }
        let types = kind.types();
        let layout =
            RowLayout::new(&types, &vec![false; types.len()]).map_err(std::io::Error::other)?;
        let enc = KeyEncoder::new(&types).map_err(std::io::Error::other)?;
        for &n in &sizes {
            if n > 1_000_000 && !large_types.iter().any(|t| t.name() == kind.name()) {
                continue;
            }
            #[expect(clippy::cast_precision_loss, reason = "row counts far below 2^53")]
            let rows = (2 * n) as f64;
            let ns = |s: f64| s * 1e9 / rows;
            #[expect(clippy::cast_precision_loss, reason = "byte counts")]
            let per_key = |b: usize| format!("{:.0}", b as f64 / n as f64);
            for stream in [Stream::Uniform, Stream::Zipf] {
                let input = Input::new(kind, &layout, &enc, stream, n);
                let (mut s, mut h, mut k) = (Vec::new(), Vec::new(), Vec::new());
                let (mut sb, mut kb) = (0, 0);
                for _ in 0..reps {
                    let (t, b, _, _) = build_rows(&layout, &input.rows, true, 5);
                    s.push(t);
                    sb = b;
                    h.push(build_rows(&layout, &input.rows, false, 5).0);
                    let (t, b, _) = build_arrow(&input.arrow);
                    k.push(t);
                    kb = b;
                }
                Line {
                    kind,
                    n,
                    op: "build",
                    stream,
                    salted: ns(median(s)),
                    hb: ns(median(h)),
                    encode: ns(input.encode),
                    kt: ns(median(k)),
                    encode_arrow: ns(input.encode_arrow),
                    salted_bytes: per_key(sb),
                    kt_bytes: per_key(kb),
                }
                .write(&mut out)?;
            }
            // Probe: a table of the N keys (the first half of the uniform
            // stream, built untimed), then 2N lookups.
            let keys = Input::new(kind, &layout, &enc, Stream::Uniform, n);
            let half = keys.rows.len().div_ceil(2);
            let (_, _, salted, salted_pages) = build_rows(&layout, &keys.rows[..half], true, 4);
            let (_, _, hb, hb_pages) = build_rows(&layout, &keys.rows[..half], false, 4);
            let (_, _, kt) = build_arrow(&keys.arrow[..half]);
            drop(keys);
            for stream in [Stream::Hit, Stream::Half, Stream::Zipf] {
                let input = Input::new(kind, &layout, &enc, stream, n);
                let (mut s, mut h, mut k) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..reps {
                    s.push(probe_rows(
                        &layout,
                        salted.as_ref(),
                        &salted_pages,
                        &input.rows,
                    ));
                    h.push(probe_rows(&layout, hb.as_ref(), &hb_pages, &input.rows));
                    k.push(probe_arrow(&kt, &input.arrow));
                }
                Line {
                    kind,
                    n,
                    op: "probe",
                    stream,
                    salted: ns(median(s)),
                    hb: ns(median(h)),
                    encode: ns(input.encode),
                    kt: ns(median(k)),
                    encode_arrow: ns(input.encode_arrow),
                    salted_bytes: "-".to_owned(),
                    kt_bytes: "-".to_owned(),
                }
                .write(&mut out)?;
            }
        }
    }
    Ok(())
}
