// SPDX-License-Identifier: Apache-2.0
//! Seeded random functions: `rand` (`random`), `randn`, `uniform`,
//! `randstr` and `uuid`.
//!
//! Ground truth: Spark 4.2's `randomExpressions.scala` (`Rand`, `Randn`,
//! `Uniform`, `RandStr`), `misc.scala` (`Uuid`) and
//! `XORShiftRandom` (`core/.../util/random/XORShiftRandom.scala`), plus the
//! `random.sql` golden (`tests/spark-golden/v4.2.0/results/random.sql.out`).
//!
//! Every function owns one value stream per partition: the stream is a pure
//! function of the seed (the folded seed argument, or the query seed for an
//! absent or NULL seed) and the partition index, advanced once per row the
//! partition produces. The analyzer only lets foldable seeds through (the
//! registry rules), so a bind reads the seed from the first row of every
//! invocation; the per-partition streams live in the partitioned body.
//!
//! `rand`/`randn` draw from Spark's `XORShiftRandom` (the MurmurHash3 seed
//! hash and the shift register below), so seeded results match Spark
//! exactly (`rand(0)` is `0.7604953758285915`). `uuid` draws its 128 bits
//! from the same generator and sets the RFC 4122 version and variant bits;
//! Spark fills those bits from a Mersenne Twister instead, so only the
//! shape (and the per-seed determinism) matches, not Spark's values. No
//! golden pins seeded `uuid` values (see `misc`).

use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{ArrayRef, Float64Array, StringArray};
use arrow_schema::DataType;
use krishiv_common::error::{KrishivError, UnableToAcquireMemory};
use krishiv_kernels::cast::{CastMode, CastOptions, TimeZoneRules, cast};

use super::super::uniform::{UniformKind, uniform_result as shared_result};
use super::args::{Inputs, internal};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds (`random` is Spark's alias of `rand`).
pub(super) const NAMES: &[&str] = &["rand", "random", "randn", "uniform", "randstr", "uuid"];

/// Spark's `XORShiftRandom`: the seed hash (Scala's `MurmurHash3`, the only
/// hashing in the stream) and the shift register.
#[derive(Debug, Clone)]
struct XorShift {
    seed: u64,
}

impl XorShift {
    /// The stream for `init` (Spark's `new XORShiftRandom(init)`).
    fn new(init: i64) -> Self {
        Self {
            seed: hash_seed(init),
        }
    }

    /// Spark's `next(bits)`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Java's `(int)` cast of the masked bits"
    )]
    fn next(&mut self, bits: u32) -> i32 {
        let mut next = self.seed ^ self.seed.wrapping_shl(21);
        next ^= next >> 35;
        next ^= next.wrapping_shl(4);
        self.seed = next;
        (next & ((1_u64 << bits) - 1)) as i32
    }

    /// `java.util.Random.nextDouble`: 53 bits over `2^53`.
    fn next_double(&mut self) -> f64 {
        let high = i64::from(self.next(26));
        let low = i64::from(self.next(27));
        ((high << 27) + low) as f64 / 9_007_199_254_740_992.0
    }

    /// `java.util.Random.nextGaussian` (the polar method): `cached` is the
    /// spare value (`haveNextNextGaussian` / `nextNextGaussian`).
    fn next_gaussian(&mut self, cached: &mut Option<f64>) -> f64 {
        if let Some(v) = cached.take() {
            return v;
        }
        loop {
            let v1 = 2.0 * self.next_double() - 1.0;
            let v2 = 2.0 * self.next_double() - 1.0;
            let s = v1 * v1 + v2 * v2;
            if s < 1.0 && s != 0.0 {
                let multiplier = (-2.0 * s.ln() / s).sqrt();
                *cached = Some(v2 * multiplier);
                return v1 * multiplier;
            }
        }
    }

    /// `java.util.Random.nextLong`.
    fn next_long(&mut self) -> i64 {
        (i64::from(self.next(32)) << 32) + i64::from(self.next(32))
    }
}

/// Spark's `XORShiftRandom.hashSeed`: the seed's big-endian bytes hashed
/// with Scala's `MurmurHash3.bytesHash`.
fn hash_seed(seed: i64) -> u64 {
    let bytes = seed.to_be_bytes();
    let low = bytes_hash(&bytes, 0x3c07_4a61);
    let high = bytes_hash(&bytes, low);
    ((i64::from(high) << 32) | (i64::from(low) & 0xffff_ffff)).cast_unsigned()
}

/// Scala 2.13's `MurmurHash3.bytesHash` (an 8-byte input has no tail).
fn bytes_hash(data: &[u8], seed: i32) -> i32 {
    let mut h = seed;
    let (chunks, _) = data.as_chunks::<4>();
    for chunk in chunks {
        let k = i32::from_le_bytes(*chunk);
        h = murmur_mix(h, k);
    }
    murmur_finalize(h, data.len())
}

/// Scala 2.13's `MurmurHash3.mix`.
fn murmur_mix(hash: i32, data: i32) -> i32 {
    murmur_mix_last(hash, data)
        .rotate_left(13)
        .wrapping_mul(5)
        .wrapping_add(0xe654_6b64u32.cast_signed())
}

/// Scala 2.13's `MurmurHash3.mixLast`.
fn murmur_mix_last(hash: i32, mut data: i32) -> i32 {
    data = data.wrapping_mul(0xcc9e_2d51u32.cast_signed());
    data = data.rotate_left(15);
    data = data.wrapping_mul(0x1b87_3593u32.cast_signed());
    hash ^ data
}

/// Scala 2.13's `MurmurHash3.finalizeHash`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the hashed seed bytes always fit"
)]
fn murmur_finalize(hash: i32, length: usize) -> i32 {
    murmur_avalanche(hash ^ length as i32)
}

/// Scala 2.13's `MurmurHash3.avalanche`.
fn murmur_avalanche(mut hash: i32) -> i32 {
    hash ^= (hash.cast_unsigned() >> 16).cast_signed();
    hash = hash.wrapping_mul(0x85eb_ca6bu32.cast_signed());
    hash ^= (hash.cast_unsigned() >> 13).cast_signed();
    hash = hash.wrapping_mul(0xc2b2_ae35u32.cast_signed());
    hash ^= (hash.cast_unsigned() >> 16).cast_signed();
    hash
}

/// SplitMix64: the per-query seed from the query start instant (fixed per
/// plan, never sampled at execution).
fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The query seed: one value per query, from the plan-fixed query start.
fn query_seed(opts: &BindOptions) -> u64 {
    opts.query_start_micros.cast_unsigned()
}

/// An unseeded (or NULL-seeded, where Spark draws randomly) call's seed:
/// the query seed mixed with a per-occurrence salt, so two unseeded calls
/// in one query draw different streams (Spark's `ResolveRandomSeed`
/// assigns each its own random seed) while one query always draws the same
/// streams. The salt counts the unseeded binds of one plan instantiation,
/// in compilation order.
fn unseeded_seed(opts: &BindOptions) -> i64 {
    use std::sync::atomic::Ordering;
    let salt = opts.seed_salt.fetch_add(1, Ordering::Relaxed);
    splitmix64(query_seed(opts).wrapping_add(salt)) as i64
}

/// One value stream: the generator and the spare Gaussian.
#[derive(Debug)]
struct Stream {
    rng: XorShift,
    gaussian: Option<f64>,
}

impl Stream {
    fn new(seed: i64) -> Self {
        Self {
            rng: XorShift::new(seed),
            gaussian: None,
        }
    }
}

/// The per-partition streams of one bound function, by partition index.
/// The seed is fixed per binding (the folded seed argument, or the query
/// seed for an absent or NULL seed), so the first bind records it and every
/// slot `j` draws `base + j`: a driver that binds a higher partition first
/// no longer seeds the lower slots with the wrong seed. The table is
/// untracked by `krishiv-memory`, but bounded by the partition count
/// (bytes), like the other per-partition counters.
#[derive(Debug, Default)]
struct Streams {
    base: Option<i64>,
    streams: Vec<Stream>,
}

impl Streams {
    /// The stream of `partition`: slot `j` draws `seed + j`, seeded once.
    /// (`seed` is the binding's base seed; the callers pass it unfolded.)
    fn stream(&mut self, partition: u32, seed: i64) -> &mut Stream {
        let base = *self.base.get_or_insert(seed);
        let index = partition as usize;
        if self.streams.len() <= index {
            let have = self.streams.len();
            self.streams.reserve(index + 1 - have);
            for slot in have..=index {
                // `slot <= index`, and `index` is a `u32` widened to
                // `usize`: `slot` fits a `u32` (the fallback never fires).
                let p = u32::try_from(slot).unwrap_or(u32::MAX);
                self.streams
                    .push(Stream::new(base.wrapping_add(i64::from(p))));
            }
        }
        &mut self.streams[index]
    }
}

/// Locks `mutex` (poison is impossible: the bodies never panic).
fn lock<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, KrishivError> {
    mutex
        .lock()
        .map_err(|_| internal("a random function's streams are poisoned".to_owned()))
}

/// The folded seed argument `i` (`None` for a NULL seed).
fn seed(x: &Inputs<'_>, i: usize) -> Result<Option<i64>, KrishivError> {
    let c = x.column(i)?;
    if c.data_type() == &DataType::Null || c.is_null(0) {
        return Ok(None);
    }
    match c.data_type() {
        DataType::Int32 => Ok(Some(i64::from(c.as_primitive::<Int32Type>().value(0)))),
        DataType::Int64 => Ok(Some(c.as_primitive::<Int64Type>().value(0))),
        t => Err(internal(format!("a random seed of type {t}"))),
    }
}

/// The cast to a bound of `uniform` (Spark's plain `Cast`, never `TRY_CAST`).
fn bound_cast(
    c: &ArrayRef,
    to: &DataType,
    ansi: bool,
    zone: &Arc<dyn TimeZoneRules>,
) -> Result<ArrayRef, KrishivError> {
    cast(
        c.as_ref(),
        to,
        &CastOptions {
            mode: if ansi {
                CastMode::Ansi
            } else {
                CastMode::Legacy
            },
            time_zone: zone.as_ref(),
        },
    )
    .map_err(KrishivError::from)
}

/// The folded numeric argument `i` as an `f64` (`None` for NULL): Spark's
/// `Cast` to `DOUBLE`.
fn double(
    x: &Inputs<'_>,
    i: usize,
    ansi: bool,
    zone: &Arc<dyn TimeZoneRules>,
) -> Result<Option<f64>, KrishivError> {
    let c = x.column(i)?;
    if c.data_type() == &DataType::Null || c.is_null(0) {
        return Ok(None);
    }
    let d = bound_cast(&c, &DataType::Float64, ansi, zone)?;
    if d.is_null(0) {
        return Ok(None);
    }
    Ok(Some(d.as_primitive::<Float64Type>().value(0)))
}

/// The folded length argument `i` as an `i32` (`None` for NULL): Spark's
/// `Cast` to `INT`.
fn length(
    x: &Inputs<'_>,
    i: usize,
    ansi: bool,
    zone: &Arc<dyn TimeZoneRules>,
) -> Result<Option<i32>, KrishivError> {
    let c = x.column(i)?;
    if c.data_type() == &DataType::Null || c.is_null(0) {
        return Ok(None);
    }
    let d = bound_cast(&c, &DataType::Int32, ansi, zone)?;
    if d.is_null(0) {
        return Ok(None);
    }
    Ok(Some(d.as_primitive::<Int32Type>().value(0)))
}

/// `rand(seed?)` / `random(seed?)`: `rng.nextDouble()` per row.
fn rand(opts: &BindOptions, args: &[BindArg]) -> Binding {
    let queries = unseeded_seed(opts);
    let seeded = !args.is_empty();
    let state = Mutex::new(Streams::default());
    Binding::partitioned(DataType::Float64, move |x, partition| {
        let seed = if seeded {
            seed(x, 0)?.unwrap_or(0)
        } else {
            queries
        };
        let mut streams = lock(&state)?;
        let s = streams.stream(partition, seed);
        let out: Vec<f64> = (0..x.rows()).map(|_| s.rng.next_double()).collect();
        Ok(Arc::new(Float64Array::from(out)) as ArrayRef)
    })
}

/// `randn(seed?)`: `rng.nextGaussian()` per row.
fn randn(opts: &BindOptions, args: &[BindArg]) -> Binding {
    let queries = unseeded_seed(opts);
    let seeded = !args.is_empty();
    let state = Mutex::new(Streams::default());
    Binding::partitioned(DataType::Float64, move |x, partition| {
        let seed = if seeded {
            seed(x, 0)?.unwrap_or(0)
        } else {
            queries
        };
        let mut streams = lock(&state)?;
        let s = streams.stream(partition, seed);
        let out: Vec<f64> = (0..x.rows())
            .map(|_| s.rng.next_gaussian(&mut s.gaussian))
            .collect();
        Ok(Arc::new(Float64Array::from(out)) as ArrayRef)
    })
}

/// `uuid(seed?)`: 128 bits per row, with the RFC 4122 version and variant
/// bits. A NULL or absent seed draws from the query seed.
fn uuid(opts: &BindOptions, args: &[BindArg]) -> Binding {
    let queries = unseeded_seed(opts);
    let seeded = !args.is_empty();
    let state = Mutex::new(Streams::default());
    Binding::partitioned(DataType::Utf8, move |x, partition| {
        let seed = if seeded {
            seed(x, 0)?.unwrap_or(queries)
        } else {
            queries
        };
        let mut streams = lock(&state)?;
        let s = streams.stream(partition, seed);
        let out: Vec<String> = (0..x.rows())
            .map(|_| {
                let most = (s.rng.next_long().cast_unsigned() & 0xffff_ffff_ffff_0fff)
                    | 0x0000_0000_0000_4000;
                let least = (s.rng.next_long().cast_unsigned() | 0x8000_0000_0000_0000)
                    & 0xbfff_ffff_ffff_ffff;
                format!(
                    "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                    (most >> 32) as u32,
                    ((most >> 16) & 0xffff) as u16,
                    (most & 0xffff) as u16,
                    ((least >> 48) & 0xffff) as u16,
                    least & 0xffff_ffff_ffff,
                )
            })
            .collect();
        Ok(Arc::new(StringArray::from(out)) as ArrayRef)
    })
}

/// The [`UniformKind`] of a `uniform` bound (an untyped `NULL` reads as
/// `DOUBLE`, as Spark casts it).
fn uniform_kind(t: &DataType) -> UniformKind {
    match t {
        DataType::Int8 => UniformKind::Byte,
        DataType::Int16 => UniformKind::Short,
        DataType::Int32 => UniformKind::Int,
        DataType::Int64 => UniformKind::Long,
        DataType::Float32 => UniformKind::Float,
        DataType::Float64 | DataType::Null => UniformKind::Double,
        DataType::Decimal128(p, s) => UniformKind::Decimal {
            precision: *p,
            scale: *s,
        },
        _ => UniformKind::Other,
    }
}

/// The Arrow result type of `uniform(min, max)` for argument types
/// `min`/`max`: Spark's `Uniform.dataType` (see `uniform_kind`).
fn uniform_result(min: &DataType, max: &DataType) -> DataType {
    match shared_result(uniform_kind(min), uniform_kind(max)) {
        UniformKind::Byte => DataType::Int8,
        UniformKind::Short => DataType::Int16,
        UniformKind::Int => DataType::Int32,
        UniformKind::Long => DataType::Int64,
        UniformKind::Float => DataType::Float32,
        UniformKind::Double | UniformKind::Other => DataType::Float64,
        UniformKind::Decimal { precision, scale } => DataType::Decimal128(precision, scale),
    }
}

/// `uniform(min, max[, seed])`: Spark's `Uniform` is a `RuntimeReplaceable`
/// over `cast(min + (max - min) * Rand(seed))`, computed here in one pass:
/// the bounds as doubles, one draw per row, then Spark's `Cast` to the
/// result type. A NULL bound is a NULL result.
fn uniform(opts: &BindOptions, args: &[BindArg]) -> Binding {
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let result = uniform_result(&types[0], &types[1]);
    let out_type = result.clone();
    let queries = unseeded_seed(opts);
    let seeded = args.len() > 2;
    let ansi = opts.ansi;
    let zone = Arc::clone(&opts.time_zone);
    let state = Mutex::new(Streams::default());
    Binding::partitioned(result, move |x, partition| {
        let seed = if seeded {
            seed(x, 2)?.unwrap_or(0)
        } else {
            queries
        };
        let (Some(lo), Some(hi)) = (double(x, 0, ansi, &zone)?, double(x, 1, ansi, &zone)?) else {
            return Ok(arrow_array::new_null_array(&out_type, x.rows()));
        };
        let mut streams = lock(&state)?;
        let s = streams.stream(partition, seed);
        let draws: Vec<f64> = (0..x.rows())
            .map(|_| lo + (hi - lo) * s.rng.next_double())
            .collect();
        let draws = Float64Array::from(draws);
        bound_cast(&(Arc::new(draws) as ArrayRef), &out_type, ansi, &zone)
    })
}

/// `randstr(length[, seed])`: `length` characters per row, each drawn with
/// `Math.abs(rng.nextInt() % 62)` and read as a digit, a lower-case or an
/// upper-case letter (Spark's `ExpressionImplUtils.randStr`). A NULL length
/// is the empty string; a negative length is
/// `INVALID_PARAMETER_VALUE.LENGTH`.
fn randstr(opts: &BindOptions, args: &[BindArg]) -> Binding {
    let queries = unseeded_seed(opts);
    let seeded = args.len() > 1;
    let ansi = opts.ansi;
    let zone = Arc::clone(&opts.time_zone);
    let state = Mutex::new(Streams::default());
    Binding::partitioned(DataType::Utf8, move |x, partition| {
        let seed = if seeded {
            seed(x, 1)?.unwrap_or(queries)
        } else {
            queries
        };
        let Some(n) = length(x, 0, ansi, &zone)? else {
            return Ok(Arc::new(StringArray::from(vec![""; x.rows()])) as ArrayRef);
        };
        if n < 0 {
            return Err(krishiv_common::error::InvalidParameterValueLength {
                function_name: "`randstr`".to_owned(),
                length: n.to_string(),
                parameter: "`length`".to_owned(),
            }
            .into());
        }
        // The output must fit a `StringArray`'s `i32` offsets: refuse before
        // drawing or allocating, as the other string builders do.
        let total = usize::try_from(n)
            .ok()
            .and_then(|n| n.checked_mul(x.rows()))
            .filter(|t| i32::try_from(*t).is_ok());
        let Some(_) = total else {
            return Err(UnableToAcquireMemory {
                requested_bytes: n.to_string(),
                received_bytes: "0".to_owned(),
            }
            .into());
        };
        let mut streams = lock(&state)?;
        let s = streams.stream(partition, seed);
        let out: Vec<String> = (0..x.rows())
            .map(|_| {
                (0..n)
                    .map(|_| {
                        // Java's `Math.abs(rng.nextInt() % 62)`: `%` keeps
                        // the dividend's sign, then `abs`.
                        let v = (s.rng.next(32) % 62).abs();
                        // `v` is `abs(nextInt() % 62)`, so in `0..62`.
                        let v = u8::try_from(v).unwrap_or(0);
                        if v < 10 {
                            (b'0' + v) as char
                        } else if v < 36 {
                            (b'a' + v - 10) as char
                        } else {
                            (b'A' + v - 36) as char
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Arc::new(StringArray::from(out)) as ArrayRef)
    })
}

/// Whether `t` is an `int` or `bigint` seed (an untyped `NULL` reads as
/// one): what the analyzer lets through for every seed but `uuid`'s.
fn is_seed(t: &DataType) -> bool {
    matches!(t, DataType::Int32 | DataType::Int64 | DataType::Null)
}

/// Binds the random function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("rand" | "random", []) => rand(opts, args),
        ("rand" | "random" | "randn" | "uuid", [t]) if is_seed(t) => match listed {
            "randn" => randn(opts, args),
            "uuid" => uuid(opts, args),
            _ => rand(opts, args),
        },
        ("randn", []) => randn(opts, args),
        ("uuid", []) => uuid(opts, args),
        ("uniform", [_, _] | [_, _, _]) => uniform(opts, args),
        ("randstr", [_] | [_, _]) => randstr(opts, args),
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
