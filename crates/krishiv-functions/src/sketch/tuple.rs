// SPDX-License-Identifier: Apache-2.0
//! Tuple sketches (Spark 4.2+, `tupleSketch*.scala`, `tupleUnion`, `tupleIntersection`, `tupleDifference`) over the
//! `datasketches` crate's `tuple` module. Spark's summaries are Java's `DoubleSummary` and `IntegerSummary`: a value
//! and the aggregation mode (`sum`, `min`, `max`, `alwaysone`) it was built with, written next to each retained hash
//! as the value (little-endian `f64` or `i32`) and one mode byte. The keys hash exactly as the Theta sketch's do.

use datasketches::codec::{SketchBytes, SketchSlice};
use datasketches::error::{Error, ErrorKind};
use datasketches::hash::value::{canonical_float, raw_bytes};
use datasketches::tuple::{
    CompactTupleSketch, SummaryCombinePolicy, SummaryPolicy, SummaryUpdatePolicy, TupleANotB,
    TupleSketchBuilder, TupleSummaryValue, TupleUnionBuilder,
};
pub use datasketches::tuple::{TupleIntersection, TupleSketch, TupleUnion};
use krishiv_common::error::{
    InternalError, KrishivError, TupleInvalidInputSketchBuffer, TupleInvalidSketchMode,
};

use super::theta::{self, CompactThetaSketch};

/// Spark's `TupleSummaryMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `sum`.
    Sum,
    /// `min`.
    Min,
    /// `max`.
    Max,
    /// `alwaysone`.
    AlwaysOne,
}

impl Mode {
    /// Spark's `TupleSummaryMode.fromString` (case-insensitive).
    ///
    /// # Errors
    /// `TUPLE_INVALID_SKETCH_MODE` naming `function` for any other text.
    pub fn parse(text: &str, function: &str) -> Result<Self, KrishivError> {
        match text.to_lowercase().as_str() {
            "sum" => Ok(Self::Sum),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "alwaysone" => Ok(Self::AlwaysOne),
            _ => Err(TupleInvalidSketchMode {
                function: format!("`{function}`"),
                mode: text.to_owned(),
                valid_modes: "sum, min, max, alwaysone".to_owned(),
            }
            .into()),
        }
    }

    fn sum_for_count(self) -> Self {
        self
    }

    fn byte(self) -> u8 {
        match self {
            Self::Sum => 0,
            Self::Min => 1,
            Self::Max => 2,
            Self::AlwaysOne => 3,
        }
    }

    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Sum),
            1 => Some(Self::Min),
            2 => Some(Self::Max),
            3 => Some(Self::AlwaysOne),
            _ => None,
        }
    }
}

/// The numeric type of a summary (`f64` for `_double`, `i32` for `_integer`) and Java's arithmetic on it.
pub trait Value: Copy + PartialOrd + std::fmt::Debug {
    /// The value of a summary no update has touched.
    fn identity(mode: Mode) -> Self;
    /// Fold `other` into `self` as the mode says (Java's `DoubleSummary.update` / `IntegerSummary.update`).
    fn fold(self, mode: Mode, other: Self) -> Self;
    /// Little-endian image of the value.
    fn write(self, out: &mut SketchBytes);
    /// Read the image back.
    fn read(cursor: &mut SketchSlice<'_>) -> Result<Self, Error>;
    /// The value widened to `f64` (a `_double` summary result).
    fn as_f64(self) -> f64;
    /// The value widened to `i64` (an `_integer` summary result).
    fn as_i64(self) -> i64;
    /// The size of the image.
    const SIZE: usize;
}

impl Value for f64 {
    fn identity(mode: Mode) -> Self {
        match mode {
            Mode::Sum => 0.0,
            Mode::Min => f64::INFINITY,
            Mode::Max => f64::NEG_INFINITY,
            Mode::AlwaysOne => 1.0,
        }
    }
    fn fold(self, mode: Mode, other: Self) -> Self {
        match mode {
            Mode::Sum => self + other,
            Mode::Min => java_min(self, other),
            Mode::Max => java_max(self, other),
            Mode::AlwaysOne => 1.0,
        }
    }
    fn write(self, out: &mut SketchBytes) {
        out.write_f64_le(self);
    }
    fn read(cursor: &mut SketchSlice<'_>) -> Result<Self, Error> {
        cursor
            .read_f64_le()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "tuple double summary"))
    }
    fn as_f64(self) -> f64 {
        self
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "never read: a double summary reports as f64"
    )]
    fn as_i64(self) -> i64 {
        self as i64
    }
    const SIZE: usize = 8;
}

impl Value for i32 {
    fn identity(mode: Mode) -> Self {
        match mode {
            Mode::Sum => 0,
            Mode::Min => i32::MAX,
            Mode::Max => i32::MIN,
            Mode::AlwaysOne => 1,
        }
    }
    fn fold(self, mode: Mode, other: Self) -> Self {
        match mode {
            Mode::Sum => self.wrapping_add(other),
            Mode::Min => self.min(other),
            Mode::Max => self.max(other),
            Mode::AlwaysOne => 1,
        }
    }
    fn write(self, out: &mut SketchBytes) {
        out.write_i32_le(self);
    }
    fn read(cursor: &mut SketchSlice<'_>) -> Result<Self, Error> {
        cursor
            .read_i32_le()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "tuple integer summary"))
    }
    fn as_f64(self) -> f64 {
        f64::from(self)
    }
    fn as_i64(self) -> i64 {
        i64::from(self)
    }
    const SIZE: usize = 4;
}

/// `Math.min` on doubles: NaN wins, `-0.0` is below `0.0`.
fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() { a } else { b }
    } else if a < b {
        a
    } else {
        b
    }
}

/// `Math.max` on doubles: NaN wins, `0.0` is above `-0.0`.
fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() { b } else { a }
    } else if a > b {
        a
    } else {
        b
    }
}

/// A summary: the value and the mode it folds with.
#[derive(Clone, Copy, Debug)]
pub struct Summary<V> {
    /// The aggregated value.
    pub value: V,
    /// The mode the summary was created with (written to the image).
    pub mode: Mode,
}

impl<V: Value> TupleSummaryValue for Summary<V> {
    fn serialize_size(&self) -> usize {
        V::SIZE + 1
    }
    fn serialize_value(&self, bytes: &mut SketchBytes) {
        self.value.write(bytes);
        bytes.write_u8(self.mode.byte());
    }
    fn deserialize_value(cursor: &mut SketchSlice<'_>) -> Result<Self, Error> {
        let value = V::read(cursor)?;
        let b = cursor
            .read_u8()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "tuple summary mode"))?;
        let mode = Mode::from_byte(b).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("unknown tuple summary mode {b}"),
            )
        })?;
        Ok(Self { value, mode })
    }
}

/// The policy of every Spark tuple operation: one mode folds updates and combines summaries.
#[derive(Clone, Copy, Debug)]
pub struct Policy<V> {
    mode: Mode,
    marker: std::marker::PhantomData<fn() -> V>,
}

impl<V> Policy<V> {
    /// The policy of `mode`.
    #[must_use]
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            marker: std::marker::PhantomData,
        }
    }
}

impl<V: Value> SummaryPolicy for Policy<V> {
    type Summary = Summary<V>;
    fn create(&self) -> Summary<V> {
        Summary {
            value: V::identity(self.mode),
            mode: self.mode,
        }
    }
}

impl<V: Value> SummaryUpdatePolicy<V> for Policy<V> {
    fn update(&self, summary: &mut Summary<V>, value: V) {
        summary.value = summary.value.fold(summary.mode, value);
    }
}

impl<V: Value> SummaryCombinePolicy for Policy<V> {
    fn combine(&self, summary: &mut Summary<V>, other: &Summary<V>) {
        summary.value = summary.value.fold(summary.mode, other.value);
    }
}

/// A compact tuple sketch of `V` summaries.
pub type Compact<V> = CompactTupleSketch<Summary<V>>;

fn invalid_buffer(function: &str) -> KrishivError {
    TupleInvalidInputSketchBuffer {
        function: format!("`{function}`"),
    }
    .into()
}

fn builder_failure(function: &str, e: &dyn std::fmt::Display) -> KrishivError {
    InternalError {
        message: format!("`{function}`: the Tuple sketch builder failed: {e}"),
    }
    .into()
}

/// A new updatable sketch of `lg_nom` (checked by [`theta::check_lg_nom`]).
///
/// # Errors
/// `SKETCH_INVALID_LG_NOM_ENTRIES` for an out-of-range size.
pub fn new_sketch<V: Value>(
    lg_nom: i32,
    mode: Mode,
    function: &str,
) -> Result<TupleSketch<Policy<V>>, KrishivError> {
    let k = theta::check_lg_nom(lg_nom, function)?;
    TupleSketchBuilder::new(Policy::<V>::new(mode))
        .lg_k(k.max(5))
        .build()
        .map_err(|e| builder_failure(function, &e))
}

/// A new union of `lg_nom`.
///
/// # Errors
/// `SKETCH_INVALID_LG_NOM_ENTRIES` for an out-of-range size.
pub fn new_union<V: Value>(
    lg_nom: i32,
    mode: Mode,
    function: &str,
) -> Result<TupleUnion<Policy<V>>, KrishivError> {
    let k = theta::check_lg_nom(lg_nom, function)?;
    TupleUnionBuilder::new(Policy::<V>::new(mode))
        .lg_k(k.max(5))
        .build()
        .map_err(|e| builder_failure(function, &e))
}

/// A new intersection.
#[must_use]
pub fn new_intersection<V: Value>(mode: Mode) -> TupleIntersection<Policy<V>> {
    TupleIntersection::new(Policy::<V>::new(mode))
}

/// `INT` and `BIGINT` keys: both hash as a `long`.
pub fn update_long<V: Value>(s: &mut TupleSketch<Policy<V>>, key: i64, value: V) {
    s.update(key, value);
}

/// `DOUBLE` keys (`FLOAT` is promoted first).
pub fn update_double<V: Value>(s: &mut TupleSketch<Policy<V>>, key: f64, value: V) {
    s.update(canonical_float::from_f64(key), value);
}

/// `STRING` keys: the bytes of the collation's sort key; the empty one is ignored.
pub fn update_bytes<V: Value>(s: &mut TupleSketch<Policy<V>>, key: &[u8], value: V) {
    if !key.is_empty() {
        s.update(raw_bytes::from_slice(key), value);
    }
}

/// `ARRAY<INT>` keys: little-endian 4-byte words; an empty array is ignored.
pub fn update_int_words<V: Value>(
    s: &mut TupleSketch<Policy<V>>,
    len: usize,
    keys: impl Iterator<Item = i32>,
    value: V,
) {
    if len > 0 {
        let mut bytes = Vec::with_capacity(len.saturating_mul(4));
        for v in keys {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        s.update(raw_bytes::from_vec(bytes), value);
    }
}

/// `ARRAY<BIGINT>` keys: little-endian 8-byte words; an empty array is ignored.
pub fn update_long_words<V: Value>(
    s: &mut TupleSketch<Policy<V>>,
    len: usize,
    keys: impl Iterator<Item = i64>,
    value: V,
) {
    if len > 0 {
        let mut bytes = Vec::with_capacity(len.saturating_mul(8));
        for v in keys {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        s.update(raw_bytes::from_vec(bytes), value);
    }
}

/// Java's image of an empty sketch.
const EMPTY_IMAGE: [u8; 8] = [1, 3, 9, 1, 0, 0x0E, 0xCC, 0x93];

/// The compact image Spark stores in a `BINARY` column (unordered, uncompressed).
#[must_use]
pub fn to_bytes<V: Value>(sketch: &Compact<V>) -> Vec<u8> {
    if sketch.is_empty() {
        // Java writes the empty image with the default seed hash and the EMPTY flag (the crate writes seed hash 0).
        return EMPTY_IMAGE.to_vec();
    }
    let mut bytes = sketch.serialize();
    // Spark's results are never flagged ordered (the crate marks a one-entry set operation result so).
    if let Some(flags) = bytes.get_mut(5) {
        *flags &= !0x10;
    }
    bytes
}

/// [`to_bytes`] for a sketch of `lg_nom`: below the library's smallest size (5) the sketch is built at 5, so a result
/// of more than 16 entries is cut to the 16 smallest hashes with theta the 17th, as Java's `lgNomEntries` 4 sketch is.
/// Entries come out in hash order (Java's are in table order).
#[must_use]
pub fn to_bytes_at<V: Value>(sketch: &Compact<V>, lg_nom: i32) -> Vec<u8> {
    const K: usize = 16;
    let bytes = to_bytes(sketch);
    if lg_nom >= 5 || sketch.num_retained() <= K {
        return bytes;
    }
    let pre = usize::from(bytes.first().copied().unwrap_or(0)) * 8;
    let n = sketch.num_retained();
    let Some(body) = bytes.get(pre..).filter(|b| b.len() % n == 0) else {
        return bytes;
    };
    let width = body.len() / n;
    let mut entries: Vec<&[u8]> = body.chunks_exact(width).collect();
    let hash = |e: &[u8]| {
        e.get(..8)
            .map_or(0, |h| u64::from_le_bytes(h.try_into().unwrap_or([0; 8])))
    };
    entries.sort_unstable_by_key(|e| hash(e));
    let theta = hash(entries[K]);
    let mut out = bytes[..8].to_vec();
    out[0] = 3;
    out[5] |= 0x10;
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&theta.to_le_bytes());
    for e in &entries[..K] {
        out.extend_from_slice(e);
    }
    out
}

/// Compact an updatable sketch.
#[must_use]
pub fn compact<V: Value>(sketch: &TupleSketch<Policy<V>>) -> Compact<V> {
    sketch.compact(false)
}

/// Read a compact tuple sketch as `TupleSketchUtils.heapify*Sketch` does for `function`.
///
/// # Errors
/// `TUPLE_INVALID_INPUT_SKETCH_BUFFER` for bytes that are no tuple sketch of `V` summaries.
pub fn from_bytes<V: Value>(bytes: &[u8], function: &str) -> Result<Compact<V>, KrishivError> {
    // A sketch of another family (a Theta one, say) is its own error, naming the family.
    if let Some(&id) = bytes.get(2)
        && id != 9
        && let Some(actual) = theta::family_name(id)
    {
        return Err(krishiv_common::error::TupleInvalidInputSketchBufferFamily {
            function: format!("`{function}`"),
            expected_family: "TUPLE".to_owned(),
            actual_family: actual.to_owned(),
        }
        .into());
    }
    CompactTupleSketch::<Summary<V>>::deserialize(bytes).map_err(|_| invalid_buffer(function))
}

/// `tuple_sketch_estimate_*`.
#[must_use]
pub fn estimate<V: Value>(sketch: &Compact<V>) -> f64 {
    sketch.estimate()
}

/// `tuple_sketch_theta_*`.
#[must_use]
pub fn theta_of<V: Value>(sketch: &Compact<V>) -> f64 {
    sketch.theta()
}

/// `tuple_union_*` and `tuple_union_theta_*`: a union of `a` and the second input; `theta` holds a Theta sketch
/// whose hashes enter with a default summary of `mode`.
///
/// # Errors
/// An invalid-buffer error for sketches the library cannot combine (a different seed).
pub fn union<V: Value>(
    a: &Compact<V>,
    b: Either<'_, V>,
    lg_nom: i32,
    mode: Mode,
    function: &str,
) -> Result<Compact<V>, KrishivError> {
    let mut u = new_union::<V>(lg_nom, mode, function)?;
    u.update(a).map_err(|_| invalid_buffer(function))?;
    match b {
        Either::Tuple(t) => u.update(t).map_err(|_| invalid_buffer(function))?,
        Either::Theta(t) => {
            let lifted = lift_theta::<V>(t, mode, function)?;
            u.update(&lifted).map_err(|_| invalid_buffer(function))?;
        }
    }
    Ok(u.to_sketch(false))
}

/// The second operand of a binary set operation.
#[derive(Debug)]
pub enum Either<'a, V: Value> {
    /// A tuple sketch.
    Tuple(&'a Compact<V>),
    /// A Theta sketch (the `_theta_` variants).
    Theta(&'a CompactThetaSketch),
}

/// A Theta sketch's hashes as a tuple sketch whose summaries are `mode`'s default (Java's `union.union(theta,
/// defaultSummary)`), built through the compact image.
fn lift_theta<V: Value>(
    t: &CompactThetaSketch,
    mode: Mode,
    function: &str,
) -> Result<Compact<V>, KrishivError> {
    if t.is_empty() {
        return from_bytes::<V>(&EMPTY_IMAGE, function);
    }
    let n = t.num_retained();
    let estimation = t.is_estimation_mode();
    let pre: u8 = if estimation {
        3
    } else if n == 1 {
        1
    } else {
        2
    };
    let default = Summary::<V> {
        value: V::identity(mode),
        mode,
    };
    let mut out = SketchBytes::with_capacity(8 * usize::from(pre) + n * (8 + V::SIZE + 1));
    out.write_u8(pre);
    out.write_u8(3);
    out.write_u8(9);
    out.write_u8(1);
    out.write_u8(0);
    out.write_u8(0x0A | if t.is_ordered() { 0x10 } else { 0 });
    out.write_u16_le(t.seed_hash());
    if pre > 1 {
        out.write_u32_le(u32::try_from(n).map_err(|_| invalid_buffer(function))?);
        out.write_u32_le(0);
    }
    if estimation {
        out.write_u64_le(t.theta64());
    }
    for e in t.iter() {
        out.write_u64_le(e.hash());
        default.serialize_value(&mut out);
    }
    from_bytes::<V>(&out.into_bytes(), function)
}

/// `tuple_intersection_*` and `tuple_intersection_theta_*`.
///
/// # Errors
/// An invalid-buffer error for sketches the library cannot combine (a different seed).
pub fn intersection<V: Value>(
    a: &Compact<V>,
    b: Either<'_, V>,
    mode: Mode,
    function: &str,
) -> Result<Compact<V>, KrishivError> {
    let mut i = new_intersection::<V>(mode);
    i.update(a).map_err(|_| invalid_buffer(function))?;
    match b {
        Either::Tuple(t) => i.update(t).map_err(|_| invalid_buffer(function))?,
        Either::Theta(t) => {
            let lifted = lift_theta::<V>(t, mode, function)?;
            i.update(&lifted).map_err(|_| invalid_buffer(function))?;
        }
    }
    i.to_sketch(false).ok_or_else(|| invalid_buffer(function))
}

/// `tuple_difference_*`: the entries of `a` not in `b`.
///
/// # Errors
/// An invalid-buffer error for sketches the library cannot combine (a different seed).
pub fn difference<V: Value>(
    a: &Compact<V>,
    b: Either<'_, V>,
    function: &str,
) -> Result<Compact<V>, KrishivError> {
    match b {
        Either::Tuple(t) => TupleANotB::default()
            .compute(a, t, false)
            .map_err(|_| invalid_buffer(function)),
        Either::Theta(t) => {
            let lifted = lift_theta::<V>(t, Mode::Sum, function)?;
            TupleANotB::default()
                .compute(a, &lifted, false)
                .map_err(|_| invalid_buffer(function))
        }
    }
}

/// `tuple_sketch_summary_double`: the retained summaries folded with `mode` (`alwaysone` counts the entries).
#[must_use]
pub fn summary_double<V: Value>(sketch: &Compact<V>, mode: Mode) -> f64 {
    let mut acc = f64::identity(mode.sum_for_count());
    for e in sketch.iter() {
        acc = acc.fold(mode.sum_for_count(), e.summary().value.as_f64());
    }
    if matches!(mode, Mode::AlwaysOne) {
        // Spark adds a 1 per entry.
        return sketch.num_retained() as f64;
    }
    acc
}

/// `tuple_sketch_summary_integer`: as [`summary_double`], accumulated in a `bigint` (Spark's `Long`).
#[must_use]
pub fn summary_integer<V: Value>(sketch: &Compact<V>, mode: Mode) -> i64 {
    let mut acc = match mode {
        Mode::Sum | Mode::AlwaysOne => 0,
        Mode::Min => i64::MAX,
        Mode::Max => i64::MIN,
    };
    for e in sketch.iter() {
        let v = e.summary().value.as_i64();
        acc = match mode {
            Mode::Sum => acc.wrapping_add(v),
            Mode::Min => acc.min(v),
            Mode::Max => acc.max(v),
            Mode::AlwaysOne => acc.wrapping_add(1),
        };
    }
    acc
}

impl<V: Value> Clone for Either<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Value> Copy for Either<'_, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// `tuple_sketch_agg_double(k, v)` of (1,1),(1,2),(2,3) on Spark 4.2.0.
    const SPARK_DOUBLE: &str = "02030901000ACC93020000000000000015F97DCBBD86A105000000000000084000C397FC1281709D1E000000000000084000";
    const SPARK_INTEGER: &str =
        "02030901000ACC93020000000000000015F97DCBBD86A1050300000000C397FC1281709D1E0300000000";

    #[test]
    fn reads_and_rewrites_spark_bytes() {
        let d = from_bytes::<f64>(&unhex(SPARK_DOUBLE), "f").unwrap();
        assert_eq!(d.estimate(), 2.0);
        assert_eq!(summary_double(&d, Mode::Sum), 6.0);
        let i = from_bytes::<i32>(&unhex(SPARK_INTEGER), "f").unwrap();
        assert_eq!(summary_integer(&i, Mode::Sum), 6);
    }

    #[test]
    fn lg_4_results_are_cut_to_sixteen_entries_with_theta_the_next_hash() {
        let build = |n: i64| {
            let mut s = new_sketch::<f64>(4, Mode::Sum, "f").unwrap();
            for v in 0..n {
                update_long(&mut s, v, 1.0);
            }
            compact(&s)
        };
        // 16 or fewer entries: untouched.
        let small = build(10);
        assert_eq!(to_bytes_at(&small, 4), to_bytes(&small));
        // More: the 16 smallest hashes of the retained set, estimation mode.
        for n in [20, 1000] {
            let c = build(n);
            let mut all: Vec<u64> = c
                .iter()
                .map(datasketches::tuple::TupleEntry::hash)
                .collect();
            all.sort_unstable();
            let cut = from_bytes::<f64>(&to_bytes_at(&c, 4), "f").unwrap();
            assert_eq!(cut.num_retained(), 16);
            assert_eq!(cut.theta64(), all[16]);
            let mut kept: Vec<u64> = cut
                .iter()
                .map(datasketches::tuple::TupleEntry::hash)
                .collect();
            kept.sort_unstable();
            assert_eq!(kept, all[..16]);
        }
        // lgNom 5 and up: no cut.
        let c = build(1000);
        assert_eq!(to_bytes_at(&c, 5), to_bytes(&c));
    }

    #[test]
    fn builds_spark_bytes() {
        let mut s = new_sketch::<f64>(12, Mode::Sum, "f").unwrap();
        update_long(&mut s, 1, 1.0);
        update_long(&mut s, 1, 2.0);
        update_long(&mut s, 2, 3.0);
        let mut got = to_bytes(&compact(&s));
        let mut want = unhex(SPARK_DOUBLE);
        assert_eq!(got.len(), want.len());
        got.truncate(16);
        want.truncate(16);
        assert_eq!(got, want);
        let mut s = new_sketch::<i32>(12, Mode::Sum, "f").unwrap();
        update_long(&mut s, 1, 1);
        update_long(&mut s, 1, 2);
        update_long(&mut s, 2, 3);
        assert_eq!(to_bytes(&compact(&s)), unhex(SPARK_INTEGER));
    }

    fn theta_of_key(k: i64) -> CompactThetaSketch {
        let mut t = theta::new_sketch(12, "f").unwrap();
        theta::update_long(&mut t, k);
        theta::compact(&mut t)
    }

    #[test]
    fn theta_variants_match_spark() {
        // tuple_sketch_agg_double(1, 3.0) with theta_sketch_agg(6)
        let mut s = new_sketch::<f64>(12, Mode::Sum, "f").unwrap();
        update_long(&mut s, 1, 3.0);
        let a = compact(&s);
        let t = theta_of_key(6);
        let want = |hex: &str| unhex(hex);
        let u = union(&a, Either::Theta(&t), 12, Mode::Min, "f").unwrap();
        assert_eq!(
            to_bytes(&u),
            want(
                "02030901000ACC93020000000000000015F97DCBBD86A105000000000000084000FE162113FB98BC10000000000000F07F01"
            )
        );
        let u = union(&a, Either::Theta(&t), 12, Mode::Max, "f").unwrap();
        assert_eq!(
            to_bytes(&u),
            want(
                "02030901000ACC93020000000000000015F97DCBBD86A105000000000000084000FE162113FB98BC10000000000000F0FF02"
            )
        );
        let t1 = theta_of_key(1);
        let i = intersection(&a, Either::Theta(&t1), Mode::Sum, "f").unwrap();
        assert_eq!(
            to_bytes(&i),
            want("01030901000ACC9315F97DCBBD86A105000000000000084000")
        );
        let t2 = theta_of_key(2);
        let d = difference(&a, Either::Theta(&t2), "f").unwrap();
        assert_eq!(
            to_bytes(&d),
            want("01030901000ACC9315F97DCBBD86A105000000000000084000")
        );
    }

    #[test]
    fn empty_image_is_javas() {
        let s = new_sketch::<f64>(12, Mode::Sum, "f").unwrap();
        assert_eq!(to_bytes(&compact(&s)), unhex("01030901000ECC93"));
        assert!(
            from_bytes::<f64>(&unhex("01030901000ECC93"), "f")
                .unwrap()
                .is_empty()
        );
    }
}
