// SPDX-License-Identifier: Apache-2.0
//! Spark string functions on Arrow `Utf8`/`LargeUtf8` (and `Binary` where
//! Spark defines the function on binary): `upper`/`lower`, `length`, the
//! `trim` family, `startswith`/`endswith`/`contains`, `replace`,
//! `lpad`/`rpad`, `substring`, `concat`, `like` ([`like()`]) and a literal
//! `split` ([`split_literal`]).
//!
//! `STRING` is valid UTF-8 (validated at the edges), so a "character" is a
//! Unicode scalar value (Spark counts code points) and every slice a kernel
//! takes lies on a character boundary. Operands are columns or literals
//! ([`StrOperand`], [`Operand`]); a NULL operand makes the row NULL, and a
//! NULL row is never evaluated. Work is per valid row and depends on the
//! row's content, so these kernels have no has-nulls axis (they skip NULL
//! rows) and ASCII rows take byte-wise fast paths.

mod case_table;
mod like;
mod split;

use arrow_array::builder::{BooleanBufferBuilder, GenericByteBuilder};
use arrow_array::types::{ByteArrayType, GenericBinaryType, GenericStringType, Int32Type};
use arrow_array::{
    Array, BooleanArray, GenericBinaryArray, GenericByteArray, GenericStringArray, Int32Array,
    OffsetSizeTrait,
};
use arrow_buffer::{ArrowNativeType, NullBuffer};

pub use like::{LikePattern, like};
pub use split::{literal_delimiter, split_literal};

use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::ops::Operand;
use crate::version::SparkVersion;

/// One string operand of a kernel: a column, or a literal (`None` is a NULL
/// literal) broadcast over the other operands.
#[derive(Debug, Clone, Copy)]
pub enum StrOperand<'a, O: OffsetSizeTrait> {
    /// A column.
    Array(&'a GenericStringArray<O>),
    /// A literal.
    Scalar(Option<&'a str>),
}

impl<'a, O: OffsetSizeTrait> StrOperand<'a, O> {
    /// The row's string (the slot's own bytes under a NULL).
    fn at(self, i: usize) -> &'a str {
        match self {
            Self::Array(a) => a.value(i),
            Self::Scalar(s) => s.unwrap_or_default(),
        }
    }
}

/// One binary operand of a kernel: a column, or a literal (`None` is NULL).
#[derive(Debug, Clone, Copy)]
pub enum BinOperand<'a, O: OffsetSizeTrait> {
    /// A column.
    Array(&'a GenericBinaryArray<O>),
    /// A literal.
    Scalar(Option<&'a [u8]>),
}

impl<'a, O: OffsetSizeTrait> BinOperand<'a, O> {
    fn at(self, i: usize) -> &'a [u8] {
        match self {
            Self::Array(a) => a.value(i),
            Self::Scalar(s) => s.unwrap_or_default(),
        }
    }
}

/// What every operand kind offers the row driver.
trait Slot {
    /// The column length; `None` for a literal.
    fn len(&self) -> Option<usize>;
    fn is_valid(&self, i: usize) -> bool;
}

impl<O: OffsetSizeTrait> Slot for StrOperand<'_, O> {
    fn len(&self) -> Option<usize> {
        match self {
            Self::Array(a) => Some(a.len()),
            Self::Scalar(_) => None,
        }
    }
    fn is_valid(&self, i: usize) -> bool {
        match self {
            Self::Array(a) => a.is_valid(i),
            Self::Scalar(s) => s.is_some(),
        }
    }
}

impl<O: OffsetSizeTrait> Slot for BinOperand<'_, O> {
    fn len(&self) -> Option<usize> {
        match self {
            Self::Array(a) => Some(a.len()),
            Self::Scalar(_) => None,
        }
    }
    fn is_valid(&self, i: usize) -> bool {
        match self {
            Self::Array(a) => a.is_valid(i),
            Self::Scalar(s) => s.is_some(),
        }
    }
}

impl Slot for Operand<'_, Int32Type> {
    fn len(&self) -> Option<usize> {
        match self {
            Self::Array(a) => Some(a.len()),
            Self::Scalar(_) => None,
        }
    }
    fn is_valid(&self, i: usize) -> bool {
        match self {
            Self::Array(a) => a.is_valid(i),
            Self::Scalar(s) => s.is_some(),
        }
    }
}

/// The value of an integer operand at row `i` (0 under a NULL).
fn int_at(o: Operand<'_, Int32Type>, i: usize) -> i32 {
    match o {
        Operand::Array(a) => a.value(i),
        Operand::Scalar(v) => v.unwrap_or_default(),
    }
}

/// The common row count of the operands, or the engine error.
fn row_count(slots: &[&dyn Slot]) -> Result<usize, KernelError> {
    let mut len = None;
    for s in slots {
        match (len, s.len()) {
            (Some(a), Some(b)) if a != b => {
                return Err(KernelError::internal(format!(
                    "string operands differ in length: {a} vs {b}"
                )));
            }
            (None, Some(b)) => len = Some(b),
            _ => {}
        }
    }
    len.ok_or_else(|| KernelError::internal("a string kernel needs at least one column operand"))
}

fn all_valid(slots: &[&dyn Slot], i: usize) -> bool {
    slots.iter().all(|s| s.is_valid(i))
}

/// `bytes` of values as an offset of type `O`, or the memory error.
pub(super) fn check_offset<O: OffsetSizeTrait>(bytes: usize) -> Result<usize, KernelError> {
    O::from_usize(bytes)
        .map(|_| bytes)
        .ok_or_else(|| KernelError::out_of_memory(bytes))
}

/// A string or binary output under construction. Offsets are checked, so a
/// result that does not fit the offset type is an error, never a panic.
struct Sink<T: ByteArrayType> {
    builder: GenericByteBuilder<T>,
    bytes: usize,
}

type StrSink<O> = Sink<GenericStringType<O>>;
type BinSink<O> = Sink<GenericBinaryType<O>>;

impl<T: ByteArrayType<Native: AsRef<[u8]>>> Sink<T> {
    fn new(rows: usize, bytes_hint: usize) -> Self {
        Self {
            builder: GenericByteBuilder::with_capacity(rows, bytes_hint),
            bytes: 0,
        }
    }

    fn push(&mut self, s: &T::Native) -> Result<(), KernelError> {
        self.bytes = check_offset::<T::Offset>(self.bytes.saturating_add(s.as_ref().len()))?;
        self.builder.append_value(s);
        Ok(())
    }

    fn push_null(&mut self) {
        self.builder.append_null();
    }

    fn finish(mut self) -> GenericByteArray<T> {
        self.builder.finish()
    }
}

/// Appends `n` bytes of room to `buf`, reporting failure as an error.
fn reserve(buf: &mut Vec<u8>, n: usize) -> Result<(), KernelError> {
    buf.try_reserve(n)
        .map_err(|_| KernelError::out_of_memory(n))
}

// ---- upper, lower, length ----

/// Which case mapping [`change_case`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Case {
    Upper,
    Lower,
}

/// The code points that gained a case mapping in Unicode 17: Spark 4.2's
/// JVM maps them, 4.0 and 4.1 leave them unchanged.
const CASED_IN_UNICODE_17: [(u32, u32); 4] = [
    (0xA7CE, 0xA7CF),
    (0xA7D2, 0xA7D5),
    (0x16EA0, 0x16EB8),
    (0x16EBB, 0x16ED3),
];

fn is_cased_in_unicode_17(c: char) -> bool {
    let c = u32::from(c);
    CASED_IN_UNICODE_17
        .iter()
        .any(|(from, to)| (*from..=*to).contains(&c))
}

/// Whether Spark `version` changes `c` under `case`: the generated table of
/// what Spark 4.2 maps ([`case_table`]), less the Unicode 17 additions
/// before 4.2. It does not depend on the toolchain's Unicode version.
fn spark_maps(c: char, case: Case, version: SparkVersion) -> bool {
    let runs = match case {
        Case::Upper => case_table::UPPER,
        Case::Lower => case_table::LOWER,
    };
    let cp = u32::from(c);
    let at = runs.partition_point(|(first, _, _)| *first <= cp);
    let listed = at
        .checked_sub(1)
        .and_then(|i| runs.get(i))
        .is_some_and(|(first, last, step)| {
            cp <= *last && cp.wrapping_sub(*first).checked_rem(*step) == Some(0)
        });
    listed && (version >= SparkVersion::V4_2 || !is_cased_in_unicode_17(c))
}

/// A Unicode case mapping: std's (the toolchain's Unicode data), or in
/// tests a stand-in for a toolchain with newer data.
trait CaseMapping {
    /// The full mapping of `s` (context rules included).
    fn string(&self, s: &str, case: Case) -> String;
    /// Whether the mapping changes the single character `c`.
    fn changes(&self, c: char, case: Case) -> bool;
}

/// `str::to_*case` / `char::to_*case`.
struct StdMapping;

impl CaseMapping for StdMapping {
    fn string(&self, s: &str, case: Case) -> String {
        match case {
            Case::Upper => s.to_uppercase(),
            Case::Lower => s.to_lowercase(),
        }
    }

    fn changes(&self, c: char, case: Case) -> bool {
        let unchanged =
            |m: &mut dyn Iterator<Item = char>| m.next() == Some(c) && m.next().is_none();
        match case {
            Case::Upper => !unchanged(&mut c.to_uppercase()),
            Case::Lower => !unchanged(&mut c.to_lowercase()),
        }
    }
}

/// The full Unicode mapping of a non-ASCII `s` (`str::to_*case` carries the
/// context rules, final sigma), as Spark `version` maps it.
fn map_case(s: &str, case: Case, version: SparkVersion) -> String {
    map_case_with(&StdMapping, s, case, version)
}

/// [`map_case`] over `mapping`. A character the mapping would change but
/// Spark `version` does not (a later Unicode version than Spark's JVM, or a
/// Unicode 17 letter before Spark 4.2) is left unchanged, so the result does
/// not move when the toolchain's Unicode data does.
fn map_case_with(mapping: &impl CaseMapping, s: &str, case: Case, version: SparkVersion) -> String {
    let held = |c: char| !spark_maps(c, case, version) && mapping.changes(c, case);
    if !s.chars().any(held) {
        return mapping.string(s, case);
    }
    // A held character stays as it is and (uncased for Spark's JVM, as an
    // unassigned or newly cased letter is) ends the context of its
    // neighbours like any other uncased character.
    let (mut out, mut start) = (String::new(), 0);
    for (at, c) in s.char_indices().filter(|(_, c)| held(*c)) {
        out.push_str(&mapping.string(s.get(start..at).unwrap_or_default(), case));
        out.push(c);
        start = at.saturating_add(c.len_utf8());
    }
    out.push_str(&mapping.string(s.get(start..).unwrap_or_default(), case));
    out
}

fn change_case<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
    case: Case,
    ctx: &KernelCtx,
) -> Result<GenericStringArray<O>, KernelError> {
    let mut sink = StrSink::new(a.len(), a.values().len());
    let mut scratch = String::new();
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
            continue;
        }
        let s = a.value(i);
        if s.is_ascii() {
            // Byte-wise, no Unicode tables.
            scratch.clear();
            scratch.push_str(s);
            match case {
                Case::Upper => scratch.make_ascii_uppercase(),
                Case::Lower => scratch.make_ascii_lowercase(),
            }
        } else {
            scratch = map_case(s, case, ctx.spark_version());
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// `upper(str)`: the full Unicode case mapping (`ß` becomes `SS`), with the
/// Unicode data of the context's Spark version.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn upper<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
    ctx: &KernelCtx,
) -> Result<GenericStringArray<O>, KernelError> {
    change_case(a, Case::Upper, ctx)
}

/// `lower(str)`: the full Unicode case mapping, with the Unicode data of the
/// context's Spark version.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn lower<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
    ctx: &KernelCtx,
) -> Result<GenericStringArray<O>, KernelError> {
    change_case(a, Case::Lower, ctx)
}

/// `length(str)` / `char_length`: the number of characters (code points).
#[must_use]
pub fn char_length<O: OffsetSizeTrait>(a: &GenericStringArray<O>) -> Int32Array {
    let lengths = (0..a.len()).map(|i| {
        // Every byte that is not a UTF-8 continuation byte starts a character.
        let n = a.value(i).bytes().filter(|b| (b & 0xC0) != 0x80).count();
        i32::try_from(n).unwrap_or(i32::MAX)
    });
    Int32Array::new(lengths.collect(), a.nulls().cloned())
}

/// `length(binary)` / `octet_length`: the number of bytes.
#[must_use]
pub fn byte_length<T: ByteArrayType>(a: &GenericByteArray<T>) -> Int32Array {
    let lengths = a
        .value_offsets()
        .windows(2)
        .map(|w| {
            let n = w
                .get(1)
                .zip(w.first())
                .map_or(0, |(e, s)| e.as_usize().saturating_sub(s.as_usize()));
            i32::try_from(n).unwrap_or(i32::MAX)
        })
        .collect();
    Int32Array::new(lengths, a.nulls().cloned())
}

// ---- trim ----

/// Which end of the string `trim` strips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrimSide {
    /// `trim` / `btrim`: both ends.
    Both,
    /// `ltrim`: the start.
    Leading,
    /// `rtrim`: the end.
    Trailing,
}

/// `trim`/`ltrim`/`rtrim`/`btrim`: removes the characters of `chars` from
/// the chosen end(s); `None` removes spaces (U+0020 only, not other
/// whitespace). An empty `chars` removes nothing.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn trim<O: OffsetSizeTrait>(
    side: TrimSide,
    input: StrOperand<'_, O>,
    chars: Option<StrOperand<'_, O>>,
) -> Result<GenericStringArray<O>, KernelError> {
    let default_chars = StrOperand::<O>::Scalar(Some(" "));
    let chars = chars.unwrap_or(default_chars);
    let slots: [&dyn Slot; 2] = [&input, &chars];
    let rows = row_count(&slots)?;
    let mut sink = StrSink::new(rows, 0);
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let (s, set) = (input.at(i), chars.at(i));
        let strip = |c: char| set.contains(c);
        let s = match side {
            TrimSide::Both => s.trim_matches(strip),
            TrimSide::Leading => s.trim_start_matches(strip),
            TrimSide::Trailing => s.trim_end_matches(strip),
        };
        sink.push(s)?;
    }
    Ok(sink.finish())
}

// ---- startswith, endswith, contains ----

/// The relation `startswith`, `endswith` and `contains` test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatchKind {
    /// `startswith(l, r)`.
    StartsWith,
    /// `endswith(l, r)`.
    EndsWith,
    /// `contains(l, r)`.
    Contains,
}

impl MatchKind {
    fn holds_str(self, haystack: &str, needle: &str) -> bool {
        match self {
            Self::StartsWith => haystack.starts_with(needle),
            Self::EndsWith => haystack.ends_with(needle),
            Self::Contains => haystack.contains(needle),
        }
    }

    fn holds_bytes(self, haystack: &[u8], needle: &[u8]) -> bool {
        match self {
            Self::StartsWith => haystack.starts_with(needle),
            Self::EndsWith => haystack.ends_with(needle),
            Self::Contains => {
                needle.is_empty() || haystack.windows(needle.len()).any(|w| w == needle)
            }
        }
    }
}

fn bool_result(
    rows: usize,
    slots: &[&dyn Slot],
    mut holds: impl FnMut(usize) -> bool,
) -> BooleanArray {
    let mut values = BooleanBufferBuilder::new(rows);
    let mut valid = BooleanBufferBuilder::new(rows);
    let mut any_null = false;
    for i in 0..rows {
        let ok = all_valid(slots, i);
        any_null |= !ok;
        valid.append(ok);
        // Values under NULL slots are false, never evaluated.
        values.append(ok && holds(i));
    }
    let nulls = any_null.then(|| NullBuffer::new(valid.finish()));
    BooleanArray::new(values.finish(), nulls)
}

/// `startswith`/`endswith`/`contains` on strings. An empty needle matches.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths or two literals.
pub fn match_str<O: OffsetSizeTrait>(
    kind: MatchKind,
    l: StrOperand<'_, O>,
    r: StrOperand<'_, O>,
) -> Result<BooleanArray, KernelError> {
    let slots: [&dyn Slot; 2] = [&l, &r];
    let rows = row_count(&slots)?;
    Ok(bool_result(rows, &slots, |i| {
        kind.holds_str(l.at(i), r.at(i))
    }))
}

/// `startswith`/`endswith`/`contains` on binary.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths or two literals.
pub fn match_binary<O: OffsetSizeTrait>(
    kind: MatchKind,
    l: BinOperand<'_, O>,
    r: BinOperand<'_, O>,
) -> Result<BooleanArray, KernelError> {
    let slots: [&dyn Slot; 2] = [&l, &r];
    let rows = row_count(&slots)?;
    Ok(bool_result(rows, &slots, |i| {
        kind.holds_bytes(l.at(i), r.at(i))
    }))
}

// ---- replace ----

/// `replace(str, search, replacement)`: every non-overlapping occurrence,
/// left to right; an empty `search` replaces nothing.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn replace<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    search: StrOperand<'_, O>,
    replacement: StrOperand<'_, O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&input, &search, &replacement];
    let rows = row_count(&slots)?;
    let mut sink = StrSink::new(rows, 0);
    let mut scratch = String::new();
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let (s, from, to) = (input.at(i), search.at(i), replacement.at(i));
        if from.is_empty() || !s.contains(from) {
            sink.push(s)?;
            continue;
        }
        scratch.clear();
        let mut last = 0;
        for (at, _) in s.match_indices(from) {
            scratch.push_str(s.get(last..at).unwrap_or_default());
            scratch.push_str(to);
            last = at.saturating_add(from.len());
        }
        scratch.push_str(s.get(last..).unwrap_or_default());
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

// ---- lpad, rpad ----

/// Which side `pad` fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadSide {
    /// `lpad`.
    Left,
    /// `rpad`.
    Right,
}

/// Where the first `len` characters of `s` end, in bytes (`len <= 0`: 0).
fn prefix_end(s: &str, len: i32) -> usize {
    let n = usize::try_from(len).unwrap_or(0);
    s.char_indices().nth(n).map_or(s.len(), |(at, _)| at)
}

/// `lpad`/`rpad(str, len, pad)`: `str` cut to `len` characters, or extended
/// to `len` with the characters of `pad` repeated. `pad = None` pads with a
/// space; an empty `pad` pads nothing; `len <= 0` gives the empty string.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit memory or the
/// offset type.
pub fn pad<O: OffsetSizeTrait>(
    side: PadSide,
    input: StrOperand<'_, O>,
    len: Operand<'_, Int32Type>,
    fill: Option<StrOperand<'_, O>>,
) -> Result<GenericStringArray<O>, KernelError> {
    let fill = fill.unwrap_or(StrOperand::Scalar(Some(" ")));
    let slots: [&dyn Slot; 3] = [&input, &len, &fill];
    let rows = row_count(&slots)?;
    let mut sink = StrSink::new(rows, 0);
    let mut scratch = String::new();
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let (s, want, pad) = (input.at(i), int_at(len, i), fill.at(i));
        let have = s.chars().count();
        let missing = usize::try_from(want).unwrap_or(0).saturating_sub(have);
        if missing == 0 || pad.is_empty() {
            sink.push(s.get(..prefix_end(s, want)).unwrap_or_default())?;
            continue;
        }
        scratch.clear();
        // The exact byte length first (a multibyte pad is wider than its
        // character count), so an oversized result is an error before any
        // allocation, and the reserve below is never outgrown.
        let pad_chars = pad.chars().count();
        let reps = missing.checked_div(pad_chars).unwrap_or(0);
        let rest = missing.checked_rem(pad_chars).unwrap_or(0);
        let rest_bytes = pad.char_indices().nth(rest).map_or(pad.len(), |(at, _)| at);
        let need = reps
            .saturating_mul(pad.len())
            .saturating_add(rest_bytes)
            .saturating_add(s.len());
        check_offset::<O>(sink.bytes.saturating_add(need))?;
        scratch
            .try_reserve(need)
            .map_err(|_| KernelError::out_of_memory(need))?;
        if side == PadSide::Right {
            scratch.push_str(s);
        }
        scratch.extend(pad.chars().cycle().take(missing));
        if side == PadSide::Left {
            scratch.push_str(s);
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// `lpad`/`rpad` on binary: bytes instead of characters. `fill = None` pads
/// with zero bytes.
///
/// # Errors
/// As [`pad`].
pub fn pad_binary<O: OffsetSizeTrait>(
    side: PadSide,
    input: BinOperand<'_, O>,
    len: Operand<'_, Int32Type>,
    fill: Option<BinOperand<'_, O>>,
) -> Result<GenericBinaryArray<O>, KernelError> {
    let fill = fill.unwrap_or(BinOperand::Scalar(Some(&[0])));
    let slots: [&dyn Slot; 3] = [&input, &len, &fill];
    let rows = row_count(&slots)?;
    let mut sink = BinSink::new(rows, 0);
    let mut scratch = Vec::new();
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let (s, want, pad) = (input.at(i), int_at(len, i), fill.at(i));
        let want = usize::try_from(want).unwrap_or(0);
        let missing = want.saturating_sub(s.len());
        if missing == 0 || pad.is_empty() {
            sink.push(s.get(..want.min(s.len())).unwrap_or_default())?;
            continue;
        }
        scratch.clear();
        check_offset::<O>(sink.bytes.saturating_add(want))?;
        reserve(&mut scratch, want)?;
        if side == PadSide::Right {
            scratch.extend_from_slice(s);
        }
        scratch.extend(pad.iter().copied().cycle().take(missing));
        if side == PadSide::Left {
            scratch.extend_from_slice(s);
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

// ---- substring ----

/// The half-open range `[start, end)` of units Spark's `substring(str, pos,
/// len)` selects from a value of `n` units: `pos` is 1-based, 0 counts as
/// 1, negative counts from the end.
fn substring_range(n: usize, pos: i32, len: i32) -> Option<(usize, usize)> {
    let n = i64::try_from(n).unwrap_or(i64::MAX);
    let start = match pos {
        1.. => i64::from(pos).saturating_sub(1),
        0 => 0,
        _ => n.saturating_add(i64::from(pos)),
    };
    let end = start
        .saturating_add(i64::from(len))
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX));
    let start = start.max(0);
    if start >= n || start >= end {
        return None;
    }
    let end = end.min(n);
    Some((
        usize::try_from(start).unwrap_or(0),
        usize::try_from(end).unwrap_or(0),
    ))
}

/// Byte offsets of the character range `[start, end)` of `s`.
fn char_range_bytes(s: &str, start: usize, end: usize) -> (usize, usize) {
    let (mut from, mut to) = (s.len(), s.len());
    for (n, (at, _)) in s.char_indices().enumerate() {
        if n == start {
            from = at;
        }
        if n == end {
            to = at;
            break;
        }
    }
    (from, to)
}

/// `substring(str, pos[, len])` in characters. `len = None` selects to the
/// end.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn substring<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    pos: Operand<'_, Int32Type>,
    len: Option<Operand<'_, Int32Type>>,
) -> Result<GenericStringArray<O>, KernelError> {
    let to_end = Operand::<Int32Type>::Scalar(Some(i32::MAX));
    let len = len.unwrap_or(to_end);
    let slots: [&dyn Slot; 3] = [&input, &pos, &len];
    let rows = row_count(&slots)?;
    let mut sink = StrSink::new(rows, 0);
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let s = input.at(i);
        // ASCII: characters are bytes.
        let chars = if s.is_ascii() {
            s.len()
        } else {
            s.chars().count()
        };
        let piece = substring_range(chars, int_at(pos, i), int_at(len, i)).map_or("", |(a, b)| {
            let (from, to) = if s.is_ascii() {
                (a, b)
            } else {
                char_range_bytes(s, a, b)
            };
            s.get(from..to).unwrap_or_default()
        });
        sink.push(piece)?;
    }
    Ok(sink.finish())
}

/// `substring(binary, pos[, len])` in bytes.
///
/// # Errors
/// As [`substring`].
pub fn substring_binary<O: OffsetSizeTrait>(
    input: BinOperand<'_, O>,
    pos: Operand<'_, Int32Type>,
    len: Option<Operand<'_, Int32Type>>,
) -> Result<GenericBinaryArray<O>, KernelError> {
    let to_end = Operand::<Int32Type>::Scalar(Some(i32::MAX));
    let len = len.unwrap_or(to_end);
    let slots: [&dyn Slot; 3] = [&input, &pos, &len];
    let rows = row_count(&slots)?;
    let mut sink = BinSink::new(rows, 0);
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let s = input.at(i);
        let piece = substring_range(s.len(), int_at(pos, i), int_at(len, i))
            .and_then(|(a, b)| s.get(a..b))
            .unwrap_or_default();
        sink.push(piece)?;
    }
    Ok(sink.finish())
}

// ---- concat ----

/// `concat(str, ...)`: NULL when any argument is NULL.
///
/// # Errors
/// `INTERNAL_ERROR` for no column operand or operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn concat<O: OffsetSizeTrait>(
    parts: &[StrOperand<'_, O>],
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: Vec<&dyn Slot> = parts.iter().map(|p| p as &dyn Slot).collect();
    let rows = row_count(&slots)?;
    let mut sink = StrSink::new(rows, 0);
    let mut scratch = String::new();
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        scratch.clear();
        for p in parts {
            scratch.push_str(p.at(i));
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// `concat(binary, ...)`: NULL when any argument is NULL.
///
/// # Errors
/// As [`concat()`].
pub fn concat_binary<O: OffsetSizeTrait>(
    parts: &[BinOperand<'_, O>],
) -> Result<GenericBinaryArray<O>, KernelError> {
    let slots: Vec<&dyn Slot> = parts.iter().map(|p| p as &dyn Slot).collect();
    let rows = row_count(&slots)?;
    let mut sink = BinSink::new(rows, 0);
    let mut scratch = Vec::new();
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        scratch.clear();
        for p in parts {
            scratch.extend_from_slice(p.at(i));
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_that_do_not_fit_are_errors() {
        assert_eq!(check_offset::<i32>(5).unwrap(), 5);
        assert_eq!(check_offset::<i64>(1 << 40).unwrap(), 1 << 40);
        let e = check_offset::<i32>(usize::MAX).unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::UnableToAcquireMemory
        );
        let mut sink = StrSink::<i32>::new(1, 0);
        sink.bytes = usize::try_from(i32::MAX).unwrap();
        assert!(sink.push("x").is_err());
        let mut sink = BinSink::<i32>::new(1, 0);
        sink.bytes = usize::try_from(i32::MAX).unwrap();
        assert!(sink.push(b"x").is_err());
    }

    /// std's mapping plus what a later Unicode version might add: `ɷ`
    /// gains an uppercase (U+A7DD, as in Unicode 18) and U+A7DD a lowercase.
    struct NewerUnicode;

    impl CaseMapping for NewerUnicode {
        fn string(&self, s: &str, case: Case) -> String {
            s.chars()
                .map(|c| match (case, c) {
                    (Case::Upper, '\u{0277}') => '\u{A7DD}'.to_string(),
                    (Case::Lower, '\u{A7DD}') => '\u{0277}'.to_string(),
                    _ => StdMapping.string(&c.to_string(), case),
                })
                .collect()
        }

        fn changes(&self, c: char, case: Case) -> bool {
            matches!(
                (case, c),
                (Case::Upper, '\u{0277}') | (Case::Lower, '\u{A7DD}')
            ) || StdMapping.changes(c, case)
        }
    }

    /// The kernel's output does not follow the toolchain's Unicode data: a
    /// mapping no Spark version has stays unapplied on every version, and the
    /// rest of the string still maps.
    #[test]
    fn a_newer_toolchain_unicode_does_not_change_the_result() {
        for v in [SparkVersion::V4_0, SparkVersion::V4_1, SparkVersion::V4_2] {
            let up = |s| map_case_with(&NewerUnicode, s, Case::Upper, v);
            let low = |s| map_case_with(&NewerUnicode, s, Case::Lower, v);
            assert_eq!(up("\u{0277}é"), "\u{0277}É");
            assert_eq!(low("\u{A7DD}É"), "\u{A7DD}é");
            assert_eq!(up("straße"), "STRASSE");
        }
    }

    /// The table gate itself: what Spark maps, by direction and version.
    #[test]
    fn spark_maps_follows_the_generated_table() {
        let v42 = SparkVersion::V4_2;
        assert!(spark_maps('a', Case::Upper, v42) && !spark_maps('a', Case::Lower, v42));
        assert!(spark_maps('A', Case::Lower, v42) && !spark_maps('A', Case::Upper, v42));
        // A step-2 run (Latin Extended-A) and its gaps.
        assert!(spark_maps('\u{0101}', Case::Upper, v42));
        assert!(!spark_maps('\u{0102}', Case::Upper, v42));
        assert!(!spark_maps('\u{0277}', Case::Upper, v42));
        assert!(!spark_maps('\u{A7DD}', Case::Lower, v42));
        assert!(!spark_maps('日', Case::Upper, v42));
        // Unicode 17 additions: 4.2 maps them, 4.1 does not.
        assert!(spark_maps('\u{A7CE}', Case::Lower, v42));
        assert!(!spark_maps('\u{A7CE}', Case::Lower, SparkVersion::V4_1));
    }

    #[test]
    fn row_counts_and_substring_ranges() {
        let a = arrow_array::StringArray::from(vec!["a", "b"]);
        let b = arrow_array::StringArray::from(vec!["a"]);
        let (x, y) = (StrOperand::<i32>::Array(&a), StrOperand::<i32>::Array(&b));
        assert!(row_count(&[&x, &y]).is_err());
        let lit = StrOperand::<i32>::Scalar(Some("x"));
        assert!(row_count(&[&lit, &lit]).is_err());
        assert_eq!(row_count(&[&lit, &x]).unwrap(), 2);
        // (pos, len) over 9 units, from Spark 4.2.
        let r = |pos, len| substring_range(9, pos, len);
        assert_eq!(r(5, 1), Some((4, 5)));
        assert_eq!(r(-3, i32::MAX), Some((6, 9)));
        assert_eq!(r(0, 3), Some((0, 3)));
        assert_eq!(r(10, 1), None);
        assert_eq!(r(2, -1), None);
        assert_eq!(r(-20, 12), Some((0, 1)));
        assert_eq!(r(i32::MAX, i32::MAX), None);
        assert_eq!(r(1, i32::MIN), None);
    }
}
