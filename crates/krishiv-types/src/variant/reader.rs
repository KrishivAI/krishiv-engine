// SPDX-License-Identifier: Apache-2.0
//! Validated borrowed reader for the Parquet Variant binary encoding (v1).
//!
//! Layout (Parquet `VariantEncoding.md`, cross-checked byte-for-byte against
//! Spark 4.2.0's `parse_json` output): a variant is a `(metadata, value)`
//! pair of byte strings. Metadata holds the field-name dictionary; the value
//! holds one nested value whose object field ids index that dictionary.
//!
//! ```text
//! metadata: version:u4/reserved:u2/size-1:u2 dict_size:u{off}
//!           offsets:(dict_size+1)*u{off} keys:u8[total]
//! value primitive: (id:u6 | 0b00) payload...
//! value short string: (len:u6 | 0b01) bytes[len]          (len <= 63)
//! value object: 0b10 ((off-1):u2 | (id-1):u2 | large:u1 | 0)
//!           count:u8/u32 ids:n*u{id} offs:(n+1)*u{off} data
//! value array: 0b11 ((off-1):u2 | large:u1 | 0:u3)
//!           count:u8/u32 offs:(n+1)*u{off} data
//! ```
//!
//! Object field ids are ordered by key bytes (not numerically); the offset
//! table is parallel to the id table, so `offsets[i]` is the start of field
//! `i` and `offsets[n]` the total data length. Array offsets are monotonic
//! and delimit each element. Every accessor re-validates the bounds it
//! touches; deeper levels stay unchecked until accessed.
//!
//! The reader is deliberately stricter than Spark 4.2.0's own validator
//! (which skips most bounds checks): anything Spark's writer emits is
//! accepted, while inconsistent bytes yield `MALFORMED_VARIANT`.

use krishiv_common::error::{KrishivError, MalformedVariant};

use crate::error::Result;

/// Builds the reader's single error: `MALFORMED_VARIANT` (no parameters).
fn malformed() -> KrishivError {
    MalformedVariant {}.into()
}

/// Parquet Variant version this reader accepts.
const VERSION: u8 = 1;
/// Mask of the version nibble in a metadata header byte.
const VERSION_MASK: u8 = 0x0F;
/// Reserved bits of a metadata header byte; must read zero.
const METADATA_RESERVED_MASK: u8 = 0x30;

/// The kind of a Variant value.
///
/// Primitive ids follow the Parquet Variant encoding. Ids 17-19 (time and
/// nanosecond timestamps) are reserved by Spark 4.2.0 but defined by the
/// encoding, so the reader accepts them although the builder here never
/// emits them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VariantKind {
    /// Primitive id 0: SQL `NULL`.
    Null,
    /// Primitive ids 1-2: `true` / `false`.
    Bool,
    /// Primitive id 3: 8-bit signed integer.
    Int8,
    /// Primitive id 4: 16-bit signed integer, little-endian.
    Int16,
    /// Primitive id 5: 32-bit signed integer, little-endian.
    Int32,
    /// Primitive id 6: 64-bit signed integer, little-endian.
    Int64,
    /// Primitive id 7: IEEE 754 binary64, little-endian.
    Double,
    /// Primitive id 14: IEEE 754 binary32, little-endian.
    Float,
    /// Primitive ids 8-10: decimal4/8/16. `precision` is derived from the
    /// stored unscaled value (significant digits, with zero counting as 1),
    /// matching Spark's `BigDecimal.precision`; `scale` is stored on the wire.
    Decimal {
        /// Total significant digits, 1..=38.
        precision: u8,
        /// Digits after the point, as stored.
        scale: u8,
    },
    /// Primitive id 11: days since 1970-01-01, little-endian.
    Date,
    /// Primitive id 12: microseconds since the epoch, a UTC instant.
    Timestamp,
    /// Primitive id 13: microseconds since the epoch, no time zone.
    TimestampNtz,
    /// Primitive id 17: microseconds since midnight.
    TimeNtz,
    /// Primitive id 18: nanoseconds since the epoch, a UTC instant.
    TimestampNanos,
    /// Primitive id 19: nanoseconds since the epoch, no time zone.
    TimestampNtzNanos,
    /// Primitive id 15: arbitrary bytes (`len:u32` + bytes).
    Binary,
    /// Short strings and primitive id 16: UTF-8 bytes.
    String,
    /// Primitive id 20: 16 bytes, big-endian (UUID string order).
    Uuid,
    /// A field map; see the module docs for the layout.
    Object,
    /// An ordered element list.
    Array,
}

/// A validated metadata dictionary: offset width, entry count, the raw
/// offset table and the concatenated key bytes.
#[derive(Debug, Clone, Copy)]
struct MetadataView<'a> {
    /// Offset width in bytes (1-4).
    offset_size: usize,
    /// Number of dictionary entries.
    dict_size: u32,
    /// `(dict_size + 1) * offset_size` bytes of little-endian offsets.
    offsets: &'a [u8],
    /// Concatenated key bytes.
    strings: &'a [u8],
}

impl<'a> MetadataView<'a> {
    /// Key bytes of dictionary entry `id` (`id < dict_size`).
    fn key(&self, id: u32) -> Result<&'a [u8], KrishivError> {
        let index = usize::try_from(id).map_err(|_| malformed())?;
        let start = read_table_u32(self.offsets, index, self.offset_size)?;
        let end = read_table_u32(
            self.offsets,
            index.checked_add(1).ok_or_else(malformed)?,
            self.offset_size,
        )?;
        // Offsets were validated monotonic and in bounds at construction,
        // so `start <= end <= strings.len()` holds here.
        self.strings.get(start..end).ok_or_else(malformed)
    }
}

/// Reads the `i`-th little-endian integer of width `1..=4` from a table.
fn read_table_u32(table: &[u8], i: usize, width: usize) -> Result<usize, KrishivError> {
    let start = i.checked_mul(width).ok_or_else(malformed)?;
    let end = start.checked_add(width).ok_or_else(malformed)?;
    let raw = table.get(start..end).ok_or_else(malformed)?;
    let mut value: u32 = 0;
    for (shift, byte) in raw.iter().enumerate() {
        let shift = u32::try_from(shift).map_err(|_| malformed())?;
        value |= u32::from(*byte) << (shift * 8);
    }
    usize::try_from(value).map_err(|_| malformed())
}

/// Adds a small constant to a buffer position, mapping overflow to
/// `MALFORMED_VARIANT`.
fn at(pos: usize, delta: usize) -> Result<usize, KrishivError> {
    pos.checked_add(delta).ok_or_else(malformed)
}

/// Reads exactly `N` bytes at `at`.
fn read_fixed<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N], KrishivError> {
    let end = at.checked_add(N).ok_or_else(malformed)?;
    bytes
        .get(at..end)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(malformed)
}

/// Parses and validates a metadata dictionary.
///
/// Checks the version nibble (must be 1), the reserved bits (must be 0, so
/// only headers `0x01`/`0x41`/`0x81`/`0xC1` pass), the offset width (1-4),
/// and that the `dict_size + 1` offsets are present, start at 0, are
/// monotonic and end exactly at the key bytes.
fn parse_metadata(metadata: &[u8]) -> Result<MetadataView<'_>, KrishivError> {
    let header = *metadata.first().ok_or_else(malformed)?;
    if header & VERSION_MASK != VERSION || header & METADATA_RESERVED_MASK != 0 {
        return Err(malformed());
    }
    let offset_size = usize::from(header >> 6) + 1;
    let dict_size = {
        let raw = metadata.get(1..1 + offset_size).ok_or_else(malformed)?;
        let mut value: u32 = 0;
        for (shift, byte) in raw.iter().enumerate() {
            let shift = u32::try_from(shift).map_err(|_| malformed())?;
            value |= u32::from(*byte) << (shift * 8);
        }
        value
    };
    let table_len = usize::try_from(dict_size)
        .map_err(|_| malformed())?
        .checked_add(1)
        .and_then(|n| n.checked_mul(offset_size))
        .ok_or_else(malformed)?;
    let table_start = 1 + offset_size;
    let table_end = table_start.checked_add(table_len).ok_or_else(malformed)?;
    let offsets = metadata.get(table_start..table_end).ok_or_else(malformed)?;
    let strings = metadata.get(table_end..).ok_or_else(malformed)?;
    let count = usize::try_from(dict_size)
        .map_err(|_| malformed())?
        .checked_add(1)
        .ok_or_else(malformed)?;
    let mut previous = 0_usize;
    for i in 0..count {
        let current = read_table_u32(offsets, i, offset_size)?;
        if i == 0 && current != 0 {
            return Err(malformed());
        }
        if current < previous || current > strings.len() {
            return Err(malformed());
        }
        previous = current;
    }
    if previous != strings.len() {
        return Err(malformed());
    }
    Ok(MetadataView {
        offset_size,
        dict_size,
        offsets,
        strings,
    })
}

/// A validated object or array header: element count, field-id width
/// (objects only), offset width, and the table/data positions.
#[derive(Debug, Clone, Copy)]
struct ContainerView {
    /// Number of fields or elements.
    count: u32,
    /// Field-id width in bytes (objects; 0 for arrays).
    id_size: usize,
    /// Offset width in bytes (1-4).
    offset_size: usize,
    /// Byte offset of the id table within the value buffer.
    ids: usize,
    /// Byte offset of the offset table within the value buffer.
    offs: usize,
    /// Byte offset of the data region within the value buffer.
    data: usize,
    /// Declared data length (`offsets[n]`).
    total: usize,
}

/// Parses and validates the framing of the container (object or array) at
/// `pos`, bounded by `end` (exclusive).
///
/// Checks the basic type, the reserved bits, the count, and that every
/// offset is present and within the declared total with the data fitting in
/// `end`. Object offsets are *not* required to be monotonic (Spark emits
/// them in input order); array offsets are. Ids are range-checked by the
/// caller, which owns the metadata.
fn parse_container(
    value: &[u8],
    pos: usize,
    end: usize,
    is_object: bool,
) -> Result<ContainerView, KrishivError> {
    let header = *value.get(pos).ok_or_else(malformed)?;
    if header & 0x03 != if is_object { 2 } else { 3 } {
        return Err(malformed());
    }
    let offset_size = usize::from((header >> 2) & 0x03) + 1;
    let (id_size, large, reserved_ok) = if is_object {
        (
            usize::from((header >> 4) & 0x03) + 1,
            header & 0x40 != 0,
            header & 0x80 == 0,
        )
    } else {
        (0, header & 0x10 != 0, header & 0xE0 == 0)
    };
    if !reserved_ok {
        return Err(malformed());
    }
    let count_len = if large { 4 } else { 1 };
    let count = if large {
        u32::from_le_bytes(read_fixed(value, at(pos, 1)?)?)
    } else {
        u32::from(*value.get(at(pos, 1)?).ok_or_else(malformed)?)
    };
    let count_usize = usize::try_from(count).map_err(|_| malformed())?;
    let ids_len = count_usize.checked_mul(id_size).ok_or_else(malformed)?;
    let offs_len = count_usize
        .checked_add(1)
        .and_then(|n| n.checked_mul(offset_size))
        .ok_or_else(malformed)?;
    let ids = pos.checked_add(1 + count_len).ok_or_else(malformed)?;
    let offs = ids.checked_add(ids_len).ok_or_else(malformed)?;
    let data = offs.checked_add(offs_len).ok_or_else(malformed)?;
    if data > end {
        return Err(malformed());
    }
    let offs_bytes = value.get(offs..data).ok_or_else(malformed)?;
    let total = read_table_u32(offs_bytes, count_usize, offset_size)?;
    if data.checked_add(total).ok_or_else(malformed)? > end {
        return Err(malformed());
    }
    let mut previous = 0_usize;
    for i in 0..count_usize {
        let current = read_table_u32(offs_bytes, i, offset_size)?;
        if current > total {
            return Err(malformed());
        }
        if !is_object {
            if current < previous {
                return Err(malformed());
            }
            previous = current;
        }
    }
    Ok(ContainerView {
        count,
        id_size,
        offset_size,
        ids,
        offs,
        data,
        total,
    })
}

/// Reads the `i`-th field id of an object view.
fn read_id_at(value: &[u8], view: &ContainerView, i: usize) -> Result<u32, KrishivError> {
    let start = view
        .ids
        .checked_add(i.checked_mul(view.id_size).ok_or_else(malformed)?)
        .ok_or_else(malformed)?;
    let end = start.checked_add(view.id_size).ok_or_else(malformed)?;
    let raw = value.get(start..end).ok_or_else(malformed)?;
    let mut id: u32 = 0;
    for (shift, byte) in raw.iter().enumerate() {
        let shift = u32::try_from(shift).map_err(|_| malformed())?;
        id |= u32::from(*byte) << (shift * 8);
    }
    Ok(id)
}

/// Reads the `i`-th offset of a container view (`i <= count`).
fn read_off_at(value: &[u8], view: &ContainerView, i: usize) -> Result<usize, KrishivError> {
    let offs_bytes = value.get(view.offs..view.data).ok_or_else(malformed)?;
    read_table_u32(offs_bytes, i, view.offset_size)
}

/// Computes the byte extent of the value at `pos` (bounded by `end`) without
/// validating its contents. Container extents come from their own declared
/// totals, so this never recurses.
fn value_extent(value: &[u8], pos: usize, end: usize) -> Result<usize, KrishivError> {
    let header = *value.get(pos).ok_or_else(malformed)?;
    let len = match header & 0x03 {
        // Primitive.
        0 => match header >> 2 {
            0..=2 => 1,
            3 => 2,
            4 => 3,
            5 | 11 | 14 => 5,
            6 | 7 | 12 | 13 | 17 | 18 | 19 => 9,
            8 => 6,
            9 => 10,
            10 => 18,
            20 => 17,
            15 | 16 => {
                let raw_len = u32::from_le_bytes(read_fixed(value, at(pos, 1)?)?);
                usize::try_from(raw_len)
                    .map_err(|_| malformed())?
                    .checked_add(5)
                    .ok_or_else(malformed)?
            }
            _ => return Err(malformed()),
        },
        // Short string.
        1 => 1 + usize::from(header >> 2),
        // Object / array: extent from the declared total.
        2 => {
            let view = parse_container(value, pos, end, true)?;
            view.data
                .checked_add(view.total)
                .and_then(|e| e.checked_sub(pos))
                .ok_or_else(malformed)?
        }
        _ => {
            let view = parse_container(value, pos, end, false)?;
            view.data
                .checked_add(view.total)
                .and_then(|e| e.checked_sub(pos))
                .ok_or_else(malformed)?
        }
    };
    if pos.checked_add(len).ok_or_else(malformed)? > end {
        return Err(malformed());
    }
    Ok(len)
}

/// Significant digits of an unscaled decimal value (zero counts as 1),
/// matching Spark's `BigDecimal.precision`.
fn decimal_precision(unscaled: i128) -> u8 {
    let mut rest = unscaled.unsigned_abs();
    if rest == 0 {
        return 1;
    }
    let mut digits: u8 = 0;
    while rest > 0 {
        rest /= 10;
        digits += 1;
    }
    digits
}

/// Validates that `value[pos..end]` holds exactly one value and returns its
/// kind. Object field ids are range-checked against `meta`; nested contents
/// stay unchecked until accessed.
fn validate_exact(
    value: &[u8],
    pos: usize,
    end: usize,
    meta: &MetadataView<'_>,
) -> Result<VariantKind, KrishivError> {
    let header = *value.get(pos).ok_or_else(malformed)?;
    let kind = match header & 0x03 {
        0 => match header >> 2 {
            0 => VariantKind::Null,
            1 | 2 => VariantKind::Bool,
            3 => VariantKind::Int8,
            4 => VariantKind::Int16,
            5 => VariantKind::Int32,
            6 => VariantKind::Int64,
            7 => VariantKind::Double,
            8..=10 => {
                let scale = *value.get(at(pos, 1)?).ok_or_else(malformed)?;
                let body = at(pos, 2)?;
                let unscaled = match header >> 2 {
                    8 => i128::from(i32::from_le_bytes(read_fixed(value, body)?)),
                    9 => i128::from(i64::from_le_bytes(read_fixed(value, body)?)),
                    _ => i128::from_le_bytes(read_fixed(value, body)?),
                };
                VariantKind::Decimal {
                    precision: decimal_precision(unscaled),
                    scale,
                }
            }
            11 => VariantKind::Date,
            12 => VariantKind::Timestamp,
            13 => VariantKind::TimestampNtz,
            14 => VariantKind::Float,
            15 => VariantKind::Binary,
            16 => VariantKind::String,
            17 => VariantKind::TimeNtz,
            18 => VariantKind::TimestampNanos,
            19 => VariantKind::TimestampNtzNanos,
            20 => VariantKind::Uuid,
            _ => return Err(malformed()),
        },
        1 => VariantKind::String,
        2 => VariantKind::Object,
        _ => VariantKind::Array,
    };
    let extent = value_extent(value, pos, end)?;
    if pos.checked_add(extent).ok_or_else(malformed)? != end {
        return Err(malformed());
    }
    if kind == VariantKind::Object {
        let view = parse_container(value, pos, end, true)?;
        let count = usize::try_from(view.count).map_err(|_| malformed())?;
        for i in 0..count {
            if read_id_at(value, &view, i)? >= meta.dict_size {
                return Err(malformed());
            }
        }
    }
    Ok(kind)
}

/// A validated borrow of one Variant value plus its shared dictionary.
///
/// `new` checks the metadata (version 1, offset widths 1-4, monotonic
/// dictionary offsets ending exactly at the key bytes) and the root value's
/// framing (every id in dictionary range, every offset inside the declared
/// total, exact consumption). Nested values are only validated when
/// accessed. All bounds use checked arithmetic and fallible slicing, so
/// malformed bytes surface as `MALFORMED_VARIANT`, never a panic.
#[derive(Debug, Clone, Copy)]
pub struct VariantRef<'a> {
    /// The metadata bytes, for the [`Self::metadata`] accessor.
    metadata: &'a [u8],
    /// The parsed dictionary.
    meta: MetadataView<'a>,
    /// Exactly one validated value.
    value: &'a [u8],
    /// The validated kind of [`Self::value`].
    kind: VariantKind,
}

impl<'a> VariantRef<'a> {
    /// Borrows `metadata` and `value` after validating both.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when any header, count, id, offset or length is
    /// missing, out of range, or inconsistent.
    pub fn new(metadata: &'a [u8], value: &'a [u8]) -> Result<Self, KrishivError> {
        let meta = parse_metadata(metadata)?;
        let kind = validate_exact(value, 0, value.len(), &meta)?;
        Ok(Self {
            metadata,
            meta,
            value,
            kind,
        })
    }

    /// The raw metadata bytes this value was validated against.
    #[must_use]
    pub fn metadata(&self) -> &'a [u8] {
        self.metadata
    }

    /// The raw bytes of exactly this value.
    #[must_use]
    pub fn value_bytes(&self) -> &'a [u8] {
        self.value
    }

    /// The kind of this value. Infallible: construction validated the header
    /// and every child accessor re-validates before building a new ref.
    #[must_use]
    pub fn kind(&self) -> VariantKind {
        self.kind
    }

    /// Builds a child ref over `self.value[start..end]` after re-validating
    /// its framing.
    fn child(&self, start: usize, end: usize) -> Result<Self, KrishivError> {
        let slice = self.value.get(start..end).ok_or_else(malformed)?;
        let kind = validate_exact(slice, 0, slice.len(), &self.meta)?;
        Ok(Self {
            metadata: self.metadata,
            meta: self.meta,
            value: slice,
            kind,
        })
    }

    /// Re-parses this value's container header (already validated at
    /// construction; re-checked here so every access stands on its own).
    fn view(&self, is_object: bool) -> Result<ContainerView, KrishivError> {
        let matches = match self.kind {
            VariantKind::Object => is_object,
            VariantKind::Array => !is_object,
            _ => false,
        };
        if !matches {
            return Err(malformed());
        }
        parse_container(self.value, 0, self.value.len(), is_object)
    }

    /// Number of fields (objects) or elements (arrays).
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` for scalar values.
    pub fn len(&self) -> Result<usize, KrishivError> {
        match self.kind {
            VariantKind::Object => {
                Ok(usize::try_from(self.view(true)?.count).map_err(|_| malformed())?)
            }
            VariantKind::Array => {
                Ok(usize::try_from(self.view(false)?.count).map_err(|_| malformed())?)
            }
            _ => Err(malformed()),
        }
    }

    /// Whether this object or array holds no fields or elements.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` for scalar values.
    pub fn is_empty(&self) -> Result<bool, KrishivError> {
        self.len().map(|n| n == 0)
    }

    /// The `i`-th array element in storage order.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not an array, `i` is out of range,
    /// or the element's bytes fail re-validation.
    pub fn get_index(&self, i: usize) -> Result<Self, KrishivError> {
        let view = self.view(false)?;
        let count = usize::try_from(view.count).map_err(|_| malformed())?;
        if i >= count {
            return Err(malformed());
        }
        let start = view
            .data
            .checked_add(read_off_at(self.value, &view, i)?)
            .ok_or_else(malformed)?;
        let next = i.checked_add(1).ok_or_else(malformed)?;
        let end = view
            .data
            .checked_add(read_off_at(self.value, &view, next)?)
            .ok_or_else(malformed)?;
        if start > end {
            return Err(malformed());
        }
        self.child(start, end)
    }

    /// The `i`-th object field value in storage (key-sorted) order.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not an object, `i` is out of range,
    /// or the field's bytes fail re-validation.
    pub fn field_at(&self, i: usize) -> Result<Self, KrishivError> {
        let view = self.view(true)?;
        let count = usize::try_from(view.count).map_err(|_| malformed())?;
        if i >= count {
            return Err(malformed());
        }
        if read_id_at(self.value, &view, i)? >= self.meta.dict_size {
            return Err(malformed());
        }
        let start = view
            .data
            .checked_add(read_off_at(self.value, &view, i)?)
            .ok_or_else(malformed)?;
        let bound = view.data.checked_add(view.total).ok_or_else(malformed)?;
        if start > bound {
            return Err(malformed());
        }
        let extent = value_extent(self.value, start, bound)?;
        let end = start.checked_add(extent).ok_or_else(malformed)?;
        self.child(start, end)
    }

    /// The dictionary id of the `i`-th object field in storage order.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not an object or `i` is out of range.
    pub fn field_id_at(&self, i: usize) -> Result<u32, KrishivError> {
        let view = self.view(true)?;
        let count = usize::try_from(view.count).map_err(|_| malformed())?;
        if i >= count {
            return Err(malformed());
        }
        let id = read_id_at(self.value, &view, i)?;
        if id >= self.meta.dict_size {
            return Err(malformed());
        }
        Ok(id)
    }

    /// The key of the `i`-th object field in storage order.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not an object, `i` is out of range,
    /// or the key bytes are not valid UTF-8.
    pub fn field_key_at(&self, i: usize) -> Result<&'a str, KrishivError> {
        let id = self.field_id_at(i)?;
        core::str::from_utf8(self.meta.key(id)?).map_err(|_| malformed())
    }

    /// The field stored under `key`, via binary search over the field ids
    /// (which are ordered by key bytes). Returns `None` when this is not an
    /// object, the key is absent, or the field's bytes fail re-validation.
    #[must_use]
    pub fn get_field(&self, key: &str) -> Option<Self> {
        if self.kind != VariantKind::Object {
            return None;
        }
        let view = self.view(true).ok()?;
        let count = usize::try_from(view.count).ok()?;
        let bound = view.data.checked_add(view.total)?;
        let mut lo = 0_usize;
        let mut hi = count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let id = read_id_at(self.value, &view, mid).ok()?;
            if id >= self.meta.dict_size {
                return None;
            }
            let stored = self.meta.key(id).ok()?;
            match stored.cmp(key.as_bytes()) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => {
                    let start = view
                        .data
                        .checked_add(read_off_at(self.value, &view, mid).ok()?)?;
                    if start > bound {
                        return None;
                    }
                    let extent = value_extent(self.value, start, bound).ok()?;
                    let end = start.checked_add(extent)?;
                    return self.child(start, end).ok();
                }
            }
        }
        None
    }

    /// Reads fixed-width payload bytes at `at`, mapping absence to
    /// `MALFORMED_VARIANT` (unreachable for validated values of the matching
    /// kind, but defensive since kinds and widths travel together).
    fn payload<const N: usize>(&self, at: usize) -> Result<[u8; N], KrishivError> {
        read_fixed(self.value, at)
    }

    /// The boolean value.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not a boolean.
    pub fn as_bool(&self) -> Result<bool, KrishivError> {
        if self.kind != VariantKind::Bool {
            return Err(malformed());
        }
        Ok(self.payload::<1>(0)? == [0x04])
    }

    /// The integer value, widened from int8/int16/int32/int64.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not an integer.
    pub fn as_i64(&self) -> Result<i64, KrishivError> {
        match self.kind {
            VariantKind::Int8 => Ok(i64::from(i8::from_le_bytes(self.payload(1)?))),
            VariantKind::Int16 => Ok(i64::from(i16::from_le_bytes(self.payload(1)?))),
            VariantKind::Int32 => Ok(i64::from(i32::from_le_bytes(self.payload(1)?))),
            VariantKind::Int64 => Ok(i64::from_le_bytes(self.payload(1)?)),
            _ => Err(malformed()),
        }
    }

    /// The binary32 value.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not a float.
    pub fn as_f32(&self) -> Result<f32, KrishivError> {
        if self.kind != VariantKind::Float {
            return Err(malformed());
        }
        Ok(f32::from_le_bytes(self.payload(1)?))
    }

    /// The binary64 value.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not a double.
    pub fn as_f64(&self) -> Result<f64, KrishivError> {
        if self.kind != VariantKind::Double {
            return Err(malformed());
        }
        Ok(f64::from_le_bytes(self.payload(1)?))
    }

    /// The `(unscaled, scale)` pair of a decimal value.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not a decimal.
    pub fn as_decimal(&self) -> Result<(i128, u8), KrishivError> {
        let scale: u8 = match self.kind {
            VariantKind::Decimal { scale, .. } => scale,
            _ => return Err(malformed()),
        };
        let unscaled = match self.value.first().ok_or_else(malformed)? >> 2 {
            8 => i128::from(i32::from_le_bytes(self.payload(2)?)),
            9 => i128::from(i64::from_le_bytes(self.payload(2)?)),
            10 => i128::from_le_bytes(self.payload(2)?),
            _ => return Err(malformed()),
        };
        Ok((unscaled, scale))
    }

    /// The string value (short or long).
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not a string or the bytes are not
    /// valid UTF-8.
    pub fn as_str(&self) -> Result<&'a str, KrishivError> {
        if self.kind != VariantKind::String {
            return Err(malformed());
        }
        let header = *self.value.first().ok_or_else(malformed)?;
        let bytes = if header & 0x03 == 1 {
            let len = usize::from(header >> 2);
            self.value.get(1..1 + len).ok_or_else(malformed)?
        } else {
            let len =
                usize::try_from(u32::from_le_bytes(self.payload(1)?)).map_err(|_| malformed())?;
            let end = 5_usize.checked_add(len).ok_or_else(malformed)?;
            self.value.get(5..end).ok_or_else(malformed)?
        };
        core::str::from_utf8(bytes).map_err(|_| malformed())
    }

    /// The binary value's bytes.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not a binary value.
    pub fn as_bytes(&self) -> Result<&'a [u8], KrishivError> {
        if self.kind != VariantKind::Binary {
            return Err(malformed());
        }
        let len = usize::try_from(u32::from_le_bytes(self.payload(1)?)).map_err(|_| malformed())?;
        let end = 5_usize.checked_add(len).ok_or_else(malformed)?;
        self.value.get(5..end).ok_or_else(malformed)
    }

    /// The UUID value's 16 bytes, big-endian.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not a UUID.
    pub fn as_uuid(&self) -> Result<[u8; 16], KrishivError> {
        if self.kind != VariantKind::Uuid {
            return Err(malformed());
        }
        self.payload(1)
    }

    /// Days since 1970-01-01.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not a date.
    pub fn as_date(&self) -> Result<i32, KrishivError> {
        if self.kind != VariantKind::Date {
            return Err(malformed());
        }
        Ok(i32::from_le_bytes(self.payload(1)?))
    }

    /// Microseconds since the epoch (timestamp or timestamp_ntz).
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` for other kinds.
    pub fn as_timestamp_micros(&self) -> Result<i64, KrishivError> {
        match self.kind {
            VariantKind::Timestamp | VariantKind::TimestampNtz => {
                Ok(i64::from_le_bytes(self.payload(1)?))
            }
            _ => Err(malformed()),
        }
    }

    /// Nanoseconds since the epoch (timestamp_nanos or timestamp_ntz_nanos).
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` for other kinds.
    pub fn as_timestamp_nanos(&self) -> Result<i64, KrishivError> {
        match self.kind {
            VariantKind::TimestampNanos | VariantKind::TimestampNtzNanos => {
                Ok(i64::from_le_bytes(self.payload(1)?))
            }
            _ => Err(malformed()),
        }
    }

    /// Microseconds since midnight.
    ///
    /// # Errors
    ///
    /// `MALFORMED_VARIANT` when this is not a time.
    pub fn as_time_micros(&self) -> Result<i64, KrishivError> {
        if self.kind != VariantKind::TimeNtz {
            return Err(malformed());
        }
        Ok(i64::from_le_bytes(self.payload(1)?))
    }
}
