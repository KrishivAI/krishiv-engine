// SPDX-License-Identifier: Apache-2.0
//! [`VariantWriter`]: builds a Variant value from typed parts, as Spark
//! 4.2.0's `VariantBuilder` does for `CAST(x AS VARIANT)`,
//! `to_variant_object` and `variant_get(..., 'variant')`.
//!
//! The byte choices follow `VariantBuilder.java` at v4.2.0: integers take
//! the smallest of int8/16/32/64; a decimal takes decimal4 when both its
//! precision and scale are at most 9, decimal8 at most 18, else decimal16;
//! strings up to 63 bytes are short strings; the metadata dictionary keeps
//! keys in first-added order (not flagged sorted), offset width from the
//! larger of the key bytes and the key count. Object fields are sorted by
//! key in Java `String.compareTo` order (UTF-16 code units); a repeated key
//! is `VARIANT_DUPLICATE_KEY` (Spark builds these values with
//! `allowDuplicateKeys = false`).
//!
//! [`VariantWriter::append_variant`] copies a value from another
//! dictionary, re-keying every object field into this writer's dictionary
//! and copying scalars byte for byte (`appendVariantImpl`).

use std::cmp::Ordering;
use std::collections::HashMap;

use krishiv_common::error::{KrishivError, VariantDuplicateKey, VariantSizeLimit};

use super::VARIANT_SIZE_LIMIT;
use super::reader::{VariantKind, VariantRef};

/// Basic type of a primitive value (low two bits of a value header).
const PRIMITIVE: u8 = 0;
/// Basic type of a short string.
const SHORT_STR: u8 = 1;
/// Basic type of an object.
const OBJECT: u8 = 2;
/// Basic type of an array.
const ARRAY: u8 = 3;
/// The longest string stored as a short string.
const MAX_SHORT_STR: usize = 63;
/// The most digits a decimal4 holds.
const DECIMAL4_DIGITS: u8 = 9;
/// The most digits a decimal8 holds.
const DECIMAL8_DIGITS: u8 = 18;

/// The header byte of primitive id `id`.
const fn primitive(id: u8) -> u8 {
    (id << 2) | PRIMITIVE
}

/// Bytes needed to store `v` unsigned (1-4).
fn int_size(v: usize) -> usize {
    match v {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0xFF_FFFF => 3,
        _ => 4,
    }
}

/// Writes the low `size` bytes of `v`, little-endian.
fn put(out: &mut Vec<u8>, v: usize, size: usize) {
    let bytes = (v as u64).to_le_bytes();
    out.extend_from_slice(&bytes[..size]);
}

fn size_limit() -> KrishivError {
    VariantSizeLimit {
        size_limit: "128.0 MiB".to_owned(),
        function_name: "`variant`".to_owned(),
    }
    .into()
}

/// Java's `String.compareTo`: UTF-16 code-unit order.
fn java_order(a: &str, b: &str) -> Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// One field of an object being written: its key id and where its value
/// starts, relative to the object's start.
#[derive(Debug, Clone, Copy)]
pub struct FieldEntry {
    id: u32,
    offset: usize,
}

/// A Variant value under construction plus its metadata dictionary.
#[derive(Debug, Default)]
pub struct VariantWriter {
    keys: Vec<String>,
    ids: HashMap<String, u32>,
    buf: Vec<u8>,
}

impl VariantWriter {
    /// An empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The current write position, the `start` of an array or object.
    #[must_use]
    pub fn position(&self) -> usize {
        self.buf.len()
    }

    /// Appends a Variant null.
    pub fn null(&mut self) {
        self.buf.push(primitive(0));
    }

    /// Appends a boolean.
    pub fn boolean(&mut self, b: bool) {
        self.buf.push(primitive(if b { 1 } else { 2 }));
    }

    /// Appends an integer in the smallest width that holds it.
    pub fn long(&mut self, v: i64) {
        if let Ok(b) = i8::try_from(v) {
            self.buf.push(primitive(3));
            self.buf.extend_from_slice(&b.to_le_bytes());
        } else if let Ok(s) = i16::try_from(v) {
            self.buf.push(primitive(4));
            self.buf.extend_from_slice(&s.to_le_bytes());
        } else if let Ok(i) = i32::try_from(v) {
            self.buf.push(primitive(5));
            self.buf.extend_from_slice(&i.to_le_bytes());
        } else {
            self.buf.push(primitive(6));
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// Appends a double.
    pub fn double(&mut self, v: f64) {
        self.buf.push(primitive(7));
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    /// Appends a float.
    pub fn float(&mut self, v: f32) {
        self.buf.push(primitive(14));
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    /// Appends `unscaled × 10^-scale`, whose precision (significant digits
    /// of `unscaled`, 1 for zero) and scale are at most 38.
    pub fn decimal(&mut self, unscaled: i128, scale: u8) {
        let precision = digits(unscaled);
        let widest = precision.max(scale);
        if widest <= DECIMAL4_DIGITS {
            self.buf.push(primitive(8));
            self.buf.push(scale);
            // Nine digits fit an i32.
            let v = i32::try_from(unscaled).unwrap_or_default();
            self.buf.extend_from_slice(&v.to_le_bytes());
        } else if widest <= DECIMAL8_DIGITS {
            self.buf.push(primitive(9));
            self.buf.push(scale);
            let v = i64::try_from(unscaled).unwrap_or_default();
            self.buf.extend_from_slice(&v.to_le_bytes());
        } else {
            self.buf.push(primitive(10));
            self.buf.push(scale);
            self.buf.extend_from_slice(&unscaled.to_le_bytes());
        }
    }

    /// Appends a date (days since 1970-01-01).
    pub fn date(&mut self, days: i32) {
        self.buf.push(primitive(11));
        self.buf.extend_from_slice(&days.to_le_bytes());
    }

    /// Appends a timestamp (UTC microseconds).
    pub fn timestamp(&mut self, micros: i64) {
        self.buf.push(primitive(12));
        self.buf.extend_from_slice(&micros.to_le_bytes());
    }

    /// Appends a timestamp without time zone (local microseconds).
    pub fn timestamp_ntz(&mut self, micros: i64) {
        self.buf.push(primitive(13));
        self.buf.extend_from_slice(&micros.to_le_bytes());
    }

    /// Appends binary bytes.
    ///
    /// # Errors
    /// `VARIANT_SIZE_LIMIT` for more than 4 GiB.
    pub fn binary(&mut self, bytes: &[u8]) -> Result<(), KrishivError> {
        let len = u32::try_from(bytes.len()).map_err(|_| size_limit())?;
        self.buf.push(primitive(15));
        self.buf.extend_from_slice(&len.to_le_bytes());
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    /// Appends a string.
    ///
    /// # Errors
    /// `VARIANT_SIZE_LIMIT` for more than 4 GiB.
    pub fn string(&mut self, s: &str) -> Result<(), KrishivError> {
        let bytes = s.as_bytes();
        if bytes.len() <= MAX_SHORT_STR {
            // The length fits the six header bits.
            #[expect(clippy::cast_possible_truncation, reason = "at most 63")]
            self.buf.push(((bytes.len() as u8) << 2) | SHORT_STR);
        } else {
            let len = u32::try_from(bytes.len()).map_err(|_| size_limit())?;
            self.buf.push(primitive(16));
            self.buf.extend_from_slice(&len.to_le_bytes());
        }
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    /// The id of `key` in the dictionary, added at the end when new.
    ///
    /// # Errors
    /// `VARIANT_SIZE_LIMIT` past `u32::MAX` keys.
    pub fn add_key(&mut self, key: &str) -> Result<u32, KrishivError> {
        if let Some(id) = self.ids.get(key) {
            return Ok(*id);
        }
        let id = u32::try_from(self.keys.len()).map_err(|_| size_limit())?;
        self.keys.push(key.to_owned());
        self.ids.insert(key.to_owned(), id);
        Ok(id)
    }

    /// A field entry for `key` whose value is about to be appended to the
    /// object started at `start`.
    ///
    /// # Errors
    /// As [`Self::add_key`].
    pub fn field(&mut self, start: usize, key: &str) -> Result<FieldEntry, KrishivError> {
        let id = self.add_key(key)?;
        Ok(FieldEntry {
            id,
            offset: self.buf.len() - start,
        })
    }

    /// Turns the values written since `start` into an array whose elements
    /// begin at `offsets` (relative to `start`).
    ///
    /// # Errors
    /// `VARIANT_SIZE_LIMIT` when the array outgrows the format.
    pub fn finish_array(&mut self, start: usize, offsets: &[usize]) -> Result<(), KrishivError> {
        let data = self.buf.len() - start;
        let size = offsets.len();
        let large = size > 0xFF;
        let offset_size = int_size(data);
        let mut header = Vec::with_capacity(1 + 4 + (size + 1) * offset_size);
        #[expect(clippy::cast_possible_truncation, reason = "offset_size is 1-4")]
        header.push(((u8::from(large) << 2) | (offset_size as u8 - 1)) << 2 | ARRAY);
        put(&mut header, size, if large { 4 } else { 1 });
        for o in offsets {
            put(&mut header, *o, offset_size);
        }
        put(&mut header, data, offset_size);
        self.splice(start, &header)
    }

    /// Turns the values written since `start` into an object of `fields`.
    ///
    /// # Errors
    /// `VARIANT_DUPLICATE_KEY` for a repeated key; `VARIANT_SIZE_LIMIT` when
    /// the object outgrows the format.
    pub fn finish_object(
        &mut self,
        start: usize,
        mut fields: Vec<FieldEntry>,
    ) -> Result<(), KrishivError> {
        let keys = &self.keys;
        let key = |id: u32| keys.get(id as usize).map_or("", String::as_str);
        fields.sort_by(|a, b| java_order(key(a.id), key(b.id)));
        if let Some(w) = fields.windows(2).find(|w| w[0].id == w[1].id) {
            return Err(VariantDuplicateKey {
                key: key(w[0].id).to_owned(),
            }
            .into());
        }
        let data = self.buf.len() - start;
        let size = fields.len();
        let large = size > 0xFF;
        let max_id = fields.iter().map(|f| f.id as usize).max().unwrap_or(0);
        let id_size = int_size(max_id);
        let offset_size = int_size(data);
        let mut header = Vec::with_capacity(1 + 4 + size * id_size + (size + 1) * offset_size);
        #[expect(clippy::cast_possible_truncation, reason = "sizes are 1-4")]
        header.push(
            ((u8::from(large) << 4) | ((id_size as u8 - 1) << 2) | (offset_size as u8 - 1)) << 2
                | OBJECT,
        );
        put(&mut header, size, if large { 4 } else { 1 });
        for f in &fields {
            put(&mut header, f.id as usize, id_size);
        }
        for f in &fields {
            put(&mut header, f.offset, offset_size);
        }
        put(&mut header, data, offset_size);
        self.splice(start, &header)
    }

    /// Inserts `header` at `start`, before the values written since.
    fn splice(&mut self, start: usize, header: &[u8]) -> Result<(), KrishivError> {
        if self.buf.len() + header.len() > VARIANT_SIZE_LIMIT {
            return Err(size_limit());
        }
        self.buf.splice(start..start, header.iter().copied());
        Ok(())
    }

    /// Appends `v` (from any dictionary), re-keying its objects into this
    /// writer's dictionary.
    ///
    /// # Errors
    /// `MALFORMED_VARIANT` for bytes that fail validation on access, and
    /// the errors of [`Self::finish_object`].
    pub fn append_variant(&mut self, v: &VariantRef<'_>) -> Result<(), KrishivError> {
        match v.kind() {
            VariantKind::Object => {
                let start = self.position();
                let n = v.len()?;
                let mut fields = Vec::with_capacity(n);
                for i in 0..n {
                    fields.push(self.field(start, v.field_key_at(i)?)?);
                    self.append_variant(&v.field_at(i)?)?;
                }
                self.finish_object(start, fields)
            }
            VariantKind::Array => {
                let start = self.position();
                let n = v.len()?;
                let mut offsets = Vec::with_capacity(n);
                for i in 0..n {
                    offsets.push(self.position() - start);
                    self.append_variant(&v.get_index(i)?)?;
                }
                self.finish_array(start, &offsets)
            }
            _ => {
                self.buf.extend_from_slice(v.value_bytes());
                Ok(())
            }
        }
    }

    /// The finished `(metadata, value)` pair.
    ///
    /// # Errors
    /// `VARIANT_SIZE_LIMIT` when either part exceeds 128 MiB.
    pub fn finish(self) -> Result<(Vec<u8>, Vec<u8>), KrishivError> {
        if self.buf.len() > VARIANT_SIZE_LIMIT {
            return Err(size_limit());
        }
        let strings: usize = self.keys.iter().map(String::len).sum();
        let n = self.keys.len();
        let offset_size = int_size(strings.max(n));
        let mut metadata = Vec::with_capacity(1 + (n + 2) * offset_size + strings);
        #[expect(clippy::cast_possible_truncation, reason = "offset_size is 1-4")]
        metadata.push(1 | ((offset_size as u8 - 1) << 6));
        put(&mut metadata, n, offset_size);
        let mut at = 0;
        for k in &self.keys {
            put(&mut metadata, at, offset_size);
            at += k.len();
        }
        put(&mut metadata, at, offset_size);
        for k in &self.keys {
            metadata.extend_from_slice(k.as_bytes());
        }
        if metadata.len() > VARIANT_SIZE_LIMIT {
            return Err(size_limit());
        }
        Ok((metadata, self.buf))
    }
}

/// Significant decimal digits of `v` (1 for zero), as `BigDecimal.precision`.
#[must_use]
pub fn digits(v: i128) -> u8 {
    let mut n = v.unsigned_abs();
    let mut d = 1;
    while n >= 10 {
        n /= 10;
        d += 1;
    }
    d
}

/// `unscaled × 10^-scale` without trailing fractional zeros, the scale never
/// below 0: Spark's `Decimal(getDecimal(...))` (`stripTrailingZeros`, then a
/// negative scale reset to 0).
#[must_use]
pub fn strip_trailing_zeros(mut unscaled: i128, mut scale: u8) -> (i128, u8) {
    while scale > 0 && unscaled % 10 == 0 {
        unscaled /= 10;
        scale -= 1;
    }
    if unscaled == 0 {
        scale = 0;
    }
    (unscaled, scale)
}
