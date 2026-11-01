// SPDX-License-Identifier: Apache-2.0
//! The stable encoding of a [`KeyFilter`] (design P2-T163 "Versioning",
//! made mandatory by P2-T171 I3): a runtime filter is a value that can be
//! shipped to a scan in another process.
//!
//! Layout, little-endian throughout:
//! - header: format version (1), [`RowHash`] id, normalization version (1);
//! - key type: a tag and its parameters (decimal precision and scale, the
//!   timestamp's time zone);
//! - flags (`nulls_pass`, `empty`), key count;
//! - range: absent, or two bounds in the key's normalized form;
//! - the statistics `IN` list: absent, or a count and normalized values;
//! - the row form: range only; a small or hashed exact set (a count and
//!   values; the hash table is rebuilt from the values on decode); or a
//!   Bloom filter in the Parquet bitset layout.
//!
//! Normalized values are what the filter compares and hashes (fixed-width
//! integers, unscaled decimals, float bit patterns, raw bytes), so the
//! encoding needs no Arrow IPC and decodes to an identical filter. Decoding
//! takes untrusted bytes: every length is checked against the input, and a
//! filter of another format, hash or normalization version is refused.

use arrow_schema::{DataType, TimeUnit};
use hashbrown::HashTable;

use super::KeyFilter;
use super::bloom::{RuntimeBloom, xxh64};
use super::hash::{RowHash, decimal_hash};
use super::key::{KeyRange, KeyValue, supported_key_type};
use super::member::{Class, Exact, Member, SMALL_SET_MAX, Small};

/// The encoding's format version.
pub const FORMAT_VERSION: u8 = 1;
/// The key normalization version (`normalize_key_bytes`).
pub const NORMALIZATION_VERSION: u8 = 1;

fn put_u32(out: &mut Vec<u8>, n: usize) {
    out.extend_from_slice(&u32::try_from(n).unwrap_or(u32::MAX).to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_u32(out, b.len());
    out.extend_from_slice(b);
}

fn put_value(out: &mut Vec<u8>, v: &KeyValue) {
    match v {
        KeyValue::Fixed(n) => {
            out.push(0);
            out.extend_from_slice(&n.to_le_bytes());
        }
        KeyValue::Bytes(b) => {
            out.push(1);
            put_bytes(out, b);
        }
    }
}

fn put_key_type(out: &mut Vec<u8>, t: &DataType) {
    use DataType as D;
    let tag: u8 = match t {
        D::Boolean => 1,
        D::Int8 => 2,
        D::Int16 => 3,
        D::Int32 => 4,
        D::Int64 => 5,
        D::Float32 => 6,
        D::Float64 => 7,
        D::Decimal128(..) => 8,
        D::Date32 => 9,
        D::Timestamp(..) => 10,
        D::Utf8 => 11,
        D::LargeUtf8 => 12,
        D::Utf8View => 13,
        D::Binary => 14,
        D::LargeBinary => 15,
        D::BinaryView => 16,
        D::Decimal64(..) => 17,
        _ => 0,
    };
    out.push(tag);
    match t {
        D::Decimal64(p, s) | D::Decimal128(p, s) => {
            out.push(*p);
            out.extend_from_slice(&s.to_le_bytes());
        }
        D::Timestamp(_, tz) => match tz {
            Some(z) => {
                out.push(1);
                put_bytes(out, z.as_bytes());
            }
            None => out.push(0),
        },
        _ => {}
    }
}

impl KeyFilter {
    /// The filter's stable encoding: a versioned header (format, row
    /// hash, normalization), the key type, the range, the `IN` list and
    /// the row form.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![FORMAT_VERSION, RowHash::CURRENT.id(), NORMALIZATION_VERSION];
        put_key_type(&mut out, &self.key_type);
        out.push(u8::from(self.nulls_pass) | (u8::from(self.empty) << 1));
        out.extend_from_slice(&self.keys.to_le_bytes());
        match &self.range {
            None => out.push(0),
            Some(KeyRange::Int(a, b)) => {
                out.push(1);
                out.extend_from_slice(&a.to_le_bytes());
                out.extend_from_slice(&b.to_le_bytes());
            }
            Some(KeyRange::Decimal(a, b)) => {
                out.push(2);
                out.extend_from_slice(&a.to_le_bytes());
                out.extend_from_slice(&b.to_le_bytes());
            }
            Some(KeyRange::Float(a, b)) => {
                out.push(3);
                out.extend_from_slice(&a.to_bits().to_le_bytes());
                out.extend_from_slice(&b.to_bits().to_le_bytes());
            }
            Some(KeyRange::Bytes(a, b)) => {
                out.push(4);
                put_bytes(&mut out, a);
                put_bytes(&mut out, b);
            }
        }
        match &self.values {
            None => out.push(0),
            Some(v) => {
                out.push(1);
                put_u32(&mut out, v.len());
                for x in v {
                    put_value(&mut out, x);
                }
            }
        }
        match &self.member {
            Member::Range => out.push(0),
            Member::Small(s) => {
                out.push(1);
                put_small(&mut out, s);
            }
            Member::Exact(e) => {
                out.push(2);
                put_exact(&mut out, e);
            }
            Member::Bloom(b) => {
                out.push(3);
                put_bytes(&mut out, &b.to_bytes());
            }
        }
        out
    }

    /// The filter `bytes` encode (see [`Self::encode`]).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for bytes that are not a filter of this format,
    /// row hash and normalization version.
    pub fn decode(bytes: &[u8]) -> crate::Result<Self> {
        let mut r = Reader(bytes);
        let header = (r.u8()?, r.u8()?, r.u8()?);
        if header != (FORMAT_VERSION, RowHash::CURRENT.id(), NORMALIZATION_VERSION) {
            return Err(bad("an unsupported version"));
        }
        let key_type = r.key_type()?;
        let class = Class::of(&key_type).ok_or_else(|| bad("an unsupported key type"))?;
        let flags = r.u8()?;
        if flags > 3 {
            return Err(bad("unknown flags"));
        }
        let keys = u64::from_le_bytes(r.array()?);
        let range = match (r.u8()?, class) {
            (0, _) => None,
            (1, Class::Int { .. }) => Some(KeyRange::Int(r.i64()?, r.i64()?)),
            (2, Class::Decimal) => Some(KeyRange::Decimal(r.i128()?, r.i128()?)),
            (3, Class::Float { .. }) => Some(KeyRange::Float(
                f64::from_bits(u64::from_le_bytes(r.array()?)),
                f64::from_bits(u64::from_le_bytes(r.array()?)),
            )),
            (4, Class::Bytes) => Some(KeyRange::Bytes(r.bytes()?.into(), r.bytes()?.into())),
            _ => return Err(bad("a range of another type")),
        };
        let values = match r.u8()? {
            0 => None,
            1 => {
                let n = r.count()?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(r.value(class)?);
                }
                Some(v)
            }
            _ => return Err(bad("an unknown value list")),
        };
        let member = match r.u8()? {
            0 => Member::Range,
            1 => Member::Small(r.small(class)?),
            2 => Member::Exact(r.exact(class)?),
            3 => Member::Bloom(
                RuntimeBloom::from_bytes(r.bytes()?).ok_or_else(|| bad("a malformed Bloom"))?,
            ),
            _ => return Err(bad("an unknown row form")),
        };
        if !r.0.is_empty() {
            return Err(bad("trailing bytes"));
        }
        Ok(Self {
            key_type,
            range,
            values,
            member,
            nulls_pass: flags & 1 != 0,
            empty: flags & 2 != 0,
            keys,
        })
    }
}

fn put_small(out: &mut Vec<u8>, s: &Small) {
    match s {
        Small::Int(v) => {
            put_u32(out, v.len());
            v.iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
        }
        Small::Wide(v) => {
            put_u32(out, v.len());
            v.iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
        }
        Small::Bytes(v) => {
            put_u32(out, v.len());
            v.iter().for_each(|x| put_bytes(out, x));
        }
    }
}

fn put_exact(out: &mut Vec<u8>, e: &Exact) {
    match e {
        Exact::Int(t) => {
            put_u32(out, t.len());
            t.iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
        }
        Exact::Wide(t) => {
            put_u32(out, t.len());
            t.iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
        }
        Exact::Bytes(t) => {
            put_u32(out, t.len());
            t.iter().for_each(|x| put_bytes(out, x));
        }
    }
}

fn bad(what: &str) -> krishiv_common::error::KrishivError {
    crate::error::internal(format!("runtime filter encoding: {what}"))
}

/// A cursor over untrusted input.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> crate::Result<&[u8]> {
        if self.0.len() < n {
            return Err(bad("truncated"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> crate::Result<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn u8(&mut self) -> crate::Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn i64(&mut self) -> crate::Result<i64> {
        Ok(i64::from_le_bytes(self.array()?))
    }

    fn i128(&mut self) -> crate::Result<i128> {
        Ok(i128::from_le_bytes(self.array()?))
    }

    /// An element count, at most the bytes left (every element takes at
    /// least one), so a corrupt count cannot force a huge allocation.
    fn count(&mut self) -> crate::Result<usize> {
        let n = usize::try_from(u32::from_le_bytes(self.array()?)).unwrap_or(usize::MAX);
        if n > self.0.len() {
            return Err(bad("a count beyond the input"));
        }
        Ok(n)
    }

    fn bytes(&mut self) -> crate::Result<&[u8]> {
        let n = usize::try_from(u32::from_le_bytes(self.array()?)).unwrap_or(usize::MAX);
        self.take(n)
    }

    fn key_type(&mut self) -> crate::Result<DataType> {
        use DataType as D;
        let t = match self.u8()? {
            1 => D::Boolean,
            2 => D::Int8,
            3 => D::Int16,
            4 => D::Int32,
            5 => D::Int64,
            6 => D::Float32,
            7 => D::Float64,
            8 => {
                let p = self.u8()?;
                let s = i8::from_le_bytes(self.array()?);
                D::Decimal128(p, s)
            }
            9 => D::Date32,
            10 => {
                let tz = match self.u8()? {
                    0 => None,
                    1 => Some(
                        std::str::from_utf8(self.bytes()?)
                            .map_err(|_| bad("a time zone that is not UTF-8"))?
                            .into(),
                    ),
                    _ => return Err(bad("an unknown time zone flag")),
                };
                D::Timestamp(TimeUnit::Microsecond, tz)
            }
            11 => D::Utf8,
            12 => D::LargeUtf8,
            13 => D::Utf8View,
            14 => D::Binary,
            15 => D::LargeBinary,
            16 => D::BinaryView,
            17 => {
                let p = self.u8()?;
                let s = i8::from_le_bytes(self.array()?);
                D::Decimal64(p, s)
            }
            _ => return Err(bad("an unknown key type")),
        };
        if !supported_key_type(&t) {
            return Err(bad("an unsupported key type"));
        }
        Ok(t)
    }

    fn value(&mut self, class: Class) -> crate::Result<KeyValue> {
        match (self.u8()?, class) {
            (0, Class::Int { .. } | Class::Float { .. } | Class::Decimal) => {
                Ok(KeyValue::Fixed(self.i128()?))
            }
            (1, Class::Bytes) => Ok(KeyValue::Bytes(self.bytes()?.into())),
            _ => Err(bad("a value of another type")),
        }
    }

    fn small(&mut self, class: Class) -> crate::Result<Small> {
        let n = self.count()?;
        // `finish` builds a small set only up to its cap, and its per-row
        // cost assumes that: a larger one is not a filter this code wrote.
        if n > SMALL_SET_MAX {
            return Err(bad("a small set beyond its cap"));
        }
        Ok(match class {
            Class::Int { .. } | Class::Float { .. } => {
                Small::Int((0..n).map(|_| self.i64()).collect::<crate::Result<_>>()?)
            }
            Class::Decimal => {
                Small::Wide((0..n).map(|_| self.i128()).collect::<crate::Result<_>>()?)
            }
            Class::Bytes => Small::Bytes(
                (0..n)
                    .map(|_| self.bytes().map(Box::from))
                    .collect::<crate::Result<_>>()?,
            ),
        })
    }

    /// An exact set, re-placed by each member's row hash.
    fn exact(&mut self, class: Class) -> crate::Result<Exact> {
        let n = self.count()?;
        Ok(match class {
            Class::Int { .. } | Class::Float { .. } => {
                let mut t = HashTable::with_capacity(n);
                for _ in 0..n {
                    let v = self.i64()?;
                    let h = class.hash64(v);
                    if t.find(h, |&x| x == v).is_none() {
                        t.insert_unique(h, v, |&x| class.hash64(x));
                    }
                }
                Exact::Int(t)
            }
            Class::Decimal => {
                let mut t = HashTable::with_capacity(n);
                for _ in 0..n {
                    let v = self.i128()?;
                    let h = decimal_hash(v);
                    if t.find(h, |&x| x == v).is_none() {
                        t.insert_unique(h, v, |&x| decimal_hash(x));
                    }
                }
                Exact::Wide(t)
            }
            Class::Bytes => {
                let mut t: HashTable<Box<[u8]>> = HashTable::with_capacity(n);
                for _ in 0..n {
                    let b = self.bytes()?;
                    let h = xxh64(b);
                    if t.find(h, |x| x.as_ref() == b).is_none() {
                        t.insert_unique(h, b.into(), |x| xxh64(x));
                    }
                }
                Exact::Bytes(t)
            }
        })
    }
}
