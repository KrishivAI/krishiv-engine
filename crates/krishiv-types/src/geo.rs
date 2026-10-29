// SPDX-License-Identifier: Apache-2.0
//! Well-known binary (WKB) for `geometry` and `geography` values: the
//! validation Spark's `ST_GeomFromWKB` / `ST_GeogFromWKB` apply, and the
//! re-encoding `ST_AsBinary` does.
//!
//! This is a library only for now: no SQL function calls it yet, so
//! `ST_GeomFromWKB` and `ST_GeogFromWKB` do not raise `WKB_PARSE_ERROR`
//! from SQL (the `st-functions` golden statements stay in the `time-geo-types`
//! known-failure category, P2-T181, until the functions are registered).
//!
//! The input is ISO WKB: the seven OGC geometry types (point to geometry
//! collection) in XY, XYZ (type + 1000), XYM (+ 2000) or XYZM (+ 3000),
//! either byte order, which may change from one nested geometry to the
//! next. Extended WKB flags (SRID, Z or M bits) are not accepted. The
//! rules follow Spark 4.2:
//!
//! - every nested geometry has its collection's dimensions, and the
//!   elements of a `Multi*` collection have its element type;
//! - ordinates are never infinite, and NaN only in points (an empty point
//!   is NaN, NaN);
//! - a line string has 0 or at least 2 points; a ring at least 4, and its
//!   last point repeats its first (X and Y);
//! - a count that is negative, or larger than the bytes left allow, is
//!   refused before any element is read;
//! - for a geography, every point has `|x| <= 180` and `|y| <= 90`;
//! - bytes after the geometry are ignored, and dropped from the output.
//!
//! Errors are `WKB_PARSE_ERROR` with Spark's message and byte position.
//! Nesting is walked with an explicit stack, so a deeply nested input
//! cannot overflow the call stack (Spark fails on such inputs, with no
//! error class); the output is never larger than the input.

use crate::error::{self, Result};

/// The byte order of encoded WKB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    /// Little-endian (`NDR`), the storage form and `ST_AsBinary`'s default.
    Ndr,
    /// Big-endian (`XDR`).
    Xdr,
}

/// Which spatial type a value belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialKind {
    /// `geometry`: Cartesian coordinates, any range.
    Geometry,
    /// `geography`: longitude and latitude in degrees.
    Geography,
}

const POINT: u32 = 1;
const LINE_STRING: u32 = 2;
const POLYGON: u32 = 3;
const MULTI_POINT: u32 = 4;
const MULTI_LINE_STRING: u32 = 5;
const MULTI_POLYGON: u32 = 6;
const COLLECTION: u32 = 7;

/// The bytes of a geometry header: byte order and type.
const HEADER: usize = 5;

/// Validates `wkb` as a value of `kind` and re-encodes it in `order`.
///
/// ```
/// use krishiv_types::geo::{rewrite, ByteOrder, SpatialKind};
///
/// // POINT (1 2), big-endian in, little-endian out.
/// let xdr = [0, 0, 0, 0, 1, 0x3F, 0xF0, 0, 0, 0, 0, 0, 0, 0x40, 0, 0, 0, 0, 0, 0, 0];
/// let ndr = rewrite(&xdr, SpatialKind::Geometry, ByteOrder::Ndr).unwrap();
/// assert_eq!(ndr[..5], [1, 1, 0, 0, 0]);
/// assert!(rewrite(&[1], SpatialKind::Geometry, ByteOrder::Ndr).is_err());
/// ```
///
/// # Errors
/// `WKB_PARSE_ERROR` with Spark's message and position (see the module
/// docs for the rules).
pub fn rewrite(wkb: &[u8], kind: SpatialKind, order: ByteOrder) -> Result<Vec<u8>> {
    if wkb.is_empty() {
        return Err(error::wkb("WKB data is empty or null", 0));
    }
    let mut r = Reader { buf: wkb, pos: 0 };
    let mut w = Writer {
        out: Vec::with_capacity(wkb.len()),
        order,
    };
    let geographic = kind == SpatialKind::Geography;
    let mut stack: Vec<Collection> = Vec::new();
    loop {
        let start = r.pos;
        if wkb.len() - start < HEADER {
            return Err(error::wkb(END, start));
        }
        let big = match r.buf.get(start) {
            Some(0) => true,
            Some(1) => false,
            other => {
                let b = other.copied().unwrap_or_default();
                return Err(error::wkb(format!("Invalid byte order {b}"), start));
            }
        };
        r.pos += 1;
        let code = r.u32(big)?;
        let parent = stack.last().map(|c| c.dims);
        let (base, dims) = split_type(code)
            .filter(|(_, d)| parent.is_none_or(|p| p == *d))
            .ok_or_else(|| {
                let signed = i32::from_ne_bytes(code.to_ne_bytes());
                error::wkb(format!("Invalid or unsupported type {signed}"), start + 1)
            })?;
        w.header(code);
        let mut finished = match base {
            POINT => {
                r.point(big, dims, true, geographic, &mut w)?;
                Some(base)
            }
            LINE_STRING => {
                r.line_string(big, dims, geographic, &mut w)?;
                Some(base)
            }
            POLYGON => {
                r.polygon(big, dims, geographic, &mut w)?;
                Some(base)
            }
            _ => {
                let what = match base {
                    MULTI_POINT => "MultiPoint points",
                    MULTI_LINE_STRING => "MultiLineString line strings",
                    MULTI_POLYGON => "MultiPolygon polygons",
                    _ => "GeometryCollection geometries",
                };
                let count = r.count(big, what, HEADER)?;
                w.u32(count);
                if count == 0 {
                    Some(base)
                } else {
                    stack.push(Collection {
                        base,
                        dims,
                        left: count,
                    });
                    None
                }
            }
        };
        // Close every collection the finished geometry completes.
        while let Some(child) = finished {
            let Some(top) = stack.last_mut() else {
                return Ok(w.out);
            };
            let expected = match top.base {
                MULTI_POINT => Some((POINT, "Expected Point in MultiPoint")),
                MULTI_LINE_STRING => Some((LINE_STRING, "Expected LineString in MultiLineString")),
                MULTI_POLYGON => Some((POLYGON, "Expected Polygon in MultiPolygon")),
                _ => None,
            };
            if let Some((want, message)) = expected
                && child != want
            {
                return Err(error::wkb(message, r.pos));
            }
            top.left -= 1;
            finished = (top.left == 0).then_some(top.base);
            if finished.is_some() {
                stack.pop();
            }
        }
    }
}

const END: &str = "Unexpected end of WKB buffer";
const INVALID_COORDINATE: &str = "Invalid coordinate value found";

/// A collection whose elements are being read.
struct Collection {
    base: u32,
    dims: usize,
    left: u32,
}

/// The base type (1 to 7) and the number of ordinates of an ISO type code.
fn split_type(code: u32) -> Option<(u32, usize)> {
    let base = code % 1000;
    let dims = match code / 1000 {
        0 => 2,
        1 | 2 => 3,
        3 => 4,
        _ => return None,
    };
    (POINT..=COLLECTION).contains(&base).then_some((base, dims))
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let bytes = self
            .buf
            .get(self.pos..)
            .and_then(|rest| rest.first_chunk::<N>())
            .copied()
            .ok_or_else(|| error::wkb(END, self.pos))?;
        self.pos += N;
        Ok(bytes)
    }

    fn u32(&mut self, big: bool) -> Result<u32> {
        let b = self.take::<4>()?;
        Ok(if big {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    }

    fn f64_bits(&mut self, big: bool) -> Result<u64> {
        let b = self.take::<8>()?;
        Ok(if big {
            u64::from_be_bytes(b)
        } else {
            u64::from_le_bytes(b)
        })
    }

    /// An element count: not negative, and at most what the bytes left
    /// can hold at `size` bytes per element.
    fn count(&mut self, big: bool, what: &str, size: usize) -> Result<u32> {
        let at = self.pos;
        let raw = self.u32(big)?;
        let n = i32::from_ne_bytes(raw.to_ne_bytes());
        if n < 0 {
            return Err(error::wkb(format!("Invalid count for {what}: {n}"), at));
        }
        let left = (self.buf.len() - self.pos) as u64;
        if u64::from(raw) * size as u64 > left {
            return Err(error::wkb(
                format!("Invalid count for {what}: exceeds remaining bytes"),
                at,
            ));
        }
        Ok(raw)
    }

    /// One point's ordinates, copied bit for bit; returns its X and Y.
    fn point(
        &mut self,
        big: bool,
        dims: usize,
        allow_nan: bool,
        geographic: bool,
        w: &mut Writer,
    ) -> Result<(f64, f64)> {
        let start = self.pos;
        let mut xy = (0.0, 0.0);
        for i in 0..dims {
            let at = self.pos;
            let bits = self.f64_bits(big)?;
            let v = f64::from_bits(bits);
            if v.is_infinite() || (v.is_nan() && !allow_nan) {
                return Err(error::wkb(INVALID_COORDINATE, at));
            }
            match i {
                0 => xy.0 = v,
                1 => xy.1 = v,
                _ => {}
            }
            w.u64(bits);
        }
        if geographic && (xy.0.abs() > 180.0 || xy.1.abs() > 90.0) {
            return Err(error::wkb(INVALID_COORDINATE, start));
        }
        Ok(xy)
    }

    fn line_string(
        &mut self,
        big: bool,
        dims: usize,
        geographic: bool,
        w: &mut Writer,
    ) -> Result<()> {
        let at = self.pos;
        let n = self.count(big, "LineString points", 8 * dims)?;
        if n == 1 {
            return Err(error::wkb("Too few points in linestring", at));
        }
        w.u32(n);
        for _ in 0..n {
            self.point(big, dims, false, geographic, w)?;
        }
        Ok(())
    }

    fn polygon(&mut self, big: bool, dims: usize, geographic: bool, w: &mut Writer) -> Result<()> {
        let rings = self.count(big, "polygon rings", 4)?;
        w.u32(rings);
        for _ in 0..rings {
            let at = self.pos;
            let n = self.count(big, "ring points", 8 * dims)?;
            if n < 4 {
                return Err(error::wkb("Too few points in ring", at));
            }
            w.u32(n);
            let first = self.point(big, dims, false, geographic, w)?;
            let mut last = first;
            for _ in 1..n {
                last = self.point(big, dims, false, geographic, w)?;
            }
            // Exact equality, so -0.0 closes on 0.0 (rings hold no NaN).
            if first != last {
                return Err(error::wkb("Ring is not closed", at));
            }
        }
        Ok(())
    }
}

struct Writer {
    out: Vec<u8>,
    order: ByteOrder,
}

impl Writer {
    fn header(&mut self, code: u32) {
        self.out.push(match self.order {
            ByteOrder::Ndr => 1,
            ByteOrder::Xdr => 0,
        });
        self.u32(code);
    }

    fn u32(&mut self, v: u32) {
        match self.order {
            ByteOrder::Ndr => self.out.extend_from_slice(&v.to_le_bytes()),
            ByteOrder::Xdr => self.out.extend_from_slice(&v.to_be_bytes()),
        }
    }

    fn u64(&mut self, v: u64) {
        match self.order {
            ByteOrder::Ndr => self.out.extend_from_slice(&v.to_le_bytes()),
            ByteOrder::Xdr => self.out.extend_from_slice(&v.to_be_bytes()),
        }
    }
}

#[cfg(test)]
mod tests;
