// SPDX-License-Identifier: Apache-2.0
//! Extended well-known text of a spatial value, as Spark 4.2 prints it
//! (`HiveResult`, the `spark-sql` form): every geometry type, `Z`/`M`/`ZM`
//! ordinates, empty geometries and nested collections.
//!
//! The rules are those of a real Spark 4.2.0 run
//! (`research/2026-11-06-spark-geo-wkt-and-geography-cast.md`):
//! - no space between the type and `(` without a dimension tag
//!   (`POINT(1 2)`), a space around it with one (`POINT Z (1 2 3)`);
//! - an empty geometry is `<TYPE>[ <tag>] EMPTY`; an empty member of a
//!   multi-geometry is a bare `EMPTY`;
//! - members are separated by `,` without a space, and a hole is another ring;
//! - the members of a collection are full text, and may nest;
//! - an ordinate that is a whole number below 2^63 prints as an integer
//!   (`-0.0` is `0`), any other as Java's `Double.toString` (`1.5`, `1.0E20`).

use super::{
    COLLECTION, LINE_STRING, MULTI_LINE_STRING, MULTI_POINT, MULTI_POLYGON, POINT, POLYGON,
};
use crate::error::{self, Result};

/// A reader over ISO WKB that has already been validated by
/// [`rewrite`](super::rewrite) (or written by the engine); a short or
/// malformed value is still an error, never a panic.
struct Reader<'a> {
    wkb: &'a [u8],
    pos: usize,
}

/// The dimension of a geometry: its ordinates per point and its tag.
#[derive(Clone, Copy)]
struct Dims {
    z: bool,
    m: bool,
}

impl Dims {
    fn count(self) -> usize {
        2 + usize::from(self.z) + usize::from(self.m)
    }

    fn tag(self) -> &'static str {
        match (self.z, self.m) {
            (false, false) => "",
            (true, false) => "Z",
            (false, true) => "M",
            (true, true) => "ZM",
        }
    }
}

impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let at = self.pos;
        let b = self
            .wkb
            .get(at..at + N)
            .and_then(|s| <[u8; N]>::try_from(s).ok())
            .ok_or_else(|| error::wkb("WKB data is too short", at))?;
        self.pos += N;
        Ok(b)
    }

    /// One geometry header: its byte order (for what follows), type and dimension.
    fn header(&mut self) -> Result<(bool, u32, Dims)> {
        let [order] = self.bytes::<1>()?;
        let big = match order {
            0 => true,
            1 => false,
            b => return Err(error::wkb(format!("Invalid byte order {b}"), self.pos - 1)),
        };
        let raw = self.bytes::<4>()?;
        let code = if big {
            u32::from_be_bytes(raw)
        } else {
            u32::from_le_bytes(raw)
        };
        let dims = match code / 1000 {
            0 => Dims { z: false, m: false },
            1 => Dims { z: true, m: false },
            2 => Dims { z: false, m: true },
            3 => Dims { z: true, m: true },
            _ => {
                return Err(error::wkb(
                    format!("Invalid or unsupported type {code}"),
                    self.pos - 4,
                ));
            }
        };
        Ok((big, code % 1000, dims))
    }

    fn count(&mut self, big: bool) -> Result<usize> {
        let raw = self.bytes::<4>()?;
        let n = if big {
            u32::from_be_bytes(raw)
        } else {
            u32::from_le_bytes(raw)
        };
        usize::try_from(n).map_err(|_| error::wkb("a count out of range", self.pos - 4))
    }

    fn ordinate(&mut self, big: bool) -> Result<f64> {
        let raw = self.bytes::<8>()?;
        Ok(if big {
            f64::from_be_bytes(raw)
        } else {
            f64::from_le_bytes(raw)
        })
    }

    /// A point's ordinates, as text; `None` for the empty point (all NaN).
    fn point(&mut self, big: bool, dims: Dims) -> Result<Option<String>> {
        let mut parts = Vec::with_capacity(dims.count());
        let mut empty = true;
        for _ in 0..dims.count() {
            let v = self.ordinate(big)?;
            empty &= v.is_nan();
            parts.push(number(v));
        }
        Ok((!empty).then(|| parts.join(" ")))
    }

    /// `n` points, comma separated.
    fn points(&mut self, big: bool, dims: Dims, n: usize) -> Result<String> {
        let mut out = Vec::new();
        for _ in 0..n {
            out.push(self.point(big, dims)?.unwrap_or_default());
        }
        Ok(out.join(","))
    }

    /// The text after the type name of a geometry of `code`: `None` when empty.
    fn body(&mut self, big: bool, code: u32, dims: Dims, depth: usize) -> Result<Option<String>> {
        Ok(match code {
            POINT => self.point(big, dims)?.map(|p| format!("({p})")),
            LINE_STRING => {
                let n = self.count(big)?;
                (n > 0)
                    .then(|| self.points(big, dims, n))
                    .transpose()?
                    .map(|p| format!("({p})"))
            }
            POLYGON => {
                let rings = self.count(big)?;
                if rings == 0 {
                    None
                } else {
                    let mut out = Vec::new();
                    for _ in 0..rings {
                        let n = self.count(big)?;
                        out.push(format!("({})", self.points(big, dims, n)?));
                    }
                    Some(format!("({})", out.join(",")))
                }
            }
            MULTI_POINT | MULTI_LINE_STRING | MULTI_POLYGON => {
                let n = self.count(big)?;
                if n == 0 {
                    None
                } else {
                    let mut out = Vec::new();
                    for _ in 0..n {
                        let (mbig, mcode, mdims) = self.header()?;
                        // A member is the single type the multi-geometry holds; this also
                        // keeps a multi-geometry from nesting in another (unbounded recursion).
                        let member = match code {
                            MULTI_POINT => POINT,
                            MULTI_LINE_STRING => LINE_STRING,
                            _ => POLYGON,
                        };
                        if mcode != member {
                            return Err(error::wkb(
                                format!("Invalid or unsupported type {mcode}"),
                                self.pos,
                            ));
                        }
                        out.push(
                            self.body(mbig, mcode, mdims, depth + 1)?
                                .unwrap_or_else(|| "EMPTY".to_owned()),
                        );
                    }
                    Some(format!("({})", out.join(",")))
                }
            }
            COLLECTION => {
                let n = self.count(big)?;
                if n == 0 {
                    None
                } else {
                    let mut out = Vec::new();
                    for _ in 0..n {
                        out.push(self.geometry(depth + 1)?);
                    }
                    Some(format!("({})", out.join(",")))
                }
            }
            other => {
                return Err(error::wkb(
                    format!("Invalid or unsupported type {other}"),
                    self.pos,
                ));
            }
        })
    }

    /// One geometry as full text, `POINT Z (1 2 3)` or `LINESTRING EMPTY`.
    fn geometry(&mut self, depth: usize) -> Result<String> {
        // A collection may nest; refuse a pathological depth.
        if depth > MAX_DEPTH {
            return Err(error::wkb("geometry collections nest too deeply", self.pos));
        }
        let (big, code, dims) = self.header()?;
        let name = match code {
            POINT => "POINT",
            LINE_STRING => "LINESTRING",
            POLYGON => "POLYGON",
            MULTI_POINT => "MULTIPOINT",
            MULTI_LINE_STRING => "MULTILINESTRING",
            MULTI_POLYGON => "MULTIPOLYGON",
            COLLECTION => "GEOMETRYCOLLECTION",
            other => {
                return Err(error::wkb(
                    format!("Invalid or unsupported type {other}"),
                    self.pos,
                ));
            }
        };
        let tag = dims.tag();
        let body = self.body(big, code, dims, depth)?;
        Ok(match (tag.is_empty(), body) {
            (true, None) => format!("{name} EMPTY"),
            (false, None) => format!("{name} {tag} EMPTY"),
            (true, Some(b)) => format!("{name}{b}"),
            (false, Some(b)) => format!("{name} {tag} {b}"),
        })
    }
}

/// The deepest nesting of geometry collections printed.
const MAX_DEPTH: usize = 64;

/// The text of `wkb` as Spark prints it, with `SRID=<n>;` in front unless
/// `srid` is 0.
///
/// # Errors
/// `WKB_PARSE_ERROR` for a value that is not well-formed ISO WKB.
pub(super) fn to_ewkt(srid: i32, wkb: &[u8]) -> Result<String> {
    let mut r = Reader { wkb, pos: 0 };
    let text = r.geometry(0)?;
    if r.pos != wkb.len() {
        return Err(error::wkb("trailing bytes after the geometry", r.pos));
    }
    Ok(if srid == 0 {
        text
    } else {
        format!("SRID={srid};{text}")
    })
}

/// An ordinate as Spark prints it: a whole number below 2^63 as an integer,
/// any other as Java's `Double.toString`.
fn number(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_owned();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "the bound only decides whether the value prints as an integer"
    )]
    let bound = i64::MAX as f64;
    if v.fract() == 0.0 && v.abs() < bound {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "whole and below 2^63, checked above"
        )]
        return (v as i64).to_string();
    }
    java_double(v)
}

/// Java's `Double.toString` (JDK 19 and later: the shortest digits that
/// round-trip): plain from 10^-3 up to 10^7, `d.dddE<exp>` outside.
fn java_double(v: f64) -> String {
    // Rust's `{:e}` gives the shortest round-trip digits: `1.5e0`, `1e20`.
    let sci = format!("{:e}", v.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if v < 0.0 { "-" } else { "" };
    if (-3..7).contains(&exp) {
        // Plain: put the decimal point `exp` places after the first digit.
        let text = if exp >= 0 {
            let point = usize::try_from(exp).unwrap_or(0) + 1;
            let (int, frac) = if digits.len() > point {
                (digits[..point].to_owned(), digits[point..].to_owned())
            } else {
                (
                    format!("{digits}{}", "0".repeat(point - digits.len())),
                    "0".to_owned(),
                )
            };
            format!("{int}.{frac}")
        } else {
            let zeros = "0".repeat(usize::try_from(-exp - 1).unwrap_or(0));
            format!("0.{zeros}{digits}")
        };
        format!("{sign}{text}")
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() { "0" } else { rest };
        format!("{sign}{first}.{rest}E{exp}")
    }
}
