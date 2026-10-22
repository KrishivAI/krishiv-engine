// SPDX-License-Identifier: Apache-2.0
//! A structural check of a Parquet footer before arrow-rs decodes it.
//!
//! The footer is a Thrift compact-protocol struct. A list or map header
//! carries an element count that the decoder trusts: it skips or collects
//! that many elements, so a few bytes that declare billions of elements
//! make it loop for minutes or reserve gigabytes. Every element takes at
//! least one byte on the wire, so a count larger than the bytes left is
//! corrupt. This walk checks that, and the nesting depth, without
//! allocating per element and in time linear in the footer's size.

/// Nesting deeper than this is corrupt: the Parquet footer is about six
/// levels deep.
const MAX_DEPTH: usize = 32;

const BOOL_TRUE: u8 = 1;
const BOOL_FALSE: u8 = 2;
const BYTE: u8 = 3;
const I16: u8 = 4;
const I32: u8 = 5;
const I64: u8 = 6;
const DOUBLE: u8 = 7;
const BINARY: u8 = 8;
const LIST: u8 = 9;
const SET: u8 = 10;
const MAP: u8 = 11;
const STRUCT: u8 = 12;

/// A value being walked.
#[derive(Clone, Copy)]
enum Frame {
    /// A struct: fields until the stop byte.
    Struct,
    /// `left` elements of one type (a list or set).
    Items { left: u64, ty: u8 },
    /// `left` keys and values, alternating from a key.
    Entries { left: u64, key: u8, value: u8 },
}

/// Checks the compact-protocol struct `footer`.
///
/// # Errors
/// What is wrong, for a truncated value, an element count larger than the
/// bytes left, an unknown type or nesting that is too deep.
pub(super) fn check(footer: &[u8]) -> Result<(), &'static str> {
    let mut walk = Walk {
        buf: footer,
        pos: 0,
    };
    let mut stack = vec![Frame::Struct];
    while let Some(top) = stack.last().copied() {
        let (ty, in_list) = match top {
            Frame::Struct => {
                let header = walk.byte()?;
                if header == 0 {
                    stack.pop();
                    continue;
                }
                if header >> 4 == 0 {
                    // A field id that does not follow its predecessor.
                    walk.varint()?;
                }
                (header & 0x0f, false)
            }
            Frame::Items { left, ty } => {
                let Some(rest) = left.checked_sub(1) else {
                    stack.pop();
                    continue;
                };
                set_top(&mut stack, Frame::Items { left: rest, ty });
                (ty, true)
            }
            Frame::Entries { left, key, value } => {
                let Some(rest) = left.checked_sub(1) else {
                    stack.pop();
                    continue;
                };
                set_top(
                    &mut stack,
                    Frame::Entries {
                        left: rest,
                        key,
                        value,
                    },
                );
                // `left` counts keys and values: an even count is a key.
                (if left % 2 == 0 { key } else { value }, true)
            }
        };
        if let Some(inner) = walk.value(ty, in_list)? {
            if stack.len() >= MAX_DEPTH {
                return Err("nested too deeply");
            }
            stack.push(inner);
        }
    }
    Ok(())
}

fn set_top(stack: &mut [Frame], frame: Frame) {
    if let Some(top) = stack.last_mut() {
        *top = frame;
    }
}

struct Walk<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Walk<'_> {
    fn left(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn byte(&mut self) -> Result<u8, &'static str> {
        let b = *self.buf.get(self.pos).ok_or("truncated")?;
        self.pos += 1;
        Ok(b)
    }

    fn skip(&mut self, n: u64) -> Result<(), &'static str> {
        let n = usize::try_from(n).map_err(|_| "truncated")?;
        if n > self.left() {
            return Err("truncated");
        }
        self.pos += n;
        Ok(())
    }

    fn varint(&mut self) -> Result<u64, &'static str> {
        let mut value = 0_u64;
        for shift in (0..70).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("varint longer than 10 bytes")
    }

    /// Steps over a value of type `ty`, or returns the frame to walk into.
    /// A boolean takes a byte in a list or map and none as a field.
    fn value(&mut self, ty: u8, in_list: bool) -> Result<Option<Frame>, &'static str> {
        match ty {
            BOOL_TRUE | BOOL_FALSE => {
                if in_list {
                    self.skip(1)?;
                }
            }
            BYTE => self.skip(1)?,
            I16 | I32 | I64 => {
                self.varint()?;
            }
            DOUBLE => self.skip(8)?,
            BINARY => {
                let len = self.varint()?;
                self.skip(len)?;
            }
            LIST | SET => {
                let header = self.byte()?;
                let mut count = u64::from(header >> 4);
                if count == 15 {
                    count = self.varint()?;
                }
                let ty = header & 0x0f;
                if count > self.left() as u64 {
                    return Err("an element count larger than the bytes left");
                }
                return Ok(Some(Frame::Items { left: count, ty }));
            }
            MAP => {
                let count = self.varint()?;
                if count == 0 {
                    return Ok(None);
                }
                // A key and a value take two bytes at least.
                if count > (self.left() / 2) as u64 {
                    return Err("an entry count larger than the bytes left");
                }
                let types = self.byte()?;
                return Ok(Some(Frame::Entries {
                    left: count * 2,
                    key: types >> 4,
                    value: types & 0x0f,
                }));
            }
            STRUCT => return Ok(Some(Frame::Struct)),
            _ => return Err("an unknown type"),
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::check;

    #[test]
    fn accepts_a_nested_footer() {
        // struct { 1: i32 = 1, 2: list<struct> [ { 1: i32 = 2 }, {} ],
        //          3: map<i32, binary> { 1: "a" }, 4: list<bool> [t, f] }
        let footer = [
            0x15, 0x02, // field 1, i32 1
            0x19, 0x2c, 0x15, 0x04, 0x00, 0x00, // field 2, list of 2 structs
            0x1b, 0x01, 0x58, 0x02, 0x01, 0x61, // field 3, map of 1 i32 -> binary
            0x19, 0x21, 0x01, 0x02, // field 4, list of 2 bools
            0x00,
        ];
        assert_eq!(check(&footer), Ok(()));
    }

    #[test]
    fn rejects_a_count_larger_than_the_bytes_left() {
        // field 1: a list of 2^31 - 1 bools in three bytes.
        let list = [0x19, 0xf1, 0xff, 0xff, 0xff, 0xff, 0x07, 0x00];
        assert_eq!(
            check(&list),
            Err("an element count larger than the bytes left")
        );
        // field 1: a map of 2^31 - 1 entries.
        let map = [0x1b, 0xff, 0xff, 0xff, 0xff, 0x07, 0x55, 0x00];
        assert_eq!(
            check(&map),
            Err("an entry count larger than the bytes left")
        );
    }

    #[test]
    fn rejects_truncated_and_unknown_values() {
        assert_eq!(check(&[0x18, 0x05, 0x61]), Err("truncated"));
        assert_eq!(check(&[0x15]), Err("truncated"));
        assert_eq!(check(&[0x1f, 0x00]), Err("an unknown type"));
        assert_eq!(check(&[0x15, 0x80]), Err("truncated"));
        assert_eq!(
            check(&[
                0x15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
            ]),
            Err("varint longer than 10 bytes")
        );
        assert_eq!(check(&[0x17]), Err("truncated"));
        assert_eq!(check(&[0x1b, 0x00, 0x00]), Ok(()));
    }

    #[test]
    fn rejects_the_fuzzed_footer() {
        // The footer of the file on which the reader reserved gigabytes.
        let footer = [
            0x1c, 0x2a, 0x1c, 0x1c, 0x1c, 0x06, 0x3d, 0x01, 0x00, 0x62, 0x0a, 0x00, 0x00, 0x00,
            0xf4, 0x07, 0x03, 0x03, 0x00, 0x16, 0xfb, 0x1c, 0x66, 0xe3, 0xd5, 0xe3, 0xf3, 0xf9,
            0xc2, 0xfe, 0x00, 0x62, 0x0a, 0x00, 0x00, 0x00, 0x54, 0x52, 0x4d, 0x01, 0x00, 0x70,
            0x46, 0xd6, 0xcc, 0x3c, 0x80, 0x00, 0x00, 0x00, 0x6a, 0x00, 0x1c, 0x1c, 0x1c, 0x06,
            0x3d, 0x01, 0x00, 0x62, 0x0a, 0x00, 0x00, 0x00, 0x5c, 0x49, 0x4d, 0x01, 0x00, 0x03,
            0xb6, 0x03, 0x72, 0x2e, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x50, 0x41, 0x45,
        ];
        assert_eq!(check(&footer), Err("an unknown type"));
    }

    #[test]
    fn rejects_nesting_that_is_too_deep() {
        // field 1 of a struct, 64 levels of it.
        let mut deep = vec![0x1c; 64];
        deep.extend([0x00; 64]);
        assert_eq!(check(&deep), Err("nested too deeply"));
    }
}
