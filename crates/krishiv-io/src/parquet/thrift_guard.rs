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
//!
//! Page headers are Thrift structs too, read by arrow-rs from the column
//! chunk bytes; [`check_pages`] walks a chunk's page headers the same way
//! before the chunk reaches the decoder (arrow-rs skips a `list<bool>` of
//! an unknown field without reading a byte per element, so a count of
//! 2^31 costs seconds of CPU per header).

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
    walk_struct(footer).map(|_| ())
}

/// What [`walk_struct`] found: the bytes the struct takes, and the value of
/// its top-level `i32` field 3 (a page header's `compressed_page_size`).
struct Walked {
    len: usize,
    field_3: Option<u64>,
}

/// Walks the compact-protocol struct at the start of `buf`.
fn walk_struct(buf: &[u8]) -> Result<Walked, &'static str> {
    let mut walk = Walk { buf, pos: 0 };
    let mut stack = vec![Frame::Struct];
    let (mut last_id, mut field_3) = (0_i64, None);
    while let Some(top) = stack.last().copied() {
        let (ty, in_list) = match top {
            Frame::Struct => {
                let header = walk.byte()?;
                if header == 0 {
                    stack.pop();
                    continue;
                }
                let id = if header >> 4 == 0 {
                    // A field id that does not follow its predecessor.
                    unzigzag(walk.varint()?)
                } else {
                    last_id.saturating_add(i64::from(header >> 4))
                };
                let ty = header & 0x0f;
                if stack.len() == 1 {
                    last_id = id;
                    if id == 3 && ty == I32 {
                        field_3 = Some(walk.varint()?);
                        continue;
                    }
                }
                (ty, false)
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
                return Err(TOO_DEEP);
            }
            stack.push(inner);
        }
    }
    Ok(Walked {
        len: walk.pos,
        field_3,
    })
}

/// Checks the page headers of the column chunk `chunk`: each is walked as
/// [`check`] walks a footer, then its `compressed_page_size` bytes of page
/// data are stepped over, up to the end of the chunk.
///
/// A count larger than the bytes left and nesting that are too deep are
/// errors (arrow-rs reads them in time that does not follow the bytes), and
/// so is a varint longer than 10 bytes or past 64 bits (arrow-rs reads it
/// with wrapping shifts and would go on to headers this walk did not see).
/// Anything else that does not parse (bytes after the last page, a
/// truncated header, an unknown type) ends the walk and is left to arrow-rs,
/// which reads pages only as far as the chunk's values go and rejects a bad
/// header in linear time.
///
/// # Errors
/// Which of the two was found.
pub(super) fn check_pages(chunk: &[u8]) -> Result<(), &'static str> {
    let mut pos = 0_usize;
    while let Some(rest) = chunk.get(pos..).filter(|r| !r.is_empty()) {
        let walked = match walk_struct(rest) {
            Ok(w) => w,
            Err(
                e @ (COUNT_TOO_LARGE | ENTRIES_TOO_MANY | TOO_DEEP | VARINT_TOO_LONG
                | VARINT_OVERFLOW),
            ) => return Err(e),
            Err(_) => return Ok(()),
        };
        let Some(next) = walked
            .field_3
            .and_then(|raw| usize::try_from(unzigzag(raw)).ok())
            .and_then(|size| size.checked_add(walked.len))
            .and_then(|step| pos.checked_add(step))
        else {
            return Ok(());
        };
        pos = next;
    }
    Ok(())
}

/// The signed value of a zigzag varint.
fn unzigzag(raw: u64) -> i64 {
    i64::from_ne_bytes(((raw >> 1) ^ (raw & 1).wrapping_neg()).to_ne_bytes())
}

const COUNT_TOO_LARGE: &str = "an element count larger than the bytes left";
const ENTRIES_TOO_MANY: &str = "an entry count larger than the bytes left";
const TOO_DEEP: &str = "nested too deeply";
const VARINT_TOO_LONG: &str = "varint longer than 10 bytes";
const VARINT_OVERFLOW: &str = "varint overflows 64 bits";

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
            // The tenth byte holds the top bit of a u64: more would be
            // shifted out silently and read as a smaller value.
            if shift == 63 && byte & 0x7f > 1 {
                return Err(VARINT_OVERFLOW);
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(VARINT_TOO_LONG)
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
                    return Err(COUNT_TOO_LARGE);
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
                    return Err(ENTRIES_TOO_MANY);
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
    use super::{check, check_pages};

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
        // A tenth byte above 1 overflows u64 (2 << 63 would read as 0, an
        // empty list); 1 is the largest u64 and a continuation is too long.
        let overflow = |last: u8| {
            let mut list = vec![0x19, 0xf1];
            list.extend([0x80; 9]);
            list.extend([last, 0x00]);
            list
        };
        assert_eq!(check(&overflow(0x02)), Err("varint overflows 64 bits"));
        assert_eq!(check(&overflow(0x7f)), Err("varint overflows 64 bits"));
        assert_eq!(
            check(&overflow(0x01)),
            Err("an element count larger than the bytes left")
        );
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

    /// A page header: `type`, `uncompressed_page_size` and
    /// `compressed_page_size` (all `n`), then `extra` fields, then `n`
    /// bytes of page data.
    fn page(n: u8, extra: &[u8]) -> Vec<u8> {
        let mut p = vec![0x15, 0x00, 0x15, n * 2, 0x15, n * 2];
        p.extend(extra);
        p.push(0x00);
        p.extend(std::iter::repeat_n(0xAB, usize::from(n)));
        p
    }

    #[test]
    fn walks_the_pages_of_a_chunk() {
        let mut chunk = page(4, &[]);
        chunk.extend(page(0, &[]));
        chunk.extend(page(3, &[0x2c, 0x15, 0x02, 0x00])); // field 5: struct
        assert_eq!(check_pages(&chunk), Ok(()));
        // Bytes after the last page, a truncated header or a size past the
        // end end the walk: arrow-rs reads (or rejects) those itself.
        chunk.extend([0x00, 0x00, 0x15]);
        assert_eq!(check_pages(&chunk), Ok(()));
        assert_eq!(check_pages(&page(9, &[])[..10]), Ok(()));
        assert_eq!(check_pages(&[]), Ok(()));
    }

    #[test]
    fn rejects_a_page_header_list_longer_than_the_chunk() {
        // Field 9 (unknown to arrow-rs, which skips it): a list of 2^31 - 1
        // bools, in the second page header.
        let mut chunk = page(2, &[]);
        chunk.extend(page(2, &[0x69, 0xf1, 0xff, 0xff, 0xff, 0xff, 0x07]));
        assert_eq!(
            check_pages(&chunk),
            Err("an element count larger than the bytes left")
        );
        // A field id given in full (zigzag) before field 3 still finds it.
        let mut long_id = vec![0x05, 0x02, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0xAB];
        long_id.extend(page(1, &[0x69, 0xf1, 0xff, 0xff, 0xff, 0xff, 0x07]));
        assert_eq!(
            check_pages(&long_id),
            Err("an element count larger than the bytes left")
        );
        // A varint arrow-rs would read with wrapping shifts (11 bytes, as
        // the uncompressed size) is rejected, not the end of the walk: the
        // decoder would read on to the next header.
        let mut overlong = vec![0x15, 0x00, 0x15];
        overlong.extend([0x80; 10]);
        overlong.extend([0x00, 0x15, 0x02, 0x00, 0xAB]);
        overlong.extend(page(2, &[0x69, 0xf1, 0xff, 0xff, 0xff, 0xff, 0x07]));
        assert_eq!(check_pages(&overlong), Err("varint longer than 10 bytes"));
    }
}
