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

/// The type nibble of the largest field type arrow-rs knows (UUID).
const MAX_FIELD_TYPE: u8 = 13;
const UUID: u8 = 13;

/// Which struct a frame reads. arrow-rs reads the fields of a page header by
/// the schema of the header, not by the wire type of each field: a known id
/// is read as its schema type whatever the wire says, and only an unknown id
/// is skipped by its wire type. A walk by wire type would take the bytes of a
/// known field differently and lose its place in the header.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Schema {
    /// Walked by wire type alone (a footer, or an unknown struct).
    Any,
    /// `PageHeader`.
    Page,
    /// `DataPageHeader`.
    DataPage,
    /// `IndexPageHeader` (no known field).
    IndexPage,
    /// `DictionaryPageHeader`.
    DictPage,
    /// `DataPageHeaderV2`.
    DataPageV2,
    /// A page index's `ColumnIndex`: walked by wire type like `Any`, except
    /// for field 1, `null_pages`, a boolean list the decoder reads a byte
    /// per element of (an unknown field's boolean list it skips unread).
    ColumnIndex,
}

/// How arrow-rs reads a field of a known struct.
enum Known {
    /// A zigzag varint (`i32` or an enum), whatever the wire type.
    Varint,
    /// A boolean, which arrow-rs reads from the wire type itself.
    Bool,
    /// A nested known struct, whatever the wire type.
    Struct(Schema),
    /// A list the decoder reads: booleans take a byte each (the other
    /// element types are walked as any list).
    List,
    /// Not read: skipped by wire type.
    Skipped,
}

/// The fields `read_thrift_without_stats` of arrow-rs 60 reads; the
/// statistics of a data page header are skipped, not read.
fn known(schema: Schema, id: i16) -> Known {
    match (schema, id) {
        (Schema::Page | Schema::DataPage, 1..=4)
        | (Schema::DictPage, 1 | 2)
        | (Schema::DataPageV2, 1..=6) => Known::Varint,
        (Schema::Page, 5) => Known::Struct(Schema::DataPage),
        (Schema::Page, 6) => Known::Struct(Schema::IndexPage),
        (Schema::Page, 7) => Known::Struct(Schema::DictPage),
        (Schema::Page, 8) => Known::Struct(Schema::DataPageV2),
        (Schema::DictPage, 3) | (Schema::DataPageV2, 7) => Known::Bool,
        (Schema::ColumnIndex, 1) => Known::List,
        _ => Known::Skipped,
    }
}

/// A value being walked.
#[derive(Clone, Copy)]
enum Frame {
    /// A struct: fields until the stop byte; `last` is the previous field id.
    Struct { schema: Schema, last: i16 },
    /// `left` elements of one type (a list or set).
    Items { left: u64, ty: u8 },
    /// `left` keys and values, alternating from a key.
    Entries { left: u64, key: u8, value: u8 },
}

const fn any_struct() -> Frame {
    Frame::Struct {
        schema: Schema::Any,
        last: 0,
    }
}

/// Checks the compact-protocol struct `footer`.
///
/// # Errors
/// What is wrong, for a truncated value, an element count larger than the
/// bytes left, an unknown type or nesting that is too deep.
pub(super) fn check(footer: &[u8]) -> Result<(), &'static str> {
    let limit = footer.len().saturating_mul(8).saturating_add(1024);
    walk_struct(footer, Schema::Any, &mut 0, limit).map(|_| ())
}

/// Checks the `ColumnIndex` at `bytes`: as [`check`], but walking its
/// boolean `null_pages` list the way the decoder reads it (a byte per
/// element), so the collections after it are reached at the right bytes.
///
/// # Errors
/// What [`check`] reports.
pub(super) fn check_column_index(bytes: &[u8]) -> Result<(), &'static str> {
    let limit = bytes.len().saturating_mul(8).saturating_add(1024);
    walk_struct(bytes, Schema::ColumnIndex, &mut 0, limit).map(|_| ())
}

/// The `numBytes` of the `BloomFilterHeader` at the start of `buf` and the
/// header's length. The header is walked first (counts, depth, work), then
/// read: its first field must be the `i32` `numBytes`, which the format
/// writes first.
///
/// # Errors
/// What is wrong with the walk, or that the struct does not start with
/// `numBytes`, or that `numBytes` is negative.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the Bloom stage and the sparse fetch of P2-T080b use it"
    )
)]
pub(super) fn bloom_header(buf: &[u8]) -> Result<(u32, usize), &'static str> {
    let limit = buf.len().saturating_mul(8).saturating_add(1024);
    let walked = walk_struct(buf, Schema::Any, &mut 0, limit)?;
    // Field 1, type i32: delta 1 in the high nibble, type 5 in the low.
    if buf.first() != Some(&0x15) {
        return Err("a Bloom header that does not start with numBytes");
    }
    let mut walk = Walk {
        buf,
        pos: 1,
        work: &mut 0,
        limit,
        page_header: false,
    };
    let raw = walk.varint()?;
    let num_bytes = u32::try_from(unzigzag(raw)).map_err(|_| "a negative Bloom bitset size")?;
    Ok((num_bytes, walked.len))
}

/// What [`walk_struct`] found: the bytes the struct takes, and the value of
/// its top-level `i32` field 3 (a page header's `compressed_page_size`).
struct Walked {
    len: usize,
    field_3: Option<u64>,
}

/// Walks the compact-protocol struct at the start of `buf`; `work` counts the
/// values stepped over, against `limit`.
fn walk_struct(
    buf: &[u8],
    schema: Schema,
    work: &mut usize,
    limit: usize,
) -> Result<Walked, &'static str> {
    let mut walk = Walk {
        buf,
        pos: 0,
        work,
        limit,
        page_header: schema == Schema::Page,
    };
    let mut stack = vec![Frame::Struct { schema, last: 0 }];
    let mut field_3 = None;
    while let Some(top) = stack.last().copied() {
        let (ty, in_list) = match top {
            Frame::Struct { schema, last } => {
                let header = walk.byte()?;
                // arrow-rs ends a struct at any byte whose type nibble is 0 (it does
                // not read a field id after it), not only at the zero byte. Walking
                // on past it as an unknown type would end this check early and leave
                // the rest of the header unchecked.
                if header & 0x0f == 0 {
                    stack.pop();
                    continue;
                }
                let ty = header & 0x0f;
                if ty > MAX_FIELD_TYPE {
                    return Err("an unknown type");
                }
                // arrow-rs keeps field ids as `i16`: a full id is the zigzag
                // value truncated to 16 bits, and a delta past `i16::MAX` fails.
                let id = if header >> 4 == 0 {
                    low16(unzigzag(walk.varint()?))
                } else if schema == Schema::Any {
                    // arrow-rs skips such a struct with the previous id always 0,
                    // so a delta never overflows there, and the id is not used.
                    0
                } else {
                    last.checked_add(i16::from(header >> 4))
                        .ok_or("a field id overflows")?
                };
                set_top(&mut stack, Frame::Struct { schema, last: id });
                match known(schema, id) {
                    Known::Varint => {
                        walk.budget()?;
                        let raw = walk.varint()?;
                        if schema == Schema::Page && id == 3 {
                            field_3 = Some(raw);
                        }
                        continue;
                    }
                    Known::Bool => {
                        if ty != BOOL_TRUE && ty != BOOL_FALSE {
                            return Err("not a boolean");
                        }
                        continue;
                    }
                    Known::Struct(child) => {
                        if stack.len() >= MAX_DEPTH {
                            return Err(TOO_DEEP);
                        }
                        stack.push(Frame::Struct {
                            schema: child,
                            last: 0,
                        });
                        continue;
                    }
                    Known::List => {
                        // The decoder reads this field as a list whatever
                        // the wire type says.
                        let (count, elem) = walk.list_header()?;
                        if elem == BOOL_TRUE || elem == BOOL_FALSE {
                            walk.skip(count)?;
                            continue;
                        }
                        if stack.len() >= MAX_DEPTH {
                            return Err(TOO_DEEP);
                        }
                        stack.push(Frame::Items {
                            left: count,
                            ty: elem,
                        });
                        continue;
                    }
                    Known::Skipped => (ty, false),
                }
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

/// The low 16 bits of `v`, as arrow-rs's `as i16` takes them.
#[expect(
    clippy::cast_possible_truncation,
    reason = "arrow-rs truncates the same way"
)]
const fn low16(v: i64) -> i16 {
    v as i16
}

/// The low 32 bits of `v`, as arrow-rs's `as i32` takes them.
#[expect(
    clippy::cast_possible_truncation,
    reason = "arrow-rs truncates the same way"
)]
const fn low32(v: i64) -> i32 {
    v as i32
}

/// Checks the page headers of the column chunk `chunk`: each is walked as
/// arrow-rs reads it (see [`Schema`]), then its `compressed_page_size` bytes
/// of page data are stepped over, up to the end of the chunk.
///
/// A count larger than the bytes left and nesting that are too deep are
/// errors (arrow-rs reads them in time that does not follow the bytes), and
/// so is a varint longer than 10 bytes or past 64 bits (arrow-rs reads it
/// with wrapping shifts and would go on to headers this walk did not see).
/// So is anything that runs past the end of the chunk, a binary or UUID
/// value, a header or a page: the pages of a chunk tile it, and arrow-rs
/// reads on past the end into whatever follows, bytes this walk never
/// checks. The work budget is one for the whole chunk. A header that does
/// not parse for another reason (an unknown type, a missing page size) ends
/// the walk: arrow-rs rejects it at once.
///
/// # Errors
/// Which of these was found.
pub(super) fn check_pages(chunk: &[u8]) -> Result<(), &'static str> {
    let mut pos = 0_usize;
    let mut work = 0_usize;
    let limit = chunk.len().saturating_mul(8).saturating_add(1024);
    while let Some(rest) = chunk.get(pos..).filter(|r| !r.is_empty()) {
        let walked = match walk_struct(rest, Schema::Page, &mut work, limit) {
            Ok(w) => w,
            Err("truncated") => return Err(HEADER_PAST_END),
            Err(
                e @ (COUNT_TOO_LARGE | ENTRIES_TOO_MANY | TOO_DEEP | TOO_MUCH_WORK
                | BINARY_PAST_END | VARINT_TOO_LONG | VARINT_OVERFLOW),
            ) => return Err(e),
            Err(_) => return Ok(()),
        };
        let Some(next) = walked
            .field_3
            .and_then(|raw| usize::try_from(low32(unzigzag(raw))).ok())
            .and_then(|size| size.checked_add(walked.len))
            .and_then(|step| pos.checked_add(step))
        else {
            return Ok(());
        };
        if next > chunk.len() {
            return Err(PAGE_PAST_END);
        }
        pos = next;
    }
    Ok(())
}

/// The signed value of a zigzag varint.
fn unzigzag(raw: u64) -> i64 {
    i64::from_ne_bytes(((raw >> 1) ^ (raw & 1).wrapping_neg()).to_ne_bytes())
}

const PAGE_PAST_END: &str = "a page runs past the end of its column chunk";
const HEADER_PAST_END: &str = "a page header runs past the end of its column chunk";
const TOO_MUCH_WORK: &str = "more values than the bytes can hold";
const BINARY_PAST_END: &str = "a binary value longer than the bytes left";
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
    /// Values stepped over so far, shared by the walks of a chunk; see
    /// [`Walk::budget`].
    work: &'a mut usize,
    limit: usize,
    /// Whether this is a page header, read by arrow-rs from a reader that
    /// runs on past the chunk. A footer is read from a slice, whose every
    /// skip is bounded, and still refuses UUIDs here as before.
    page_header: bool,
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

    /// Counts one value and refuses a walk that does far more work than it
    /// has bytes: arrow-rs steps over a boolean element of a list or map
    /// without reading a byte (see [`Walk::value`]), so nested collections of
    /// them cost time that their size on the wire does not bound.
    fn budget(&mut self) -> Result<(), &'static str> {
        *self.work += 1;
        if *self.work > self.limit {
            return Err(TOO_MUCH_WORK);
        }
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
    ///
    /// A boolean takes a byte in a list or map on the wire, but arrow-rs
    /// steps over an unknown field's booleans without reading one, so this
    /// does the same: walking by the specification would put this check at
    /// other bytes than the decoder's, and let it miss what the decoder
    /// then reads (a map of tens of millions of entries, say).
    /// A list or set header: the element count (checked against the bytes
    /// left) and the element type.
    fn list_header(&mut self) -> Result<(u64, u8), &'static str> {
        let header = self.byte()?;
        let mut count = u64::from(header >> 4);
        if count == 15 {
            count = self.varint()?;
        }
        if count > self.left() as u64 {
            return Err(COUNT_TOO_LARGE);
        }
        Ok((count, header & 0x0f))
    }

    fn value(&mut self, ty: u8, _in_list: bool) -> Result<Option<Frame>, &'static str> {
        self.budget()?;
        match ty {
            BOOL_TRUE | BOOL_FALSE => {}
            BYTE => self.skip(1)?,
            I16 | I32 | I64 => {
                self.varint()?;
            }
            DOUBLE => self.skip(8)?,
            BINARY => {
                let len = self.varint()?;
                // arrow-rs skips a binary field by copying `len` bytes from a
                // reader that goes on past the end of the buffer, so a length
                // that overruns it lands the decoder in bytes this walk never
                // sees (the next column chunk's, say).
                if usize::try_from(len).map_or(true, |n| n > self.left()) {
                    return Err(BINARY_PAST_END);
                }
                self.skip(len)?;
            }
            UUID if self.page_header => {
                // The same copy, of 16 bytes.
                if self.left() < 16 {
                    return Err(BINARY_PAST_END);
                }
                self.skip(16)?;
            }
            LIST | SET => {
                let (count, ty) = self.list_header()?;
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
            STRUCT => return Ok(Some(any_struct())),
            _ => return Err("an unknown type"),
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BINARY_PAST_END, COUNT_TOO_LARGE, ENTRIES_TOO_MANY, HEADER_PAST_END, PAGE_PAST_END,
        TOO_MUCH_WORK, check, check_pages,
    };

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
        assert_eq!(check(&[0x18, 0x05, 0x61]), Err(BINARY_PAST_END));
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

    /// A fuzzed column chunk (`parquet_read`, found by the weekly run): its
    /// first header holds a set of nine booleans, which arrow-rs steps over
    /// without reading a byte each, so its next field starts 9 bytes earlier
    /// than the specification puts it, a map of tens of millions of entries
    /// that costs seconds to skip. A walk by the specification lost the
    /// header's thread and saw none of it.
    #[test]
    fn follows_arrow_rs_over_booleans_in_a_collection() {
        let chunk: &[u8] = &[
            0x8a, 0x92, 0x8b, 0xfc, 0xff, 0xc0, 0x19, 0x72, 0xa8, 0xa6, 0x8b, 0xfc, 0xff, 0x3b,
            0x7a, 0x49, 0xc6, 0xba, 0x8b, 0xfc, 0xff, 0xb6, 0xda, 0x20, 0xe4, 0xce, 0x8b, 0xfc,
            0xff, 0x31, 0x3b, 0xf8, 0x01, 0xe3, 0x8b, 0xfc, 0xff, 0xac, 0x9b, 0xcf, 0x1f, 0xf7,
            0x8b, 0xfc, 0xff, 0xa2, 0x5c, 0x7e, 0x5b, 0x1f, 0x8c, 0xfc, 0xff, 0x1d, 0xbd, 0x55,
            0x79, 0x33, 0x8c, 0xfc, 0xff, 0x98, 0x1d, 0x2d, 0x97, 0x47, 0x8c, 0xfc, 0xff, 0x13,
            0x7e, 0x04, 0xb5, 0x5b, 0x8c, 0xfc, 0xff, 0x8e, 0xde, 0xdb, 0xd2, 0x6f, 0x8c, 0xfc,
            0xff, 0x09, 0x3f, 0xb3, 0xf0, 0x83, 0x8c, 0xfc, 0xff, 0xff, 0xff, 0x61, 0x2c, 0xac,
            0x8c, 0xfc, 0xff, 0x7a, 0x60, 0x39, 0x4a, 0xc0, 0x8c, 0xfc, 0xff, 0xf5, 0xc0, 0x10,
            0x68, 0xd4, 0x8c, 0xfc, 0xff, 0x70, 0x21, 0xe8, 0x85, 0xe8, 0x8c, 0xfc, 0xff, 0xeb,
            0x81, 0xbf, 0xa3, 0xfc, 0x8c, 0xfc, 0xff, 0x66, 0xe2, 0x96, 0xc1, 0x10, 0x8d, 0xfc,
            0xff, 0x5c, 0xa3, 0x45, 0xfd, 0x38, 0x8d, 0xfc, 0xff, 0xd7, 0x03, 0x1d, 0x1b, 0x4d,
            0x8d, 0xfc, 0xff, 0x52, 0x64, 0xf4, 0x38, 0x61, 0x8d, 0xfc, 0xff, 0xcd, 0xc4, 0xcb,
            0x56, 0x75, 0x8d, 0xfc, 0xff, 0x48, 0x25, 0xa3, 0x74, 0x89, 0x8d, 0xfc, 0xff, 0xc3,
            0x85, 0x7a, 0x92, 0x9d, 0x8d, 0xfc, 0xff, 0xb9, 0x46, 0x29, 0xce, 0xc5, 0x8d, 0xfc,
            0xff, 0x34, 0xa7, 0x00, 0xec, 0xd9, 0x8d, 0xfc, 0xff, 0xaf, 0x07, 0xd8, 0x09, 0xee,
            0x8d, 0xfc, 0xff, 0x2a, 0x68, 0xaf, 0x27, 0x02, 0x8e, 0xfc, 0xff, 0xa5, 0xc8, 0x86,
            0x45, 0x16, 0x8e, 0xfc, 0xff, 0x20, 0x29, 0x5e, 0x63, 0x2a, 0x8e, 0xfc, 0xff, 0x16,
            0xea, 0x0c, 0x9f, 0x52, 0x8e, 0xfc, 0xff, 0x91, 0x4a, 0xe4, 0xbc, 0x66, 0x8e, 0xfc,
            0xff, 0x0c, 0xab, 0xbb, 0xda, 0x7a, 0x8e, 0xfc, 0xff, 0x87, 0x0b, 0x93, 0xf8, 0x8e,
            0x8e, 0xfc, 0xff, 0x02, 0x6c, 0x6a, 0x16, 0xa3, 0x8e, 0xfc, 0xff, 0x7d, 0xcc, 0x41,
            0x34, 0xb7, 0x8e, 0xfc, 0xff, 0x73, 0x8d, 0xf0, 0x6f, 0xdf, 0x8e, 0xfc, 0xff, 0xee,
            0xed, 0xc7, 0x8d, 0xf3, 0x8e, 0xfc, 0xff, 0x69, 0x4e, 0x9f, 0xab, 0x07, 0x8f, 0xfc,
            0xff, 0xe4, 0xae, 0x76, 0xc9, 0x1b, 0x8f, 0xfc, 0xff, 0x5f, 0x0f, 0x4e, 0xe7, 0x2f,
            0x8f, 0xfc, 0xff, 0xda, 0x6f, 0x25, 0x05, 0x44, 0x8f, 0xfc, 0xff, 0xd0, 0x30, 0xd4,
            0x40, 0x6c, 0x8f, 0xfc, 0xff, 0x4b, 0x91, 0xab, 0x5e, 0x80, 0x8f, 0xfc, 0xff, 0xc6,
            0xf1, 0x82, 0x7c, 0x94, 0x8f, 0xfc, 0xff, 0x41, 0x52, 0x5a, 0x9a, 0xa8, 0x8f, 0xfc,
            0xff, 0xbc, 0xb2, 0x31, 0xb8, 0xbc, 0x8f, 0xfc, 0xff, 0x37, 0x13, 0x09, 0xd6, 0xd0,
            0x8f, 0xfc, 0xff, 0x2d, 0xd4, 0xb7, 0x11, 0xf9, 0x8f, 0xfc, 0xff, 0xa8, 0x34, 0x8f,
            0x2f, 0x0d, 0x90, 0xfc, 0xff, 0x23, 0x95, 0x66, 0x4d, 0x21, 0x90, 0xfc, 0xff, 0x9e,
            0xf5, 0x3d, 0x6b, 0x35, 0x90, 0xfc, 0xff, 0x19, 0x56, 0x15, 0x89, 0x49, 0x90, 0xfc,
            0xff, 0x94, 0xb6, 0xec, 0xa6, 0x5d, 0x90, 0xfc, 0xff, 0x8a, 0x77, 0x9b, 0xe2, 0x85,
            0x90, 0xfc, 0xff, 0x05, 0xd8, 0x72, 0x00, 0x9a, 0x90, 0xfc, 0xff, 0x80, 0x38, 0x4a,
            0x1e, 0xae, 0x90, 0xfc, 0xff, 0xfb, 0x98, 0x21, 0x3c, 0xc2, 0x90, 0xfc, 0xff, 0x76,
            0xf9, 0xf8, 0x59, 0xd6, 0x90, 0xfc, 0xff, 0xf1, 0x59, 0xd0, 0x77, 0xea, 0x90, 0xfc,
            0xff, 0xe7, 0x1a, 0x7f, 0xb3, 0x12, 0x91, 0xfc, 0xff, 0x62, 0x7b, 0x56, 0xd1, 0x26,
            0x91, 0xfc, 0xff, 0xdd, 0xdb, 0x2d, 0xef, 0x3a, 0x91, 0xfc, 0xff, 0x58, 0x3c, 0x05,
            0x0d, 0x4f, 0x91, 0xfc, 0xff, 0xd3, 0x9c, 0xdc, 0x2a, 0x63, 0x91, 0xfc, 0xff, 0x4e,
            0xfd, 0xb3, 0x48, 0x77, 0x91, 0xfc, 0xff, 0x44, 0xbe, 0x62, 0x84, 0x9f, 0x91, 0xfc,
            0xff, 0xbf, 0x1e, 0x3a, 0xa2, 0xb3, 0x91, 0xfc, 0xff, 0x3a, 0x7f, 0x11, 0xc0, 0xc7,
            0x91, 0xfc, 0xff, 0xb5, 0xdf, 0xe8, 0xdd, 0xdb, 0x91, 0xfc, 0xff, 0x30, 0x40, 0xc0,
            0xfb, 0xef, 0x91, 0xfc, 0xff, 0xab, 0xa0, 0x97, 0x19, 0x04, 0x92, 0xfc, 0xff, 0xa1,
            0x61, 0x46, 0x55, 0x2c, 0x92, 0xfc, 0xff, 0x1c, 0xc2, 0x1d, 0x73, 0x40,
        ];
        assert_eq!(check_pages(chunk), Err(ENTRIES_TOO_MANY));
        // A set of nine booleans takes no bytes here: the stop byte that
        // follows it ends the struct, and the nine bytes after are left over.
        assert_eq!(check(&[0x8a, 0x92, 0, 0, 0, 0, 0, 0, 0, 0, 0]), Ok(()));
    }

    #[test]
    fn nested_collections_of_booleans_are_bounded_by_the_bytes() {
        // 100 lists of 100 booleans each, in a list of 100 lists: 10^4 values
        // of no bytes, more than 8 times the 107 bytes that hold them.
        let mut v = vec![0x19, 0xf9, 0x64];
        for _ in 0..100 {
            v.extend([0xf2, 0x64]);
        }
        v.push(0x00);
        assert_eq!(check(&v), Err(TOO_MUCH_WORK));
    }

    /// A fuzzed column chunk (`parquet_read`, nightly run): its first
    /// header holds field 3 as a binary of 11,456 bytes in a chunk of 459. arrow-rs
    /// reads it as a varint (the page size) and then a map of 253 million entries.
    #[test]
    fn takes_a_known_field_by_its_schema_type_not_its_wire_type() {
        let mut chunk = vec![
            0x38, 0xc0, 0x59, 0xf4, 0x2b, 0x67, 0x1f, 0x0f, 0x04, 0x5e, 0xfa, 0x54, 0x1b, 0x8b,
            0xc7, 0x95, 0xbd, 0x9b, 0x1b, 0xc3, 0xf1, 0x12, 0x08, 0xbb, 0xf9, 0x93, 0xf5, 0x78,
            0x17, 0x24, 0x9f, 0x7e, 0x35, 0x4a, 0x7c, 0xbe, 0xa4, 0x78, 0x6e, 0x4c, 0xe0, 0xa7,
            0xf6, 0xb6, 0x6a, 0xbc, 0xd2, 0x42, 0x64, 0x7c, 0xd6, 0xc4, 0xf7, 0x00, 0x42, 0x60,
            0x0e, 0xbf, 0x1d, 0xe8, 0x29, 0x44, 0xf5, 0xeb,
        ];
        chunk.resize(459, 0x41);
        // arrow-rs reads field 3 as a varint whatever its wire type says, so the walk
        // takes this header as arrow-rs does and reaches a map of 253 million
        // entries in it.
        assert_eq!(check_pages(&chunk), Err(ENTRIES_TOO_MANY));
    }

    /// A header: `type`, `uncompressed_page_size` and the page size with the
    /// given wire type byte for field 3, then `extra`, a stop and `n` bytes.
    fn page_with(f3: u8, n: u8, extra: &[u8]) -> Vec<u8> {
        let mut p = vec![0x15, 0x00, 0x15, n * 2, f3, n * 2];
        p.extend(extra);
        p.push(0x00);
        p.extend(std::iter::repeat_n(0xAB, usize::from(n)));
        p
    }

    /// A field 9 list of 2^31 - 1 booleans: the header that must be reached.
    const BOMB: [u8; 7] = [0x69, 0xf1, 0xff, 0xff, 0xff, 0xff, 0x07];

    #[test]
    fn reads_the_page_size_whatever_its_wire_type() {
        // Field 3 sent as an i64 (wire type 6): arrow-rs still reads the page
        // size from it, so the second header is the one it reads next.
        let mut chunk = page_with(0x16, 2, &[]);
        chunk.extend(page(2, &BOMB));
        assert_eq!(check_pages(&chunk), Err(COUNT_TOO_LARGE));
    }

    #[test]
    fn reads_a_known_struct_whatever_its_wire_type() {
        // Field 5 sent as a boolean (wire type 1): arrow-rs reads a data page
        // header from the bytes that follow, up to its own stop byte.
        let mut chunk = page_with(0x15, 2, &[0x21, 0x15, 0x02, 0x00]);
        chunk.extend(page(2, &BOMB));
        assert_eq!(check_pages(&chunk), Err(COUNT_TOO_LARGE));
    }

    #[test]
    fn truncates_a_full_field_id_to_16_bits() {
        // Field id 65539 given in full: arrow-rs keeps its low 16 bits, 3.
        let mut chunk = vec![
            0x15, 0x00, 0x15, 0x04, 0x05, 0x86, 0x80, 0x08, 0x04, 0x00, 0xAB, 0xAB,
        ];
        chunk.extend(page(2, &BOMB));
        assert_eq!(check_pages(&chunk), Err(COUNT_TOO_LARGE));
    }

    #[test]
    fn steps_over_a_uuid_as_arrow_rs_does() {
        // An unknown field 9 of type UUID (13), 16 bytes, then the next page.
        let mut uuid = vec![0x6d];
        uuid.extend([0x11; 16]);
        let mut chunk = page(2, &uuid);
        chunk.extend(page(2, &BOMB));
        assert_eq!(check_pages(&chunk), Err(COUNT_TOO_LARGE));
        // One that runs past the chunk.
        let mut short = page(0, &[0x6d, 0x11, 0x11]);
        short.truncate(short.len());
        assert_eq!(check_pages(&short), Err(BINARY_PAST_END));
    }

    #[test]
    fn ids_in_a_skipped_struct_do_not_end_the_walk() {
        // Unknown field 9 is a struct whose first field has the full id 32767
        // and is followed by a delta-1 field: arrow-rs skips it with the
        // previous id always 0, and goes on to the next header.
        let skipped = [0x6c, 0x05, 0xfe, 0xff, 0x03, 0x00, 0x15, 0x00, 0x00];
        let mut chunk = page(2, &skipped);
        chunk.extend(page(2, &BOMB));
        assert_eq!(check_pages(&chunk), Err(COUNT_TOO_LARGE));
    }

    #[test]
    fn the_work_budget_is_one_for_the_chunk() {
        // 300 headers, each with a list of 200 booleans: every list is smaller
        // than the bytes left, but together they cost far more than the bytes.
        let header = [
            0x15, 0x00, 0x15, 0x00, 0x15, 0x00, 0x69, 0xf2, 0xc8, 0x01, 0x00,
        ];
        let chunk: Vec<u8> = header
            .iter()
            .copied()
            .cycle()
            .take(header.len() * 300)
            .collect();
        assert_eq!(check_pages(&chunk), Err(TOO_MUCH_WORK));
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
        // A stop byte after the last page ends the walk with it.
        chunk.extend([0x00, 0x00, 0x15]);
        assert_eq!(check_pages(&chunk), Ok(()));
        // A header cut short by the end of the chunk: arrow-rs would read on
        // into the bytes after the chunk.
        assert_eq!(check_pages(&page(9, &[])[..5]), Err(HEADER_PAST_END));
        // A page that claims more bytes than the chunk holds: the next header
        // would be read from past the chunk.
        assert_eq!(check_pages(&page(9, &[])[..10]), Err(PAGE_PAST_END));
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
