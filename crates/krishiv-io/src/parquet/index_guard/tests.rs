// SPDX-License-Identifier: Apache-2.0
//! The guard of the page index and Bloom filters: real files from arrow-rs
//! for the accepted cases, hand-built Thrift for the hostile ones.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;

use super::*;

const ROWS: i64 = 1000;
const ROW_COUNT: u64 = 1000;
const PAGE_ROWS: usize = 100;
const PAGES: usize = 10;

/// A one-column file of `ROWS` sorted rows in pages of `PAGE_ROWS`, with
/// page statistics (so a column and an offset index) and a Bloom filter.
fn file() -> (Bytes, Arc<ParquetMetaData>) {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from_iter_values(0..ROWS))],
    )
    .expect("batch");
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .set_bloom_filter_enabled(true)
        .build();
    let mut out = Vec::new();
    {
        let mut w = ArrowWriter::try_new(&mut out, schema, Some(props)).expect("writer");
        w.write(&batch).expect("write");
        w.close().expect("close");
    }
    let bytes = Bytes::from(out);
    let meta = ParquetMetaDataReader::new()
        .parse_and_finish(&bytes)
        .expect("footer");
    (bytes, Arc::new(meta))
}

fn chunk_of(meta: &ParquetMetaData) -> Range<u64> {
    let (start, len) = meta.row_group(0).column(0).byte_range();
    start..start + len
}

fn slice(bytes: &Bytes, r: &Range<u64>) -> Vec<u8> {
    bytes[usize::try_from(r.start).expect("start")..usize::try_from(r.end).expect("end")].to_vec()
}

// ---- a tiny compact-protocol writer for hostile inputs ----

fn varint(mut n: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = u8::try_from(n & 0x7f).expect("7 bits");
        n >>= 7;
        if n == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

fn zigzag(n: i64) -> Vec<u8> {
    varint(u64::from_ne_bytes(((n << 1) ^ (n >> 63)).to_ne_bytes()))
}

/// One `PageLocation`: offset, compressed size and first row index.
fn location(offset: i64, size: i64, first_row: i64) -> Vec<u8> {
    let mut out = vec![0x16];
    out.extend(zigzag(offset));
    out.push(0x15);
    out.extend(zigzag(size));
    out.push(0x16);
    out.extend(zigzag(first_row));
    out.push(0x00);
    out
}

/// An `OffsetIndex` of the given `(offset, size, first_row)` pages.
fn offset_index_bytes(pages: &[(i64, i64, i64)]) -> Vec<u8> {
    let mut out = vec![0x19];
    // A list header: size below 15 in the high nibble, element type STRUCT.
    assert!(pages.len() < 15);
    out.push(u8::try_from(pages.len()).expect("small") << 4 | 0x0c);
    for (o, s, r) in pages {
        out.extend(location(*o, *s, *r));
    }
    out.push(0x00);
    out
}

// ---- accepted cases ----

#[test]
fn a_real_file_passes_every_check() {
    let (bytes, meta) = file();
    let col = meta.row_group(0).column(0);
    let chunk = chunk_of(&meta);
    let d = declared(col, bytes.len() as u64, &chunk).expect("ranges");
    let (ci, oi, bf) = (
        d.column_index.expect("column index"),
        d.offset_index.expect("offset index"),
        d.bloom.expect("bloom"),
    );
    let offsets = offset_index(&slice(&bytes, &oi), &chunk, ROW_COUNT).expect("offset index");
    let pages = offsets.page_locations().len();
    assert_eq!(pages, PAGES);
    let columns =
        column_index(&slice(&bytes, &ci), PhysicalType::INT64, pages).expect("column index");
    assert_eq!(usize::try_from(columns.num_pages()), Ok(pages));
    let (bitset, header) = bloom_header(&slice(&bytes, &bf), &bf).expect("bloom header");
    assert!(bitset >= 32 && bitset % 32 == 0);
    assert_eq!(header as u64 + bitset, bf.end - bf.start);
    sparse_pages(
        &slice(&bytes, &chunk),
        chunk.start,
        chunk.start,
        offsets.page_locations(),
    )
    .expect("page headers");
}

#[test]
fn a_chunk_with_chunk_statistics_only_has_no_column_index() {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from_iter_values(0..10))],
    )
    .expect("batch");
    let props = WriterProperties::builder()
        .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk)
        .build();
    let mut out = Vec::new();
    let mut w = ArrowWriter::try_new(&mut out, schema, Some(props)).expect("writer");
    w.write(&batch).expect("write");
    w.close().expect("close");
    let bare = Bytes::from(out);
    let m = ParquetMetaDataReader::new()
        .parse_and_finish(&bare)
        .expect("footer");
    let chunk = chunk_of(&m);
    let d = declared(m.row_group(0).column(0), bare.len() as u64, &chunk).expect("ranges");
    // arrow-rs writes an offset index whatever the statistics level; the
    // column index and the Bloom filter are what the settings turn off.
    assert!(d.column_index.is_none() && d.bloom.is_none(), "{d:?}");
    assert!(d.offset_index.is_some());
}

// ---- declared ranges ----

#[test]
fn ranges_outside_the_file_or_over_the_data_are_rejected() {
    let file_size = 1000;
    let chunk = 100..400;
    let ok = range("column index", Some(500), Some(50), file_size, &chunk);
    assert_eq!(ok, Ok(Some(500..550)));
    for (offset, length, why) in [
        (Some(-1), Some(10), "negative"),
        (Some(500), Some(0), "not positive"),
        (Some(500), Some(-3), "not positive"),
        (Some(990), Some(20), "past the end of the file"),
        (Some(i64::MAX), Some(i32::MAX), "past the end of the file"),
        (Some(350), Some(100), "overlaps"),
        (Some(50), Some(100), "overlaps"),
        (Some(500), None, "half a range"),
        (None, Some(5), "half a range"),
    ] {
        let err = range("column index", offset, length, file_size, &chunk).expect_err(why);
        assert!(err.0.contains(why), "{offset:?} {length:?}: {err:?}");
    }
    // A Bloom filter that predates its length is simply not used.
    assert_eq!(
        range("Bloom filter", Some(500), None, file_size, &chunk),
        Ok(None)
    );
}

// ---- offset index ----

#[test]
fn a_bad_offset_index_is_rejected() {
    let chunk = 100..1000;
    let good = offset_index_bytes(&[(100, 100, 0), (200, 100, 50), (300, 100, 90)]);
    assert!(offset_index(&good, &chunk, 100).is_ok());
    for (pages, why) in [
        (
            vec![(100, 100, 0), (950, 100, 50)],
            "past the end of its column chunk",
        ),
        (vec![(100, 100, 0), (150, 100, 50)], "not increasing"),
        (
            vec![(100, 100, 0), (200, 100, 50), (300, 100, 50)],
            "first row indexes are not increasing",
        ),
        (
            vec![(100, 100, 0), (200, 100, 40), (300, 100, 20)],
            "first row indexes are not increasing",
        ),
        (vec![(100, 100, 5)], "does not start at row 0"),
        (vec![(100, 100, 0), (200, 100, 100)], "past the last row"),
        (vec![(100, 0, 0)], "not positive"),
        (vec![(-5, 10, 0)], "negative"),
    ] {
        let err = offset_index(&offset_index_bytes(&pages), &chunk, 100).expect_err(why);
        assert!(err.0.contains(why), "{pages:?}: {err:?}");
    }
    assert!(offset_index(&offset_index_bytes(&[]), &chunk, 100).is_err());
}

#[test]
fn a_huge_list_count_is_rejected_before_decoding() {
    // field 1 is a list of 2^31 booleans in two bytes of data.
    let mut bytes = vec![0x19, 0xf2];
    bytes.extend(varint(1 << 31));
    bytes.push(0x00);
    for f in [
        offset_index(&bytes, &(0..10), 10).map(|_| ()),
        column_index(&bytes, PhysicalType::INT64, 1).map(|_| ()),
    ] {
        let err = f.expect_err("a huge count");
        assert!(err.0.contains("larger than the bytes left"), "{err:?}");
    }
    // And truncated or empty bytes are errors, never panics.
    for bytes in [&[][..], &[0x19][..], &[0x19, 0x1c][..], &[0xff; 40][..]] {
        assert!(offset_index(bytes, &(0..10), 10).is_err());
        assert!(column_index(bytes, PhysicalType::INT32, 1).is_err());
    }
}

#[test]
fn the_column_index_must_match_the_offset_index() {
    let (bytes, meta) = file();
    let col = meta.row_group(0).column(0);
    let chunk = chunk_of(&meta);
    let d = declared(col, bytes.len() as u64, &chunk).expect("ranges");
    let ci = slice(&bytes, &d.column_index.expect("column index"));
    let err = column_index(&ci, PhysicalType::INT64, 3).expect_err("a different count");
    assert!(err.0.contains("different pages"), "{err:?}");
    // Read as another physical type is a decode error or a type mismatch, not a panic.
    let _ = column_index(&ci, PhysicalType::BYTE_ARRAY, 10);
}

// ---- Bloom header ----

/// A `BloomFilterHeader` of `num_bytes` with the three one-variant unions.
fn bloom_header_bytes(num_bytes: i64) -> Vec<u8> {
    let mut out = vec![0x15];
    out.extend(zigzag(num_bytes));
    for _ in 0..3 {
        out.extend([0x1c, 0x1c, 0x00, 0x00]);
    }
    out.push(0x00);
    out
}

#[test]
fn a_bloom_header_declaring_a_bad_size_is_rejected() {
    let ok = bloom_header_bytes(64);
    let declared = 0..(ok.len() as u64 + 64);
    assert_eq!(bloom_header(&ok, &declared), Ok((64, ok.len())));
    for (size, why) in [
        (0_i64, "whole 32-byte blocks"),
        (33, "whole 32-byte blocks"),
        (1 << 31, "larger than 128 MiB"),
        (BLOOM_MAX_BYTES as i64 + 32, "larger than 128 MiB"),
        (-32, "negative"),
        (128, "do not fit"),
    ] {
        let err = bloom_header(&bloom_header_bytes(size), &declared).expect_err(why);
        assert!(err.0.contains(why), "{size}: {err:?}");
    }
    // A header that does not start with numBytes, and a truncated one.
    let mut other = ok.clone();
    other[0] = 0x25;
    assert!(bloom_header(&other, &declared).is_err());
    assert!(bloom_header(&ok[..3], &declared).is_err());
    assert!(bloom_header(&[], &declared).is_err());
}

// ---- sparse page headers ----

/// A page header of `compressed` bytes of data, with an unknown field 20
/// holding a `list<bool>` of `count` elements.
fn page_with_list(compressed: i64, count: u64) -> Vec<u8> {
    let mut out = vec![0x15, 0x00, 0x15];
    out.extend(zigzag(compressed));
    out.push(0x15);
    out.extend(zigzag(compressed));
    out.push(0x09);
    out.extend(zigzag(20));
    out.push(0xf2);
    out.extend(varint(count));
    out.push(0x00);
    out
}

#[test]
fn a_hostile_page_header_inside_a_sparse_range_is_rejected() {
    let mut page = page_with_list(4, 1 << 31);
    page.extend([0; 4]);
    let location = PageLocation {
        offset: 1000,
        compressed_page_size: i32::try_from(page.len()).expect("small"),
        first_row_index: 0,
    };
    // The page lies inside the pushed bytes: it is walked and refused.
    let err = sparse_pages(&page, 1000, 1000, std::slice::from_ref(&location)).expect_err("walked");
    assert!(err.0.contains("larger than the bytes left"), "{err:?}");
    // A page outside the pushed range is not this call's to check.
    assert!(sparse_pages(&page, 5000, 5000, std::slice::from_ref(&location)).is_ok());
    // A location that overflows is skipped, not a panic.
    let wild = PageLocation {
        offset: i64::MAX,
        compressed_page_size: i32::MAX,
        first_row_index: 0,
    };
    assert!(sparse_pages(&page, 1000, 1000, &[wild]).is_ok());
    // Among many pages, the one the bytes hold is found and walked; the
    // pages before and after it are not this range's.
    let len = i32::try_from(page.len()).expect("small");
    let many: Vec<PageLocation> = (0..1000_i64)
        .map(|i| PageLocation {
            offset: 1000 + i * i64::from(len),
            compressed_page_size: len,
            first_row_index: i,
        })
        .collect();
    let at = 1000 + 500 * u64::try_from(len).expect("positive");
    let err = sparse_pages(&page, at, 1000, &many).expect_err("walked");
    assert!(err.0.contains("larger than the bytes left"), "{err:?}");
    // Bytes that start inside a page hold no whole page: nothing to walk.
    assert!(sparse_pages(&page[1..], at + 1, 1000, &many).is_ok());
}

/// A `ColumnIndex` of two INT64 pages whose `nan_counts` list has one
/// element: arrow-rs indexes it by page, so the guard must refuse it.
#[test]
fn a_short_nan_counts_list_is_rejected() {
    let le = |v: i64| v.to_le_bytes().to_vec();
    let binary = |b: Vec<u8>| {
        let mut out = varint(b.len() as u64);
        out.extend(b);
        out
    };
    let list_header = |n: u8, elem: u8| (n << 4) | elem;
    let mut bytes = Vec::new();
    // 1: null_pages = [false, false]
    bytes.extend([0x19, list_header(2, 2), 2, 2]);
    // 2: min_values, 3: max_values (binary)
    for field_delta in [1_u8, 1] {
        bytes.push((field_delta << 4) | 0x09);
        bytes.push(list_header(2, 8));
        bytes.extend(binary(le(1)));
        bytes.extend(binary(le(2)));
    }
    // 4: boundary_order = 0 (i32)
    bytes.extend([0x15, 0x00]);
    // 5: null_counts = [0, 0]
    bytes.extend([0x19, list_header(2, 6)]);
    bytes.extend(zigzag(0));
    bytes.extend(zigzag(0));
    // 8: nan_counts = [0] (one element for two pages); field delta 3
    bytes.extend([0x39, list_header(1, 6)]);
    bytes.extend(zigzag(0));
    bytes.push(0x00);
    let err = column_index(&bytes, PhysicalType::INT64, 2).expect_err("short nan counts");
    assert!(err.0.contains("NaN counts"), "{err:?}");
}

/// A `ColumnIndex` whose `null_pages` list holds three booleans (a byte
/// each for the decoder) and whose maximum bounds declare 2^31-1 elements:
/// the walk must reach that count at the right bytes and refuse it.
#[test]
fn a_huge_count_after_null_pages_is_reached_and_refused() {
    let mut bytes = vec![0x19, (3 << 4) | 2, 2, 2, 2];
    // 2: min_values, three 8-byte binaries
    bytes.extend([0x19, (3 << 4) | 8]);
    for _ in 0..3 {
        bytes.push(8);
        bytes.extend([0; 8]);
    }
    // 3: max_values declaring 2^31 - 1 elements
    bytes.extend([0x19, 0xf8]);
    bytes.extend(varint((1 << 31) - 1));
    bytes.push(0x00);
    let err = column_index(&bytes, PhysicalType::INT64, 3).expect_err("a huge count");
    assert!(err.0.contains("larger than the bytes left"), "{err:?}");
}

/// An unknown boolean-list field is skipped unread, as arrow-rs skips it:
/// consuming its bytes would lose alignment with the decoder (a reviewer's
/// 11-byte payload whose later bytes the decoder reads as a 2^31-element
/// list). The guard must still refuse it.
#[test]
fn an_unknown_boolean_list_is_walked_like_the_decoder_does() {
    let bytes = [
        0x09, 0x28, 0x72, 0x19, 0xf2, 0xff, 0xff, 0xff, 0xff, 0x07, 0x00,
    ];
    let err = column_index(&bytes, PhysicalType::INT64, 1).expect_err("a huge count");
    assert!(err.0.contains("larger than the bytes left"), "{err:?}");
}

/// A Bloom header that does not name the block / XXH64 / uncompressed
/// filter the bitset is read as is refused, so it cannot prune.
#[test]
fn a_bloom_header_of_another_kind_is_refused() {
    let declared = 0..4096;
    let ok = bloom_header_bytes(64);
    assert!(bloom_header(&ok, &declared).is_ok());
    // The compression union holds member 2 (a field the reader does not know).
    let mut other_compression = ok;
    let n = other_compression.len();
    other_compression[n - 4] = 0x2c;
    assert!(bloom_header(&other_compression, &declared).is_err());
    // The algorithm is missing.
    let mut missing = vec![0x15];
    missing.extend(zigzag(64));
    for _ in 0..2 {
        missing.extend([0x1c, 0x1c, 0x00, 0x00]);
    }
    missing.push(0x00);
    assert!(bloom_header(&missing, &declared).is_err());
}
