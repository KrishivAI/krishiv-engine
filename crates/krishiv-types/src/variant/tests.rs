// SPDX-License-Identifier: Apache-2.0
//! Tests for the Variant codec: the Spark 4.2.0 ground-truth fixture table,
//! reader edge cases, error classes, width boundaries, size limits and
//! property tests (round-trip validity plus a no-panic fuzz).

use krishiv_common::error::{ErrorClass, KrishivError};
use proptest::prelude::*;

use super::{VARIANT_SIZE_LIMIT, VariantBuilder, VariantKind, VariantRef};

/// Decodes lowercase hex (the test fixtures' notation).
fn hex_bytes(hex: &str) -> Vec<u8> {
    assert!(hex.len().is_multiple_of(2), "odd hex: {hex}");
    let nibble = |c: u8| match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => panic!("bad hex char: {c}"),
    };
    hex.as_bytes()
        .chunks(2)
        .map(|pair| nibble(pair[0]) << 4 | nibble(pair[1]))
        .collect()
}

/// Whether a result is the reader's `MALFORMED_VARIANT` failure.
fn is_malformed<T>(result: Result<T, KrishivError>) -> bool {
    matches!(
        result.map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    )
}

/// Builds a `VariantRef` over leaked hand-written bytes.
fn read_pair(meta_hex: &str, value_hex: &str) -> VariantRef<'static> {
    let metadata: &'static [u8] = Box::leak(hex_bytes(meta_hex).into_boxed_slice());
    let value: &'static [u8] = Box::leak(hex_bytes(value_hex).into_boxed_slice());
    VariantRef::new(metadata, value).unwrap()
}

/// Builds `text` and checks the exact metadata/value bytes.
fn assert_bytes(text: &str, meta_hex: &str, value_hex: &str) -> VariantRef<'static> {
    let (metadata, value) =
        VariantBuilder::from_json(text).unwrap_or_else(|e| panic!("{text:?}: builder failed: {e}"));
    assert_eq!(metadata, hex_bytes(meta_hex), "{text:?} metadata");
    assert_eq!(value, hex_bytes(value_hex), "{text:?} value");
    // Leak the buffers so the ref can be returned with a static lifetime;
    // every assertion below only reads.
    let metadata: &'static [u8] = Box::leak(metadata.into_boxed_slice());
    let value: &'static [u8] = Box::leak(value.into_boxed_slice());
    VariantRef::new(metadata, value).unwrap_or_else(|e| panic!("{text:?}: reader failed: {e}"))
}

/// Ground truth observed on Spark 4.2.0 (see the module docs), matching the
/// task's fixture table byte-for-byte.
#[test]
fn spark_ground_truth_bytes() {
    // Scalars.
    assert_eq!(
        assert_bytes("null", "010000", "00").kind(),
        VariantKind::Null
    );
    assert_eq!(assert_bytes("true", "010000", "04").as_bool(), Ok(true));
    assert_eq!(assert_bytes("false", "010000", "08").as_bool(), Ok(false));
    assert_eq!(assert_bytes("\"\"", "010000", "01").as_str(), Ok(""));
    // Integers: smallest fitting width.
    assert_eq!(assert_bytes("1", "010000", "0c01").as_i64(), Ok(1));
    assert_eq!(assert_bytes("-128", "010000", "0c80").as_i64(), Ok(-128));
    assert_eq!(assert_bytes("128", "010000", "108000").as_i64(), Ok(128));
    assert_eq!(assert_bytes("-129", "010000", "107fff").as_i64(), Ok(-129));
    assert_eq!(
        assert_bytes("40000", "010000", "14409c0000").as_i64(),
        Ok(40_000)
    );
    assert_eq!(
        assert_bytes("3000000000", "010000", "18005ed0b200000000").as_i64(),
        Ok(3_000_000_000)
    );
    assert_eq!(assert_bytes("-0", "010000", "0c00").as_i64(), Ok(0));
    // Big integers: decimal past int64, double past 38 digits.
    let big23 = assert_bytes(
        "12345678901234567890123",
        "010000",
        "2800cb444271764eb6429d02000000000000",
    );
    assert_eq!(
        big23.kind(),
        VariantKind::Decimal {
            precision: 23,
            scale: 0
        }
    );
    assert_eq!(
        assert_bytes(
            "123456789012345678901234567890123456789",
            "010000",
            "1c800558693a38d747"
        )
        .kind(),
        VariantKind::Double
    );
    // Fractions: decimals with the scale as written.
    let one = assert_bytes("1.0", "010000", "20010a000000");
    assert_eq!(
        one.kind(),
        VariantKind::Decimal {
            precision: 2,
            scale: 1
        }
    );
    assert_eq!(one.as_decimal(), Ok((10, 1)));
    let one_fifty = assert_bytes("1.50", "010000", "200296000000");
    assert_eq!(one_fifty.as_decimal(), Ok((150, 2)));
    let neg_zero = assert_bytes("-0.0", "010000", "200100000000");
    assert_eq!(neg_zero.as_decimal(), Ok((0, 1)));
    let tenth = assert_bytes("0.1000", "010000", "2004e8030000");
    assert_eq!(tenth.as_decimal(), Ok((1000, 4)));
    let micro = assert_bytes("0.000001", "010000", "200601000000");
    assert_eq!(micro.as_decimal(), Ok((1, 6)));
    let wide8 = assert_bytes("1234567890.12", "010000", "2402141a99be1c000000");
    assert_eq!(
        wide8.kind(),
        VariantKind::Decimal {
            precision: 12,
            scale: 2
        }
    );
    let wide16 = assert_bytes(
        "1234567890123456789.5",
        "010000",
        "2801d70a1feb8ca954ab0000000000000000",
    );
    assert_eq!(
        wide16.kind(),
        VariantKind::Decimal {
            precision: 20,
            scale: 1
        }
    );
    let wide16b = assert_bytes(
        "123456789012345678.9",
        "010000",
        "28011581e97df41022110000000000000000",
    );
    assert_eq!(
        wide16b.kind(),
        VariantKind::Decimal {
            precision: 19,
            scale: 1
        }
    );
    assert_eq!(
        assert_bytes(
            "12345678901234567890.123456789012345678901",
            "010000",
            "1ce1639d31956ae543"
        )
        .kind(),
        VariantKind::Double
    );
    // Exponents always mean double, with overflow to infinity.
    assert_eq!(
        assert_bytes("1e2", "010000", "1c0000000000005940").as_f64(),
        Ok(100.0)
    );
    assert_eq!(
        assert_bytes("1.0e1", "010000", "1c0000000000002440").as_f64(),
        Ok(10.0)
    );
    assert_eq!(
        assert_bytes("1e400", "010000", "1c000000000000f07f").as_f64(),
        Ok(f64::INFINITY)
    );
    let max = assert_bytes("1.7976931348623157E308", "010000", "1cffffffffffffef7f");
    assert_eq!(max.as_f64(), Ok(f64::MAX));
    // Strings.
    assert_eq!(
        assert_bytes("\"hi\"", "010000", "096869").as_str(),
        Ok("hi")
    );
    let long70 = "x".repeat(70);
    let long70_json = format!("\"{long70}\"");
    let long70_hex = format!("4046000000{}", "78".repeat(70));
    assert_eq!(
        assert_bytes(&long70_json, "010000", &long70_hex).as_str(),
        Ok(long70.as_str())
    );
    assert_eq!(
        assert_bytes("\"a\\\"b\u{e9}\\n\"", "010000", "19612262c3a90a").as_str(),
        Ok("a\"b\u{e9}\n")
    );
    assert_eq!(
        assert_bytes("\"\u{1f600}\"", "010000", "11f09f9880").as_str(),
        Ok("\u{1f600}")
    );
    // Empty containers.
    assert_eq!(assert_bytes("[]", "010000", "030000").len(), Ok(0));
    assert_eq!(assert_bytes("{}", "010000", "020000").len(), Ok(0));
    // Arrays.
    let arr = assert_bytes("[1,\"a\",null]", "010000", "0303000204050c01056100");
    assert_eq!(arr.len(), Ok(3));
    assert_eq!(arr.get_index(0).unwrap().as_i64(), Ok(1));
    assert_eq!(arr.get_index(1).unwrap().as_str(), Ok("a"));
    assert_eq!(arr.get_index(2).unwrap().kind(), VariantKind::Null);
    let nested = assert_bytes("[[],{}]", "010000", "0302000306030000020000");
    assert_eq!(nested.len(), Ok(2));
    assert_eq!(nested.get_index(0).unwrap().len(), Ok(0));
    assert_eq!(nested.get_index(1).unwrap().len(), Ok(0));
    // Objects: ids sorted by key bytes, values in input order.
    let obj = assert_bytes(
        "{\"b\":1,\"a\":[1,2.5,\"x\",null],\"c\":{\"d\":true}}",
        "0104000102030462616364",
        "02030100020200141a0c0103040002080a0b0c01200119000000057800020103000104",
    );
    assert_eq!(obj.len(), Ok(3));
    assert_eq!(obj.field_key_at(0), Ok("a"));
    assert_eq!(obj.field_key_at(1), Ok("b"));
    assert_eq!(obj.field_key_at(2), Ok("c"));
    assert_eq!(obj.get_field("b").unwrap().unwrap().as_i64(), Ok(1));
    let arr_a = obj.get_field("a").unwrap().unwrap();
    assert_eq!(arr_a.len(), Ok(4));
    assert_eq!(arr_a.get_index(1).unwrap().as_decimal(), Ok((25, 1)));
    assert_eq!(arr_a.get_index(2).unwrap().as_str(), Ok("x"));
    assert_eq!(
        obj.get_field("c")
            .unwrap()
            .unwrap()
            .get_field("d")
            .unwrap()
            .unwrap()
            .as_bool(),
        Ok(true)
    );
    assert!(obj.get_field("missing").unwrap().is_none());
    let obj2 = assert_bytes(
        "{\"a\":{\"a\":1},\"b\":[{\"a\":2}]}",
        "01020001026162",
        "0202000100071202010000020c010301000702010000020c02",
    );
    assert_eq!(
        obj2.get_field("a")
            .unwrap()
            .unwrap()
            .get_field("a")
            .unwrap()
            .unwrap()
            .as_i64(),
        Ok(1)
    );
    assert_eq!(
        obj2.get_field("b")
            .unwrap()
            .unwrap()
            .get_index(0)
            .unwrap()
            .get_field("a")
            .unwrap()
            .unwrap()
            .as_i64(),
        Ok(2)
    );
    let obj3 = assert_bytes(
        "{\"é\":1,\"a\":{\"é\":2}}",
        "0102000203c3a961",
        "020201000200090c0102010000020c02",
    );
    assert_eq!(obj3.get_field("é").unwrap().unwrap().as_i64(), Ok(1));
    assert_eq!(
        obj3.get_field("a")
            .unwrap()
            .unwrap()
            .get_field("é")
            .unwrap()
            .unwrap()
            .as_i64(),
        Ok(2)
    );
    let ctrl = assert_bytes("{\"k\":\"\\u0001\"}", "010100016b", "02010000020501");
    assert_eq!(ctrl.get_field("k").unwrap().unwrap().as_str(), Ok("\u{1}"));
    // Trailing text after the first value is ignored.
    assert_bytes("{\"a\":1} x", "0101000161", "02010000020c01");
}

/// Malformed inputs fail as `MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION`
/// with Spark's parameters (`badRecord` is the whole input).
#[test]
fn malformed_inputs_are_rejected_with_spark_parameters() {
    let cases = [
        "{\"a\":1,\"a\":2}",
        "{\"a\":",
        "01",
        "NaN",
        "[1,]",
        "1.",
        ".5",
        "-",
        "",
        "   ",
        "{\"a\"}",
        "{\"a\":1,}",
        "[,]",
        "[1",
        "{\"a\":1",
        "\"abc",
        "\"\\ud800\"",
        "\"ab\\ud800\"",
        "\"\\udc00\\ud800\"",
        "\"a\x01b\"",
        "00",
        "00.1",
        "2.e3",
        "1e",
        "1e+",
        "+1",
        "- 1",
        "nul",
        "NULL",
        "True",
        "nullx",
        "12ab",
        "0,",
        "{1:2}",
        "{'a':1}",
        "[1 2]",
        "{\"a\":{\"b\":1,\"b\":2}}",
        "null\u{e9}",
    ];
    for text in cases {
        let err = VariantBuilder::from_json(text).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::MalformedRecordInParsingWithoutSuggestion,
            "{text:?}"
        );
        let params = err.message_parameters();
        assert!(
            params.contains(&("badRecord", text.to_owned())),
            "{text:?}: {params:?}"
        );
        assert!(
            params.contains(&("failFastMode", "FAILFAST".to_owned())),
            "{text:?}: {params:?}"
        );
    }
}

/// Well-formed inputs with ignored trailing text or extra whitespace.
#[test]
fn trailing_text_and_whitespace_rules() {
    for text in [
        "  1  ",
        "\t\n{\"a\":1}\r\n",
        "1 2",
        "{\"a\":1} {\"b\":2}",
        "{\"a\":1}x",
        "12 ab",
        "null null",
        "\"a\"\"b\"",
        "null,",
        "true]",
        "[1]x",
        "\"a\" ,",
        "[] ,",
    ] {
        VariantBuilder::from_json(text).unwrap_or_else(|e| panic!("{text:?}: should parse: {e}"));
    }
    // A number must end at whitespace or end of input; a literal must not be
    // followed by an identifier character.
    for text in [
        "0,",
        "0]",
        "0}",
        "0:",
        "0\"",
        "null1",
        "true1",
        "falsex",
        "null_",
        "nul\u{e9}",
        "null$",
        "true$",
        "false$",
    ] {
        assert_eq!(
            VariantBuilder::from_json(text).map_err(|e| e.error_class()),
            Err(ErrorClass::MalformedRecordInParsingWithoutSuggestion),
            "{text:?}"
        );
    }
}

/// Integer, decimal-width and scale boundaries.
#[test]
fn number_boundaries() {
    // A 9-digit integer is still an integer, not a decimal.
    assert_eq!(
        assert_bytes("999999999", "010000", "14ffc99a3b").as_i64(),
        Ok(999_999_999)
    );
    assert_eq!(
        assert_bytes("9999999999", "010000", "18ffe30b5402000000").as_i64(),
        Ok(9_999_999_999)
    );
    assert_eq!(
        assert_bytes("2147483647", "010000", "14ffffff7f").as_i64(),
        Ok(2_147_483_647)
    );
    assert_eq!(
        assert_bytes("2147483648", "010000", "180000008000000000").as_i64(),
        Ok(2_147_483_648)
    );
    assert_eq!(
        assert_bytes("-2147483649", "010000", "18ffffff7fffffffff").as_i64(),
        Ok(-2_147_483_649)
    );
    assert_eq!(
        assert_bytes("9223372036854775807", "010000", "18ffffffffffffff7f").as_i64(),
        Ok(9_223_372_036_854_775_807)
    );
    // Width follows max(precision, scale).
    assert_eq!(
        assert_bytes("99999999.9", "010000", "2001ffc99a3b").kind(),
        VariantKind::Decimal {
            precision: 9,
            scale: 1
        }
    );
    assert_eq!(
        assert_bytes("999999999.9", "010000", "2401ffe30b5402000000").kind(),
        VariantKind::Decimal {
            precision: 10,
            scale: 1
        }
    );
    assert_eq!(
        assert_bytes("0.000000001", "010000", "200901000000").kind(),
        VariantKind::Decimal {
            precision: 1,
            scale: 9
        }
    );
    assert_eq!(
        assert_bytes("0.00000000001", "010000", "240b0100000000000000").kind(),
        VariantKind::Decimal {
            precision: 1,
            scale: 11
        }
    );
    // Scale 38 stays decimal; scale 39 falls back to double.
    assert_eq!(
        assert_bytes("0.00000000001", "010000", "240b0100000000000000").as_decimal(),
        Ok((1, 11))
    );
    let s38 = "0.".to_owned() + &"0".repeat(37) + "1";
    assert_eq!(
        VariantBuilder::from_json(&s38).unwrap().1,
        hex_bytes("282601000000000000000000000000000000")
    );
    let s39 = "0.".to_owned() + &"0".repeat(38) + "1";
    assert_eq!(
        VariantRef::new(
            &[0x01, 0x00, 0x00],
            &VariantBuilder::from_json(&s39).unwrap().1
        )
        .unwrap()
        .kind(),
        VariantKind::Double
    );
    // Precision 38 with scale 0 is the widest decimal; 39 digits is double.
    let p38 = "9".repeat(38);
    assert_eq!(VariantBuilder::from_json(&p38).unwrap().1[0], 0x28);
    let p39 = "9".repeat(39);
    assert_eq!(VariantBuilder::from_json(&p39).unwrap().1[0], 0x1C);
}

/// Short strings hold up to 63 bytes; 64 bytes switch to the long form.
#[test]
fn string_length_boundary() {
    let short63 = "y".repeat(63);
    let (meta, value) = VariantBuilder::from_json(&format!("\"{short63}\"")).unwrap();
    assert_eq!(meta, hex_bytes("010000"));
    assert_eq!(value[0], 0xFD);
    assert_eq!(value.len(), 64);
    let long64 = "y".repeat(64);
    let (meta, value) = VariantBuilder::from_json(&format!("\"{long64}\"")).unwrap();
    assert_eq!(meta, hex_bytes("010000"));
    assert_eq!(value, hex_bytes(&format!("4040000000{}", "79".repeat(64))));
    let back = VariantRef::new(&meta, &value).unwrap();
    assert_eq!(back.as_str(), Ok(long64.as_str()));
}

/// The `is_large` bit flips past 255 elements; field-id widths are minimal.
#[test]
fn container_size_boundaries() {
    let object = |n: usize| {
        let doc = format!(
            "{{{}}}",
            (0..n)
                .map(|i| format!("\"k{i:03}\":{i}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        VariantBuilder::from_json(&doc).unwrap()
    };
    let (meta, value) = object(255);
    assert_eq!(
        value[0], 0x06,
        "255 fields: small, 1-byte ids, 2-byte offsets"
    );
    assert_eq!(VariantRef::new(&meta, &value).unwrap().len(), Ok(255));
    let (meta, value) = object(256);
    assert_eq!(
        value[0], 0x46,
        "256 fields: large, 1-byte ids (max id 255), 2-byte offsets"
    );
    assert_eq!(VariantRef::new(&meta, &value).unwrap().len(), Ok(256));
    let (meta, value) = object(300);
    assert_eq!(
        value[0], 0x56,
        "300 fields: large, 2-byte ids, 2-byte offsets"
    );
    let back = VariantRef::new(&meta, &value).unwrap();
    assert_eq!(back.len(), Ok(300));
    assert_eq!(back.get_field("k299").unwrap().unwrap().as_i64(), Ok(299));

    let array =
        |n: usize| VariantBuilder::from_json(&format!("[{}]", vec!["1"; n].join(","))).unwrap();
    assert_eq!(array(255).1[0], 0x07);
    assert_eq!(array(256).1[0], 0x17);
    assert_eq!(
        VariantRef::new(&array(256).0, &array(256).1).unwrap().len(),
        Ok(256)
    );
}

/// Metadata offset widths follow the largest dictionary offset.
#[test]
fn metadata_offset_widths() {
    // One 255-byte key: offsets [0, 255] fit one byte.
    let (meta, _) = VariantBuilder::from_json(&format!("{{\"{}\":1}}", "k".repeat(255))).unwrap();
    assert_eq!(meta[0], 0x01);
    // One 256-byte key: offsets [0, 256] need two bytes.
    let (meta, _) = VariantBuilder::from_json(&format!("{{\"{}\":1}}", "k".repeat(256))).unwrap();
    assert_eq!(meta[0], 0x41);
    assert_eq!(meta.len(), 1 + 2 + 2 * 2 + 256);
    // 75 KiB of keys need three-byte offsets.
    let doc = format!(
        "{{{}}}",
        (0..300)
            .map(|i| format!("\"k{i:05}{}\":1", "x".repeat(250)))
            .collect::<Vec<_>>()
            .join(",")
    );
    let (meta, value) = VariantBuilder::from_json(&doc).unwrap();
    assert_eq!(meta[0], 0x81);
    let back = VariantRef::new(&meta, &value).unwrap();
    assert_eq!(back.len(), Ok(300));
}

/// Three-byte value offsets (data past 64 KiB).
#[test]
fn value_offset_widths() {
    let doc = format!("[{}]", vec!["1"; 40_000].join(","));
    let (meta, value) = VariantBuilder::from_json(&doc).unwrap();
    assert_eq!(value[0], 0x1B, "large array, 3-byte offsets");
    let back = VariantRef::new(&meta, &value).unwrap();
    assert_eq!(back.len(), Ok(40_000));
    assert_eq!(back.get_index(39_999).unwrap().as_i64(), Ok(1));
}

/// Four-byte metadata and value offsets (past 16 MiB).
#[test]
fn wide_offsets() {
    // 18 MiB of dictionary keys: 4-byte metadata offsets.
    let doc = format!(
        "{{{}}}",
        (0..400)
            .map(|i| format!("\"k{i:05}{}\":1", "x".repeat(45_000)))
            .collect::<Vec<_>>()
            .join(",")
    );
    let (meta, value) = VariantBuilder::from_json(&doc).unwrap();
    assert_eq!(meta[0], 0xC1);
    assert_eq!(VariantRef::new(&meta, &value).unwrap().len(), Ok(400));
    // 17 MiB of array data: 4-byte value offsets.
    let doc = format!("[{}]", vec!["1"; 8_500_000].join(","));
    let (meta, value) = VariantBuilder::from_json(&doc).unwrap();
    assert_eq!(value[0] & 0x03, 3);
    assert_eq!((value[0] >> 2) & 0x03, 3, "4-byte offsets");
    assert_eq!(VariantRef::new(&meta, &value).unwrap().len(), Ok(8_500_000));
}

/// Deep nesting stays iterative in the reader while the builder shares
/// Spark 4.2.0's 1000-level cap (depth 1000 works for arrays and objects,
/// depth 1001 fails as malformed).
#[test]
fn deep_nesting_is_iterative() {
    // Depth 1000 arrays work and read back iteratively.
    let doc = format!("{}{}{}", "[".repeat(1000), "1", "]".repeat(1000));
    let (meta, value) = VariantBuilder::from_json(&doc).unwrap();
    let mut back = VariantRef::new(&meta, &value).unwrap();
    for _ in 0..1000 {
        back = back.get_index(0).unwrap();
    }
    assert_eq!(back.as_i64(), Ok(1));
    // Depth 1000 objects work and read back through `get_field`.
    let doc = format!("{}{}{}", "{\"a\":".repeat(1000), "1", "}".repeat(1000));
    let (meta, value) = VariantBuilder::from_json(&doc).unwrap();
    let mut back = VariantRef::new(&meta, &value).unwrap();
    for _ in 0..1000 {
        back = back.get_field("a").unwrap().unwrap();
    }
    assert_eq!(back.as_i64(), Ok(1));
    // Depth 1001 fails for both shapes with Spark's malformed-record error.
    for doc in [
        format!("{}{}{}", "[".repeat(1001), "1", "]".repeat(1001)),
        format!("{}{}{}", "{\"a\":".repeat(1001), "1", "}".repeat(1001)),
    ] {
        assert_eq!(
            VariantBuilder::from_json(&doc).map_err(|e| e.error_class()),
            Err(ErrorClass::MalformedRecordInParsingWithoutSuggestion),
        );
    }
}

/// Hand-built values for kinds the builder never emits, plus malformed
/// variants of every shape the validator must catch.
#[test]
fn reader_primitive_kinds() {
    fn read_value(value_hex: &str) -> VariantRef<'static> {
        read_pair("010000", value_hex)
    }
    let read = read_value;
    assert_eq!(read("380000c03f").as_f32(), Ok(1.5_f32));
    assert_eq!(read("2c39300000").as_date(), Ok(12_345));
    assert_eq!(
        read("30c0ba8a3cd5620400").as_timestamp_micros(),
        Ok(1_234_567_890_123_456)
    );
    assert_eq!(read("34ffffffffffffffff").as_timestamp_micros(), Ok(-1));
    assert_eq!(
        read("3c0400000000ff4142").as_bytes(),
        Ok([0x00, 0xFF, 0x41, 0x42].as_slice())
    );
    assert_eq!(
        read("5012345678123456781234567812345678").as_uuid(),
        Ok(hex_bytes("12345678123456781234567812345678")
            .try_into()
            .unwrap())
    );
    // Spec ids 17-19 (time, nanosecond timestamps): reserved by Spark 4.2.0.
    assert_eq!(
        read(&format!("44{}", "00".repeat(8))).as_time_micros(),
        Ok(0)
    );
    assert_eq!(
        read(&format!("48{}", "00".repeat(8))).as_timestamp_nanos(),
        Ok(0)
    );
    assert_eq!(
        read(&format!("4c{}", "00".repeat(8))).as_timestamp_nanos(),
        Ok(0)
    );
    assert_eq!(read("44ffffffffffffffff").as_time_micros(), Ok(-1));
    // Wrong-kind getters fail with MALFORMED_VARIANT, never a panic.
    let one = read("0c01");
    assert_eq!(
        one.as_bool().map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    );
    assert_eq!(
        one.as_f64().map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    );
    assert_eq!(
        one.as_str().map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    );
    assert_eq!(
        one.len().map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    );
    assert_eq!(
        read("04").as_i64().map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    );
    assert_eq!(
        read("00").as_bool().map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    );
}

/// The reader accepts every metadata offset width and the large-bit forms.
#[test]
fn reader_widths_and_large_forms() {
    // Two-byte metadata offsets: dict ["a"].
    let back = read_pair("4101000000010061", "00");
    assert_eq!(back.kind(), VariantKind::Null);
    // Three-byte metadata offsets.
    let back = read_pair("8101000000000001000061", "00");
    assert_eq!(back.kind(), VariantKind::Null);
    // Four-byte metadata offsets.
    let back = read_pair("c101000000000000000100000061", "00");
    assert_eq!(back.kind(), VariantKind::Null);
    // Small object with 2-byte field ids (the encoding allows wider ids than
    // the minimum).
    let back = read_pair("0101000161", "1201000000020c01");
    assert_eq!(back.len(), Ok(1));
    assert_eq!(back.get_field("a").unwrap().unwrap().as_i64(), Ok(1));
    // Small object with 3-byte field ids.
    let back = read_pair("0101000161", "220100000000020c01");
    assert_eq!(back.get_field("a").unwrap().unwrap().as_i64(), Ok(1));
    // Small array with 3-byte offsets.
    let back = read_pair("0101000161", "0b0100000001000000");
    assert_eq!(back.len(), Ok(1));
    assert_eq!(back.get_index(0).unwrap().kind(), VariantKind::Null);
    // Large empty object and array.
    let back = read_pair("0101000161", "420000000000");
    assert_eq!(back.len(), Ok(0));
    let back = read_pair("0101000161", "130000000000");
    assert_eq!(back.len(), Ok(0));
}

/// Malformed bytes yield MALFORMED_VARIANT, never a panic.
#[test]
fn reader_rejects_malformed_bytes() {
    let bad_values = [
        "",
        "0c",
        "1000",
        "1400",
        "14000000",
        "18ffffffffffffff",
        "1c000000",
        "2001",
        "200100",
        "240100000000000000",
        "2801000000000000000000000000000000",
        "2000ffffff7f",
        "200a01000000",
        "2400ffffffffffffff7f",
        "24130100000000000000",
        "2800ffffffffffffffffffffffffffffff7f",
        "28270100000000000000000000000000000000",
        "5000",
        "4001000000",
        "4002",
        "4002000000",
        "0968",
        "fc",
        "54",
        "ff",
        "c0",
        "0200",
        "0201",
        "020100",
        "02010000",
        "0201000001",
        "0201000002000c",
        "0201000002000c0100",
        "0300",
        "0301",
        "030100",
        "03010000000c01",
        "0c0100",
    ];
    for value in bad_values {
        assert!(
            is_malformed(VariantRef::new(&hex_bytes("010000"), &hex_bytes(value))),
            "value {value}"
        );
    }
    let bad_metas = [
        "",
        "00",
        "02",
        "11",
        "21",
        "31",
        "810000",
        "c1",
        "0101",
        "010100",
        "010100036162",
        "010200010002006162",
        "010100010062aa",
        "11020001026261",
        "11020001026161",
    ];
    for meta in bad_metas {
        assert!(
            is_malformed(VariantRef::new(&hex_bytes(meta), &hex_bytes("00"))),
            "meta {meta}"
        );
    }
    // Sorted-flagged metadata with sorted unique keys is accepted.
    let back = read_pair("11020001026162", "020200010002040c010c02");
    assert_eq!(back.len(), Ok(2));
    assert_eq!(back.get_field("a").unwrap().unwrap().as_i64(), Ok(1));
    assert_eq!(back.get_field("b").unwrap().unwrap().as_i64(), Ok(2));
    // Field id outside the dictionary.
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("010000"),
        &hex_bytes("020101000500030c01")
    )));
    // Array offsets must be monotonic here (the writer always emits them so).
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("010000"),
        &hex_bytes("030200040002000c010c02")
    )));
    // Object ids out of key order are malformed.
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("01020001026162"),
        &hex_bytes("020201000002040c010c02")
    )));
    // Duplicate object ids are malformed.
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("01020001026261"),
        &hex_bytes("020200000100050c010c0200")
    )));
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("01020001026162"),
        &hex_bytes("02020000000002040c010c02")
    )));
    // Out-of-range and wrong-kind access on valid values.
    let back = read_pair("010000", "0303000204050c01056100");
    assert!(is_malformed(back.get_index(3)));
    assert!(is_malformed(back.field_at(0)));
    assert!(back.get_field("a").unwrap().is_none());
    let obj = read_pair("0101000161", "02010000020c01");
    assert!(is_malformed(obj.get_index(0)));
    assert!(is_malformed(obj.field_at(1)));
    assert_eq!(
        obj.field_id_at(1).map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    );
    assert_eq!(
        obj.field_key_at(1).map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    );
}

/// Decimal widths reject unscaled values with more digits than they hold and
/// scales past the same maximum.
#[test]
fn decimal_bounds_are_validated() {
    // decimal4 holds 9 digits: 2147483647 has 10.
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("010000"),
        &hex_bytes("2000ffffff7f")
    )));
    // decimal4 scale past 9.
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("010000"),
        &hex_bytes("200a01000000")
    )));
    // decimal8 holds 18 digits: i64::MAX has 19.
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("010000"),
        &hex_bytes("2400ffffffffffffff7f")
    )));
    // decimal8 scale past 18.
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("010000"),
        &hex_bytes("24130100000000000000")
    )));
    // decimal16 holds 38 digits: 10^38 has 39.
    let mut decimal16 = vec![0x28, 0x00];
    decimal16.extend_from_slice(&10_i128.pow(38).to_le_bytes());
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("010000"),
        &decimal16
    )));
    // decimal16 at i128::MAX (39 digits) is malformed.
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("010000"),
        &hex_bytes("2800ffffffffffffffffffffffffffffff7f")
    )));
    // decimal16 scale 39 past the 38 maximum.
    assert!(is_malformed(VariantRef::new(
        &hex_bytes("010000"),
        &hex_bytes("28270100000000000000000000000000000000")
    )));
}

/// `get_field` distinguishes an absent key from corrupt bytes.
#[test]
fn get_field_reports_corruption() {
    let obj = read_pair("0101000161", "02010000020c01");
    assert!(obj.get_field("missing").unwrap().is_none());
    assert!(
        read_pair("010000", "0303000204050c01056100")
            .get_field("a")
            .unwrap()
            .is_none()
    );
    // A present key whose value bytes are truncated errors.
    let corrupt = read_pair("0101000161", "02010000010c");
    assert_eq!(
        corrupt
            .get_field("a")
            .map(|_| ())
            .map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedVariant)
    );
}

/// Spark-accepted quirk the reader deliberately tolerates: unsorted object
/// offsets (Spark emits values in input order) pass validation and read back.
/// Duplicate or out-of-key-order field ids are rejected (see
/// `reader_rejects_malformed_bytes`).
#[test]
fn reader_tolerates_spark_quirks() {
    // {"b":1,"a":2}: ids [1,0] (key order), offsets [2,0,4] (input order).
    let back = read_pair("01020001026261", "020201000200040c010c02");
    assert_eq!(back.get_field("a").unwrap().unwrap().as_i64(), Ok(2));
    assert_eq!(back.get_field("b").unwrap().unwrap().as_i64(), Ok(1));
}

/// The value buffer is capped at 128 MiB with Spark's error and parameters.
/// The aggregate limit is reached with many strings each below the 20,000,000
/// UTF-16-unit per-string limit.
#[test]
fn variant_size_limit() {
    assert_eq!(VARIANT_SIZE_LIMIT, 128 * 1024 * 1024);
    // 127 one-MiB strings plus a tail that fills the buffer exactly: small
    // array (2 header bytes), `(128 + 1) * 4` offset bytes, and each long
    // string costs 5 header bytes plus its payload.
    let one_mib = "x".repeat(1_048_576);
    let mut exact = String::with_capacity(134_300_000);
    exact.push('[');
    for _ in 0..127 {
        exact.push('"');
        exact.push_str(&one_mib);
        exact.push('"');
        exact.push(',');
    }
    let tail = "x".repeat(1_047_418);
    exact.push('"');
    exact.push_str(&tail);
    exact.push('"');
    exact.push(']');
    let (meta, value) = VariantBuilder::from_json(&exact).unwrap();
    assert_eq!(value.len(), VARIANT_SIZE_LIMIT);
    assert_eq!(VariantRef::new(&meta, &value).unwrap().len(), Ok(128));
    // One payload byte more exceeds the aggregate limit.
    let mut over = String::with_capacity(134_300_000);
    over.push('[');
    for _ in 0..127 {
        over.push('"');
        over.push_str(&one_mib);
        over.push('"');
        over.push(',');
    }
    let tail = "x".repeat(1_047_419);
    over.push('"');
    over.push_str(&tail);
    over.push('"');
    over.push(']');
    let err = VariantBuilder::from_json(&over).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::VariantSizeLimit);
    let params = err.message_parameters();
    assert!(
        params.contains(&("sizeLimit", "128.0 MiB".to_owned())),
        "{params:?}"
    );
    assert!(
        params.contains(&("functionName", "`parse_json`".to_owned())),
        "{params:?}"
    );
}

/// Spark 4.2.0 string-value limits: 20,000,000 UTF-16 code units parse,
/// 20,000,001 fail; a char outside the BMP counts 2.
#[test]
fn string_value_limits() {
    let ok = format!("\"{}\"", "x".repeat(20_000_000));
    VariantBuilder::from_json(&ok).unwrap();
    let over = format!("\"{}\"", "x".repeat(20_000_001));
    assert_eq!(
        VariantBuilder::from_json(&over).map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedRecordInParsingWithoutSuggestion)
    );
    // Outside the BMP counts 2 units: 19,999,998 `x` plus one emoji is
    // exactly 20,000,000 units and parses.
    let ok_emoji = format!("\"{}{}\"", "x".repeat(19_999_998), "\u{1F600}");
    VariantBuilder::from_json(&ok_emoji).unwrap();
    // One more `x` is 20,000,001 units and fails.
    let over_emoji = format!("\"{}{}\"", "x".repeat(19_999_999), "\u{1F600}");
    assert_eq!(
        VariantBuilder::from_json(&over_emoji).map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedRecordInParsingWithoutSuggestion)
    );
    // Ten million emoji are 20,000,000 units and parse; one more fails.
    let ok_many = format!("\"{}\"", "\u{1F600}".repeat(10_000_000));
    VariantBuilder::from_json(&ok_many).unwrap();
    let over_many = format!("\"{}\"", "\u{1F600}".repeat(10_000_001));
    assert_eq!(
        VariantBuilder::from_json(&over_many).map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedRecordInParsingWithoutSuggestion)
    );
}

/// Spark 4.2.0 field-name limits: 50,000 UTF-16 code units parse, 50,001
/// fail; 50,000 `é` parse, so the count is characters, not bytes.
#[test]
fn field_name_limits() {
    let ok = format!("{{\"{}\":1}}", "a".repeat(50_000));
    VariantBuilder::from_json(&ok).unwrap();
    let over = format!("{{\"{}\":1}}", "a".repeat(50_001));
    assert_eq!(
        VariantBuilder::from_json(&over).map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedRecordInParsingWithoutSuggestion)
    );
    // Multi-byte BMP chars count as one unit, not by bytes.
    let ok_e = format!("{{\"{}\":1}}", "\u{e9}".repeat(50_000));
    VariantBuilder::from_json(&ok_e).unwrap();
    let over_e = format!("{{\"{}\":1}}", "\u{e9}".repeat(50_001));
    assert_eq!(
        VariantBuilder::from_json(&over_e).map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedRecordInParsingWithoutSuggestion)
    );
    // Outside the BMP counts 2 units: 49,998 `a` plus one emoji is 50,000.
    let ok_emoji = format!("{{\"{}{}\":1}}", "a".repeat(49_998), "\u{1F600}");
    VariantBuilder::from_json(&ok_emoji).unwrap();
    let over_emoji = format!("{{\"{}{}\":1}}", "a".repeat(49_999), "\u{1F600}");
    assert_eq!(
        VariantBuilder::from_json(&over_emoji).map_err(|e| e.error_class()),
        Err(ErrorClass::MalformedRecordInParsingWithoutSuggestion)
    );
}

/// Spark 4.2.0 number digit limits: integer-part digits above 1000 fail (a
/// leading `-` is not counted), and integer plus fraction digits above 1001
/// fail; exponent digits are not limited.
#[test]
fn number_digit_limits() {
    for text in [
        "9".repeat(1000),
        format!("-{}", "9".repeat(1000)),
        format!("0.{}", "0".repeat(1000)),
        format!("1.{}", "0".repeat(1000)),
        format!("1e{}", "0".repeat(999)),
    ] {
        VariantBuilder::from_json(&text)
            .unwrap_or_else(|e| panic!("{text:.40}...: should parse: {e}"));
    }
    for text in [
        "9".repeat(1001),
        format!("-{}", "9".repeat(1001)),
        format!("0.{}", "0".repeat(1001)),
        format!("{}.{}", "9".repeat(1000), "0".repeat(1000)),
        format!("{}.1", "9".repeat(1001)),
    ] {
        assert_eq!(
            VariantBuilder::from_json(&text).map_err(|e| e.error_class()),
            Err(ErrorClass::MalformedRecordInParsingWithoutSuggestion),
            "{text:.40}..."
        );
    }
}

/// Iterating a large array through `get_index` exercises the cached
/// validated container header instead of rescanning it per access.
#[test]
fn array_iteration_is_linear() {
    let doc = format!("[{}]", vec!["1"; 100_000].join(","));
    let (meta, value) = VariantBuilder::from_json(&doc).unwrap();
    let back = VariantRef::new(&meta, &value).unwrap();
    assert_eq!(back.len(), Ok(100_000));
    let mut sum: i64 = 0;
    for i in 0..100_000 {
        sum += back.get_index(i).unwrap().as_i64().unwrap();
    }
    assert_eq!(sum, 100_000);
}

/// Random JSON documents for the round-trip property test.
#[derive(Debug, Clone)]
enum Doc {
    /// JSON `null`.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// Integer lexeme (optional `-` plus digits, no leading zeros).
    Int(String),
    /// Fraction lexeme `int.frac` (no exponent).
    Dec {
        /// Integer digits (may carry a leading `-`).
        int: String,
        /// Fraction digits.
        frac: String,
    },
    /// Exponent lexeme.
    Float(String),
    /// Decoded string value.
    Str(String),
    /// Array elements.
    Arr(Vec<Doc>),
    /// Object entries (keys may repeat; rendering keeps the first).
    Obj(Vec<(String, Doc)>),
}

/// Expected value, computed independently from the builder's rules.
#[derive(Debug, Clone, PartialEq)]
enum Expect {
    /// SQL null.
    Null,
    /// Boolean.
    Bool(bool),
    /// Integer (fits `i64`).
    Int(i64),
    /// Decimal: unscaled value, scale, precision.
    Dec(i128, u8, u8),
    /// Double: exact bits.
    Float(u64),
    /// String value.
    Str(String),
    /// Array elements in order.
    Arr(Vec<Expect>),
    /// Object entries in first-seen order.
    Obj(Vec<(String, Expect)>),
}

/// JSON-escapes `s` into `out` (decoded string to JSON string literal).
fn escape_into(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                out.push_str("\\u");
                for shift in [12, 8, 4, 0] {
                    let nibble = (u32::from(c) >> shift) & 0xF;
                    out.push(char::from_digit(nibble, 16).unwrap_or('?'));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Renders a document to JSON text (object duplicates keep the first, so the
/// builder never sees a duplicate key).
fn render_doc(doc: &Doc) -> String {
    let mut out = String::new();
    render_into(&mut out, doc);
    out
}

/// Renders a document to JSON text.
fn render_into(out: &mut String, doc: &Doc) {
    match doc {
        Doc::Null => out.push_str("null"),
        Doc::Bool(true) => out.push_str("true"),
        Doc::Bool(false) => out.push_str("false"),
        Doc::Int(lexeme) | Doc::Float(lexeme) => out.push_str(lexeme),
        Doc::Dec { int, frac } => {
            out.push_str(int);
            out.push('.');
            out.push_str(frac);
        }
        Doc::Str(string) => escape_into(out, string),
        Doc::Arr(elements) => {
            out.push('[');
            for (i, element) in elements.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                render_into(out, element);
            }
            out.push(']');
        }
        Doc::Obj(entries) => {
            out.push('{');
            let mut first = true;
            let mut seen: Vec<&str> = Vec::new();
            for (key, value) in entries {
                if seen.contains(&key.as_str()) {
                    continue;
                }
                seen.push(key);
                if !first {
                    out.push(',');
                }
                first = false;
                escape_into(out, key);
                out.push(':');
                render_into(out, value);
            }
            out.push('}');
        }
    }
}

/// Significant digits of an integer-plus-fraction digit pair (leading zeros
/// free, all-zero counts as 1), matching `BigDecimal.precision`.
fn decimal_precision(int_digits: &str, frac_digits: &str) -> usize {
    let int = int_digits.strip_prefix('-').unwrap_or(int_digits);
    let combined = format!("{int}{frac_digits}");
    let stripped = combined.trim_start_matches('0');
    if stripped.is_empty() {
        1
    } else {
        stripped.len()
    }
}

/// Classifies an integer lexeme the way the builder does.
fn expect_int(lexeme: &str) -> Expect {
    let digits = lexeme.strip_prefix('-').unwrap_or(lexeme);
    if digits.len() > 38 {
        return Expect::Float(lexeme.parse::<f64>().unwrap_or(f64::NAN).to_bits());
    }
    // At most 38 digits always fit `i128`.
    let magnitude: i128 = digits.parse().unwrap_or(0);
    let value = if lexeme.starts_with('-') {
        -magnitude
    } else {
        magnitude
    };
    match i64::try_from(value) {
        Ok(int) => Expect::Int(int),
        Err(_) => Expect::Dec(value, 0, u8::try_from(digits.len()).unwrap_or(u8::MAX)),
    }
}

/// Classifies a fraction lexeme the way the builder does.
fn expect_dec(int: &str, frac: &str) -> Expect {
    let lexeme = format!("{int}.{frac}");
    let precision = decimal_precision(int, frac);
    let scale = frac.len();
    if precision > 38 || scale > 38 {
        return Expect::Float(lexeme.parse::<f64>().unwrap_or(f64::NAN).to_bits());
    }
    let digits = format!("{}{}", int.strip_prefix('-').unwrap_or(int), frac);
    let unscaled: i128 = digits.parse().unwrap_or(0);
    let unscaled = if int.starts_with('-') {
        -unscaled
    } else {
        unscaled
    };
    Expect::Dec(
        unscaled,
        u8::try_from(scale).unwrap_or(u8::MAX),
        u8::try_from(precision).unwrap_or(u8::MAX),
    )
}

/// Classifies a document the way the builder does (duplicate object keys
/// keep the first, so rendering never produces duplicates).
fn expect_doc(doc: &Doc) -> Expect {
    match doc {
        Doc::Null => Expect::Null,
        Doc::Bool(bool) => Expect::Bool(*bool),
        Doc::Int(lexeme) => expect_int(lexeme),
        Doc::Dec { int, frac } => expect_dec(int, frac),
        Doc::Float(lexeme) => Expect::Float(lexeme.parse::<f64>().unwrap_or(f64::NAN).to_bits()),
        Doc::Str(string) => Expect::Str(string.clone()),
        Doc::Arr(elements) => Expect::Arr(elements.iter().map(expect_doc).collect()),
        Doc::Obj(entries) => {
            let mut seen: Vec<&str> = Vec::new();
            let mut out = Vec::new();
            for (key, value) in entries {
                if seen.contains(&key.as_str()) {
                    continue;
                }
                seen.push(key);
                out.push((key.clone(), expect_doc(value)));
            }
            Expect::Obj(out)
        }
    }
}

/// Checks a built value against its independent expectation.
fn check_value(back: &VariantRef<'_>, expected: &Expect) {
    match expected {
        Expect::Null => assert_eq!(back.kind(), VariantKind::Null),
        Expect::Bool(bool) => {
            assert_eq!(back.kind(), VariantKind::Bool);
            assert_eq!(back.as_bool(), Ok(*bool));
        }
        Expect::Int(int) => {
            assert!(matches!(
                back.kind(),
                VariantKind::Int8 | VariantKind::Int16 | VariantKind::Int32 | VariantKind::Int64
            ));
            assert_eq!(back.as_i64(), Ok(*int));
        }
        Expect::Dec(unscaled, scale, precision) => {
            assert_eq!(
                back.kind(),
                VariantKind::Decimal {
                    precision: *precision,
                    scale: *scale
                }
            );
            assert_eq!(back.as_decimal(), Ok((*unscaled, *scale)));
        }
        Expect::Float(bits) => {
            assert_eq!(back.kind(), VariantKind::Double);
            assert_eq!(back.as_f64().map(f64::to_bits), Ok(*bits));
        }
        Expect::Str(string) => {
            assert_eq!(back.kind(), VariantKind::String);
            assert_eq!(back.as_str(), Ok(string.as_str()));
        }
        Expect::Arr(items) => {
            assert_eq!(back.kind(), VariantKind::Array);
            assert_eq!(back.len(), Ok(items.len()));
            for (i, item) in items.iter().enumerate() {
                check_value(&back.get_index(i).unwrap(), item);
            }
        }
        Expect::Obj(fields) => {
            assert_eq!(back.kind(), VariantKind::Object);
            assert_eq!(back.len(), Ok(fields.len()));
            for (key, value) in fields {
                let child = back.get_field(key).unwrap().unwrap();
                check_value(&child, value);
            }
        }
    }
}

/// A digit string of exactly `len` digits (no leading zero unless `len` is
/// 1).
fn digits(len: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(0..10_u8, len).prop_map(move |values| {
        let mut out = String::with_capacity(len);
        for (i, digit) in values.iter().enumerate() {
            let mut byte = b'0' + digit;
            if i == 0 && len > 1 && byte == b'0' {
                byte = b'1';
            }
            out.push(char::from(byte));
        }
        out
    })
}

/// An integer lexeme: small values, boundaries, big and huge magnitudes.
fn arb_int_lex() -> impl Strategy<Value = String> {
    let big = |len: usize| {
        digits(len).prop_flat_map(|magnitude| {
            any::<bool>().prop_map(move |neg| {
                if neg {
                    format!("-{magnitude}")
                } else {
                    magnitude.clone()
                }
            })
        })
    };
    prop_oneof![
        any::<i64>().prop_map(|value| value.to_string()),
        big(19),
        big(20),
        big(30),
        big(38),
        big(39),
        big(45),
    ]
}

/// A fraction magnitude: unsigned integer and fraction digits.
fn arb_dec_parts() -> impl Strategy<Value = (String, String)> {
    ((1..20_usize), (1..40_usize))
        .prop_flat_map(|(int_len, frac_len)| (digits(int_len), digits(frac_len)))
}

/// A fraction lexeme as `(int_digits, frac_digits)`; the integer part may be
/// negated (rendering keeps the sign on the integer part).
fn arb_dec_lex() -> impl Strategy<Value = (String, String)> {
    (any::<bool>(), arb_dec_parts())
        .prop_map(|(neg, (int, frac))| (if neg { format!("-{int}") } else { int }, frac))
}

/// An exponent lexeme, including overflow and underflow magnitudes.
fn arb_float_lex() -> impl Strategy<Value = String> {
    let exp = prop_oneof![
        Just("e2"),
        Just("e-3"),
        Just("E+10"),
        Just("e0"),
        Just("e400"),
        Just("e-400"),
        Just("E308")
    ];
    let base = prop_oneof![
        digits(3),
        arb_dec_parts().prop_map(|(int, frac)| format!("{int}.{frac}")),
    ];
    (base, exp, any::<bool>()).prop_map(|(base, exp, neg)| {
        if neg {
            format!("-{base}{exp}")
        } else {
            format!("{base}{exp}")
        }
    })
}

/// Interesting strings: empty, ASCII, multi-byte, escapes, boundaries.
fn arb_str() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::new()),
        Just("a".to_owned()),
        Just("hi".to_owned()),
        Just("\u{e9}".to_owned()),
        Just("\u{1f600}".to_owned()),
        Just("a\"b\\c".to_owned()),
        Just("x".repeat(63)),
        Just("x".repeat(64)),
        Just("x".repeat(200)),
        prop::collection::vec(any::<char>(), 0..24)
            .prop_map(|chars| chars.into_iter().collect::<String>()),
    ]
}

/// Interesting keys: tiny alphabets (dictionary reuse), unicode, empty.
fn arb_key() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("a".to_owned()),
        Just("b".to_owned()),
        Just("c".to_owned()),
        Just("abc".to_owned()),
        Just(String::new()),
        Just("\u{e9}".to_owned()),
        Just("k".repeat(300)),
        prop::collection::vec(prop_oneof![Just('a'), Just('b')], 1..5)
            .prop_map(|chars| chars.into_iter().collect::<String>()),
    ]
}

/// A random JSON document (depth-bounded); object keys may repeat (rendering
/// keeps the first so the builder never sees duplicates).
fn arb_doc() -> impl Strategy<Value = Doc> {
    let leaf = prop_oneof![
        Just(Doc::Null),
        any::<bool>().prop_map(Doc::Bool),
        arb_int_lex().prop_map(Doc::Int),
        arb_dec_lex().prop_map(|(int, frac)| Doc::Dec { int, frac }),
        arb_float_lex().prop_map(Doc::Float),
        arb_str().prop_map(Doc::Str),
    ];
    leaf.prop_recursive(3, 32, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Doc::Arr),
            prop::collection::vec((arb_key(), inner), 0..4).prop_map(Doc::Obj),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Every generated document builds to bytes the reader accepts, with the
    /// expected structure and leaf values.
    #[test]
    fn builder_round_trip(doc in arb_doc()) {
        let text = render_doc(&doc);
        let (meta, value) = VariantBuilder::from_json(&text)
            .unwrap_or_else(|e| panic!("{text:?}: builder failed: {e}"));
        let back = VariantRef::new(&meta, &value)
            .unwrap_or_else(|e| panic!("{text:?}: reader failed: {e}"));
        check_value(&back, &expect_doc(&doc));
    }

    /// Arbitrary bytes into the reader and every accessor: no panics, only
    /// `MALFORMED_VARIANT` (proptest fails the test on a panic).
    #[test]
    fn fuzz_never_panics(
        meta in prop::collection::vec(any::<u8>(), 0..96),
        value in prop::collection::vec(any::<u8>(), 0..96),
    ) {
        if let Ok(back) = VariantRef::new(&meta, &value) {
        let _ = back.kind();
        let _ = back.metadata();
        let _ = back.value_bytes();
        let _ = back.is_empty();
        if let Ok(len) = back.len() {
            for i in 0..len.saturating_add(2) {
                let _ = back.get_index(i);
                let _ = back.field_at(i);
                let _ = back.field_id_at(i);
                let _ = back.field_key_at(i);
            }
            // One level of nesting on whatever children exist.
            for i in 0..len {
                if let Ok(child) = back.get_index(i).or_else(|_| back.field_at(i)) {
                    let _ = child.kind();
                    let _ = child.len();
                    let _ = child.get_index(0);
                    let _ = child.field_at(0);
                    let _ = child.get_field("a");
                }
            }
        }
        for key in ["", "a", "b", "\u{e9}", "\u{1f600}", "\u{0}", "zzzz"] {
            let _ = back.get_field(key);
        }
        let _ = back.as_bool();
        let _ = back.as_i64();
        let _ = back.as_f32();
        let _ = back.as_f64();
        let _ = back.as_decimal();
        let _ = back.as_str();
        let _ = back.as_bytes();
        let _ = back.as_uuid();
        let _ = back.as_date();
        let _ = back.as_timestamp_micros();
        let _ = back.as_timestamp_nanos();
        let _ = back.as_time_micros();
        }
    }
}
