// SPDX-License-Identifier: Apache-2.0
//! Tests of Spark's partition path segments: the escape table, the
//! `CAST(v AS STRING)` rendering and the directory layout.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Int32Array,
    Int64Array, LargeBinaryArray, StringArray, StringViewArray, TimestampMicrosecondArray,
};

use super::{
    DEFAULT_PARTITION, check_partition_column, escape_partition_value, partition_dir_name,
    partition_segments, partition_texts, spark_partitioned_file_name,
};

/// Every ASCII code point against the expected table, written out: the
/// C0 controls U+0001..U+001F, DEL, and `" # % ' * / : = ? \ { [ ] ^`
/// escape; U+0000, `}`, space and everything else stays.
#[test]
fn the_escape_table_covers_all_ascii() {
    const ESCAPES: &str = "\"#%'*/:=?\\{[]^";
    for byte in 0_u8..=127 {
        let c = byte as char;
        let text = c.to_string();
        let got = escape_partition_value(&text);
        let escapable = (0x01..=0x1F).contains(&byte) || byte == 0x7F || ESCAPES.contains(c);
        if escapable {
            assert_eq!(got, format!("%{byte:02X}"), "byte {byte:#04X}");
        } else {
            assert_eq!(got, text, "byte {byte:#04X}");
        }
    }
    // Notably kept: NUL, `}`, space, `+`, `,`, `;`, `(`, `)`, `-`, `.`, `_`.
    for kept in ["\0", "}", " ", "+", ",", ";", "()", "-._"] {
        assert_eq!(escape_partition_value(kept), kept);
    }
}

#[test]
fn non_ascii_and_empty_stay_or_default() {
    assert_eq!(escape_partition_value("héllo \u{1F600}"), "héllo \u{1F600}");
    // NULL and the empty string both become the default partition.
    let nulls: ArrayRef = Arc::new(StringArray::from(vec![None, Some(""), Some("x")]));
    let segments = partition_segments(std::slice::from_ref(&nulls), Some(0)).unwrap();
    assert_eq!(
        segments,
        vec![
            vec![DEFAULT_PARTITION.to_owned()],
            vec![DEFAULT_PARTITION.to_owned()],
            vec!["x".to_owned()],
        ]
    );
}

#[test]
fn values_render_as_cast_to_string() {
    let texts = |column: ArrayRef| partition_texts(&column, Some(0)).unwrap();
    assert_eq!(
        texts(Arc::new(StringViewArray::from(vec!["a", "b c"]))),
        [Some("a".to_owned()), Some("b c".to_owned())]
    );
    assert_eq!(
        texts(Arc::new(Int32Array::from(vec![Some(7), Some(-42), None]))),
        [Some("7".to_owned()), Some("-42".to_owned()), None]
    );
    assert_eq!(
        texts(Arc::new(Int64Array::from(vec![i64::MIN]))),
        [Some(i64::MIN.to_string())]
    );
    let decimal = Decimal128Array::from(vec![Some(12345), Some(-5), None])
        .with_precision_and_scale(10, 2)
        .unwrap();
    assert_eq!(
        texts(Arc::new(decimal)),
        [Some("123.45".to_owned()), Some("-0.05".to_owned()), None]
    );
    assert_eq!(
        texts(Arc::new(Date32Array::from(vec![0, -1]))),
        [Some("1970-01-01".to_owned()), Some("1969-12-31".to_owned())]
    );
    assert_eq!(
        texts(Arc::new(
            TimestampMicrosecondArray::from(vec![0, 1_500_000]).with_timezone("UTC")
        )),
        [
            Some("1970-01-01 00:00:00".to_owned()),
            Some("1970-01-01 00:00:01.5".to_owned())
        ]
    );
    assert_eq!(
        texts(Arc::new(BooleanArray::from(vec![true, false]))),
        [Some("true".to_owned()), Some("false".to_owned())]
    );
}

#[test]
fn a_value_with_slash_and_equals_stays_one_segment() {
    let column: ArrayRef = Arc::new(StringArray::from(vec!["a/b=c"]));
    let segments = partition_segments(std::slice::from_ref(&column), Some(0)).unwrap();
    assert_eq!(segments, vec![vec!["a%2Fb%3Dc".to_owned()]]);
    assert_eq!(
        partition_dir_name(&["k".to_owned()], &segments[0]),
        "k=a%2Fb%3Dc"
    );
}

#[test]
fn names_follow_spark_byte_for_byte() {
    assert_eq!(
        spark_partitioned_file_name(0, "uuid", 0, ".snappy.parquet"),
        "part-00000-uuid.c000.snappy.parquet"
    );
    assert_eq!(
        spark_partitioned_file_name(7, "uuid", 12, ".csv"),
        "part-00007-uuid.c012.csv"
    );
    assert_eq!(
        partition_dir_name(
            &["c1".to_owned(), "c2".to_owned()],
            &["v1".to_owned(), DEFAULT_PARTITION.to_owned()],
        ),
        "c1=v1/c2=__HIVE_DEFAULT_PARTITION__"
    );
    let one: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    assert_eq!(
        partition_texts(&one, Some(0)).unwrap(),
        [Some("1".to_owned())]
    );
}

#[test]
fn timestamps_render_in_the_session_zone() {
    use arrow_array::TimestampMicrosecondArray;
    let ts: ArrayRef =
        Arc::new(TimestampMicrosecondArray::from(vec![Some(0_i64)]).with_timezone("UTC"));
    // 1970-01-01 00:00:00 UTC is 05:30 at +05:30 (19800 seconds).
    assert_eq!(
        partition_texts(&ts, Some(0)).unwrap(),
        [Some("1970-01-01 00:00:00".to_owned())]
    );
    assert_eq!(
        partition_texts(&ts, Some(19_800)).unwrap(),
        [Some("1970-01-01 05:30:00".to_owned())]
    );
    // A region zone renders them in its rules (Spark 4.2.0: the epoch in Los Angeles is 16:00 the day before).
    let la = krishiv_kernels::datetime::resolve_zone("America/Los_Angeles").unwrap();
    assert_eq!(
        super::partition_texts_in(&ts, Some(&la)).unwrap(),
        [Some("1969-12-31 16:00:00".to_owned())]
    );
    // Without a known session zone they cannot be rendered.
    let e = partition_texts(&ts, None).unwrap_err();
    assert!(
        e.to_string()
            .contains("session time zone that is not known"),
        "{e}"
    );
    // Other types do not depend on the zone.
    let ints: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![7]));
    assert_eq!(
        partition_texts(&ints, None).unwrap(),
        [Some("7".to_owned())]
    );
}

#[test]
fn column_names_are_escaped_like_values() {
    use crate::partition_dir_name;
    assert_eq!(
        partition_dir_name(
            &["a/b".to_owned(), "c=d".to_owned()],
            &["1".to_owned(), "2".to_owned()]
        ),
        "a%2Fb=1/c%3Dd=2"
    );
}

#[test]
fn partition_columns_of_different_lengths_are_refused() {
    let a: ArrayRef = Arc::new(StringArray::from(vec!["x", "y"]));
    let b: ArrayRef = Arc::new(StringArray::from(vec!["z"]));
    let e = partition_segments(&[a, b], Some(0)).unwrap_err();
    assert!(e.to_string().contains("different lengths"), "{e}");
}

#[test]
fn many_stars_before_a_missing_suffix_stay_cheap() {
    // 4 million calls without memoization; a few hundred with it.
    let glob = crate::partition::glob::Glob::parse("********b").unwrap();
    let name = "a".repeat(40);
    assert!(!glob.matches(&[name.as_str()]));
    assert!(glob.matches(&[format!("{name}b").as_str()]));
}

#[test]
fn repeated_brace_groups_are_bounded() {
    // 2^30 alternatives would exhaust memory; the pattern is refused before they are built.
    let pattern = "{a,b}".repeat(30);
    let err = crate::partition::glob::Glob::parse(&pattern).unwrap_err();
    assert!(err.contains("alternatives"), "{err}");
    // A reasonable number still works.
    assert!(crate::partition::glob::Glob::parse(&"{a,b}".repeat(8)).is_ok());
}

/// A binary value renders as its bytes' UTF-8 text (Spark 4.2.0 `B=SPARK`, `b=é`), a malformed sequence as U+FFFD,
/// on every binary carrier; NULL stays NULL and the empty value is the default partition.
#[test]
fn binary_values_render_as_their_utf8_text() {
    let values: Vec<Option<&[u8]>> = vec![
        Some(b"SPARK"),
        Some("é".as_bytes()),
        Some(&[0xFF, b'a']),
        None,
        Some(b""),
    ];
    let want = vec![
        Some("SPARK".to_owned()),
        Some("é".to_owned()),
        Some("\u{FFFD}a".to_owned()),
        None,
        Some(String::new()),
    ];
    let columns: [ArrayRef; 3] = [
        Arc::new(BinaryArray::from(values.clone())),
        Arc::new(LargeBinaryArray::from(values.clone())),
        Arc::new(BinaryViewArray::from(values)),
    ];
    for column in &columns {
        assert_eq!(partition_texts(column, None).unwrap(), want);
        check_partition_column("b", column.data_type()).unwrap();
        assert_eq!(
            partition_segments(&[Arc::clone(column)], None).unwrap()[4],
            [DEFAULT_PARTITION]
        );
    }
}

/// Malformed UTF-8 decodes as Java's `new String(bytes, UTF_8)` (Spark 4.2.0 `probe4.out`): one U+FFFD per
/// maximal subpart, and one for a whole encoded surrogate.
#[test]
fn malformed_binary_partition_values_decode_as_java_does() {
    let cases: [(&[u8], usize); 8] = [
        (&[0xED, 0xA0, 0x80], 1),
        (&[0xED, 0xBF, 0xBF], 1),
        (&[0xF0, 0x80, 0x80], 3),
        (&[0x80], 1),
        (&[0xC3], 1),
        (&[0xE2, 0x82], 1),
        (&[0xFF], 1),
        (&[0xF4, 0x90, 0x80, 0x80], 4),
    ];
    for (bytes, replacements) in cases {
        let column: ArrayRef = Arc::new(BinaryArray::from(vec![Some(bytes)]));
        let text = partition_texts(&column, None).unwrap()[0].clone().unwrap();
        assert_eq!(text, "\u{FFFD}".repeat(replacements), "{bytes:02X?}");
    }
    // Valid text around a malformed byte stays.
    let column: ArrayRef = Arc::new(BinaryArray::from(vec![Some(&b"a\xEDb"[..])]));
    assert_eq!(
        partition_texts(&column, None).unwrap()[0].as_deref(),
        Some("a\u{FFFD}b")
    );
}
