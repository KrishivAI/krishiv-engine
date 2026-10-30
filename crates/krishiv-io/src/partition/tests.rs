// SPDX-License-Identifier: Apache-2.0
//! Tests of Spark's partition path segments: the escape table, the
//! `CAST(v AS STRING)` rendering and the directory layout.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, Int32Array, Int64Array, StringArray,
    StringViewArray, TimestampMicrosecondArray,
};

use super::{
    DEFAULT_PARTITION, escape_partition_value, partition_dir_name, partition_segments,
    partition_texts, spark_partitioned_file_name,
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
    let segments = partition_segments(std::slice::from_ref(&nulls)).unwrap();
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
    let texts = |column: ArrayRef| partition_texts(&column).unwrap();
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
    let segments = partition_segments(std::slice::from_ref(&column)).unwrap();
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
    assert_eq!(partition_texts(&one).unwrap(), [Some("1".to_owned())]);
}
