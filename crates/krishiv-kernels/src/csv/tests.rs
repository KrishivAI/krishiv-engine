// SPDX-License-Identifier: Apache-2.0
//! Tests of the CSV tokenizer and converters.

use super::*;
use arrow_array::StringArray;
use arrow_array::cast::AsArray;
use arrow_schema::{Field, Fields};

/// Tokenizes one record.
fn tokens_of(text: &str, syntax: Syntax) -> Vec<Token> {
    let tok = Tokenizer::new(syntax);
    let mut out = Vec::new();
    tok.record(text, 0, &mut out).unwrap();
    out
}

#[test]
fn tokenizer_quotes_and_escapes() {
    let s = Syntax::default();
    assert_eq!(
        tokens_of("a,\"b,c\",d", s.clone()),
        [
            Some("a".to_owned()),
            Some("b,c".to_owned()),
            Some("d".to_owned())
        ]
    );
    // An escaped quote inside a quoted value.
    assert_eq!(
        tokens_of("\"a\\\"b\",c", s.clone()),
        [Some("a\"b".to_owned()), Some("c".to_owned())]
    );
    // Empty unquoted is null; quoted empty uses `empty_value`.
    assert_eq!(tokens_of(",", s.clone()), [None, None]);
    assert_eq!(tokens_of("\"\"", s), [Some(String::new())]);
}

#[test]
fn tokenizer_sep_and_newlines_in_quotes() {
    let s = Syntax {
        sep: "|".to_owned(),
        ..Syntax::default()
    };
    assert_eq!(
        tokens_of("a|b|c", s),
        [
            Some("a".to_owned()),
            Some("b".to_owned()),
            Some("c".to_owned())
        ]
    );
    let s = Syntax::default();
    assert_eq!(
        tokens_of("\"a\nb\",c", s),
        [Some("a\nb".to_owned()), Some("c".to_owned())]
    );
}

#[test]
fn from_csv_builds_structs() {
    let fields: Fields = vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8View, true),
    ]
    .into();
    let kinds = vec![FieldKind::Int(32), FieldKind::Str];
    let input = StringArray::from(vec![Some("1,hi"), None]);
    let o = FromCsvOptions {
        syntax: Syntax::default(),
        mode: ParseMode::Permissive,
        null_value: String::new(),
        nan: "NaN".to_owned(),
        pos_inf: "Inf".to_owned(),
        neg_inf: "-Inf".to_owned(),
        date_pattern: None,
        time_pattern: None,
        timestamp_pattern: None,
        ntz_pattern: None,
        fallback: false,
    };
    let zone = crate::cast::FixedOffset(0);
    let out = from_csv(
        &input,
        &fields,
        &kinds,
        None,
        &o,
        &zone,
        &|_| Ok(None),
        &|_| Ok(None),
    )
    .unwrap();
    assert_eq!(out.len(), 2);
    assert!(!out.is_null(0));
    assert!(out.is_null(1));
    assert_eq!(
        out.column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        1
    );
}

#[test]
fn struct_to_csv_quotes() {
    use arrow_array::StringViewArray;
    let fields: Fields = vec![Field::new("v", DataType::Utf8View, true)].into();
    let col = StructArray::new(
        fields,
        vec![Arc::new(StringViewArray::from(vec![Some("a,b"), None])) as ArrayRef],
        None,
    );
    let o = ToCsvOptions {
        binary: crate::binary_style::BinaryStyle::default(),
        syntax: Syntax::default(),
        quote_all: false,
        escape_quotes: true,
        line_sep: None,
        null_value: String::new(),
        date_pattern: None,
        time_pattern: None,
        timestamp_pattern: None,
        ntz_pattern: None,
    };
    let out = struct_to_csv(
        &col,
        &[FieldKind::Str],
        &o,
        &|_| Err(KernelError::unsupported("no date")),
        &|_| Err(KernelError::unsupported("no ts")),
        &|_| Err(KernelError::unsupported("no ntz")),
    )
    .unwrap();
    assert_eq!(out.value(0), "\"a,b\"");
    assert_eq!(out.value(1), "");
}

/// `to_csv` kernel: every scalar kind, nested values (struct, list,
/// large list, map) the way Spark prints them, trimming, empty values
/// and nulls.
#[test]
fn struct_to_csv_writes_every_kind_and_nested_values() {
    use arrow_array::builder::{Int32Builder, ListBuilder, MapBuilder, StringBuilder};
    use arrow_array::{
        BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
        Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, NullArray, StringArray,
        TimestampMicrosecondArray,
    };
    let mut list = ListBuilder::new(Int32Builder::new());
    list.values().append_value(1);
    list.values().append_null();
    list.append(true);
    let list: ArrayRef = Arc::new(list.finish());
    let mut map = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    map.keys().append_value("k");
    map.values().append_value(7);
    map.append(true).unwrap();
    let map: ArrayRef = Arc::new(map.finish());
    let inner: ArrayRef = Arc::new(StructArray::new(
        vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Float32, true),
            Field::new("c", DataType::Float64, true),
            Field::new("d", DataType::Boolean, true),
            Field::new("e", DataType::Utf8, true),
            Field::new("f", DataType::LargeUtf8, true),
            Field::new("g", DataType::Decimal128(5, 2), true),
            Field::new("h", DataType::Date32, true),
            Field::new(
                "i",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            ),
            Field::new("j", DataType::Timestamp(TimeUnit::Microsecond, None), true),
            Field::new("k", DataType::Int8, true),
            Field::new("l", DataType::Int16, true),
        ]
        .into(),
        vec![
            Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            Arc::new(Float32Array::from(vec![1.5])),
            Arc::new(Float64Array::from(vec![2.5])),
            Arc::new(BooleanArray::from(vec![true])),
            Arc::new(StringArray::from(vec!["s"])),
            Arc::new(LargeStringArray::from(vec!["t"])),
            Arc::new(
                Decimal128Array::from(vec![725])
                    .with_precision_and_scale(5, 2)
                    .unwrap(),
            ),
            Arc::new(Date32Array::from(vec![18_263])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![1_577_934_245_000_000]).with_timezone("UTC"),
            ),
            Arc::new(TimestampMicrosecondArray::from(vec![1_577_934_245_000_000])),
            Arc::new(Int8Array::from(vec![8])),
            Arc::new(Int16Array::from(vec![16])),
        ],
        None,
    ));
    let col = StructArray::new(
        vec![
            Field::new("list", list.data_type().clone(), true),
            Field::new("map", map.data_type().clone(), true),
            Field::new("inner", inner.data_type().clone(), true),
            Field::new("n", DataType::Int32, true),
            Field::new("bin", DataType::Binary, true),
            Field::new("void", DataType::Null, true),
            Field::new("s", DataType::Utf8, true),
        ]
        .into(),
        vec![
            list,
            map,
            inner,
            Arc::new(Int32Array::from(vec![None])) as ArrayRef,
            Arc::new(BinaryArray::from(vec![&b"ab"[..]])),
            Arc::new(NullArray::new(1)),
            Arc::new(StringArray::from(vec!["  x  "])),
        ],
        None,
    );
    let kinds: Vec<FieldKind> = col
        .fields()
        .iter()
        .map(|f| FieldKind::of_write(f.data_type()).unwrap())
        .collect();
    let syntax = Syntax {
        ltrim: true,
        rtrim: true,
        ..Syntax::default()
    };
    let o = ToCsvOptions {
        binary: crate::binary_style::BinaryStyle::default(),
        syntax,
        quote_all: false,
        escape_quotes: true,
        line_sep: None,
        null_value: "NULL".to_owned(),
        date_pattern: None,
        time_pattern: None,
        timestamp_pattern: None,
        ntz_pattern: None,
    };
    let out = struct_to_csv(
        &col,
        &kinds,
        &o,
        &|d| Ok(format!("D{d}")),
        &|t| Ok(format!("T{t}")),
        &|t| Ok(format!("N{t}")),
    )
    .unwrap();
    assert_eq!(
        out.value(0),
        "\"[1, null]\",{k -> 7},\"{1, 1.5, 2.5, true, s, t, 7.25, D18263, T1577934245000000, \
         N1577934245000000, 8, 16}\",NULL,[61 62],NULL,x"
    );
}

/// A nested value of a type CSV has no text for is refused.
#[test]
fn nested_values_of_exotic_types_are_refused() {
    use arrow_array::builder::{DurationMicrosecondBuilder, ListBuilder};
    let mut list = ListBuilder::new(DurationMicrosecondBuilder::new());
    list.values().append_value(1);
    list.append(true);
    let list: ArrayRef = Arc::new(list.finish());
    let col = StructArray::new(
        vec![Field::new("l", list.data_type().clone(), true)].into(),
        vec![list],
        None,
    );
    let o = ToCsvOptions {
        binary: crate::binary_style::BinaryStyle::default(),
        syntax: Syntax::default(),
        quote_all: false,
        escape_quotes: true,
        line_sep: None,
        null_value: String::new(),
        date_pattern: None,
        time_pattern: None,
        timestamp_pattern: None,
        ntz_pattern: None,
    };
    let err = struct_to_csv(
        &col,
        &[FieldKind::Nested],
        &o,
        &|_| Ok(String::new()),
        &|_| Ok(String::new()),
        &|_| Ok(String::new()),
    )
    .unwrap_err();
    assert!(err.to_string().contains("CSV"), "{err}");
}

/// `from_csv` kernel: every scalar kind converts, malformed values
/// become NULL under PERMISSIVE with the parsed row in the FAILFAST
/// error, and the corrupt-record column keeps the text.
#[test]
fn from_csv_converts_every_kind_and_reports_the_parsed_row() {
    use arrow_array::types::{Int64Type, TimestampMicrosecondType};
    let fields: Fields = vec![
        Field::new("b", DataType::Boolean, true),
        Field::new("l", DataType::Int64, true),
        Field::new("f", DataType::Float32, true),
        Field::new("d", DataType::Float64, true),
        Field::new("dec", DataType::Decimal128(5, 2), true),
        Field::new("s", DataType::Utf8, true),
        Field::new("bin", DataType::Binary, true),
        Field::new("dt", DataType::Date32, true),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "ntz",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("_corrupt", DataType::Utf8, true),
    ]
    .into();
    let kinds: Vec<FieldKind> = fields
        .iter()
        .map(|f| FieldKind::of(f.data_type()).unwrap())
        .collect();
    let options = |mode| FromCsvOptions {
        syntax: Syntax::default(),
        mode,
        null_value: String::new(),
        nan: "NaN".to_owned(),
        pos_inf: "Inf".to_owned(),
        neg_inf: "-Inf".to_owned(),
        date_pattern: None,
        time_pattern: None,
        timestamp_pattern: None,
        ntz_pattern: None,
        fallback: false,
    };
    let zone = crate::cast::FixedOffset(0);
    let ts = |s: &str| -> Result<Option<i64>, KernelError> { Ok((s == "T").then_some(5)) };
    let ntz = |s: &str| -> Result<Option<i64>, KernelError> { Ok((s == "N").then_some(6)) };
    let good = "true,9,1.5,2.5,7.25,str,bin,2020-01-02,T,N";
    let bad = "true,x,1.5,2.5,7.25,str,bin,2020-01-02,T,N";
    let input = StringArray::from(vec![Some(good), Some(bad), None]);
    let out = from_csv(
        &input,
        &fields,
        &kinds,
        Some(10),
        &options(ParseMode::Permissive),
        &zone,
        &ts,
        &ntz,
    )
    .unwrap();
    assert!(out.column(0).as_boolean().value(0));
    assert_eq!(out.column(1).as_primitive::<Int64Type>().value(0), 9);
    assert!(out.column(1).is_null(1));
    assert_eq!(
        out.column(8)
            .as_primitive::<TimestampMicrosecondType>()
            .value(0),
        5
    );
    assert_eq!(
        out.column(9)
            .as_primitive::<TimestampMicrosecondType>()
            .value(0),
        6
    );
    assert_eq!(out.column(10).as_string::<i32>().value(1), bad);
    assert!(out.is_null(2));
    // A failed field under FAILFAST shows the parsed row.
    let err = from_csv(
        &StringArray::from(vec![bad]),
        &fields,
        &kinds,
        Some(10),
        &options(ParseMode::FailFast),
        &zone,
        &ts,
        &ntz,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("[true,null,1.5,2.5,7.25,str,[62 69 6E],"),
        "{err}"
    );
    assert_eq!(err.row(), Some(0));
    // The error names the failing input row, after a NULL one.
    let err = from_csv(
        &StringArray::from(vec![None, Some(bad)]),
        &fields,
        &kinds,
        Some(10),
        &options(ParseMode::FailFast),
        &zone,
        &ts,
        &ntz,
    )
    .unwrap_err();
    assert_eq!(err.row(), Some(1), "{err}");
}
