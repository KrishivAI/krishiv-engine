// SPDX-License-Identifier: Apache-2.0
//! The protobuf codec against the golden descriptor sets (`tests/spark-golden/functions/from_to_protobuf/fixtures`); the Spark
//! results themselves are checked by `krishiv-session/tests/protobuf_goldens.rs`.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, StructArray};
use krishiv_types::DataType;

use super::{Call, DescriptorArg, DescriptorSet, ParseMode, ProtoOptions};

const GOLDEN: &[u8] = include_bytes!(
    "../../../../tests/spark-golden/functions/from_to_protobuf/fixtures/krishiv_golden.desc"
);
const TYPES: &[u8] = include_bytes!(
    "../../../../tests/spark-golden/functions/from_to_protobuf/fixtures/krishiv_golden_types.desc"
);

fn decode(
    desc: &[u8],
    message: &str,
    options: &[(&str, &str)],
    rows: &[Option<&[u8]>],
) -> Result<ArrayRef, krishiv_common::error::KrishivError> {
    let set = DescriptorSet::parse(desc)?;
    super::Decoder::new(&set, message, options.iter().copied())?.decode(rows)
}

#[test]
fn a_singular_scalar_takes_its_last_value_and_a_zero_reads_as_null() {
    // i32 = 1, then i32 = 7 (last wins); s32 = 0 is the zero of a proto3 scalar without presence.
    let out = decode(
        GOLDEN,
        "krishiv.golden.Scalars",
        &[],
        &[Some(&[0x08, 0x01, 0x08, 0x07, 0x28, 0x00]), None],
    )
    .unwrap();
    let st = out.as_struct();
    let i32s = st.column(0).as_primitive::<Int32Type>();
    assert_eq!(i32s.value(0), 7);
    assert!(st.column(4).is_null(0));
    assert!(st.is_null(1));
}

#[test]
fn repeated_scalars_read_packed_and_unpacked_and_a_message_field_merges() {
    // Outer { nums: [1, 2] packed, nums: 3 unpacked, inner: {name: "a"}, inner: {n: 5} }.
    let bytes = [
        0x12, 0x02, 0x01, 0x02, 0x10, 0x03, 0x0a, 0x03, 0x0a, 0x01, 0x61, 0x0a, 0x02, 0x10, 0x05,
    ];
    let out = decode(GOLDEN, "krishiv.golden.Outer", &[], &[Some(&bytes)]).unwrap();
    let st = out.as_struct();
    let nums = st.column(1).as_list::<i32>().value(0);
    assert_eq!(
        nums.as_primitive::<Int32Type>().values().to_vec(),
        vec![1, 2, 3]
    );
    let inner: &StructArray = st.column(0).as_struct();
    assert_eq!(inner.column(0).as_string_view().value(0), "a");
    assert_eq!(inner.column(1).as_primitive::<Int32Type>().value(0), 5);
}

#[test]
fn a_malformed_message_fails_fast_or_is_null_when_permissive() {
    let bad: &[Option<&[u8]>] = &[Some(&[0x0a, 0x05, 0x01])];
    let err = decode(GOLDEN, "krishiv.golden.Scalars", &[], bad).unwrap_err();
    assert_eq!(err.error_class().name(), "MALFORMED_PROTOBUF_MESSAGE");
    let out = decode(
        GOLDEN,
        "krishiv.golden.Scalars",
        &[("MODE", "permissive")],
        bad,
    )
    .unwrap();
    assert!(out.is_null(0));
}

#[test]
fn a_known_field_with_the_wrong_wire_type_fails_the_message() {
    // Field 1 (int32) arrives length-delimited.
    let err = decode(
        GOLDEN,
        "krishiv.golden.Scalars",
        &[],
        &[Some(&[0x0a, 0x00])],
    )
    .unwrap_err();
    assert_eq!(err.error_class().name(), "MALFORMED_PROTOBUF_MESSAGE");
}

#[test]
fn a_recursive_message_needs_a_depth_and_is_cut_at_it() {
    let set = DescriptorSet::parse(GOLDEN).unwrap();
    let err = super::Decoder::new(&set, "krishiv.golden.Node", std::iter::empty()).unwrap_err();
    assert_eq!(err.error_class().name(), "RECURSIVE_PROTOBUF_SCHEMA");
    let d = super::Decoder::new(&set, "Node", [("recursive.fields.max.depth", "2")]).unwrap();
    assert_eq!(
        d.spark_type().simple_string(),
        "struct<v:int,next:struct<v:int>>"
    );
}

#[test]
fn an_empty_wrapper_element_of_a_repeated_unwrapped_field_stays_a_null() {
    // Wkt.i32vs = [{value: 5}, {}, {value: 7}] under unwrap.primitive.wrapper.types: Spark gives [5, NULL, 7].
    let bytes = [
        0x82, 0x01, 0x02, 0x08, 0x05, 0x82, 0x01, 0x00, 0x82, 0x01, 0x02, 0x08, 0x07,
    ];
    let opts = [("unwrap.primitive.wrapper.types", "true")];
    let set = DescriptorSet::parse(TYPES).unwrap();
    let d = super::Decoder::new(&set, "Wkt", opts.iter().copied()).unwrap();
    // Spark says `array<int NOT NULL>`; Krishiv admits the NULL element rather than drop it (README divergence).
    let DataType::Struct(st) = d.spark_type() else {
        panic!("not a struct")
    };
    let field = st.fields().iter().find(|f| f.name() == "i32vs").unwrap();
    let DataType::Array(arr) = field.data_type() else {
        panic!("not an array")
    };
    assert!(arr.contains_null());
    let out = d.decode(&[Some(&bytes)]).unwrap();
    let list = out
        .as_struct()
        .column_by_name("i32vs")
        .unwrap()
        .as_list::<i32>()
        .value(0);
    let ints = list.as_primitive::<Int32Type>();
    assert_eq!(ints.len(), 3);
    assert_eq!(
        (ints.value(0), ints.is_null(1), ints.value(2)),
        (5, true, 7)
    );
}

#[test]
fn upcast_unsigned_widens_uint32_and_uint64() {
    let set = DescriptorSet::parse(GOLDEN).unwrap();
    let d = super::Decoder::new(&set, "Scalars", [("upcast.unsigned.ints", "true")]).unwrap();
    let DataType::Struct(st) = d.spark_type() else {
        panic!("not a struct")
    };
    assert_eq!(st.fields()[2].data_type(), &DataType::Long);
    assert_eq!(st.fields()[3].data_type().sql(), "DECIMAL(20,0)");
    // fixed32/fixed64 keep their signed carrier.
    assert_eq!(st.fields()[6].data_type(), &DataType::Integer);
    let out = d
        .decode(&[Some(&[0x18, 0xff, 0xff, 0xff, 0xff, 0x0f])])
        .unwrap();
    assert_eq!(
        out.as_struct()
            .column(2)
            .as_primitive::<Int64Type>()
            .value(0),
        4_294_967_295
    );
}

#[test]
fn a_timestamp_beyond_the_microsecond_range_is_malformed() {
    let set = DescriptorSet::parse(TYPES).unwrap();
    let d = super::Decoder::new(&set, "TsOnly", std::iter::empty()).unwrap();
    // ts = { seconds: 9223372036855 } overflows `seconds * 1000 * 1000`.
    let seconds = 9_223_372_036_855u64;
    let mut inner = vec![0x08];
    let mut v = seconds;
    while v >= 0x80 {
        inner.push((v as u8) | 0x80);
        v >>= 7;
    }
    inner.push(v as u8);
    let mut msg = vec![0x0a, inner.len() as u8];
    msg.extend(inner);
    let e = d.decode(&[Some(&msg)]).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::MalformedProtobufMessage
    );
}

#[test]
fn encoding_then_decoding_gives_the_values_back() {
    let set = DescriptorSet::parse(GOLDEN).unwrap();
    let input = DataType::from(krishiv_types::StructType::new([
        krishiv_types::StructField::new("i32", DataType::Integer, false),
        krishiv_types::StructField::new("str", DataType::String, true),
    ]));
    let enc = super::Encoder::new(&set, "Scalars", &input).unwrap();
    let ints: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![7, 0]));
    let strs: ArrayRef = Arc::new(arrow_array::StringViewArray::from(vec![Some("z"), None]));
    let arrow = input.to_arrow();
    let ArrowFields(fields) = ArrowFields::of(&arrow);
    let array: ArrayRef = Arc::new(StructArray::new(fields, vec![ints, strs], None));
    let bytes = enc.encode(&array).unwrap();
    assert_eq!(
        bytes[0].as_deref(),
        Some(&[0x08, 0x07, 0x72, 0x01, b'z'][..])
    );
    // A zero and a NULL are not written.
    assert_eq!(bytes[1].as_deref(), Some(&[][..]));
}

struct ArrowFields(arrow_schema::Fields);

impl ArrowFields {
    fn of(t: &arrow_schema::DataType) -> Self {
        match t {
            arrow_schema::DataType::Struct(f) => Self(f.clone()),
            _ => panic!("not a struct"),
        }
    }
}

#[test]
fn option_values_parse_like_scala() {
    let o = ProtoOptions::parse([
        ("Enums.As.Ints", "TRUE"),
        ("recursive.fields.max.depth", "+3"),
        ("mode", "bogus"),
    ])
    .unwrap();
    assert!(o.enums_as_ints);
    assert_eq!(o.recursive_depth, 3);
    assert_eq!(o.mode, ParseMode::Permissive);
    for bad in [
        ("emit.default.values", "yes"),
        ("recursive.fields.max.depth", " 2"),
    ] {
        let e = ProtoOptions::parse([bad]).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "STDS_INVALID_OPTION_VALUE.WITH_MESSAGE"
        );
    }
}

#[test]
fn a_call_without_a_descriptor_names_a_java_class() {
    let call = Call::new("a.b.C", DescriptorArg::None, Vec::new(), false).unwrap();
    assert_eq!(
        call.decoder().unwrap_err().error_class().name(),
        "CANNOT_LOAD_PROTOBUF_CLASS"
    );
    // `to_protobuf` reads an empty path (and fails to parse it); `from_protobuf` takes it as no descriptor.
    assert!(
        Call::new(
            "a.b.C",
            DescriptorArg::Path(String::new()),
            Vec::new(),
            true
        )
        .is_err()
    );
    assert!(
        Call::new(
            "a.b.C",
            DescriptorArg::Path(String::new()),
            Vec::new(),
            false
        )
        .is_ok()
    );
}

#[test]
fn a_descriptor_path_is_never_read_by_the_call() {
    // The analyzer reads a path after the access check and passes the bytes; a path reaching the call is refused, not read.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    for to in [false, true] {
        let e = Call::new("M", DescriptorArg::Path(path.to_owned()), Vec::new(), to).unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    }
    // File contents that are empty are an empty set (no message), not "no descriptor".
    let call = Call::new("M", DescriptorArg::File(Vec::new()), Vec::new(), false).unwrap();
    assert_eq!(
        call.decoder().unwrap_err().error_class().name(),
        "PROTOBUF_MESSAGE_NOT_FOUND"
    );
}

#[test]
fn descriptor_sets_find_top_level_messages_by_short_or_full_name() {
    let set = DescriptorSet::parse(GOLDEN).unwrap();
    assert!(set.message("Scalars").is_ok());
    assert!(set.message("krishiv.golden.Scalars").is_ok());
    // An enum is not a message; neither is a name that is not defined.
    for name in ["krishiv.golden.Color", "Unknown", "Counts"] {
        assert_eq!(
            set.message(name).unwrap_err().error_class().name(),
            "PROTOBUF_MESSAGE_NOT_FOUND",
            "{name}"
        );
    }
}

#[test]
fn bytes_that_are_not_a_descriptor_set_do_not_parse() {
    for bad in [&[0x0a, 0x0b][..], &[0xff][..]] {
        assert_eq!(
            DescriptorSet::parse(bad).unwrap_err().error_class().name(),
            "CANNOT_PARSE_PROTOBUF_DESCRIPTOR"
        );
    }
    // An empty set is a set with no messages.
    assert!(DescriptorSet::parse(&[]).unwrap().message("a").is_err());
}

#[test]
fn a_missing_descriptor_file_is_not_found_and_a_directory_does_not_parse() {
    let e = DescriptorSet::read_file("x.desc", "file:/no/such/dir/x.desc").unwrap_err();
    assert_eq!(e.error_class().name(), "PROTOBUF_DESCRIPTOR_FILE_NOT_FOUND");
    assert!(e.to_string().contains("x.desc"), "{e}");
    let e = DescriptorSet::read_file("d", env!("CARGO_MANIFEST_DIR")).unwrap_err();
    assert_eq!(e.error_class().name(), "CANNOT_PARSE_PROTOBUF_DESCRIPTOR");
}

/// Hand-built `FileDescriptorSet` bytes for descriptors `protoc` would refuse to write.
mod build {
    pub(super) fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    pub(super) fn len(number: u64, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        varint(&mut out, number << 3 | 2);
        varint(&mut out, body.len() as u64);
        out.extend_from_slice(body);
        out
    }

    pub(super) fn int(number: u64, v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        varint(&mut out, number << 3);
        varint(&mut out, v);
        out
    }

    /// A field: `label` 1 optional / 3 repeated, `ty` 11 message or a scalar kind, `type_name` for messages.
    pub(super) fn field(name: &str, number: u64, label: u64, ty: u64, type_name: &str) -> Vec<u8> {
        let mut f = len(1, name.as_bytes());
        f.extend(int(3, number));
        f.extend(int(4, label));
        f.extend(int(5, ty));
        if !type_name.is_empty() {
            f.extend(len(6, type_name.as_bytes()));
        }
        f
    }

    /// A message of `fields` and `nested` messages, a map entry when `map_entry`.
    pub(super) fn message(
        name: &str,
        fields: &[Vec<u8>],
        nested: &[Vec<u8>],
        map_entry: bool,
    ) -> Vec<u8> {
        let mut m = len(1, name.as_bytes());
        for f in fields {
            m.extend(len(2, f));
        }
        for n in nested {
            m.extend(len(3, n));
        }
        if map_entry {
            m.extend(len(7, &int(7, 1)));
        }
        m
    }

    /// A proto3 file of package `t` holding `messages`, importing `deps`, as a one-file set.
    pub(super) fn file(name: &str, deps: &[&str], messages: &[Vec<u8>]) -> Vec<u8> {
        let mut f = len(1, name.as_bytes());
        f.extend(len(2, b"t"));
        for d in deps {
            f.extend(len(3, d.as_bytes()));
        }
        for m in messages {
            f.extend(len(4, m));
        }
        f.extend(len(12, b"proto3"));
        len(1, &f)
    }
}

#[test]
fn a_map_entry_without_its_key_and_value_fields_does_not_parse() {
    // `message M { repeated E m = 1; message E { option map_entry = true; } }`: no key, no value; then a key only.
    for entry_fields in [
        Vec::new(),
        vec![build::field("key", 1, 1, 9, "")],
        vec![build::field("value", 2, 1, 9, "")],
    ] {
        let entry = build::message("E", &entry_fields, &[], true);
        let m = build::message(
            "M",
            &[build::field("m", 1, 3, 11, ".t.M.E")],
            &[entry],
            false,
        );
        let set = build::file("a.proto", &[], &[m]);
        assert_eq!(
            DescriptorSet::parse(&set).unwrap_err().error_class().name(),
            "CANNOT_PARSE_PROTOBUF_DESCRIPTOR"
        );
    }
}

#[test]
fn a_map_entry_with_key_and_value_reads_as_a_map() {
    let entry = build::message(
        "E",
        &[
            build::field("key", 1, 1, 9, ""),
            build::field("value", 2, 1, 5, ""),
        ],
        &[],
        true,
    );
    let m = build::message(
        "M",
        &[build::field("m", 1, 3, 11, ".t.M.E")],
        &[entry],
        false,
    );
    let set = DescriptorSet::parse(&build::file("a.proto", &[], &[m])).unwrap();
    let d = super::Decoder::new(&set, "M", std::iter::empty()).unwrap();
    assert_eq!(d.spark_type().simple_string(), "struct<m:map<string,int>>");
}

/// `M0 { M1 f = 1; }`, `M1 { M2 f = 1; }`, ... `M<n-1> { int32 v = 1; }`: a chain of `n` distinct (not recursive) types.
fn chain(n: usize) -> Vec<u8> {
    let messages: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let f = if i + 1 == n {
                build::field("v", 1, 1, 5, "")
            } else {
                build::field("f", 1, 1, 11, &format!(".t.M{}", i + 1))
            };
            build::message(&format!("M{i}"), &[f], &[], false)
        })
        .collect();
    build::file("a.proto", &[], &messages)
}

#[test]
fn a_chain_of_distinct_types_past_the_nesting_cap_is_refused_not_a_stack_overflow() {
    let set = DescriptorSet::parse(&chain(200_000)).unwrap();
    let err = super::Decoder::new(&set, "M0", std::iter::empty()).unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    let input = DataType::from(krishiv_types::StructType::new([
        krishiv_types::StructField::new("v", DataType::Integer, true),
    ]));
    // The encoder walks the input type, which stays shallow; it never follows the chain.
    let _ = super::Encoder::new(&set, "M0", &input);
}

#[test]
fn a_descriptor_whose_fields_each_refer_twice_to_the_next_type_is_refused_by_the_expansion_budget()
{
    // 40 types, two fields each of the next one: about 2^40 expanded messages from a small descriptor.
    let n = 40;
    let messages: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let fields = if i + 1 == n {
                vec![build::field("v", 1, 1, 5, "")]
            } else {
                let t = format!(".t.M{}", i + 1);
                vec![
                    build::field("f", 1, 1, 11, &t),
                    build::field("g", 2, 1, 11, &t),
                ]
            };
            build::message(&format!("M{i}"), &fields, &[], false)
        })
        .collect();
    let set = DescriptorSet::parse(&build::file("a.proto", &[], &messages)).unwrap();
    let err = super::Decoder::new(&set, "M0", std::iter::empty()).unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn two_malformed_embedded_pieces_do_not_merge_into_a_valid_message() {
    // Root { Child c = 1; } Child { int32 x = 1; }: `0A01 08` and `0A01 01` are truncated / tag-0 bodies whose concatenation is `0801`.
    let child = build::message("Child", &[build::field("x", 1, 1, 5, "")], &[], false);
    let root = build::message(
        "Root",
        &[build::field("c", 1, 1, 11, ".t.Child")],
        &[],
        false,
    );
    let set = build::file("a.proto", &[], &[child, root]);
    let bad: &[Option<&[u8]>] = &[Some(&[0x0a, 0x01, 0x08, 0x0a, 0x01, 0x01])];
    let err = decode(&set, "Root", &[], bad).unwrap_err();
    assert_eq!(err.error_class().name(), "MALFORMED_PROTOBUF_MESSAGE");
}

#[cfg(unix)]
#[test]
fn a_descriptor_file_that_is_a_symbolic_link_is_read_as_spark_reads_it() {
    let dir = std::env::temp_dir().join(format!("krishiv-pb-link-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let real = dir.join("real.desc");
    std::fs::write(&real, GOLDEN).unwrap();
    let link = dir.join("link.desc");
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(&real, &link).unwrap();
    assert!(DescriptorSet::read_file("real", real.to_str().unwrap()).is_ok());
    assert_eq!(
        DescriptorSet::read_file("link", link.to_str().unwrap()).unwrap(),
        GOLDEN
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_chain_of_distinct_types_within_the_cap_reads() {
    let set = DescriptorSet::parse(&chain(super::schema::MAX_NESTING)).unwrap();
    assert!(super::Decoder::new(&set, "M0", std::iter::empty()).is_ok());
    let set = DescriptorSet::parse(&chain(super::schema::MAX_NESTING + 1)).unwrap();
    assert!(super::Decoder::new(&set, "M0", std::iter::empty()).is_err());
}

#[test]
fn a_missing_import_is_protobuf_dependency_not_found() {
    // Spark's `ProtobufUtils.buildFileDescriptor`: the first import (depth first, files in set order) the set does not hold.
    let m = build::message("M", &[build::field("v", 1, 1, 5, "")], &[], false);
    let mut set = build::file("a.proto", &["b.proto"], &[m]);
    set.extend(build::file("b.proto", &["c.proto", "d.proto"], &[]));
    let err = DescriptorSet::parse(&set).unwrap_err();
    assert_eq!(err.error_class().name(), "PROTOBUF_DEPENDENCY_NOT_FOUND");
    assert!(err.to_string().contains("c.proto"), "{err}");
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(512))]

    /// Any bytes as a descriptor set: parsed or refused, never a panic.
    #[test]
    fn any_bytes_as_a_descriptor_set_never_panic(bytes in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..512)) {
        if let Ok(set) = DescriptorSet::parse(&bytes) {
            for m in set.pool.top_level() {
                let _ = super::Decoder::new(&set, m.full_name(), std::iter::empty());
            }
        }
    }

    /// The golden descriptor set with a byte flipped: parsed or refused, and every message's reader built or refused.
    #[test]
    fn a_damaged_golden_descriptor_never_panics(at in 0usize..5638, to in proptest::prelude::any::<u8>()) {
        let mut bytes = TYPES.to_vec();
        let at = at % bytes.len();
        bytes[at] = to;
        if let Ok(set) = DescriptorSet::parse(&bytes) {
            for m in set.pool.top_level() {
                let _ = super::Decoder::new(&set, m.full_name(), [("recursive.fields.max.depth", "3")]);
            }
        }
    }

    /// Any bytes as a message of every golden message type, fail-fast and permissive: values or an error, never a panic.
    #[test]
    fn any_bytes_as_a_message_never_panic(bytes in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..256)) {
        for desc in [GOLDEN, TYPES] {
            let set = DescriptorSet::parse(desc).unwrap();
            for m in set.pool.top_level() {
                for mode in ["FAILFAST", "PERMISSIVE"] {
                    let opts = [("mode", mode), ("recursive.fields.max.depth", "3"), ("convert.any.fields.to.json", "true")];
                    if let Ok(d) = super::Decoder::new(&set, m.full_name(), opts) {
                        let _ = d.decode(&[Some(&bytes)]);
                    }
                }
            }
        }
    }
}

const ANY_OPTS: &[(&str, &str)] = &[("convert.any.fields.to.json", "true")];

fn any_of(type_name: &str, value: &[u8]) -> Vec<u8> {
    let mut a = build::len(1, format!("type.googleapis.com/{type_name}").as_bytes());
    a.extend(build::len(2, value));
    a
}

/// The `details` column of an `AnyOnly` row holding `any` (and `many` holding each of `more`), as JSON text.
fn any_json(any: &[u8], more: &[Vec<u8>]) -> Vec<Option<String>> {
    let mut row = build::len(2, any);
    for m in more {
        row.extend(build::len(3, m));
    }
    let out = decode(
        TYPES,
        "krishiv.golden.types.AnyOnly",
        ANY_OPTS,
        &[Some(row.as_slice())],
    )
    .unwrap();
    let s = out.as_struct();
    let details = s.column_by_name("details").unwrap().as_string_view();
    let many = s.column_by_name("many").unwrap().as_list::<i32>();
    let mut got = vec![(!details.is_null(0)).then(|| details.value(0).to_owned())];
    let items = many.value(0);
    got.extend(items.as_string_view().iter().map(|v| v.map(str::to_owned)));
    got
}

#[test]
fn an_any_holding_a_message_prints_every_field_kind_as_protobuf_java_json() {
    let payload = [build::len(1, b"n"), build::int(2, 4)].concat();
    let mut kinds = Vec::new();
    kinds.extend(build::len(1, &[1, 2])); // packed enum list
    kinds.extend(build::int(2, 2));
    kinds.extend(build::len(3, &[1, 2, 3]));
    kinds.extend(build::len(4, b"a\"b\n\x01"));
    kinds.extend(build::len(5, &1.5f32.to_le_bytes()));
    kinds.extend(build::len(6, &f64::NAN.to_le_bytes()));
    kinds.extend(build::len(7, &[1, 0]));
    kinds.extend(build::len(8, &[6]));
    kinds.extend(build::len(9, &[7]));
    kinds.extend(build::len(10, &[5]));
    kinds.extend(build::len(
        11,
        &[build::int(1, 7), build::len(2, b"x")].concat(),
    ));
    kinds.extend(build::len(
        12,
        &[build::int(1, 9), build::len(2, &[1, 2, 3])].concat(),
    ));
    kinds.extend(build::len(
        13,
        &[build::int(1, 1), build::int(2, 1)].concat(),
    ));
    kinds.extend(build::len(
        14,
        &[build::len(1, b"k"), build::len(2, &payload)].concat(),
    ));
    kinds.extend(build::len(
        15,
        &[build::int(1, 3), build::int(2, 4)].concat(),
    ));
    kinds.extend(build::len(17, &payload));
    kinds.extend(build::len(20, &payload));
    kinds.extend(build::len(20, &payload));
    kinds.extend(build::len(24, &payload));
    kinds.extend(build::len(25, &[1, 2]));
    kinds.extend(build::len(27, &build::int(1, 1)));
    kinds.extend(build::len(28, b"camel"));
    kinds.extend(build::len(30, b"j"));
    let any = any_of("krishiv.golden.types.Kinds", &kinds);
    let json = any_json(&any, std::slice::from_ref(&any));
    let first = json[0].as_deref().unwrap();
    assert!(
        first.starts_with("{\"@type\":\"type.googleapis.com/krishiv.golden.types.Kinds\","),
        "{first}"
    );
    for part in [
        "\"shades\":[\"LIGHT\",\"DARK\"]",
        "\"shade\":\"DARK\"",
        "\"bins\":[\"AQID\"]",
        "\"strs\":[\"a\\\"b\\n\\u0001\"]",
        "\"flts\":[1.5]",
        "\"dbls\":[\"NaN\"]",
        "\"bools\":[true,false]",
        "\"s64s\":[\"3\"]",
        "\"u32s\":[7]",
        "\"u64s\":[\"5\"]",
        "\"i2s\":{\"7\":\"x\"}",
        "\"l2b\":{\"9\":\"AQID\"}",
        "\"b2e\":{\"true\":\"LIGHT\"}",
        "\"s2p\":{\"k\":{\"name\":\"n\",\"id\":4}}",
        "\"u2u\":{\"3\":\"4\"}",
        "\"p\":{\"name\":\"n\",\"id\":4}",
        "\"ps\":[{\"name\":\"n\",\"id\":4},{\"name\":\"n\",\"id\":4}]",
        "\"opayload\":{\"name\":\"n\",\"id\":4}",
        "\"nested\":{\"inside\":\"INSIDE_ONE\"}",
    ] {
        assert!(first.contains(part), "{part} missing from {first}");
    }
    assert_eq!(json[1], json[0]);
}

#[test]
fn an_any_holding_a_well_known_type_prints_its_value_form() {
    let ts = [build::int(1, 1_700_000_000), build::int(2, 123_000_000)].concat();
    let dur = [build::int(1, 5), build::int(2, 500)].concat();
    let any_ts = any_of("google.protobuf.Timestamp", &ts);
    let any_dur = any_of("google.protobuf.Duration", &dur);
    let any_int = any_of("google.protobuf.Int32Value", &build::int(1, 9));
    let any_str = any_of("google.protobuf.StringValue", &build::len(1, b"s"));
    let any_empty = any_of("google.protobuf.Empty", &[]);
    let inner = any_of("google.protobuf.Int32Value", &build::int(1, 3));
    let any_any = any_of("google.protobuf.Any", &inner);
    let json = any_json(&any_ts, &[any_dur, any_int, any_str, any_empty, any_any]);
    assert_eq!(
        json[0].as_deref(),
        Some(
            "{\"@type\":\"type.googleapis.com/google.protobuf.Timestamp\",\"value\":\"2023-11-14T22:13:20.123Z\"}"
        )
    );
    assert_eq!(
        json[1].as_deref(),
        Some(
            "{\"@type\":\"type.googleapis.com/google.protobuf.Duration\",\"value\":\"5.000000500s\"}"
        )
    );
    assert_eq!(
        json[2].as_deref(),
        Some("{\"@type\":\"type.googleapis.com/google.protobuf.Int32Value\",\"value\":9}")
    );
    assert_eq!(
        json[3].as_deref(),
        Some("{\"@type\":\"type.googleapis.com/google.protobuf.StringValue\",\"value\":\"s\"}")
    );
    assert_eq!(
        json[4].as_deref(),
        Some("{\"@type\":\"type.googleapis.com/google.protobuf.Empty\"}")
    );
    assert!(json[5].as_deref().unwrap().contains("\"value\":{\"@type\""));
}

#[test]
fn an_any_with_an_unknown_type_is_malformed_and_a_struct_inside_is_refused() {
    let unknown = any_of("no.such.Type", &[1, 2]);
    let row = build::len(2, &unknown);
    let strict = decode(
        TYPES,
        "krishiv.golden.types.AnyOnly",
        ANY_OPTS,
        &[Some(&row)],
    );
    assert!(strict.is_err());
    let permissive = decode(
        TYPES,
        "krishiv.golden.types.AnyOnly",
        &[
            ("convert.any.fields.to.json", "true"),
            ("mode", "PERMISSIVE"),
        ],
        &[Some(&row)],
    )
    .unwrap();
    assert!(permissive.is_null(0));
    let refused = any_of("google.protobuf.Struct", &[]);
    let row = build::len(2, &refused);
    assert!(
        decode(
            TYPES,
            "krishiv.golden.types.AnyOnly",
            ANY_OPTS,
            &[Some(&row)]
        )
        .is_err()
    );
}

#[test]
fn a_drive_loses_the_slashes_before_it() {
    assert_eq!(
        super::without_slashes_before_drive("/C:/x/g.desc"),
        "C:/x/g.desc"
    );
    assert_eq!(super::without_slashes_before_drive("///d:"), "d:");
    assert_eq!(
        super::without_slashes_before_drive("/tmp/g.desc"),
        "/tmp/g.desc"
    );
    assert_eq!(super::without_slashes_before_drive("/ab:/x"), "/ab:/x");
}
