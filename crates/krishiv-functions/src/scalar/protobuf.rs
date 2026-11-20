// SPDX-License-Identifier: Apache-2.0
//! `from_protobuf` and `to_protobuf`: protobuf messages to and from a column (Spark's `ProtobufDataToCatalyst` and
//! `CatalystDataToProtobuf`), on the codec of `krishiv-io`. The analyzer checks the arguments (see `registry/protobuf.rs`); the
//! message name, descriptor and options are literals here. A descriptor path arrives as the binary literal of the file's bytes,
//! which the analyzer read once under the access policy; a string path that reaches here is refused by `Call`, never read.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::KrishivError;
use krishiv_io::protobuf::{Call, DescriptorArg, spark_type_of};

use super::args::{Inputs, internal, to_binary, to_utf8, unsupported};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["from_protobuf", "to_protobuf"];

/// Binds `from_protobuf` or `to_protobuf`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let binding = if listed == "from_protobuf" {
        from(args)
    } else {
        to(args)
    };
    Some((listed, binding))
}

fn literal<'a>(arg: &'a BindArg, what: &str) -> Result<&'a ArrayRef, KrishivError> {
    match arg {
        BindArg::Literal(a) => Ok(a),
        BindArg::Column { .. } => Err(unsupported(format!(
            "a protobuf {what} that is not a literal"
        ))),
    }
}

fn is_string(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View
    )
}

fn is_binary(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::Binary | ArrowType::LargeBinary | ArrowType::BinaryView
    )
}

/// The descriptor argument: a literal string or binary (NULL is none).
fn descriptor_of(arg: &BindArg) -> Result<DescriptorArg, KrishivError> {
    let a = literal(arg, "descriptor")?;
    if a.is_null(0) || matches!(a.data_type(), ArrowType::Null) {
        return Ok(DescriptorArg::None);
    }
    if is_string(a.data_type()) {
        let s = to_utf8(a.as_ref())?;
        return Ok(DescriptorArg::Path(s.value(0).to_owned()));
    }
    if is_binary(a.data_type()) {
        let b = to_binary(a.as_ref())?;
        return Ok(DescriptorArg::Bytes(b.value(0).to_vec()));
    }
    Ok(DescriptorArg::None)
}

/// The options literal as entries (a NULL or empty map is empty).
fn options_of(arg: &BindArg) -> Result<Vec<(String, String)>, KrishivError> {
    let a = literal(arg, "options map")?;
    if a.is_null(0) || !matches!(a.data_type(), ArrowType::Map(..)) {
        return Ok(Vec::new());
    }
    let map = a.as_map();
    let offsets = map.value_offsets();
    let (Some(&start), Some(&end)) = (offsets.first(), offsets.get(1)) else {
        return Ok(Vec::new());
    };
    let entries = map.entries();
    let keys = to_utf8(entries.column(0).as_ref())?;
    let values = to_utf8(entries.column(1).as_ref())?;
    let mut out = Vec::new();
    for i in usize::try_from(start).unwrap_or(0)..usize::try_from(end).unwrap_or(0) {
        if keys.is_valid(i) {
            out.push((
                keys.value(i).to_owned(),
                if values.is_valid(i) {
                    values.value(i).to_owned()
                } else {
                    String::new()
                },
            ));
        }
    }
    Ok(out)
}

/// The call from the arguments after the first (Spark's constructors: a three-argument call's third is the descriptor when it is a
/// literal string or binary, the options when it is a map).
fn call_of(name: &str, args: &[BindArg], to: bool) -> Result<Call, KrishivError> {
    let message = {
        let a = literal(&args[1], "message name")?;
        if !is_string(a.data_type()) || a.is_null(0) {
            return Err(bad_types(name, args));
        }
        to_utf8(a.as_ref())?.value(0).to_owned()
    };
    let (descriptor, options) = match args {
        [_, _] => (DescriptorArg::None, Vec::new()),
        [_, _, third] => match third.data_type() {
            ArrowType::Map(..) => (DescriptorArg::None, options_of(third)?),
            t if (is_string(&t) || is_binary(&t)) && matches!(third, BindArg::Literal(_)) => {
                (descriptor_of(third)?, Vec::new())
            }
            _ => (DescriptorArg::None, Vec::new()),
        },
        [_, _, desc, options] => (descriptor_of(desc)?, options_of(options)?),
        _ => return Err(bad_types(name, args)),
    };
    Call::new(&message, descriptor, options, to)
}

/// `from_protobuf(binary, messageName[, descFilePath][, options])`.
fn from(args: &[BindArg]) -> Result<Binding, KrishivError> {
    if args.len() < 2 || !is_binary(&args[0].data_type()) {
        return Err(bad_types("from_protobuf", args));
    }
    let decoder = Arc::new(call_of("from_protobuf", args, false)?.decoder()?);
    let target = decoder.arrow_type().clone();
    Ok(Binding::new(target, move |x: &Inputs<'_>| {
        let col = x.column(0)?;
        // Borrow the payloads from whichever binary carrier the column has (no copy of the input).
        let a = col.as_ref();
        let rows: Vec<Option<&[u8]>> = match a.data_type() {
            ArrowType::Binary => {
                let b = a.as_binary::<i32>();
                (0..b.len())
                    .map(|i| b.is_valid(i).then(|| b.value(i)))
                    .collect()
            }
            ArrowType::LargeBinary => {
                let b = a.as_binary::<i64>();
                (0..b.len())
                    .map(|i| b.is_valid(i).then(|| b.value(i)))
                    .collect()
            }
            ArrowType::BinaryView => {
                let b = a.as_binary_view();
                (0..b.len())
                    .map(|i| b.is_valid(i).then(|| b.value(i)))
                    .collect()
            }
            _ => return Err(internal("from_protobuf: a non-binary input".to_owned())),
        };
        decoder.decode(&rows)
    }))
}

/// `to_protobuf(struct, messageName[, descFilePath][, options])`.
fn to(args: &[BindArg]) -> Result<Binding, KrishivError> {
    if args.len() < 2 || !matches!(args[0].data_type(), ArrowType::Struct(_)) {
        return Err(bad_types("to_protobuf", args));
    }
    let input = spark_type_of(&args[0].data_type(), args[0].nullable())?;
    let encoder = Arc::new(call_of("to_protobuf", args, true)?.encoder(&input)?);
    Ok(Binding::new(
        ArrowType::BinaryView,
        move |x: &Inputs<'_>| {
            let col: ArrayRef = x.column(0)?;
            let encoded = encoder.encode(&col)?;
            let mut out = arrow_array::builder::BinaryViewBuilder::new();
            for v in &encoded {
                match v {
                    Some(b) => out.append_value(b),
                    None => out.append_null(),
                }
            }
            Ok(Arc::new(out.finish()) as ArrayRef)
        },
    ))
}
