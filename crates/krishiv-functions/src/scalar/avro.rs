// SPDX-License-Identifier: Apache-2.0
//! `from_avro` and `to_avro`: Avro binary datums to and from a column
//! (Spark's `AvroDataToCatalyst` and `CatalystDataToAvro`), on the datum
//! codec of `krishiv-io` that the Avro file reader and writer share.
//!
//! The analyzer checks the arguments (see `registry/avro.rs`); the schema
//! and the options map are literals here. `mode` is `FAILFAST` (the
//! default; a bad datum raises) or `PERMISSIVE` (a bad datum is NULL, a
//! record a struct of NULLs); `DROPMALFORMED` is `PARSE_MODE_UNSUPPORTED`
//! and any other text behaves as `PERMISSIVE`, as in Spark. The other
//! options of the Avro data source (`avroSchema`, union naming,
//! `recursiveFieldMaxDepth`, rebase and positional matching) are refused as
//! `KRISHIV_UNSUPPORTED_FEATURE` until they have goldens of their own.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BinaryArray};
use arrow_schema::{DataType as ArrowDataType, Field};
use krishiv_common::error::{
    KrishivError, MalformedRecordInParsingWithoutSuggestion, ParseModeUnsupported,
};
use krishiv_io::FormatError;
use krishiv_io::avro::datum::{DatumDecoder, DatumEncoder};

use super::args::{Inputs, internal, to_utf8, unsupported};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["from_avro", "to_avro"];

/// Options Spark's Avro source reads that are not implemented here.
const UNSUPPORTED_OPTIONS: &[&str] = &[
    "avroschema",
    "enablestableidentifiersforuniontype",
    "stableidentifierprefixforuniontype",
    "recursivefieldmaxdepth",
    "datetimerebasemode",
    "positionalfieldmatching",
];

/// Binds `from_avro` or `to_avro`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let binding = match listed {
        "from_avro" => from(args),
        _ => to(args),
    };
    Some((listed, binding))
}

/// A format error as a classified one.
fn classified(e: FormatError, what: &str) -> KrishivError {
    match e {
        FormatError::Other(e) | FormatError::Failed(e) => e,
        other => unsupported(format!("{what}: {other}")),
    }
}

/// The literal string argument, `None` for NULL.
fn literal_string(arg: &BindArg) -> Result<Option<String>, KrishivError> {
    let BindArg::Literal(a) = arg else {
        return Err(unsupported(
            "an Avro schema that is not a literal".to_owned(),
        ));
    };
    let s = to_utf8(a.as_ref())?;
    Ok(s.is_valid(0).then(|| s.value(0).to_owned()))
}

/// The options literal as a map of lower-cased keys (a NULL or empty map is
/// empty).
fn options_of(arg: Option<&BindArg>) -> Result<BTreeMap<String, String>, KrishivError> {
    let mut out = BTreeMap::new();
    let Some(BindArg::Literal(a)) = arg else {
        return Ok(out);
    };
    if a.is_null(0) || !matches!(a.data_type(), ArrowDataType::Map(..)) {
        return Ok(out);
    }
    let map = a.as_map();
    let offsets = map.value_offsets();
    let (Some(&start), Some(&end)) = (offsets.first(), offsets.get(1)) else {
        return Err(internal("an options map without a first row".to_owned()));
    };
    if start == end {
        return Ok(out);
    }
    let entries = map.entries();
    let keys = to_utf8(entries.column(0).as_ref())?;
    let values = to_utf8(entries.column(1).as_ref())?;
    for i in usize::try_from(start).unwrap_or(0)..usize::try_from(end).unwrap_or(0) {
        if keys.is_valid(i) {
            out.insert(
                keys.value(i).to_ascii_lowercase(),
                if values.is_valid(i) {
                    values.value(i).to_owned()
                } else {
                    String::new()
                },
            );
        }
    }
    Ok(out)
}

/// `from_avro(binary, jsonSchema[, options])`.
fn from(args: &[BindArg]) -> Result<Binding, KrishivError> {
    let (data, schema, options) = match args {
        [d, s] => (d, s, None),
        [d, s, o] => (d, s, Some(o)),
        _ => return Err(bad_types("from_avro", args)),
    };
    if !super::args::is_binary(&data.data_type()) {
        return Err(bad_types("from_avro", args));
    }
    let schema = literal_string(schema)?
        .ok_or_else(|| unsupported("from_avro with a NULL schema".to_owned()))?;
    let options = options_of(options)?;
    if let Some(k) = options
        .keys()
        .find(|k| UNSUPPORTED_OPTIONS.contains(&k.as_str()))
    {
        return Err(unsupported(format!("the from_avro option {k}")));
    }
    let permissive = match options.get("mode").map(|m| m.to_ascii_uppercase()) {
        Some(m) if m == "FAILFAST" => false,
        Some(m) if m == "DROPMALFORMED" => {
            return Err(ParseModeUnsupported {
                func_name: "from_avro".to_owned(),
                mode: "DROPMALFORMED".to_owned(),
            }
            .into());
        }
        // PERMISSIVE, and any text Spark's `ParseMode.fromString` does not
        // know (it falls back to PERMISSIVE), read leniently.
        Some(_) => true,
        None => false,
    };
    // The analyzer's result type is the all-nullable form, which the lenient
    // decoder produces; a strict decode is cast to it.
    let lenient = DatumDecoder::new(&schema, None, true).map_err(|e| classified(e, "from_avro"))?;
    let target = lenient.data_type().clone();
    let decoder =
        DatumDecoder::new(&schema, None, permissive).map_err(|e| classified(e, "from_avro"))?;
    Ok(Binding::new(target, move |x: &Inputs<'_>| {
        let col = x.column(0)?;
        let bin: BinaryArray = super::args::to_binary(col.as_ref())?;
        let rows: Vec<Option<&[u8]>> = (0..bin.len())
            .map(|i| bin.is_valid(i).then(|| bin.value(i)))
            .collect();
        decoder.decode(&rows).map_err(|e| match e {
            FormatError::Corrupt(_) => MalformedRecordInParsingWithoutSuggestion {
                bad_record: String::new(),
                fail_fast_mode: "FAILFAST".to_owned(),
            }
            .into(),
            other => classified(other, "from_avro"),
        })
    }))
}

/// `to_avro(value[, jsonSchema])`.
fn to(args: &[BindArg]) -> Result<Binding, KrishivError> {
    let (value, schema) = match args {
        [v] => (v, None),
        [v, s] => (v, Some(s)),
        _ => return Err(bad_types("to_avro", args)),
    };
    let explicit = match schema {
        Some(s) => literal_string(s)?,
        None => None,
    };
    let encoder = match &explicit {
        Some(json) => DatumEncoder::with_schema(json),
        None => DatumEncoder::infer(&Field::new("value", value.data_type(), value.nullable())),
    }
    .map_err(|e| classified(e, "to_avro"))?;
    let encoder = Arc::new(encoder);
    Ok(Binding::new(
        ArrowDataType::BinaryView,
        move |x: &Inputs<'_>| {
            let col: ArrayRef = x.column(0)?;
            let encoded = encoder.encode(&col).map_err(|e| classified(e, "to_avro"))?;
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
