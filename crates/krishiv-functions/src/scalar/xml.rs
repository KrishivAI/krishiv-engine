// SPDX-License-Identifier: Apache-2.0
//! `from_xml` and `to_xml`: XML documents into structs and back.
//!
//! The model is Spark's XML data source: the text of `from_xml` is one XML
//! document whose root element is the row (its name is not checked, so
//! `rowTag` does not select anything); attributes read as fields named
//! `attributePrefix` + name (default `_`), the text of an element that also
//! has attributes or children as the field named `valueTag` (default
//! `_VALUE`), repeated children as an array, and an element with attributes
//! or children read as a `MAP<STRING, T>` as one entry per attribute and
//! child. Field names match without regard to case. A value that does not
//! convert is NULL (`PERMISSIVE`) or fails the call (`FAILFAST`); text that is
//! not well-formed XML gives a row of NULLs. `to_xml` writes the inverse:
//! NULL fields are left out, an array writes one element per item, and
//! `indent` (default four spaces) pretty-prints.
//!
//! Options this module does not implement are
//! `KRISHIV_UNSUPPORTED_FEATURE` naming the function and option.

use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder, Float64Builder, Int8Builder,
    Int16Builder, Int32Builder, Int64Builder, LargeStringBuilder, StringBuilder, StringViewBuilder,
    TimestampMicrosecondBuilder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, MapArray, StructArray};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Fields, TimeUnit};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_kernels::cast::TimeZoneRules;
use krishiv_kernels::csv::{ConvError, Converter, FieldKind, Val};
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose, Zone};
use krishiv_kernels::error::KernelError;
use krishiv_kernels::text::{
    java_decimal_string, java_double_string, java_float_string32, spark_string_to_timestamp,
    spark_string_to_timestamp_ntz,
};
use quick_xml::Reader;
use quick_xml::events::Event;

use super::args::{Inputs, is_string};
use super::csv::{datetime_option, literal_string, map_pairs, struct_fields};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["from_xml", "to_xml"];

/// How a malformed row is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Permissive,
    FailFast,
}

/// Options of both functions, resolved once at bind.
struct Opts {
    mode: Mode,
    attribute_prefix: String,
    value_tag: String,
    row_tag: String,
    /// Text reading as NULL (`from_xml`), or written for a NULL field
    /// (`to_xml`, when the option is given).
    null_value: Option<String>,
    ignore_surrounding_spaces: bool,
    /// `to_xml` indent: `None` is the default (four spaces).
    indent: Option<String>,
    date: Option<DateTimePattern>,
    timestamp: Option<DateTimePattern>,
    ntz: Option<DateTimePattern>,
}

impl Opts {
    fn parse(pairs: &[(String, String)], func: &str) -> Result<Self, KrishivError> {
        let mut o = Self {
            mode: Mode::Permissive,
            attribute_prefix: "_".to_owned(),
            value_tag: "_VALUE".to_owned(),
            row_tag: "ROW".to_owned(),
            null_value: None,
            ignore_surrounding_spaces: true,
            indent: None,
            date: None,
            timestamp: None,
            ntz: None,
        };
        let purpose = if func == "to_xml" {
            Purpose::Format
        } else {
            Purpose::Parse
        };
        let pattern = |v: &str, name: &str| -> Result<DateTimePattern, KrishivError> {
            // `datetime_option` picks the purpose from the function name.
            let _ = purpose;
            datetime_option(
                v,
                if func == "to_xml" {
                    "to_csv"
                } else {
                    "from_csv"
                },
                name,
            )
        };
        for (k, v) in pairs {
            match k.to_ascii_lowercase().as_str() {
                "mode" => {
                    o.mode = match v.to_ascii_uppercase().as_str() {
                        "PERMISSIVE" => Mode::Permissive,
                        "FAILFAST" => Mode::FailFast,
                        "DROPMALFORMED" => {
                            return Err(krishiv_common::error::ParseModeUnsupported {
                                func_name: format!("`{func}`"),
                                mode: "DROPMALFORMED".to_owned(),
                            }
                            .into());
                        }
                        _ => return Err(unsupported(func, &format!("mode={v}"))),
                    };
                }
                "attributeprefix" => o.attribute_prefix = v.clone(),
                "valuetag" => o.value_tag = v.clone(),
                "rowtag" => o.row_tag = v.clone(),
                "nullvalue" => o.null_value = Some(v.clone()),
                "ignoresurroundingspaces" => {
                    o.ignore_surrounding_spaces = !v.eq_ignore_ascii_case("false");
                }
                "indent" => o.indent = Some(v.clone()),
                "dateformat" => o.date = Some(pattern(v, "dateFormat")?),
                "timestampformat" => o.timestamp = Some(pattern(v, "timestampFormat")?),
                "timestampntzformat" => o.ntz = Some(pattern(v, "timestampNTZFormat")?),
                // Accepted and ignored: they select or label input this
                // function does not read, or change nothing it writes.
                "declaration" | "charset" | "encoding" => {}
                _ => return Err(unsupported(func, &format!("option {k}"))),
            }
        }
        Ok(o)
    }
}

fn unsupported(func: &str, what: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!("{func} with {what}"),
    }
    .into()
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

// ---------------------------------------------------------------------------
// A parsed element tree.
// ---------------------------------------------------------------------------

/// One element: its attributes, child elements, own text and raw content.
#[derive(Debug, Default)]
struct Elem {
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<Elem>,
    /// The element's own text (character data and entity references).
    text: String,
    /// The raw text between the start and end tags.
    inner: String,
}

/// The root element of `text`, or `None` when it is not a well-formed
/// document with an element.
fn parse_document(text: &str) -> Option<Elem> {
    let mut reader = Reader::from_str(text);
    reader.config_mut().trim_text(false);
    let mut stack: Vec<(Elem, usize)> = Vec::new();
    let mut root: Option<Elem> = None;
    loop {
        let before = usize::try_from(reader.buffer_position()).ok()?;
        let event = reader.read_event().ok()?;
        let after = usize::try_from(reader.buffer_position()).ok()?;
        match event {
            Event::Start(e) => stack.push((start_elem(&e)?, after)),
            Event::Empty(e) => {
                let elem = start_elem(&e)?;
                match stack.last_mut() {
                    Some((parent, _)) => parent.children.push(elem),
                    None => {
                        root = Some(elem);
                        break;
                    }
                }
            }
            Event::End(_) => {
                let (mut elem, start) = stack.pop()?;
                elem.inner = text.get(start..before)?.to_owned();
                match stack.last_mut() {
                    Some((parent, _)) => parent.children.push(elem),
                    None => {
                        // Whatever follows the root element is not read.
                        root = Some(elem);
                        break;
                    }
                }
            }
            Event::Text(t) => {
                if let Some((elem, _)) = stack.last_mut() {
                    elem.text.push_str(&t.decode().ok()?);
                }
            }
            Event::CData(c) => {
                if let Some((elem, _)) = stack.last_mut() {
                    elem.text.push_str(&c.decode().ok()?);
                }
            }
            Event::GeneralRef(r) => {
                if let Some((elem, _)) = stack.last_mut() {
                    let resolved = if r.is_char_ref() {
                        r.resolve_char_ref().ok()??.to_string()
                    } else {
                        match r.decode().ok()?.as_ref() {
                            "lt" => "<".to_owned(),
                            "gt" => ">".to_owned(),
                            "amp" => "&".to_owned(),
                            "quot" => "\"".to_owned(),
                            "apos" => "'".to_owned(),
                            _ => return None,
                        }
                    };
                    elem.text.push_str(&resolved);
                }
            }
            Event::Eof => break,
            Event::Comment(_) | Event::Decl(_) | Event::PI(_) | Event::DocType(_) => {}
        }
    }
    if stack.is_empty() { root } else { None }
}

fn start_elem(e: &quick_xml::events::BytesStart<'_>) -> Option<Elem> {
    let mut attrs = Vec::new();
    for a in e.attributes() {
        let a = a.ok()?;
        attrs.push((
            String::from_utf8(a.key.as_ref().to_vec()).ok()?,
            a.normalized_value(quick_xml::XmlVersion::Implicit1_0)
                .ok()?
                .into_owned(),
        ));
    }
    Some(Elem {
        name: String::from_utf8(e.name().as_ref().to_vec()).ok()?,
        attrs,
        ..Elem::default()
    })
}

// ---------------------------------------------------------------------------
// Reading: element tree to values.
// ---------------------------------------------------------------------------

/// A converted value.
#[derive(Debug, Clone)]
enum V {
    Null,
    Leaf(Val),
    List(Vec<V>),
    Struct(Vec<V>),
    Map(Vec<(String, V)>),
}

/// What reading a document needs.
struct Reading<'a> {
    o: &'a Opts,
    conv: Converter<'a>,
    zone: &'a dyn TimeZoneRules,
    today: i32,
}

impl Reading<'_> {
    /// A text as a value of `dt`; sets `bad` when it does not convert.
    fn leaf(&self, dt: &DataType, raw: &str, bad: &mut bool) -> Result<V, KrishivError> {
        let text = if self.o.ignore_surrounding_spaces {
            raw.trim()
        } else {
            raw
        };
        let Ok(kind) = FieldKind::of(dt) else {
            return Err(unsupported("from_xml", &format!("a field of type {dt}")));
        };
        let null_text = self.o.null_value.as_deref();
        if null_text == Some(text) && !(text.is_empty() && kind == FieldKind::Str) {
            return Ok(V::Null);
        }
        if text.is_empty() {
            // An empty element is NULL, except for a string, which is empty.
            return Ok(if kind == FieldKind::Str {
                V::Leaf(Val::Text(String::new()))
            } else {
                V::Null
            });
        }
        let fatal = |e: KernelError| e.error().clone();
        let micros = |parsed: Result<Option<i64>, KrishivError>, bad: &mut bool| match parsed {
            Ok(Some(m)) => V::Leaf(Val::I64(m)),
            _ => {
                *bad = true;
                V::Null
            }
        };
        Ok(match kind {
            FieldKind::Timestamp => micros(
                match &self.o.timestamp {
                    Some(p) => p.parse_timestamp(text, self.zone),
                    None => spark_string_to_timestamp(text, self.zone, self.today),
                },
                bad,
            ),
            FieldKind::TimestampNtz => micros(
                match &self.o.ntz {
                    Some(p) => p.parse_local(text, false),
                    None => Ok(spark_string_to_timestamp_ntz(text)),
                },
                bad,
            ),
            _ => match self.conv.convert(&kind, Some(text)) {
                Ok(Val::Null) => V::Null,
                Ok(v) => V::Leaf(v),
                Err(ConvError::Malformed) => {
                    *bad = true;
                    V::Null
                }
                Err(ConvError::Fatal(e)) => return Err(fatal(e)),
            },
        })
    }

    /// An element as a value of `dt`.
    fn value(&self, dt: &DataType, e: &Elem, bad: &mut bool) -> Result<V, KrishivError> {
        match dt {
            DataType::Struct(fields) => self.row(fields, e, bad),
            DataType::Map(entries, _) => {
                let DataType::Struct(kv) = entries.data_type() else {
                    return Err(internal("a map without entries".to_owned()));
                };
                let Some(vt) = kv.get(1).map(|f| f.data_type().clone()) else {
                    return Err(internal("a map without values".to_owned()));
                };
                let mut out = Vec::new();
                for (k, v) in &e.attrs {
                    let key = format!("{}{k}", self.o.attribute_prefix);
                    out.push((key, self.leaf(&vt, v, bad)?));
                }
                for c in &e.children {
                    out.push((c.name.clone(), self.value(&vt, c, bad)?));
                }
                Ok(V::Map(out))
            }
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 if !e.children.is_empty() => {
                // An element with children read as text is its raw content.
                Ok(V::Leaf(Val::Text(e.inner.trim().to_owned())))
            }
            DataType::List(_) | DataType::LargeList(_) => {
                Err(unsupported("from_xml", "an array inside an array"))
            }
            _ => self.leaf(dt, &e.text, bad),
        }
    }

    /// An element as a struct of `fields`.
    fn row(&self, fields: &Fields, e: &Elem, bad: &mut bool) -> Result<V, KrishivError> {
        let find = |name: &str| {
            fields
                .iter()
                .position(|f| f.name().eq_ignore_ascii_case(name))
        };
        let mut vals: Vec<V> = fields.iter().map(|_| V::Null).collect();
        let mut lists: Vec<Option<Vec<V>>> = fields.iter().map(|_| None).collect();
        let scalar = |f: &Field| {
            !matches!(
                f.data_type(),
                DataType::Struct(_)
                    | DataType::Map(..)
                    | DataType::List(_)
                    | DataType::LargeList(_)
            )
        };
        for (k, v) in &e.attrs {
            if let Some(i) = find(&format!("{}{k}", self.o.attribute_prefix))
                && scalar(&fields[i])
            {
                vals[i] = self.leaf(fields[i].data_type(), v, bad)?;
            }
        }
        if let Some(i) = find(&self.o.value_tag)
            && scalar(&fields[i])
            && !e.text.trim().is_empty()
        {
            vals[i] = self.leaf(fields[i].data_type(), &e.text, bad)?;
        }
        let mut stopped = vec![false; fields.len()];
        for c in &e.children {
            let Some(i) = find(&c.name) else { continue };
            match fields[i].data_type() {
                DataType::List(item) | DataType::LargeList(item) => {
                    if stopped[i] {
                        continue;
                    }
                    let mut item_bad = false;
                    let v = self.value(item.data_type(), c, &mut item_bad)?;
                    if item_bad {
                        // The items before the first bad one are kept.
                        *bad = true;
                        stopped[i] = true;
                        lists[i].get_or_insert_with(Vec::new);
                    } else {
                        lists[i].get_or_insert_with(Vec::new).push(v);
                    }
                }
                dt => vals[i] = self.value(dt, c, bad)?,
            }
        }
        for (i, l) in lists.into_iter().enumerate() {
            if let Some(items) = l {
                vals[i] = V::List(items);
            }
        }
        Ok(V::Struct(vals))
    }
}

// ---------------------------------------------------------------------------
// Values to Arrow.
// ---------------------------------------------------------------------------

fn leaf_of<'a>(v: &'a V, dt: &DataType) -> Result<Option<&'a Val>, KrishivError> {
    match v {
        V::Null => Ok(None),
        V::Leaf(x) => Ok(Some(x)),
        _ => Err(internal(format!("a nested value for a {dt} column"))),
    }
}

fn narrow<T: TryFrom<i64>>(x: &Val, dt: &DataType) -> Result<T, KrishivError> {
    match x {
        Val::I64(n) => T::try_from(*n).map_err(|_| internal(format!("{n} out of range for {dt}"))),
        _ => Err(internal(format!("a non-integer value for {dt}"))),
    }
}

/// The values `vals` as an array of `dt`.
fn build(dt: &DataType, vals: &[V]) -> Result<ArrayRef, KrishivError> {
    macro_rules! primitive {
        ($builder:expr, $get:expr) => {{
            let mut b = $builder;
            for v in vals {
                match leaf_of(v, dt)? {
                    None => b.append_null(),
                    Some(x) => b.append_value($get(x)?),
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }};
    }
    Ok(match dt {
        DataType::Boolean => primitive!(BooleanBuilder::new(), |x: &Val| match x {
            Val::Bool(b) => Ok(*b),
            _ => Err(internal("a non-boolean value".to_owned())),
        }),
        DataType::Int8 => primitive!(Int8Builder::new(), |x: &Val| narrow::<i8>(x, dt)),
        DataType::Int16 => primitive!(Int16Builder::new(), |x: &Val| narrow::<i16>(x, dt)),
        DataType::Int32 => primitive!(Int32Builder::new(), |x: &Val| narrow::<i32>(x, dt)),
        DataType::Int64 => primitive!(Int64Builder::new(), |x: &Val| narrow::<i64>(x, dt)),
        DataType::Float32 => primitive!(Float32Builder::new(), |x: &Val| match x {
            Val::F32(f) => Ok(*f),
            _ => Err(internal("a non-float value".to_owned())),
        }),
        DataType::Float64 => primitive!(Float64Builder::new(), |x: &Val| match x {
            Val::F64(f) => Ok(*f),
            _ => Err(internal("a non-double value".to_owned())),
        }),
        DataType::Decimal128(p, s) => {
            let b = Decimal128Builder::new()
                .with_precision_and_scale(*p, *s)
                .map_err(|e| internal(e.to_string()))?;
            primitive!(b, |x: &Val| match x {
                Val::I128(n) => Ok(*n),
                _ => Err(internal("a non-decimal value".to_owned())),
            })
        }
        DataType::Date32 => primitive!(Date32Builder::new(), |x: &Val| narrow::<i32>(x, dt)),
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let b = TimestampMicrosecondBuilder::new().with_timezone_opt(tz.clone());
            primitive!(b, |x: &Val| narrow::<i64>(x, dt))
        }
        DataType::Utf8 => primitive!(StringBuilder::new(), text_of),
        DataType::LargeUtf8 => primitive!(LargeStringBuilder::new(), text_of),
        DataType::Utf8View => primitive!(StringViewBuilder::new(), text_of),
        DataType::Struct(fields) => {
            let mut valid = Vec::with_capacity(vals.len());
            let mut cols: Vec<Vec<V>> = fields.iter().map(|_| Vec::new()).collect();
            for v in vals {
                match v {
                    V::Struct(items) => {
                        valid.push(true);
                        for (c, item) in cols.iter_mut().zip(items) {
                            c.push(item.clone());
                        }
                    }
                    V::Null => {
                        valid.push(false);
                        for c in &mut cols {
                            c.push(V::Null);
                        }
                    }
                    _ => return Err(internal("a non-struct value".to_owned())),
                }
            }
            let arrays = fields
                .iter()
                .zip(&cols)
                .map(|(f, c)| build(f.data_type(), c))
                .collect::<Result<Vec<_>, _>>()?;
            Arc::new(
                StructArray::try_new(fields.clone(), arrays, Some(NullBuffer::from(valid)))
                    .map_err(|e| internal(e.to_string()))?,
            )
        }
        DataType::List(item) => {
            let mut offsets = vec![0i32];
            let mut valid = Vec::with_capacity(vals.len());
            let mut flat: Vec<V> = Vec::new();
            for v in vals {
                if let V::List(items) = v {
                    valid.push(true);
                    flat.extend(items.iter().cloned());
                } else {
                    valid.push(false);
                }
                offsets.push(
                    i32::try_from(flat.len()).map_err(|_| internal("too many items".to_owned()))?,
                );
            }
            Arc::new(
                arrow_array::ListArray::try_new(
                    Arc::clone(item),
                    OffsetBuffer::new(offsets.into()),
                    build(item.data_type(), &flat)?,
                    Some(NullBuffer::from(valid)),
                )
                .map_err(|e| internal(e.to_string()))?,
            )
        }
        DataType::Map(entries, sorted) => {
            let DataType::Struct(kv) = entries.data_type() else {
                return Err(internal("a map without entries".to_owned()));
            };
            let (Some(kf), Some(vf)) = (kv.first(), kv.get(1)) else {
                return Err(internal("a map without key and value".to_owned()));
            };
            let mut offsets = vec![0i32];
            let mut valid = Vec::with_capacity(vals.len());
            let (mut keys, mut values): (Vec<V>, Vec<V>) = (Vec::new(), Vec::new());
            for v in vals {
                if let V::Map(items) = v {
                    valid.push(true);
                    for (k, x) in items {
                        keys.push(V::Leaf(Val::Text(k.clone())));
                        values.push(x.clone());
                    }
                } else {
                    valid.push(false);
                }
                offsets.push(
                    i32::try_from(keys.len())
                        .map_err(|_| internal("too many entries".to_owned()))?,
                );
            }
            let entries_array = StructArray::try_new(
                kv.clone(),
                vec![
                    build(kf.data_type(), &keys)?,
                    build(vf.data_type(), &values)?,
                ],
                None,
            )
            .map_err(|e| internal(e.to_string()))?;
            Arc::new(
                MapArray::try_new(
                    Arc::clone(entries),
                    OffsetBuffer::new(offsets.into()),
                    entries_array,
                    Some(NullBuffer::from(valid)),
                    *sorted,
                )
                .map_err(|e| internal(e.to_string()))?,
            )
        }
        other => return Err(unsupported("from_xml", &format!("a field of type {other}"))),
    })
}

fn text_of(x: &Val) -> Result<String, KrishivError> {
    match x {
        Val::Text(t) => Ok(t.clone()),
        _ => Err(internal("a non-string value".to_owned())),
    }
}

/// A row as Spark prints a failed record: `[1,null]` (nested fields as
/// `null`; only the top-level scalars are shown).
fn row_text(vals: &[V], fields: &Fields) -> String {
    let parts: Vec<String> = vals
        .iter()
        .zip(fields.iter())
        .map(|(v, f)| match v {
            V::Leaf(Val::Bool(b)) => b.to_string(),
            V::Leaf(Val::I64(n)) => n.to_string(),
            V::Leaf(Val::I128(n)) => match f.data_type() {
                DataType::Decimal128(_, s) => java_decimal_string(*n, *s),
                _ => n.to_string(),
            },
            V::Leaf(Val::F32(x)) => java_float_string32(*x),
            V::Leaf(Val::F64(x)) => java_double_string(*x),
            V::Leaf(Val::Text(t)) => t.clone(),
            _ => "null".to_owned(),
        })
        .collect();
    format!("[{}]", parts.join(","))
}

/// Binds `from_xml(xmlStr, schema[, options])`.
fn bind_from_xml(args: &[BindArg], opts: &BindOptions) -> Result<Binding, KrishivError> {
    if !(2..=3).contains(&args.len()) || !is_string(&args[0].data_type()) {
        return Err(bad_types("from_xml", args));
    }
    let ddl = literal_string(&args[1])?;
    let fields = struct_fields(&ddl)?;
    let mut parsed = Opts::parse(&[], "from_xml")?;
    if args.len() == 3 {
        if !matches!(args[2].data_type(), DataType::Map(_, _)) {
            return Err(bad_types("from_xml", args));
        }
        parsed = Opts::parse(&map_pairs(&args[2])?, "from_xml")?;
    }
    // Fail at bind for a field type the reader cannot hold.
    for f in &fields {
        check_read_type(f.data_type())?;
    }
    let return_type = DataType::Struct(fields.clone());
    let zone: Arc<dyn TimeZoneRules> = Arc::clone(&opts.time_zone);
    let today = krishiv_kernels::datetime::current_date(opts.query_start_micros, zone.as_ref())
        .map_err(KrishivError::from)?;
    Ok(Binding::new(return_type, move |x: &Inputs<'_>| {
        let docs = x.strings(0)?;
        let reading = Reading {
            o: &parsed,
            conv: Converter {
                null_value: "",
                nan: "NaN",
                pos_inf: "Infinity",
                neg_inf: "-Infinity",
                date: parsed.date.as_ref(),
                fallback: false,
            },
            zone: zone.as_ref(),
            today,
        };
        let mut rows: Vec<V> = Vec::with_capacity(docs.len());
        for i in 0..docs.len() {
            if docs.is_null(i) {
                rows.push(V::Null);
                continue;
            }
            let mut bad = false;
            let row = match parse_document(docs.value(i)) {
                Some(root) => reading.row(&fields, &root, &mut bad)?,
                None => {
                    // Not a document, or an empty one: a row of NULLs.
                    bad = !docs.value(i).trim().is_empty();
                    V::Struct(fields.iter().map(|_| V::Null).collect())
                }
            };
            if bad && parsed.mode == Mode::FailFast {
                let text = match &row {
                    V::Struct(vals) => row_text(vals, &fields),
                    _ => "[]".to_owned(),
                };
                return Err(KernelError::malformed_record(i, &text).error().clone());
            }
            rows.push(row);
        }
        build(&DataType::Struct(fields.clone()), &rows)
    }))
}

/// A column type `from_xml` can build.
fn check_read_type(dt: &DataType) -> Result<(), KrishivError> {
    match dt {
        DataType::Struct(fields) => fields
            .iter()
            .try_for_each(|f| check_read_type(f.data_type())),
        DataType::List(f) | DataType::LargeList(f) => {
            if matches!(f.data_type(), DataType::List(_) | DataType::LargeList(_)) {
                return Err(unsupported("from_xml", "an array inside an array"));
            }
            check_read_type(f.data_type())
        }
        DataType::Map(e, _) => check_read_type(e.data_type()),
        other => FieldKind::of(other)
            .map(|_| ())
            .map_err(|_| unsupported("from_xml", &format!("a field of type {other}"))),
    }
}

// ---------------------------------------------------------------------------
// Writing: arrays to text.
// ---------------------------------------------------------------------------

/// What writing a document needs.
struct Writing<'a> {
    o: &'a Opts,
    zone: &'a dyn TimeZoneRules,
    date: &'a DateTimePattern,
    timestamp: &'a DateTimePattern,
    ntz: &'a DateTimePattern,
}

fn escape(text: &str, attribute: bool) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '"' if attribute => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
    out
}

/// The text of a string-carrier value.
fn string_at(a: &ArrayRef, i: usize) -> Option<String> {
    if let Some(s) = a.as_string_view_opt() {
        return Some(s.value(i).to_owned());
    }
    if let Some(s) = a.as_string_opt::<i32>() {
        return Some(s.value(i).to_owned());
    }
    a.as_string_opt::<i64>().map(|s| s.value(i).to_owned())
}

impl Writing<'_> {
    /// A scalar value as its text, `None` when the type is not a scalar.
    fn scalar(&self, a: &ArrayRef, i: usize) -> Result<Option<String>, KrishivError> {
        let t = a.data_type();
        if let Some(s) = string_at(a, i) {
            return Ok(Some(s));
        }
        Ok(Some(match t {
            DataType::Boolean => a.as_boolean().value(i).to_string(),
            DataType::Int8 => a.as_primitive::<Int8Type>().value(i).to_string(),
            DataType::Int16 => a.as_primitive::<Int16Type>().value(i).to_string(),
            DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
            DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
            DataType::Float32 => java_float_string32(a.as_primitive::<Float32Type>().value(i)),
            DataType::Float64 => java_double_string(a.as_primitive::<Float64Type>().value(i)),
            DataType::Decimal128(_, s) => {
                java_decimal_string(a.as_primitive::<Decimal128Type>().value(i), *s)
            }
            DataType::Date32 => self
                .date
                .format_date(a.as_primitive::<Date32Type>().value(i))?,
            DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => {
                self.timestamp.format_timestamp(
                    a.as_primitive::<TimestampMicrosecondType>().value(i),
                    Zone {
                        id: "UTC",
                        rules: self.zone,
                    },
                )?
            }
            DataType::Timestamp(TimeUnit::Microsecond, None) => self
                .ntz
                .format_local(a.as_primitive::<TimestampMicrosecondType>().value(i))?,
            _ => return Ok(None),
        }))
    }

    fn newline(&self, out: &mut String, depth: usize) {
        if let Some(unit) = self.indent() {
            out.push('\n');
            for _ in 0..depth {
                out.push_str(unit);
            }
        }
    }

    fn indent(&self) -> Option<&str> {
        match self.o.indent.as_deref() {
            Some("") => None,
            Some(s) => Some(s),
            None => Some("    "),
        }
    }

    /// The element `name` for row `i` of `a` (a null is left out, or written
    /// with `nullValue` when the option is given).
    fn element(
        &self,
        out: &mut String,
        name: &str,
        a: &ArrayRef,
        i: usize,
        depth: usize,
    ) -> Result<(), KrishivError> {
        if a.is_null(i) {
            if let Some(n) = &self.o.null_value {
                self.newline(out, depth);
                out.push_str(&format!("<{name}>{}</{name}>", escape(n, false)));
            }
            return Ok(());
        }
        match a.data_type() {
            DataType::Struct(_) => self.struct_element(out, name, a.as_struct(), i, depth),
            DataType::List(_) => {
                let list = a.as_list::<i32>();
                let items = list.value(i);
                for j in 0..items.len() {
                    self.element(out, name, &items, j, depth)?;
                }
                Ok(())
            }
            DataType::Map(..) => {
                let map = a.as_map();
                let (keys, values) = (map.keys(), map.values());
                let range = map.value_offsets();
                let (from, to) = (
                    usize::try_from(range[i]).unwrap_or(0),
                    usize::try_from(range[i + 1]).unwrap_or(0),
                );
                self.newline(out, depth);
                out.push_str(&format!("<{name}>"));
                for j in from..to {
                    let key = string_at(keys, j).ok_or_else(|| {
                        unsupported("to_xml", "a map with keys that are not strings")
                    })?;
                    self.element(out, &key, values, j, depth + 1)?;
                }
                self.newline(out, depth);
                out.push_str(&format!("</{name}>"));
                Ok(())
            }
            _ => {
                let text = self.scalar(a, i)?.ok_or_else(|| {
                    unsupported("to_xml", &format!("a field of type {}", a.data_type()))
                })?;
                self.newline(out, depth);
                out.push_str(&format!("<{name}>{}</{name}>", escape(&text, false)));
                Ok(())
            }
        }
    }

    fn struct_element(
        &self,
        out: &mut String,
        name: &str,
        s: &StructArray,
        i: usize,
        depth: usize,
    ) -> Result<(), KrishivError> {
        let prefix = self.o.attribute_prefix.as_str();
        let mut attributes = String::new();
        let mut text = String::new();
        let mut body = String::new();
        for (f, col) in s.fields().iter().zip(s.columns()) {
            if col.is_null(i) {
                // A NULL field is left out, or written as `nullValue`.
                self.element(&mut body, f.name(), col, i, depth + 1)?;
                continue;
            }
            if f.name() == &self.o.value_tag
                && let Some(v) = self.scalar(col, i)?
            {
                text.push_str(&escape(&v, false));
            } else if let Some(attr) = f.name().strip_prefix(prefix)
                && !prefix.is_empty()
                && let Some(v) = self.scalar(col, i)?
            {
                attributes.push_str(&format!(" {attr}=\"{}\"", escape(&v, true)));
            } else {
                self.element(&mut body, f.name(), col, i, depth + 1)?;
            }
        }
        // The document starts with its root element, not a line break.
        if depth > 0 {
            self.newline(out, depth);
        }
        if body.is_empty() && text.is_empty() {
            out.push_str(&format!("<{name}{attributes}/>"));
        } else {
            out.push_str(&format!("<{name}{attributes}>{text}{body}"));
            if !body.is_empty() {
                self.newline(out, depth);
            }
            out.push_str(&format!("</{name}>"));
        }
        Ok(())
    }
}

/// Binds `to_xml(struct[, options])`.
fn bind_to_xml(args: &[BindArg], opts: &BindOptions) -> Result<Binding, KrishivError> {
    if args.is_empty() || args.len() > 2 || !matches!(args[0].data_type(), DataType::Struct(_)) {
        return Err(bad_types("to_xml", args));
    }
    let mut parsed = Opts::parse(&[], "to_xml")?;
    if args.len() == 2 {
        if !matches!(args[1].data_type(), DataType::Map(_, _)) {
            return Err(bad_types("to_xml", args));
        }
        parsed = Opts::parse(&map_pairs(&args[1])?, "to_xml")?;
    }
    let default = |p: &str| DateTimePattern::new(p, Purpose::Format);
    let date = parsed
        .date
        .clone()
        .map_or_else(|| default("yyyy-MM-dd"), Ok)?;
    let timestamp = parsed
        .timestamp
        .clone()
        .map_or_else(|| default("yyyy-MM-dd'T'HH:mm:ss.SSSXXX"), Ok)?;
    let ntz = parsed
        .ntz
        .clone()
        .map_or_else(|| default("yyyy-MM-dd'T'HH:mm:ss.SSS"), Ok)?;
    let zone: Arc<dyn TimeZoneRules> = Arc::clone(&opts.time_zone);
    Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        let col = x.plain_column(0)?;
        let rows = col.as_struct();
        let writing = Writing {
            o: &parsed,
            zone: zone.as_ref(),
            date: &date,
            timestamp: &timestamp,
            ntz: &ntz,
        };
        let mut out = arrow_array::builder::StringBuilder::new();
        for i in 0..rows.len() {
            if rows.is_null(i) {
                out.append_null();
                continue;
            }
            let mut text = String::new();
            writing.struct_element(&mut text, &parsed.row_tag, rows, i, 0)?;
            // Without pretty-printing the document ends with a line feed.
            if writing.indent().is_none() {
                text.push('\n');
            }
            out.append_value(text);
        }
        Ok(Arc::new(out.finish()) as ArrayRef)
    }))
}

/// Binds the XML function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let result = match listed {
        "from_xml" => bind_from_xml(args, opts),
        _ => bind_to_xml(args, opts),
    };
    Some((listed, result))
}
