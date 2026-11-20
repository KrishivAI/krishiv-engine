// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2's `MetricViewFactory.fromYAML` (`sql/catalyst/.../metricview/serde/`): Jackson's data
//! binding over SnakeYAML events, rebuilt over `saphyr-parser` events.
//!
//! Two reads of one document, as Spark makes them: `YAMLVersion` (only `version`, unknown keys
//! skipped), then the strict `MetricViewV01`. What Jackson does that shows in the messages:
//! - A scalar bound to a `String` is its text whatever it looks like (`0.10` stays `0.10`); a plain
//!   `~`, `null`, `Null`, `NULL` or empty scalar is null; an alias is its anchor's name as a string.
//! - The case classes are built through their constructor (property-based creators). An unknown
//!   key is held back until every constructor parameter has been seen; if one is still missing at
//!   the end of the mapping, the first missing required one (by index) is reported before it.
//!   Once all have been seen, an unknown key is reported where it stands.
//! - A value error is reported at its token's start, an unknown key at the end of the last event
//!   read; both name the path down to them (the "reference chain").
//! - Duplicate keys: the last wins. Only the first document is read.
//!
//! A malformed document gives the event parser's message, not SnakeYAML's (design §2).

use krishiv_sql::{ParserOptions, parse_multipart_identifier, parse_query};
use saphyr_parser::{Event, Parser, ScalarStyle, Span};

use super::{MetricViewColumn, MetricViewSource, MetricViewSpec};

const SERDE: &str = "org.apache.spark.sql.metricview.serde";
const SEQ_WRAPPER: &str =
    "com.fasterxml.jackson.module.scala.deser.GenericFactoryDeserializerResolver$BuilderWrapper";

/// Reads `text`; the error is the `<message>` of `INVALID_METRIC_VIEW_YAML`.
pub(super) fn read(text: &str, options: &ParserOptions) -> Result<MetricViewSpec, String> {
    let mut events = Events::new(text);
    let version = match read_version(&mut events) {
        Ok(v) => v,
        Err(Fail::Bound(m)) => return Err(format!("Failed to parse YAML: {m}")),
        Err(Fail::Invalid(m)) => return Err(m),
    };
    match version.as_deref() {
        Some("0.1") => {}
        other => return Err(format!("Invalid YAML version: {}", other.unwrap_or("null"))),
    }
    events.rewind();
    read_v01(&mut events, options).map_err(|f| match f {
        Fail::Bound(m) => format!("Failed to parse YAML: {m}"),
        Fail::Invalid(m) => m,
    })
}

/// Why a read failed: Jackson's (or a Java runtime) message, wrapped as `Failed to parse YAML: ..`
/// by Spark; or one of Spark's own validation messages, which it passes through bare.
enum Fail {
    Bound(String),
    Invalid(String),
}

/// What a plain or tagged scalar is to Jackson's YAML parser: the token it becomes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Shape {
    Str,
    Int,
    Float,
    True,
    False,
    Null,
}

#[derive(Debug)]
enum Tok {
    Scalar(String, Shape),
    MapStart,
    MapEnd,
    SeqStart,
    SeqEnd,
    /// The end of the first document.
    End,
}

/// A line and a column, both from 1, as Jackson prints them.
type Loc = (usize, usize);

struct Item {
    tok: Tok,
    start: Loc,
    end: Loc,
}

/// The first document's events, read up to its end or the first syntax error, which is raised
/// only when a read reaches it (Jackson reads no further than the value it binds).
struct Events {
    items: Vec<Item>,
    failure: Option<(String, Loc)>,
    next: usize,
    current: usize,
}

fn loc(m: saphyr_parser::Marker) -> Loc {
    (m.line(), m.col().saturating_add(1))
}

fn at((line, column): Loc) -> String {
    format!("\n at [Source: (StringReader); line: {line}, column: {column}]")
}

fn chain(path: &[String]) -> String {
    if path.is_empty() {
        String::new()
    } else {
        format!(" (through reference chain: {})", path.join("->"))
    }
}

fn shape(text: &str, style: ScalarStyle, tag: Option<&saphyr_parser::Tag>) -> Shape {
    if let Some(t) = tag.filter(|t| t.is_yaml_core_schema()) {
        match t.suffix.as_str() {
            "str" => return Shape::Str,
            "null" => return Shape::Null,
            "int" => return Shape::Int,
            "float" => return Shape::Float,
            "bool" => {
                return if text.eq_ignore_ascii_case("true") {
                    Shape::True
                } else {
                    Shape::False
                };
            }
            _ => {}
        }
    }
    if style != ScalarStyle::Plain {
        return Shape::Str;
    }
    match text {
        "" | "~" | "null" | "Null" | "NULL" => Shape::Null,
        "true" | "True" | "TRUE" => Shape::True,
        "false" | "False" | "FALSE" => Shape::False,
        _ if is_int(text) => Shape::Int,
        _ if is_float(text) => Shape::Float,
        _ => Shape::Str,
    }
}

#[cfg(test)]
pub(super) fn shape_of(text: &str, style: ScalarStyle) -> Shape {
    shape(text, style, None)
}

fn unsigned(text: &str) -> &str {
    text.strip_prefix(['-', '+']).unwrap_or(text)
}

fn is_int(text: &str) -> bool {
    let t = unsigned(text);
    t.starts_with(|c: char| c.is_ascii_digit()) && t.chars().all(|c| c.is_ascii_digit() || c == '_')
}

fn is_float(text: &str) -> bool {
    let t = unsigned(text);
    if matches!(t, ".inf" | ".Inf" | ".INF") || matches!(text, ".nan" | ".NaN" | ".NAN") {
        return true;
    }
    let (mantissa, exponent) = match t.find(['e', 'E']) {
        Some(i) => (t.get(..i).unwrap_or(""), t.get(i + 1..)),
        None => (t, None),
    };
    let mut digits = 0;
    let mut dots = 0;
    for c in mantissa.chars() {
        match c {
            '0'..='9' => digits += 1,
            '.' => dots += 1,
            '_' => {}
            _ => return false,
        }
    }
    let exponent_ok = exponent.is_none_or(|e| {
        let e = unsigned(e);
        !e.is_empty() && e.chars().all(|c| c.is_ascii_digit())
    });
    digits > 0 && dots == 1 && exponent_ok
}

impl Events {
    fn new(text: &str) -> Events {
        let mut items = Vec::new();
        let mut failure = None;
        for event in Parser::new_from_str(text) {
            let (event, span): (Event<'_>, Span) = match event {
                Ok(e) => e,
                Err(e) => {
                    failure = Some((e.info().to_owned(), loc(*e.marker())));
                    break;
                }
            };
            let tok = match event {
                Event::Nothing | Event::StreamStart | Event::DocumentStart(_) => continue,
                Event::DocumentEnd | Event::StreamEnd => Tok::End,
                Event::Alias(_) => {
                    // The anchor's name, as Jackson's YAML parser hands an alias over.
                    let name: String = text
                        .chars()
                        .skip(span.start.index().saturating_add(1))
                        .take(span.end.index().saturating_sub(span.start.index() + 1))
                        .collect();
                    Tok::Scalar(name, Shape::Str)
                }
                Event::Scalar(value, style, _, tag) => {
                    let s = shape(&value, style, tag.as_deref());
                    Tok::Scalar(value.into_owned(), s)
                }
                Event::SequenceStart(..) => Tok::SeqStart,
                Event::SequenceEnd => Tok::SeqEnd,
                Event::MappingStart(..) => Tok::MapStart,
                Event::MappingEnd => Tok::MapEnd,
            };
            let end = matches!(tok, Tok::End);
            items.push(Item {
                tok,
                start: loc(span.start),
                end: loc(span.end),
            });
            if end {
                break;
            }
        }
        Events {
            items,
            failure,
            next: 0,
            current: 0,
        }
    }

    fn rewind(&mut self) {
        self.next = 0;
        self.current = 0;
    }

    /// Moves to the next event.
    fn advance(&mut self) -> Result<(), Fail> {
        if self.next < self.items.len() {
            self.current = self.next;
            self.next += 1;
            return Ok(());
        }
        match &self.failure {
            Some((info, l)) => Err(Fail::Bound(format!("{info}{}", at(*l)))),
            // A document without events (an empty body).
            None => Err(Fail::Bound(format!(
                "No content to map due to end-of-input{}",
                at((1, 1))
            ))),
        }
    }

    fn item(&self) -> Option<&Item> {
        self.items.get(self.current)
    }

    fn tok(&self) -> &Tok {
        self.item().map_or(&Tok::End, |i| &i.tok)
    }

    /// Jackson's token location: where the current event starts.
    fn start(&self) -> Loc {
        self.item().map_or((1, 1), |i| i.start)
    }

    /// Jackson's current location: where the last event read ends.
    fn end(&self) -> Loc {
        self.item().map_or((1, 1), |i| i.end)
    }

    /// Skips the current value (its children too).
    fn skip(&mut self) -> Result<(), Fail> {
        let mut depth = 0usize;
        loop {
            match self.tok() {
                Tok::MapStart | Tok::SeqStart => depth += 1,
                Tok::MapEnd | Tok::SeqEnd => depth = depth.saturating_sub(1),
                Tok::Scalar(..) | Tok::End => {}
            }
            if depth == 0 {
                return Ok(());
            }
            self.advance()?;
        }
    }
}

/// A case class Jackson binds through its constructor.
struct Bean {
    class: &'static str,
    props: &'static [&'static str],
    required: &'static [bool],
    known: &'static str,
    ignore_unknown: bool,
}

const YAML_VERSION: Bean = Bean {
    class: "YAMLVersion",
    props: &["version"],
    required: &[false],
    known: "1 known property: \"version\"",
    ignore_unknown: true,
};

const V01: Bean = Bean {
    class: "MetricViewV01",
    props: &["version", "source", "filter", "dimensions", "measures"],
    required: &[true, true, false, false, false],
    known: "5 known properties: \"version\", \"source\", \"dimensions\", \"filter\", \"measures\"",
    ignore_unknown: false,
};

const COLUMN: Bean = Bean {
    class: "ColumnV01",
    props: &["name", "expr"],
    required: &[true, true],
    known: "2 known properties: \"name\", \"expr\"",
    ignore_unknown: false,
};

/// A bound `ColumnV01` (`name`, `expr`), or `None` for a null element.
type RawColumn = Option<[Option<String>; 2]>;

/// A bound constructor parameter.
enum Value {
    Text(Option<String>),
    Columns(Option<Vec<RawColumn>>),
}

impl Value {
    fn text(v: Option<Value>) -> Option<String> {
        match v {
            Some(Value::Text(t)) => t,
            _ => None,
        }
    }

    fn columns(v: Option<Value>) -> Vec<RawColumn> {
        match v {
            Some(Value::Columns(c)) => c.unwrap_or_default(),
            _ => Vec::new(),
        }
    }
}

fn property(class: &str, prop: &str) -> String {
    format!("{SERDE}.{class}[\"{prop}\"]")
}

/// `Cannot deserialize value of type <ty> from <token>`, for a token of the wrong shape.
fn mismatch(events: &Events, ty: &str, path: &[String]) -> Fail {
    let token = match events.tok() {
        Tok::MapStart => "Object value (token `JsonToken.START_OBJECT`)",
        Tok::SeqStart => "Array value (token `JsonToken.START_ARRAY`)",
        Tok::Scalar(_, Shape::Int) => "Integer value (token `JsonToken.VALUE_NUMBER_INT`)",
        Tok::Scalar(_, Shape::Float) => {
            "Floating-point value (token `JsonToken.VALUE_NUMBER_FLOAT`)"
        }
        Tok::Scalar(_, Shape::True) => "Boolean value (token `JsonToken.VALUE_TRUE`)",
        Tok::Scalar(_, Shape::False) => "Boolean value (token `JsonToken.VALUE_FALSE`)",
        Tok::Scalar(_, Shape::Null) => "Null value (token `JsonToken.VALUE_NULL`)",
        _ => "String value (token `JsonToken.VALUE_STRING`)",
    };
    Fail::Bound(format!(
        "Cannot deserialize value of type `{ty}` from {token}{}{}",
        at(events.start()),
        chain(path)
    ))
}

/// A scalar where a case class was expected: no single-argument creator takes it.
fn no_scalar_creator(events: &Events, class: &str, path: &[String]) -> Fail {
    let (argument, value) = match events.tok() {
        Tok::Scalar(t, Shape::Int) => ("int/Int", format!("Number value ({t})")),
        Tok::Scalar(t, Shape::Float) => ("double/Double", format!("Number value ({t})")),
        Tok::Scalar(t, Shape::True | Shape::False) => {
            ("boolean/Boolean", format!("Boolean value ({t})"))
        }
        Tok::Scalar(t, _) => ("String", format!("String value ('{t}')")),
        _ => ("String", String::from("String value ('')")),
    };
    Fail::Bound(format!(
        "Cannot construct instance of `{SERDE}.{class}` (although at least one Creator exists): no \
         {argument}-argument constructor/factory method to deserialize from {value}{}{}",
        at(events.start()),
        chain(path)
    ))
}

fn unrecognized(events: &Events, bean: &Bean, name: &str, path: &mut Vec<String>) -> Fail {
    path.push(property(bean.class, name));
    Fail::Bound(format!(
        "Unrecognized field \"{name}\" (class {SERDE}.{}), not marked as ignorable ({}){}{}",
        bean.class,
        bean.known,
        at(events.end()),
        chain(path)
    ))
}

/// A `String` (or `Option[String]`) parameter.
fn text(events: &Events, path: &[String]) -> Result<Option<String>, Fail> {
    match events.tok() {
        Tok::Scalar(_, Shape::Null) => Ok(None),
        Tok::Scalar(t, _) => Ok(Some(t.clone())),
        _ => Err(mismatch(events, "java.lang.String", path)),
    }
}

/// A `Seq[ColumnV01]` parameter.
fn columns(events: &mut Events, path: &mut Vec<String>) -> Result<Option<Vec<RawColumn>>, Fail> {
    match events.tok() {
        Tok::Scalar(_, Shape::Null) => return Ok(None),
        Tok::SeqStart => {}
        _ => {
            let ty = format!("scala.collection.immutable.Seq<{SERDE}.ColumnV01>");
            return Err(mismatch(events, &ty, path));
        }
    }
    let mut out = Vec::new();
    loop {
        events.advance()?;
        if matches!(events.tok(), Tok::SeqEnd) {
            return Ok(Some(out));
        }
        path.push(format!("{SEQ_WRAPPER}[{}]", out.len()));
        let column = match events.tok() {
            Tok::Scalar(_, Shape::Null) => None,
            Tok::Scalar(..) => return Err(no_scalar_creator(events, COLUMN.class, path)),
            Tok::SeqStart => return Err(mismatch(events, &format!("{SERDE}.ColumnV01"), path)),
            _ => {
                let mut v = bean(events, &COLUMN, path)?.into_iter();
                Some([
                    Value::text(v.next().flatten()),
                    Value::text(v.next().flatten()),
                ])
            }
        };
        path.pop();
        out.push(column);
    }
}

/// Binds the mapping at the current event to `bean`'s constructor parameters, in their order.
fn bean(
    events: &mut Events,
    bean: &Bean,
    path: &mut Vec<String>,
) -> Result<Vec<Option<Value>>, Fail> {
    let n = bean.props.len();
    let mut values: Vec<Option<Value>> = (0..n).map(|_| None).collect();
    let mut seen = vec![false; n];
    let mut held: Option<String> = None;
    let mut built = false;
    let mut moved = false;
    loop {
        if !moved {
            events.advance()?;
        }
        moved = false;
        let name = match events.tok() {
            Tok::MapEnd => break,
            Tok::Scalar(t, _) => t.clone(),
            _ => return Err(mismatch(events, &format!("{SERDE}.{}", bean.class), path)),
        };
        events.advance()?;
        let Some(i) = bean.props.iter().position(|p| *p == name) else {
            if bean.ignore_unknown {
                events.skip()?;
            } else if built {
                return Err(unrecognized(events, bean, &name, path));
            } else {
                held.get_or_insert(name);
                events.skip()?;
            }
            continue;
        };
        path.push(property(bean.class, &name));
        let value = if bean.class == V01.class && i >= 3 {
            Value::Columns(columns(events, path)?)
        } else {
            Value::Text(text(events, path)?)
        };
        path.pop();
        if let (Some(slot), Some(s)) = (values.get_mut(i), seen.get_mut(i)) {
            *slot = Some(value);
            *s = true;
        }
        if !built && seen.iter().all(|s| *s) {
            // Every parameter is in: the object is built here, and a held-back key is reported.
            built = true;
            events.advance()?;
            moved = true;
            if let Some(name) = held.take() {
                return Err(unrecognized(events, bean, &name, path));
            }
        }
    }
    if !built {
        let missing = (0..n).find(|&i| {
            !seen.get(i).copied().unwrap_or(true) && bean.required.get(i).copied().unwrap_or(false)
        });
        if let Some(i) = missing {
            let prop = bean.props.get(i).copied().unwrap_or_default();
            path.push(property(bean.class, prop));
            return Err(Fail::Bound(format!(
                "Missing required creator property '{prop}' (index {i}){}{}",
                at(events.start()),
                chain(path)
            )));
        }
        if let Some(name) = held {
            return Err(unrecognized(events, bean, &name, path));
        }
    }
    Ok(values)
}

/// The first read, `readValue(yaml, classOf[YAMLVersion])`: the version, or null.
fn read_version(events: &mut Events) -> Result<Option<String>, Fail> {
    events.advance()?;
    match events.tok() {
        Tok::MapStart => {}
        // `readValue` returns null, and `yamlVersion.version` throws.
        Tok::Scalar(_, Shape::Null) | Tok::End => {
            return Err(Fail::Bound(String::from(
                "Cannot invoke \"org.apache.spark.sql.metricview.serde.YAMLVersion.version()\" \
                 because \"yamlVersion\" is null",
            )));
        }
        Tok::Scalar(..) => return Err(no_scalar_creator(events, YAML_VERSION.class, &[])),
        _ => return Err(mismatch(events, &format!("{SERDE}.YAMLVersion"), &[])),
    }
    let mut values = bean(events, &YAML_VERSION, &mut Vec::new())?;
    Ok(Value::text(values.pop().flatten()))
}

/// The second read, `MetricViewV01`, then `toCanonical`.
fn read_v01(events: &mut Events, options: &ParserOptions) -> Result<MetricViewSpec, Fail> {
    events.advance()?;
    let mut values = bean(events, &V01, &mut Vec::new())?.into_iter();
    let _version = values.next();
    let source = Value::text(values.next().flatten());
    let filter = Value::text(values.next().flatten());
    let dimensions = Value::columns(values.next().flatten());
    let measures = Value::columns(values.next().flatten());
    let canonical = |list: Vec<RawColumn>| {
        list.into_iter()
            .map(|c| {
                // A null element: `column.toCanonical(..)` on null.
                let [name, expr] = c.ok_or_else(|| {
                    Fail::Bound(String::from(
                        "Cannot invoke \"org.apache.spark.sql.metricview.serde.ColumnBase.toCanonical(int, \
                         boolean)\" because \"column\" is null",
                    ))
                })?;
                // Spark keeps a null name or expression and fails later; Krishiv refuses it here.
                match (name, expr) {
                    (Some(name), Some(expr)) => Ok(MetricViewColumn { name, expr }),
                    (None, _) => Err(Fail::Bound(String::from("null value for property 'name'"))),
                    (_, None) => Err(Fail::Bound(String::from("null value for property 'expr'"))),
                }
            })
            .collect::<Result<Vec<_>, Fail>>()
    };
    let dimensions = canonical(dimensions)?;
    let measures = canonical(measures)?;
    let source = source.ok_or_else(|| {
        Fail::Bound(String::from(
            "Cannot invoke \"String.isEmpty()\" because \"sourceText\" is null",
        ))
    })?;
    let source = if source.is_empty() {
        return Err(Fail::Invalid(String::from("Source cannot be empty")));
    } else if parse_multipart_identifier(&source, options).is_ok() {
        MetricViewSource::Asset(source)
    } else if parse_query(&source, options).is_ok() {
        MetricViewSource::Sql(source)
    } else {
        return Err(Fail::Invalid(format!("Invalid source: {source}")));
    };
    Ok(MetricViewSpec {
        source,
        filter,
        dimensions,
        measures,
    })
}
