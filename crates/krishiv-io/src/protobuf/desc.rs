// SPDX-License-Identifier: Apache-2.0
//! A `FileDescriptorSet` (`protoc --include_imports --descriptor_set_out`) read into the descriptors the codec needs: messages,
//! fields (kind, label, oneof, packing, defaults), enums and the map entries. Read with our own wire reader: a protobuf library
//! (`prost`) may not sit under `krishiv-plan` (`xtask/layers.toml`, ADR 0003), and the schema is a small subset of
//! `descriptor.proto`. Extensions, services, options other than `map_entry`, `packed` and `proto3_optional`, and editions are ignored
//! or refused.

use std::collections::HashMap;
use std::sync::Arc;

use super::wire::{Malformed, Raw, tagged_values};

/// What a field holds.
#[derive(Debug, Clone)]
pub(crate) enum Kind {
    Double,
    Float,
    Int32,
    Int64,
    Uint32,
    Uint64,
    Sint32,
    Sint64,
    Fixed32,
    Fixed64,
    Sfixed32,
    Sfixed64,
    Bool,
    String,
    Bytes,
    Message(MessageDescriptor),
    Enum(EnumDescriptor),
}

#[derive(Debug, Clone, Copy)]
enum Ty {
    Scalar(u8),
    Message(usize),
    Group(usize),
    Enum(usize),
}

#[derive(Debug)]
struct FieldData {
    name: String,
    number: u32,
    /// 1 optional, 2 required, 3 repeated.
    label: u8,
    ty: Ty,
    oneof: Option<usize>,
    proto3_optional: bool,
    packed: Option<bool>,
    default: Option<String>,
}

#[derive(Debug)]
struct OneofData {
    fields: Vec<usize>,
}

#[derive(Debug)]
struct MessageData {
    name: String,
    full_name: String,
    proto2: bool,
    fields: Vec<FieldData>,
    oneofs: Vec<OneofData>,
    map_entry: bool,
    by_number: HashMap<u32, usize>,
}

#[derive(Debug)]
struct EnumData {
    name: String,
    proto2: bool,
    values: Vec<(String, i32)>,
}

/// Every message and enum of a set, in set order.
#[derive(Debug, Default)]
pub(crate) struct Pool {
    messages: Vec<MessageData>,
    enums: Vec<EnumData>,
    message_names: HashMap<String, usize>,
    /// The top-level messages of each file, files in set order.
    top_level: Vec<usize>,
}

/// A message type.
#[derive(Debug, Clone)]
pub(crate) struct MessageDescriptor {
    pool: Arc<Pool>,
    id: usize,
}

/// A field of a message.
#[derive(Debug, Clone)]
pub(crate) struct FieldDescriptor {
    msg: MessageDescriptor,
    index: usize,
}

/// An enum type.
#[derive(Debug, Clone)]
pub(crate) struct EnumDescriptor {
    pool: Arc<Pool>,
    id: usize,
}

/// A real or synthetic (proto3 `optional`) oneof.
#[derive(Debug, Clone)]
pub(crate) struct OneofDescriptor {
    msg: MessageDescriptor,
    index: usize,
}

/// One value of an enum.
#[derive(Debug, Clone)]
pub(crate) struct EnumValue {
    name: String,
    number: i32,
}

impl EnumValue {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn number(&self) -> i32 {
        self.number
    }
}

/// A descriptor set this reader does not build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Refused {
    /// A feature it does not read (editions).
    Unsupported(String),
    /// An import the set does not hold.
    MissingDependency(String),
}

/// Reads a `FileDescriptorSet`. `Err` for bytes that are not one or an unresolvable type name; `Ok(Err(..))` for a file this
/// reader refuses (editions) or an import that is not in the set.
pub(crate) fn parse(bytes: &[u8]) -> Result<Result<Arc<Pool>, Refused>, Malformed> {
    let mut files: Vec<FileProto> = Vec::new();
    for (number, raw) in tagged_values(bytes)? {
        if let (1, Raw::Len(b)) = (number, raw) {
            files.push(file_proto(b)?);
        }
    }
    if let Some(missing) = missing_dependency(&files) {
        return Ok(Err(Refused::MissingDependency(missing)));
    }
    if files.iter().any(|f| f.syntax == "editions") {
        return Ok(Err(Refused::Unsupported(
            "protobuf editions descriptors".to_owned(),
        )));
    }
    // Full names first (a field may name a type of a later file), then the bodies.
    let mut pool = Pool::default();
    let mut pending: Vec<(usize, &MessageProto)> = Vec::new();
    let mut enum_ids: HashMap<String, usize> = HashMap::new();
    for f in &files {
        let proto2 = f.syntax != "proto3";
        let prefix = if f.package.is_empty() {
            String::new()
        } else {
            format!("{}.", f.package)
        };
        for m in &f.messages {
            let id = declare(&mut pool, &mut enum_ids, m, &prefix, proto2, &mut pending)?;
            pool.top_level.push(id);
        }
        for e in &f.enums {
            declare_enum(&mut pool, &mut enum_ids, e, &prefix, proto2);
        }
    }
    for (id, m) in &pending {
        let fields = fields_of(&pool, &enum_ids, m, &pool.messages[*id].full_name)?;
        let (fields, oneofs) = fields;
        let by_number = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.number, i))
            .collect();
        // A map entry is `key = 1` and `value = 2`, singular (what `protoc` writes for a `map<K, V>` field): anything else would
        // leave a map field without its key or value.
        if pool.messages[*id].map_entry && !is_map_entry_shape(&fields) {
            return Err(Malformed);
        }
        let data = &mut pool.messages[*id];
        data.fields = fields;
        data.oneofs = oneofs;
        data.by_number = by_number;
    }
    Ok(Ok(Arc::new(pool)))
}

/// The first import the set does not hold, in the order Spark's `ProtobufUtils.buildFileDescriptor` meets it: files in set order,
/// each file's imports in order, an import's own imports before the next one (depth first). Walked with a stack of its own, so a
/// long chain of imports cannot overflow the thread's.
fn missing_dependency(files: &[FileProto]) -> Option<String> {
    let by_name: HashMap<&str, &FileProto> = files.iter().map(|f| (f.name.as_str(), f)).collect();
    let mut done: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for root in files {
        // (file, index of its next import to visit).
        let mut stack: Vec<(&FileProto, usize)> = vec![(root, 0)];
        while let Some((file, next)) = stack.pop() {
            if done.contains(file.name.as_str()) {
                continue;
            }
            let Some(dep) = file.dependencies.get(next) else {
                done.insert(file.name.as_str());
                continue;
            };
            stack.push((file, next + 1));
            match by_name.get(dep.as_str()) {
                None => return Some(dep.clone()),
                // An import cycle (which protobuf-java refuses) is not followed round again.
                Some(d)
                    if !done.contains(d.name.as_str())
                        && !stack.iter().any(|(f, _)| f.name == d.name) =>
                {
                    stack.push((d, 0));
                }
                Some(_) => {}
            }
        }
    }
    None
}

fn is_map_entry_shape(fields: &[FieldData]) -> bool {
    let singular = |number: u32| {
        fields
            .iter()
            .filter(|f| f.number == number)
            .map(|f| f.label != 3)
            .eq([true])
    };
    fields.len() == 2 && singular(1) && singular(2)
}

fn declare<'a>(
    pool: &mut Pool,
    enum_ids: &mut HashMap<String, usize>,
    m: &'a MessageProto,
    prefix: &str,
    proto2: bool,
    pending: &mut Vec<(usize, &'a MessageProto)>,
) -> Result<usize, Malformed> {
    let full_name = format!("{prefix}{}", m.name);
    let id = pool.messages.len();
    pool.messages.push(MessageData {
        name: m.name.clone(),
        full_name: full_name.clone(),
        proto2,
        fields: Vec::new(),
        oneofs: Vec::new(),
        map_entry: m.map_entry,
        by_number: HashMap::new(),
    });
    pool.message_names.insert(full_name.clone(), id);
    pending.push((id, m));
    let nested_prefix = format!("{full_name}.");
    for n in &m.nested {
        declare(pool, enum_ids, n, &nested_prefix, proto2, pending)?;
    }
    for e in &m.enums {
        declare_enum(pool, enum_ids, e, &nested_prefix, proto2);
    }
    Ok(id)
}

fn declare_enum(
    pool: &mut Pool,
    enum_ids: &mut HashMap<String, usize>,
    e: &EnumProto,
    prefix: &str,
    proto2: bool,
) {
    let full_name = format!("{prefix}{}", e.name);
    enum_ids.insert(full_name, pool.enums.len());
    pool.enums.push(EnumData {
        name: e.name.clone(),
        proto2,
        values: e.values.clone(),
    });
}

type Fields = (Vec<FieldData>, Vec<OneofData>);

fn fields_of(
    pool: &Pool,
    enum_ids: &HashMap<String, usize>,
    m: &MessageProto,
    _full: &str,
) -> Result<Fields, Malformed> {
    let mut fields = Vec::with_capacity(m.fields.len());
    let mut oneofs: Vec<OneofData> = (0..m.oneof_count)
        .map(|_| OneofData { fields: Vec::new() })
        .collect();
    for (i, f) in m.fields.iter().enumerate() {
        let target = |kind: &str| -> Result<usize, Malformed> {
            let name = f.type_name.strip_prefix('.').unwrap_or(&f.type_name);
            match kind {
                "enum" => enum_ids.get(name).copied().ok_or(Malformed),
                _ => pool.message_names.get(name).copied().ok_or(Malformed),
            }
        };
        let ty = match f.ty {
            10 => Ty::Group(target("message")?),
            11 => Ty::Message(target("message")?),
            14 => Ty::Enum(target("enum")?),
            1..=9 | 12 | 13 | 15..=18 => Ty::Scalar(f.ty),
            _ => return Err(Malformed),
        };
        let oneof = match f.oneof_index {
            Some(o) if o < oneofs.len() => {
                oneofs[o].fields.push(i);
                Some(o)
            }
            Some(_) => return Err(Malformed),
            None => None,
        };
        fields.push(FieldData {
            name: f.name.clone(),
            number: f.number,
            label: f.label,
            ty,
            oneof,
            proto3_optional: f.proto3_optional,
            packed: f.packed,
            default: f.default.clone(),
        });
    }
    Ok((fields, oneofs))
}

// ---- the descriptor.proto subset ----

#[derive(Default)]
struct FileProto {
    name: String,
    package: String,
    dependencies: Vec<String>,
    messages: Vec<MessageProto>,
    enums: Vec<EnumProto>,
    syntax: String,
}

#[derive(Default)]
struct MessageProto {
    name: String,
    fields: Vec<FieldProto>,
    nested: Vec<MessageProto>,
    enums: Vec<EnumProto>,
    oneof_count: usize,
    map_entry: bool,
}

#[derive(Default)]
struct FieldProto {
    name: String,
    number: u32,
    label: u8,
    ty: u8,
    type_name: String,
    default: Option<String>,
    oneof_index: Option<usize>,
    proto3_optional: bool,
    packed: Option<bool>,
}

#[derive(Default)]
struct EnumProto {
    name: String,
    values: Vec<(String, i32)>,
}

fn text(b: &[u8]) -> Result<String, Malformed> {
    String::from_utf8(b.to_vec()).map_err(|_| Malformed)
}

fn file_proto(b: &[u8]) -> Result<FileProto, Malformed> {
    let mut f = FileProto::default();
    for (n, raw) in tagged_values(b)? {
        match (n, raw) {
            (1, Raw::Len(v)) => f.name = text(v)?,
            (2, Raw::Len(v)) => f.package = text(v)?,
            (3, Raw::Len(v)) => f.dependencies.push(text(v)?),
            (4, Raw::Len(v)) => f.messages.push(message_proto(v, 0)?),
            (5, Raw::Len(v)) => f.enums.push(enum_proto(v)?),
            (12, Raw::Len(v)) => f.syntax = text(v)?,
            _ => {}
        }
    }
    Ok(f)
}

fn message_proto(b: &[u8], depth: u32) -> Result<MessageProto, Malformed> {
    if depth > 32 {
        return Err(Malformed);
    }
    let mut m = MessageProto::default();
    for (n, raw) in tagged_values(b)? {
        match (n, raw) {
            (1, Raw::Len(v)) => m.name = text(v)?,
            (2, Raw::Len(v)) => m.fields.push(field_proto(v)?),
            (3, Raw::Len(v)) => m.nested.push(message_proto(v, depth + 1)?),
            (4, Raw::Len(v)) => m.enums.push(enum_proto(v)?),
            (7, Raw::Len(v)) => {
                for (on, oraw) in tagged_values(v)? {
                    if let (7, Raw::Varint(x)) = (on, oraw) {
                        m.map_entry = x != 0;
                    }
                }
            }
            (8, Raw::Len(_)) => m.oneof_count += 1,
            _ => {}
        }
    }
    Ok(m)
}

fn field_proto(b: &[u8]) -> Result<FieldProto, Malformed> {
    let mut f = FieldProto::default();
    for (n, raw) in tagged_values(b)? {
        match (n, raw) {
            (1, Raw::Len(v)) => f.name = text(v)?,
            (3, Raw::Varint(v)) => f.number = u32::try_from(v).map_err(|_| Malformed)?,
            (4, Raw::Varint(v)) => f.label = u8::try_from(v).map_err(|_| Malformed)?,
            (5, Raw::Varint(v)) => f.ty = u8::try_from(v).map_err(|_| Malformed)?,
            (6, Raw::Len(v)) => f.type_name = text(v)?,
            (7, Raw::Len(v)) => f.default = Some(text(v)?),
            (9, Raw::Varint(v)) => f.oneof_index = Some(usize::try_from(v).map_err(|_| Malformed)?),
            (17, Raw::Varint(v)) => f.proto3_optional = v != 0,
            (8, Raw::Len(v)) => {
                for (on, oraw) in tagged_values(v)? {
                    if let (2, Raw::Varint(x)) = (on, oraw) {
                        f.packed = Some(x != 0);
                    }
                }
            }
            _ => {}
        }
    }
    if f.number == 0 || f.name.is_empty() {
        return Err(Malformed);
    }
    Ok(f)
}

fn enum_proto(b: &[u8]) -> Result<EnumProto, Malformed> {
    let mut e = EnumProto::default();
    for (n, raw) in tagged_values(b)? {
        match (n, raw) {
            (1, Raw::Len(v)) => e.name = text(v)?,
            (2, Raw::Len(v)) => {
                let (mut name, mut number) = (String::new(), 0i32);
                for (vn, vraw) in tagged_values(v)? {
                    match (vn, vraw) {
                        (1, Raw::Len(s)) => name = text(s)?,
                        (2, Raw::Varint(x)) => number = x as i32,
                        _ => {}
                    }
                }
                e.values.push((name, number));
            }
            _ => {}
        }
    }
    Ok(e)
}

// ---- the descriptors ----

impl Pool {
    /// The top-level messages, files in set order.
    pub(crate) fn top_level(self: &Arc<Self>) -> Vec<MessageDescriptor> {
        self.top_level
            .iter()
            .map(|id| MessageDescriptor {
                pool: Arc::clone(self),
                id: *id,
            })
            .collect()
    }

    /// The message of a full name (`pkg.Outer.Inner`).
    pub(crate) fn message_by_full_name(self: &Arc<Self>, name: &str) -> Option<MessageDescriptor> {
        self.message_names.get(name).map(|id| MessageDescriptor {
            pool: Arc::clone(self),
            id: *id,
        })
    }
}

impl MessageDescriptor {
    fn data(&self) -> &MessageData {
        &self.pool.messages[self.id]
    }

    pub(crate) fn name(&self) -> &str {
        &self.data().name
    }

    pub(crate) fn full_name(&self) -> &str {
        &self.data().full_name
    }

    /// The fields in declaration order.
    pub(crate) fn fields(&self) -> std::vec::IntoIter<FieldDescriptor> {
        (0..self.data().fields.len())
            .map(|index| FieldDescriptor {
                msg: self.clone(),
                index,
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub(crate) fn get_field(&self, number: u32) -> Option<FieldDescriptor> {
        self.data()
            .by_number
            .get(&number)
            .map(|index| FieldDescriptor {
                msg: self.clone(),
                index: *index,
            })
    }

    /// The `key` and `value` fields of a map entry message (`parse` refuses a map entry without both).
    pub(crate) fn map_entry_fields(&self) -> Option<(FieldDescriptor, FieldDescriptor)> {
        Some((self.get_field(1)?, self.get_field(2)?))
    }

    fn is_map_entry(&self) -> bool {
        self.data().map_entry
    }
}

impl FieldDescriptor {
    fn data(&self) -> &FieldData {
        &self.msg.data().fields[self.index]
    }

    pub(crate) fn name(&self) -> &str {
        &self.data().name
    }

    pub(crate) fn full_name(&self) -> String {
        format!("{}.{}", self.msg.full_name(), self.name())
    }

    pub(crate) fn number(&self) -> u32 {
        self.data().number
    }

    pub(crate) fn kind(&self) -> Kind {
        let pool = &self.msg.pool;
        match self.data().ty {
            Ty::Scalar(t) => match t {
                1 => Kind::Double,
                2 => Kind::Float,
                3 => Kind::Int64,
                4 => Kind::Uint64,
                5 => Kind::Int32,
                6 => Kind::Fixed64,
                7 => Kind::Fixed32,
                8 => Kind::Bool,
                9 => Kind::String,
                12 => Kind::Bytes,
                13 => Kind::Uint32,
                15 => Kind::Sfixed32,
                16 => Kind::Sfixed64,
                17 => Kind::Sint32,
                _ => Kind::Sint64,
            },
            Ty::Message(id) | Ty::Group(id) => Kind::Message(MessageDescriptor {
                pool: Arc::clone(pool),
                id,
            }),
            Ty::Enum(id) => Kind::Enum(EnumDescriptor {
                pool: Arc::clone(pool),
                id,
            }),
        }
    }

    pub(crate) fn is_group(&self) -> bool {
        matches!(self.data().ty, Ty::Group(_))
    }

    pub(crate) fn is_required(&self) -> bool {
        self.data().label == 2
    }

    fn is_repeated(&self) -> bool {
        self.data().label == 3
    }

    /// A map field: repeated entries of a map entry message.
    pub(crate) fn is_map(&self) -> bool {
        self.is_repeated() && matches!(self.kind(), Kind::Message(m) if m.is_map_entry())
    }

    /// A repeated field that is not a map.
    pub(crate) fn is_list(&self) -> bool {
        self.is_repeated() && !self.is_map()
    }

    /// Whether a repeated scalar is written packed: proto3 unless `packed = false`, proto2 only with `packed = true`.
    pub(crate) fn is_packed(&self) -> bool {
        self.is_repeated()
            && !matches!(self.kind(), Kind::String | Kind::Bytes | Kind::Message(_))
            && self.data().packed.unwrap_or(!self.is_proto2())
    }

    /// Whether the field is in a proto2 file.
    pub(crate) fn is_proto2(&self) -> bool {
        self.msg.data().proto2
    }

    /// `hasField` is meaningful: proto2 singular fields, proto3 `optional`, oneof members and messages.
    pub(crate) fn supports_presence(&self) -> bool {
        if self.is_repeated() {
            return false;
        }
        matches!(self.data().ty, Ty::Message(_) | Ty::Group(_))
            || self.data().oneof.is_some()
            || self.data().proto3_optional
            || self.is_proto2()
    }

    pub(crate) fn containing_oneof(&self) -> Option<OneofDescriptor> {
        self.data().oneof.map(|index| OneofDescriptor {
            msg: self.msg.clone(),
            index,
        })
    }

    /// Whether proto2 `[default = ..]` is given.
    pub(crate) fn has_explicit_default(&self) -> bool {
        self.data().default.is_some()
    }

    /// The explicit default as the wire value `Raw` reads (`None` without one). A bytes or string default is leaked into the
    /// returned buffer owned by the caller.
    pub(crate) fn explicit_default(&self) -> Option<DefaultValue> {
        let text = self.data().default.as_deref()?;
        Some(match self.kind() {
            Kind::Bool => DefaultValue::Varint(u64::from(text == "true")),
            Kind::Int32 | Kind::Int64 => DefaultValue::Varint(text.parse::<i64>().ok()? as u64),
            Kind::Uint32 | Kind::Uint64 => DefaultValue::Varint(text.parse::<u64>().ok()?),
            Kind::Sint32 => {
                let v = text.parse::<i32>().ok()?;
                DefaultValue::Varint(u64::from(((v << 1) ^ (v >> 31)) as u32))
            }
            Kind::Sint64 => {
                let v = text.parse::<i64>().ok()?;
                DefaultValue::Varint(((v << 1) ^ (v >> 63)) as u64)
            }
            Kind::Fixed32 => DefaultValue::Fixed32(text.parse::<u32>().ok()?),
            Kind::Sfixed32 => DefaultValue::Fixed32(text.parse::<i32>().ok()? as u32),
            Kind::Fixed64 => DefaultValue::Fixed64(text.parse::<u64>().ok()?),
            Kind::Sfixed64 => DefaultValue::Fixed64(text.parse::<i64>().ok()? as u64),
            Kind::Float => DefaultValue::Fixed32((float_text(text)? as f32).to_bits()),
            Kind::Double => DefaultValue::Fixed64(float_text(text)?.to_bits()),
            Kind::String => DefaultValue::Bytes(text.as_bytes().to_vec()),
            Kind::Bytes => DefaultValue::Bytes(unescape(text)),
            Kind::Enum(e) => {
                DefaultValue::Varint(i64::from(e.get_value_by_name(text)?.number()) as u64)
            }
            Kind::Message(_) => return None,
        })
    }
}

/// A proto2 default as a wire value.
#[derive(Debug, Clone)]
pub(crate) enum DefaultValue {
    Varint(u64),
    Fixed32(u32),
    Fixed64(u64),
    Bytes(Vec<u8>),
}

fn float_text(t: &str) -> Option<f64> {
    match t {
        "inf" => Some(f64::INFINITY),
        "-inf" => Some(f64::NEG_INFINITY),
        "nan" | "-nan" => Some(f64::NAN),
        _ => t.parse().ok(),
    }
}

/// protoc's C-style escapes of a `bytes` default.
fn unescape(t: &str) -> Vec<u8> {
    let b = t.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'\\' || i + 1 >= b.len() {
            out.push(b[i]);
            i += 1;
            continue;
        }
        i += 1;
        match b[i] {
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'a' => out.push(7),
            b'b' => out.push(8),
            b'f' => out.push(12),
            b'v' => out.push(11),
            b'x' | b'X' => {
                let mut v = 0u32;
                let mut n = 0;
                while n < 2 && i + 1 < b.len() && b[i + 1].is_ascii_hexdigit() {
                    v = v * 16 + char::from(b[i + 1]).to_digit(16).unwrap_or(0);
                    i += 1;
                    n += 1;
                }
                out.push(v as u8);
            }
            c @ b'0'..=b'7' => {
                let mut v = u32::from(c - b'0');
                let mut n = 1;
                while n < 3 && i + 1 < b.len() && (b'0'..=b'7').contains(&b[i + 1]) {
                    v = v * 8 + u32::from(b[i + 1] - b'0');
                    i += 1;
                    n += 1;
                }
                out.push(v as u8);
            }
            c => out.push(c),
        }
        i += 1;
    }
    out
}

impl EnumDescriptor {
    fn data(&self) -> &EnumData {
        &self.pool.enums[self.id]
    }

    pub(crate) fn name(&self) -> &str {
        &self.data().name
    }

    /// The first value declared with `number` (aliases name the first).
    pub(crate) fn get_value(&self, number: i32) -> Option<EnumValue> {
        self.data()
            .values
            .iter()
            .find(|(_, n)| *n == number)
            .map(|(name, number)| EnumValue {
                name: name.clone(),
                number: *number,
            })
    }

    pub(crate) fn get_value_by_name(&self, name: &str) -> Option<EnumValue> {
        self.data()
            .values
            .iter()
            .find(|(n, _)| n == name)
            .map(|(name, number)| EnumValue {
                name: name.clone(),
                number: *number,
            })
    }

    pub(crate) fn values(&self) -> Vec<EnumValue> {
        self.data()
            .values
            .iter()
            .map(|(name, number)| EnumValue {
                name: name.clone(),
                number: *number,
            })
            .collect()
    }

    /// A proto2 enum is closed: a number it does not define is not a value.
    pub(crate) fn is_closed(&self) -> bool {
        self.data().proto2
    }
}

impl OneofDescriptor {
    pub(crate) fn fields(&self) -> Vec<FieldDescriptor> {
        self.msg.data().oneofs[self.index]
            .fields
            .iter()
            .map(|index| FieldDescriptor {
                msg: self.msg.clone(),
                index: *index,
            })
            .collect()
    }

    /// proto3 `optional` is a oneof of one field.
    pub(crate) fn is_synthetic(&self) -> bool {
        let fields = &self.msg.data().oneofs[self.index].fields;
        fields.len() == 1 && self.msg.data().fields[fields[0]].proto3_optional
    }
}
