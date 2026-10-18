// SPDX-License-Identifier: Apache-2.0
//! Generates the Spark Connect Rust types from the vendored `.proto` files.
//!
//! ```text
//! cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --write [VERSION]
//! cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --check [VERSION]
//! ```
//!
//! Run it from the repository root. `VERSION` is the vendored Spark minor
//! (default `4.2`; `v<VERSION>` is `v4_2`). These files come out, all under
//! `crates/krishiv-connect`:
//!
//! - `src/proto/v<VERSION>/spark.connect.pb.rs`: the `prost` message types;
//! - `src/proto/v<VERSION>/layout.pb.rs`: for every message, its name, the size
//!   of its Rust type and what each of its fields holds. The decoder walks
//!   the bytes with it to bound the nesting depth and the decoded size of a
//!   plan, and to reject fields the version does not define, before `prost`
//!   recurses into it;
//! - `src/proto/v<VERSION>/debug.pb.rs`: `Debug` for every message, enum and
//!   `oneof`. It prints the structure of a plan but no string or byte
//!   string (SQL text, literals, names, paths), map or `Any` payload, only
//!   their sizes, so no `{:?}` of a decoded plan leaks a credential;
//! - `src/proto/v<VERSION>/scrub.pb.rs`: `Scrub` for every message that can
//!   hold a credential (option values, paths, `Any` payloads) or a message
//!   that can: it overwrites them with zeros before a decoded plan is
//!   dropped (ADR 0038);
//! - `src/proto/v<VERSION>/oneofs.pb.rs`: for every `oneof` variant, its field
//!   name, number and type, so an error can name the proto field that is not
//!   supported without formatting the (possibly secret) message;
//! - `tests/fixtures/spark-connect-<VERSION>.fds`: the compiled descriptor
//!   set, used by the golden tests to read protobuf JSON.
//!
//! `--check` regenerates in memory and fails when a file on disk differs, so
//! CI notices a proto bump without a regeneration. The compiler is `protox`,
//! a pure-Rust implementation: no `protoc` is needed on any machine.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use heck::{ToSnakeCase, ToUpperCamelCase};
use prost::Message;
use prost_build::Module;
use prost_types::field_descriptor_proto::{Label, Type};
use prost_types::{DescriptorProto, FieldDescriptorProto, FileDescriptorSet};

/// Messages the decoder starts from; they get a named constant in
/// `layout.pb.rs`.
const ROOTS: &[&str] = &[
    ".spark.connect.Plan",
    ".spark.connect.Relation",
    ".spark.connect.Expression",
];

/// `oneof`s whose variants differ so much in size that the enum would be
/// large (clippy `large_enum_variant`): their variants are boxed. None is
/// on the plan decoding path (`Relation` and `Expression` hold their
/// variants inline).
const BOXED_ONEOFS: &[&str] = &[
    ".spark.connect.Plan.op_type",
    ".spark.connect.Command.command_type",
    ".spark.connect.AnalyzePlanRequest.analyze",
    ".spark.connect.PipelineCommand.command_type",
    ".spark.connect.PipelineCommand.DefineFlow.details",
];

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("connect-protogen: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut mode = None;
    let mut version = "4.2".to_owned();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--write" | "--check" => mode = Some(arg),
            v if !v.starts_with('-') => version = v.to_owned(),
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    let mode = mode.ok_or("pass --write or --check")?;
    let root = PathBuf::from("crates/krishiv-connect");
    let outputs = generate(&root, &version)?;
    let mut stale = Vec::new();
    for (path, bytes) in &outputs {
        if mode == "--write" {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(path, bytes)?;
        } else if std::fs::read(path).ok().as_deref() != Some(bytes.as_slice()) {
            stale.push(path.display().to_string());
        }
    }
    if stale.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "generated files are out of date: {}; run the generator with --write",
            stale.join(", ")
        )
        .into())
    }
}

/// The generated files, by path.
fn generate(root: &Path, version: &str) -> Result<Vec<(PathBuf, Vec<u8>)>> {
    let proto_dir = Path::new("proto/spark-connect").join(version);
    let mut files = Vec::new();
    for entry in std::fs::read_dir(proto_dir.join("spark/connect"))? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if name.ends_with(".proto") {
            files.push(format!("spark/connect/{name}"));
        }
    }
    files.sort();
    let fds = protox::compile(&files, [&proto_dir])?;

    let mut config = prost_build::Config::new();
    // Deterministic iteration order for map fields (read options).
    config.btree_map(["."]);
    // The vendored comments are Spark's; they are not rustdoc.
    config.disable_comments(["."]);
    for path in BOXED_ONEOFS {
        config.boxed(path);
    }
    // `debug.pb.rs` writes every `Debug`, so none is derived.
    config.skip_debug(["."]);
    let requests = fds
        .file
        .iter()
        .map(|f| (Module::from_protobuf_package_name(f.package()), f.clone()))
        .collect();
    let module = config
        .generate(requests)?
        .into_iter()
        .find_map(|(module, code)| module.parts().eq(["spark", "connect"]).then_some(code))
        .ok_or("no spark.connect module generated")?;

    let header = format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
         // @generated by tools/connect-protogen from proto/spark-connect/{version}; do not edit.\n"
    );
    let package = Package::new(&fds);
    let module_dir = root
        .join("src/proto")
        .join(format!("v{}", version.replace('.', "_")));
    Ok(vec![
        (
            module_dir.join("spark.connect.pb.rs"),
            format!("{header}{module}").into_bytes(),
        ),
        (
            module_dir.join("layout.pb.rs"),
            layout_table(&header, &fds)?.into_bytes(),
        ),
        (
            module_dir.join("debug.pb.rs"),
            redacting_debug(&header, &package)?.into_bytes(),
        ),
        (
            module_dir.join("scrub.pb.rs"),
            scrubbing(&header, &package)?.into_bytes(),
        ),
        (
            module_dir.join("oneofs.pb.rs"),
            oneof_variants(&header, &fds)?.into_bytes(),
        ),
        (
            root.join(format!("tests/fixtures/spark-connect-{version}.fds")),
            fds.encode_to_vec(),
        ),
    ])
}

/// For every message of the package, by id (the index in name order): its
/// name, the size of its Rust type (a constant for map entries, which have
/// none) and, sorted by number, what each field holds. A message from
/// outside the package (`google.protobuf.Any`, `Timestamp`) is a leaf: prost
/// decodes it without recursing into the package.
fn layout_table(header: &str, fds: &FileDescriptorSet) -> Result<String> {
    let mut messages: BTreeMap<String, &DescriptorProto> = BTreeMap::new();
    for file in &fds.file {
        if file.package() != "spark.connect" {
            continue;
        }
        for m in &file.message_type {
            collect(".spark.connect", m, &mut messages);
        }
    }
    let ids: BTreeMap<&str, usize> = messages
        .keys()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    if u16::try_from(ids.len()).is_err() {
        return Err("too many messages for u16 ids".into());
    }
    let mut out = String::from(header);
    out.push_str("\nuse super::{Field, Layout, MAP_ENTRY_SIZE};\n\n/// Message ids, by name.\n");
    for root in ROOTS {
        let id = ids
            .get(root)
            .ok_or_else(|| format!("{root} is not a message"))?;
        let name = root.rsplit('.').next().unwrap_or(root).to_uppercase();
        writeln!(out, "pub(crate) const {name}: u16 = {id};")?;
    }
    writeln!(
        out,
        "\n/// Every message of the package, by id.\n\
         pub(crate) static MESSAGES: [Layout; {}] = [",
        ids.len()
    )?;
    for (name, message) in &messages {
        let mut fields = Vec::new();
        for f in &message.field {
            let kind = match f.r#type() {
                Type::Message | Type::Group => ids.get(f.type_name()).map_or_else(
                    || "Field::Bytes".to_owned(),
                    |id| format!("Field::Message({id})"),
                ),
                Type::String | Type::Bytes => "Field::Bytes".to_owned(),
                Type::Bool => "Field::Varint(1)".to_owned(),
                Type::Int32 | Type::Uint32 | Type::Sint32 | Type::Enum => {
                    "Field::Varint(4)".to_owned()
                }
                Type::Int64 | Type::Uint64 | Type::Sint64 => "Field::Varint(8)".to_owned(),
                Type::Fixed32
                | Type::Sfixed32
                | Type::Float
                | Type::Fixed64
                | Type::Sfixed64
                | Type::Double => "Field::Fixed".to_owned(),
            };
            fields.push((u32::try_from(f.number())?, kind));
        }
        fields.sort_unstable();
        let list: Vec<String> = fields
            .iter()
            .map(|(n, kind)| format!("({n}, {kind})"))
            .collect();
        let short = name.trim_start_matches(".spark.connect.");
        let size = if message.options.as_ref().is_some_and(|o| o.map_entry()) {
            "MAP_ENTRY_SIZE".to_owned()
        } else {
            format!("size_of::<super::v4_2::{}>()", rust_path(short))
        };
        writeln!(
            out,
            "    Layout {{ name: {short:?}, size: {size}, fields: &[{}] }},",
            list.join(", ")
        )?;
    }
    out.push_str("];\n");
    Ok(out)
}

/// The Rust path prost gives a message, from its name in the package
/// (`Read.DataSource` is `read::DataSource`).
fn rust_path(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let (last, modules) = parts.split_last().unwrap_or((&name, &[]));
    modules
        .iter()
        .map(|m| m.to_snake_case())
        .chain([last.to_upper_camel_case()])
        .collect::<Vec<_>>()
        .join("::")
}

fn collect<'a>(
    prefix: &str,
    m: &'a DescriptorProto,
    out: &mut BTreeMap<String, &'a DescriptorProto>,
) {
    let name = format!("{prefix}.{}", m.name());
    for nested in &m.nested_type {
        collect(&name, nested, out);
    }
    out.insert(name, m);
}

/// `impl <oneof enum> { fn variant(&self) -> Variant }` for every oneof of
/// the package. The Rust names follow prost's: the enum is the camel-cased
/// oneof name in the snake-cased module of its message (and of the messages
/// it is nested in); a variant is the camel-cased field name.
fn oneof_variants(header: &str, fds: &FileDescriptorSet) -> Result<String> {
    let mut out = String::from(header);
    out.push_str("\nuse super::v4_2::*;\nuse super::Variant;\n");
    for file in fds.file.iter().filter(|f| f.package() == "spark.connect") {
        for message in &file.message_type {
            oneofs_of(message, &mut Vec::new(), &mut out)?;
        }
    }
    Ok(out)
}

fn oneofs_of(message: &DescriptorProto, path: &mut Vec<String>, out: &mut String) -> Result<()> {
    path.push(message.name().to_snake_case());
    for nested in &message.nested_type {
        oneofs_of(nested, path, out)?;
    }
    // The module of this message holds its oneof enums, except at the top
    // level, where the message's own module has the same name.
    let module = path.join("::");
    for (index, oneof) in message.oneof_decl.iter().enumerate() {
        let fields: Vec<_> = message
            .field
            .iter()
            .filter(|f| f.oneof_index == Some(i32::try_from(index).unwrap_or(-1)))
            .collect();
        // `optional` fields of proto3 are one-field oneofs in the
        // descriptor; prost makes them `Option`s, not enums.
        if fields.iter().all(|f| f.proto3_optional()) {
            continue;
        }
        writeln!(
            out,
            "\nimpl {module}::{} {{\n    pub(crate) const fn variant(&self) -> Variant {{\n        match self {{",
            oneof.name().to_upper_camel_case()
        )?;
        for f in fields {
            writeln!(
                out,
                "            Self::{}(_) => Variant {{ field: {:?}, number: {}, type_name: {:?} }},",
                f.name().to_upper_camel_case(),
                f.name(),
                f.number(),
                if f.type_name().is_empty() {
                    format!("{:?}", f.r#type())
                } else {
                    f.type_name().trim_start_matches('.').to_owned()
                },
            )?;
        }
        out.push_str("        }\n    }\n}\n");
    }
    path.pop();
    Ok(())
}

/// The messages and enums of the package, by full name
/// (`.spark.connect.Read.DataSource`).
struct Package<'a> {
    messages: BTreeMap<String, &'a DescriptorProto>,
    enums: Vec<String>,
}

/// How the generated code treats a field.
#[derive(Clone, PartialEq, Eq)]
enum Kind {
    /// A `map`: `None` when its values are strings (option maps), else the
    /// full name of the value message (empty for other scalar values).
    Map(Option<String>),
    /// A `string` or `bytes`.
    Text,
    /// A `google.protobuf.Any`.
    Any,
    /// A message of the package, by full name.
    Message(String),
    /// An enum, by full name.
    Enum(String),
    /// Anything else.
    Scalar,
}

/// How many values a field holds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Count {
    One,
    Optional,
    Repeated,
}

impl<'a> Package<'a> {
    fn new(fds: &'a FileDescriptorSet) -> Self {
        fn visit<'a>(
            prefix: &str,
            m: &'a DescriptorProto,
            messages: &mut BTreeMap<String, &'a DescriptorProto>,
            enums: &mut Vec<String>,
        ) {
            let name = format!("{prefix}.{}", m.name());
            for nested in &m.nested_type {
                visit(&name, nested, messages, enums);
            }
            for e in &m.enum_type {
                enums.push(format!("{name}.{}", e.name()));
            }
            messages.insert(name, m);
        }
        let mut messages = BTreeMap::new();
        let mut enums = Vec::new();
        for file in fds.file.iter().filter(|f| f.package() == "spark.connect") {
            for m in &file.message_type {
                visit(".spark.connect", m, &mut messages, &mut enums);
            }
            for e in &file.enum_type {
                enums.push(format!(".spark.connect.{}", e.name()));
            }
        }
        enums.sort();
        Self { messages, enums }
    }

    /// Whether prost marks the field `#[deprecated]`: it does for every
    /// deprecated field but a map.
    fn is_deprecated(&self, f: &FieldDescriptorProto) -> bool {
        f.options.as_ref().is_some_and(|o| o.deprecated()) && !self.is_map_entry(f.type_name())
    }

    fn is_map_entry(&self, name: &str) -> bool {
        self.messages
            .get(name)
            .and_then(|m| m.options.as_ref())
            .is_some_and(|o| o.map_entry())
    }

    /// The messages that are not map entries, by full name.
    fn structs(&self) -> impl Iterator<Item = (&String, &&'a DescriptorProto)> {
        self.messages.iter().filter(|(n, _)| !self.is_map_entry(n))
    }

    fn kind(&self, f: &FieldDescriptorProto) -> (Kind, Count) {
        let count = if f.label() == Label::Repeated {
            Count::Repeated
        } else if f.proto3_optional() {
            Count::Optional
        } else {
            Count::One
        };
        let kind = match f.r#type() {
            Type::String | Type::Bytes => Kind::Text,
            Type::Enum => Kind::Enum(f.type_name().to_owned()),
            Type::Message | Type::Group if f.type_name() == ".google.protobuf.Any" => Kind::Any,
            Type::Message | Type::Group if self.is_map_entry(f.type_name()) => {
                let value = self.messages[f.type_name()]
                    .field
                    .iter()
                    .find(|v| v.number() == 2);
                let value = match value.map(FieldDescriptorProto::r#type) {
                    Some(Type::String) => None,
                    Some(Type::Message) => value.map(|v| v.type_name().to_owned()),
                    _ => Some(String::new()),
                };
                return (Kind::Map(value), Count::Repeated);
            }
            Type::Message | Type::Group if self.messages.contains_key(f.type_name()) => {
                Kind::Message(f.type_name().to_owned())
            }
            _ => Kind::Scalar,
        };
        (kind, count)
    }

    /// The messages whose `Scrub` has something to do: those with an option
    /// map, a `path`/`paths` string or an `Any`, and those holding one of
    /// them.
    fn scrubbed(&self) -> BTreeMap<&str, ()> {
        let mut out: BTreeMap<&str, ()> = BTreeMap::new();
        loop {
            let before = out.len();
            for (name, m) in self.structs() {
                let needs = m.field.iter().any(|f| match self.kind(f).0 {
                    Kind::Map(None) | Kind::Any => true,
                    Kind::Text => is_path(f),
                    Kind::Message(t) | Kind::Map(Some(t)) => out.contains_key(t.as_str()),
                    Kind::Enum(_) | Kind::Scalar => false,
                });
                if needs {
                    out.insert(name, ());
                }
            }
            if out.len() == before {
                return out;
            }
        }
    }
}

/// A string field that holds a path or a URL (a presigned URL carries its
/// credential).
fn is_path(f: &FieldDescriptorProto) -> bool {
    matches!(f.name(), "path" | "paths")
}

/// The Rust path of a message or enum from its full name
/// (`.spark.connect.Read.DataSource` is `read::DataSource`).
fn rust_type(full: &str) -> String {
    rust_path(full.trim_start_matches(".spark.connect."))
}

/// The Rust module holding the nested types of a message
/// (`.spark.connect.Read.DataSource` is `read::data_source`).
fn rust_module(full: &str) -> String {
    full.trim_start_matches(".spark.connect.")
        .split('.')
        .map(ToSnakeCase::to_snake_case)
        .collect::<Vec<_>>()
        .join("::")
}

fn short_name(full: &str) -> &str {
    full.rsplit('.').next().unwrap_or(full)
}

/// The fields of `m` outside `oneof`s (a proto3 `optional` is one), and
/// its `oneof`s with their fields.
fn split_fields(
    m: &DescriptorProto,
) -> (
    Vec<&FieldDescriptorProto>,
    Vec<(&str, Vec<&FieldDescriptorProto>)>,
) {
    let plain = m
        .field
        .iter()
        .filter(|f| f.oneof_index.is_none() || f.proto3_optional())
        .collect();
    let oneofs = m
        .oneof_decl
        .iter()
        .enumerate()
        .map(|(index, o)| {
            let members = m
                .field
                .iter()
                .filter(|f| f.oneof_index == i32::try_from(index).ok() && !f.proto3_optional())
                .collect::<Vec<_>>();
            (o.name(), members)
        })
        .filter(|(_, members)| !members.is_empty())
        .collect();
    (plain, oneofs)
}

const EXPECT_DEPRECATED: &str =
    "    #[expect(deprecated, reason = \"the generated code reads every field\")]\n";

/// The `Debug` expression of a value of kind `kind` held in `v` (an
/// expression of type `&T` for one value, or of the field's type).
fn debug_value(kind: &Kind, count: Count, v: &str) -> String {
    match (kind, count) {
        (Kind::Map(_), _) => format!("&Redacted::len({v}.len(), \"entries\")"),
        (Kind::Text, Count::One) => format!("&Redacted::len({v}.len(), \"bytes\")"),
        (Kind::Text, Count::Optional) => {
            format!("&{v}.as_ref().map(|v| Redacted::len(v.len(), \"bytes\"))")
        }
        (Kind::Text, Count::Repeated) => format!("&Redacted::len({v}.len(), \"values\")"),
        (Kind::Any, Count::Repeated) => {
            format!("&{v}.iter().map(Redacted::any).collect::<Vec<_>>()")
        }
        (Kind::Any, _) => format!("&{v}.as_ref().map(Redacted::any)"),
        (Kind::Enum(e), Count::One) => format!("&EnumValue::<{}>::of({v})", rust_type(e)),
        _ => format!("&{v}"),
    }
}

/// `Debug` for every message, enum and `oneof` of the package: strings,
/// byte strings, maps and `Any` payloads print as their sizes.
fn redacting_debug(header: &str, package: &Package<'_>) -> Result<String> {
    let mut out = String::from(header);
    out.push_str("\nuse super::v4_2::*;\nuse super::{EnumValue, Redacted};\n");
    for (name, m) in package.structs() {
        let (plain, oneofs) = split_fields(m);
        let deprecated = plain.iter().any(|f| package.is_deprecated(f));
        writeln!(
            out,
            "\nimpl ::core::fmt::Debug for {} {{\n{}    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {{\n        f.debug_struct({:?})",
            rust_type(name),
            if deprecated { EXPECT_DEPRECATED } else { "" },
            short_name(name),
        )?;
        let mut fields: Vec<(i32, String, String)> = plain
            .iter()
            .map(|f| {
                let (kind, count) = package.kind(f);
                let value = debug_value(&kind, count, &format!("self.{}", field_ident(f.name())));
                (f.number(), f.name().to_owned(), value)
            })
            .collect();
        for (oneof, members) in &oneofs {
            let first = members.iter().map(|f| f.number()).min().unwrap_or(0);
            fields.push((
                first,
                (*oneof).to_owned(),
                format!("&self.{}", field_ident(oneof)),
            ));
        }
        fields.sort();
        for (_, field, value) in fields {
            writeln!(out, "            .field({field:?}, {value})")?;
        }
        out.push_str("            .finish()\n    }\n}\n");
        for (oneof, members) in &oneofs {
            writeln!(
                out,
                "\nimpl ::core::fmt::Debug for {}::{} {{\n{}    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {{\n        match self {{",
                rust_module(name),
                oneof.to_upper_camel_case(),
                if members.iter().any(|f| package.is_deprecated(f)) {
                    EXPECT_DEPRECATED
                } else {
                    ""
                },
            )?;
            for f in members {
                let (kind, _) = package.kind(f);
                let variant = f.name().to_upper_camel_case();
                let value = match kind {
                    Kind::Text => "&Redacted::len(inner.len(), \"bytes\")".to_owned(),
                    Kind::Any => "&Redacted::any(inner)".to_owned(),
                    Kind::Map(_) | Kind::Message(_) | Kind::Enum(_) | Kind::Scalar => {
                        "inner".to_owned()
                    }
                };
                writeln!(
                    out,
                    "            Self::{variant}(inner) => f.debug_tuple({variant:?}).field({value}).finish(),"
                )?;
            }
            out.push_str("        }\n    }\n}\n");
        }
    }
    for e in &package.enums {
        writeln!(
            out,
            "\nimpl ::core::fmt::Debug for {} {{\n    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {{\n        f.write_str(self.as_str_name())\n    }}\n}}",
            rust_type(e)
        )?;
    }
    Ok(out)
}

/// `Scrub` for every message that can hold a credential, directly or in a
/// message it holds: option map values, `path`/`paths` strings and `Any`
/// payloads are overwritten with zeros, and the messages that can hold
/// them are scrubbed in turn.
fn scrubbing(header: &str, package: &Package<'_>) -> Result<String> {
    let scrubbed = package.scrubbed();
    let mut out = String::from(header);
    out.push_str("\nuse super::v4_2::*;\nuse super::{Scrub, wipe, wipe_bytes};\n");
    for (name, m) in package.structs() {
        if !scrubbed.contains_key(name.as_str()) {
            continue;
        }
        let (plain, oneofs) = split_fields(m);
        let mut body = String::new();
        let mut deprecated = false;
        let recurses = |kind: &Kind| match kind {
            Kind::Message(t) | Kind::Map(Some(t)) => scrubbed.contains_key(t.as_str()),
            _ => false,
        };
        for f in &plain {
            let (kind, count) = package.kind(f);
            let v = format!("self.{}", field_ident(f.name()));
            let line = match (&kind, count) {
                (Kind::Map(None), _) => format!("for v in {v}.values_mut() {{ wipe(v); }}"),
                (Kind::Map(Some(_)), _) if recurses(&kind) => {
                    format!("for v in {v}.values_mut() {{ v.scrub(); }}")
                }
                (Kind::Text, Count::One) if is_path(f) => format!("wipe(&mut {v});"),
                (Kind::Text, Count::Optional) if is_path(f) => {
                    format!("if let Some(v) = &mut {v} {{ wipe(v); }}")
                }
                (Kind::Text, Count::Repeated) if is_path(f) => {
                    format!("for v in &mut {v} {{ wipe(v); }}")
                }
                (Kind::Any, Count::Repeated) => {
                    format!("for v in &mut {v} {{ wipe_bytes(&mut v.value); }}")
                }
                (Kind::Any, _) => {
                    format!("if let Some(v) = &mut {v} {{ wipe_bytes(&mut v.value); }}")
                }
                (Kind::Message(_), Count::Repeated) if recurses(&kind) => {
                    format!("for v in &mut {v} {{ v.scrub(); }}")
                }
                (Kind::Message(_), _) if recurses(&kind) => {
                    format!("if let Some(v) = &mut {v} {{ v.scrub(); }}")
                }
                _ => continue,
            };
            deprecated |= package.is_deprecated(f);
            writeln!(body, "        {line}")?;
        }
        for (oneof, members) in &oneofs {
            let mut arms = Vec::new();
            for f in members {
                let (kind, _) = package.kind(f);
                let action = match kind {
                    Kind::Any => "wipe_bytes(&mut v.value)",
                    Kind::Text if is_path(f) => "wipe(v)",
                    _ if recurses(&kind) => "v.scrub()",
                    _ => continue,
                };
                deprecated |= package.is_deprecated(f);
                arms.push((
                    format!(
                        "{}::{}::{}(v)",
                        rust_module(name),
                        oneof.to_upper_camel_case(),
                        f.name().to_upper_camel_case()
                    ),
                    action,
                ));
            }
            if let [(pattern, action)] = arms.as_slice() {
                writeln!(
                    body,
                    "        if let Some({pattern}) = &mut self.{} {{ {action}; }}",
                    field_ident(oneof)
                )?;
                continue;
            }
            let mut arms: Vec<String> = arms
                .iter()
                .map(|(pattern, action)| format!("{pattern} => {action},"))
                .collect();
            if arms.is_empty() {
                continue;
            }
            if arms.len() < members.len() {
                arms.push("_ => {}".to_owned());
            }
            writeln!(
                body,
                "        if let Some(v) = &mut self.{} {{\n            match v {{\n{}\n            }}\n        }}",
                field_ident(oneof),
                arms.iter()
                    .map(|a| format!("                {a}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )?;
        }
        writeln!(
            out,
            "\nimpl Scrub for {} {{\n{}    fn scrub(&mut self) {{\n{body}    }}\n}}",
            rust_type(name),
            if deprecated { EXPECT_DEPRECATED } else { "" },
        )?;
    }
    Ok(out)
}

/// The Rust field name prost gives a proto field: snake case, with a raw
/// identifier for a keyword.
fn field_ident(name: &str) -> String {
    const KEYWORDS: &[&str] = &[
        "type", "ref", "self", "super", "crate", "match", "mod", "move", "use", "fn", "in",
    ];
    let snake = name.to_snake_case();
    if KEYWORDS.contains(&snake.as_str()) {
        format!("r#{snake}")
    } else {
        snake
    }
}
