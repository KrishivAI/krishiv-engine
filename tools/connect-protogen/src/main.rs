// SPDX-License-Identifier: Apache-2.0
//! Generates the Spark Connect Rust types from the vendored `.proto` files.
//!
//! ```text
//! cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --write [VERSION]
//! cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --check [VERSION]
//! ```
//!
//! Run it from the repository root. `VERSION` is the vendored Spark minor
//! (default `4.2`; `v<VERSION>` is `v4_2`). Three files come out, all under `crates/krishiv-connect`:
//!
//! - `src/proto/v<VERSION>/spark.connect.pb.rs`: the `prost` message types;
//! - `src/proto/v<VERSION>/layout.pb.rs`: for every message, its name, the size
//!   of its Rust type and what each of its fields holds. The decoder walks
//!   the bytes with it to bound the nesting depth and the decoded size of a
//!   plan, and to reject fields the version does not define, before `prost`
//!   recurses into it;
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
use prost_types::field_descriptor_proto::Type;
use prost_types::{DescriptorProto, FileDescriptorSet};

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
    let module_dir = root.join("src/proto").join(format!("v{}", version.replace('.', "_")));
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
                Type::Message | Type::Group => ids
                    .get(f.type_name())
                    .map_or_else(|| "Field::Bytes".to_owned(), |id| format!("Field::Message({id})")),
                Type::String | Type::Bytes => "Field::Bytes".to_owned(),
                Type::Bool => "Field::Varint(1)".to_owned(),
                Type::Int32 | Type::Uint32 | Type::Sint32 | Type::Enum => {
                    "Field::Varint(4)".to_owned()
                }
                Type::Int64 | Type::Uint64 | Type::Sint64 => "Field::Varint(8)".to_owned(),
                Type::Fixed32 | Type::Sfixed32 | Type::Float | Type::Fixed64 | Type::Sfixed64
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
