// SPDX-License-Identifier: Apache-2.0
//! The shape of the `krishiv.analyze.v1` JSON (design `P3-T063-analyze-api.md` §2), as a hand-written
//! description ([`analyze_v1`]) rather than a derived one: no schema crate (ADR 0028).
//!
//! `cargo xtask docs-analyze` writes [`json_schema`] to `docs/reference/analyze/krishiv.analyze.v1.schema.json`
//! (`--check` compares, and checks the file is additive over the base branch's); the session's tests check
//! every analyze golden against [`analyze_v1`].
//!
//! The closed vocabularies come from the enums that own them where this crate owns them ([`Transform`],
//! [`FilterKind`], [`Privilege`], [`Securable`] kinds, by exhaustive matches); the statement classes and the
//! analysis-level warning codes are listed here and checked against the session's enums by its tests.
//!
//! Versioning: the published file lets v1 grow. No object closes its members (no `additionalProperties`),
//! so a reader ignores a member it does not know; `required` lists the members v1 has had from the start,
//! and a member added within v1 is optional (never `required`); an enum may gain values, so a reader accepts
//! one it does not know. Removing a member or a value, adding a `required` entry, closing an object or
//! changing a meaning is `krishiv.analyze.v2` (`cargo xtask docs-analyze --check` refuses it). The
//! session's golden validator stays strict: it checks [`analyze_v1`], which has no unknown member.
//!
//! Not listed: the directory of a managed table. A run of `DROP TABLE`, `TRUNCATE TABLE` or `INSERT` on a
//! managed table also checks `MODIFY` (and `CREATE`) on the table's directory, but only when the
//! directory exists and on its resolved path, so listing it would take file-system I/O and would not be
//! deterministic.

use std::fmt::Write as _;

use super::{FilterKind, Leaf, LineageWarning, Transform};
use crate::analyzer::{Privilege, Securable};

/// The `schema_version` value.
pub const SCHEMA_VERSION: &str = "krishiv.analyze.v1";

/// The JSON Schema `$id` of the schema file.
pub const SCHEMA_ID: &str = "https://github.com/KrishivAI/krishiv-engine/docs/reference/analyze/krishiv.analyze.v1.schema.json";

/// The statement classes, as `StatementClass::name` prints them (checked by the session's tests).
pub const STATEMENT_CLASSES: [&str; 5] = ["READ", "WRITE", "DDL", "SESSION", "ADMIN"];

/// The `explain` values, as `ExplainKind` prints them (checked by the session's tests).
pub const EXPLAIN_KINDS: [&str; 2] = ["PLAN", "ANALYZE"];

/// The analysis-level warning codes: the lineage's ([`LineageWarning`]) and the session's (checked by its
/// tests), sorted.
pub const WARNING_CODES: [&str; 4] = [
    "LINEAGE_OPAQUE_UDF",
    "UNSUPPORTED_READ_SHAPE",
    "USER_CODE_RAN_AT_ANALYSIS",
    "VIEW_NOT_EXPANDED",
];

/// Every privilege, in declaration order.
pub const PRIVILEGES: [Privilege; 10] = [
    Privilege::Select,
    Privilege::Modify,
    Privilege::Create,
    Privilege::Drop,
    Privilege::Monitor,
    Privilege::Usage,
    Privilege::Browse,
    Privilege::Operate,
    Privilege::Manage,
    Privilege::Impersonate,
];

/// The securable kinds, as [`securable_kind`] names them.
pub const SECURABLE_KINDS: [&str; 8] = [
    "table",
    "view",
    "function",
    "path",
    "namespace",
    "system",
    "secret",
    "connection",
];

/// Every transform, weakest first.
pub const TRANSFORMS: [Transform; 9] = [
    Transform::Literal,
    Transform::Identity,
    Transform::Expression,
    Transform::Generated,
    Transform::Window,
    Transform::Aggregate,
    Transform::Opaque,
    Transform::Masked,
    Transform::View,
];

/// Every filter kind.
pub const FILTER_KINDS: [FilterKind; 6] = [
    FilterKind::Filter,
    FilterKind::Join,
    FilterKind::GroupBy,
    FilterKind::Sort,
    FilterKind::Window,
    FilterKind::Conditional,
];

/// Every lineage warning.
pub const LINEAGE_WARNINGS: [LineageWarning; 2] = [
    LineageWarning::LineageOpaqueUdf,
    LineageWarning::ViewNotExpanded,
];

/// The JSON `kind` of a securable (an exhaustive match: a new kind does not compile until it is named).
#[must_use]
pub const fn securable_kind(securable: &Securable) -> &'static str {
    match securable {
        Securable::Table(_) => "table",
        Securable::View(_) => "view",
        Securable::Function(_) => "function",
        Securable::Path(_) => "path",
        Securable::Namespace(_) => "namespace",
        Securable::System(_) => "system",
        Securable::Secret { .. } => "secret",
        Securable::Connection(_) => "connection",
    }
}

/// The name parts of a securable in the JSON: a path is one part, its redacted text (`OptionValue`'s
/// `Display`, as the 42501 denial prints it); a secret is its scope.
#[must_use]
pub fn securable_name(securable: &Securable) -> Vec<String> {
    match securable {
        Securable::Table(t) | Securable::View(t) => t.parts(),
        Securable::Function(n) | Securable::Namespace(n) => n.clone(),
        Securable::Path(p) => vec![p.to_string()],
        Securable::System(n) | Securable::Connection(n) => vec![n.clone()],
        Securable::Secret { scope } => vec![scope.clone()],
    }
}

/// The leaf forms, as [`leaf_key`] names them: `(key, description, the name is one string)`.
pub const LEAF_FORMS: [(&str, &str, bool); 3] = [
    ("table", "A catalog table's name parts.", false),
    (
        "path",
        "Files read by path: the redacted path, as the 42501 denial prints it.",
        true,
    ),
    (
        "view",
        "A view that is not expanded (a DEFINER view): its name parts; nothing under it is shown.",
        false,
    ),
];

/// The JSON key of a leaf's form (an exhaustive match: a new form does not compile until it is named).
#[must_use]
pub const fn leaf_key(leaf: &Leaf) -> &'static str {
    match leaf {
        Leaf::Table { .. } => "table",
        Leaf::Path { .. } => "path",
        Leaf::View { .. } => "view",
    }
}

/// A leaf's name: its name parts, or one string (a path's redacted URL; see [`LEAF_FORMS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafName<'a> {
    /// A table's or view's name parts.
    Parts(&'a [String]),
    /// A path's redacted text.
    Text(&'a str),
}

/// A leaf's name and column.
#[must_use]
pub fn leaf_name(leaf: &Leaf) -> (LeafName<'_>, &str) {
    match leaf {
        Leaf::Table { name, column } | Leaf::View { name, column } => {
            (LeafName::Parts(name), column)
        }
        Leaf::Path { url, column } => (LeafName::Text(url), column),
    }
}

/// One shape of a JSON value.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Shape {
    /// A string.
    String,
    /// `true` or `false`.
    Boolean,
    /// A non-negative integer.
    Count,
    /// Exactly this string.
    Const(&'static str),
    /// One of these strings.
    Enum(Vec<&'static str>),
    /// `null` or the inner shape.
    Nullable(Box<Self>),
    /// An array of the inner shape.
    Array(Box<Self>),
    /// An object with these fields, each `required` (every one has been in v1 from the start). The
    /// published schema does not close it (a reader ignores a member it does not know); the session's
    /// golden validator does.
    Object(Vec<Field>),
    /// Exactly one of these shapes.
    OneOf(Vec<Self>),
}

/// A field of an [`Shape::Object`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// The member name.
    pub name: &'static str,
    /// What it holds.
    pub description: &'static str,
    /// Its shape.
    pub shape: Shape,
}

const fn field(name: &'static str, description: &'static str, shape: Shape) -> Field {
    Field {
        name,
        description,
        shape,
    }
}

fn array(shape: Shape) -> Shape {
    Shape::Array(Box::new(shape))
}

fn nullable(shape: Shape) -> Shape {
    Shape::Nullable(Box::new(shape))
}

/// The leaf forms ([`LEAF_FORMS`]), each with `extra` fields first.
fn leaves(extra: &[Field], tail: &[Field]) -> Shape {
    let column = || {
        field(
            "column",
            "The column (a `_metadata` field is dotted).",
            Shape::String,
        )
    };
    Shape::OneOf(
        LEAF_FORMS
            .into_iter()
            .map(|(key, description, text)| {
                let leaf = field(
                    key,
                    description,
                    if text {
                        Shape::String
                    } else {
                        array(Shape::String)
                    },
                );
                let mut fields = extra.to_vec();
                fields.push(leaf);
                fields.push(column());
                fields.extend_from_slice(tail);
                Shape::Object(fields)
            })
            .collect(),
    )
}

/// The `krishiv.analyze.v1` document.
#[must_use]
pub fn analyze_v1() -> Shape {
    let securable = Shape::Object(vec![
        field(
            "kind",
            "The securable's kind.",
            Shape::Enum(SECURABLE_KINDS.to_vec()),
        ),
        field(
            "name",
            "Its name parts (a path: its redacted text; a secret: its scope).",
            array(Shape::String),
        ),
        field(
            "display",
            "The securable as the 42501 denial text prints it (redacted by `Securable`'s `Display`, ADR 0038).",
            Shape::String,
        ),
        field(
            "privilege",
            "The privilege the statement needs on it.",
            Shape::Enum(privilege_names()),
        ),
        field(
            "columns",
            "The columns its reads reference, sorted (empty for a target).",
            array(Shape::String),
        ),
    ]);
    let output = Shape::Object(vec![
        field("ordinal", "The output position, from 0.", Shape::Count),
        field("name", "The output column's name.", Shape::String),
        field(
            "type",
            "The type's `simpleString`, for people.",
            Shape::String,
        ),
        field(
            "type_json",
            "The type's lossless JSON (`DataType.json`: nested nullability, metadata), as a string.",
            Shape::String,
        ),
        field("nullable", "Whether it may be null.", Shape::Boolean),
        field(
            "declared_type",
            "The `CHAR(n)`/`VARCHAR(n)` declaration, if any.",
            nullable(Shape::String),
        ),
    ]);
    let via_view = field(
        "via_view",
        "The outermost expanded (temporary or INVOKER) view the leaf was read through, by its name parts.",
        nullable(array(Shape::String)),
    );
    let lineage = Shape::Object(vec![
        field("ordinal", "The output position, from 0.", Shape::Count),
        field("output", "The output column's name.", Shape::String),
        field(
            "sources",
            "The leaves the value depends on, sorted.",
            array(leaves(&[], &[via_view])),
        ),
        field(
            "transform",
            "The strongest step on any path.",
            Shape::Enum(TRANSFORMS.iter().map(|t| t.name()).collect()),
        ),
    ]);
    let filter_kind = field(
        "kind",
        "Why the column decides which rows or groups.",
        Shape::Enum(FILTER_KINDS.iter().map(|k| k.name()).collect()),
    );
    Shape::Object(vec![
        field(
            "schema_version",
            "The document version.",
            Shape::Const(SCHEMA_VERSION),
        ),
        field(
            "statement_class",
            "The statement's class, as `system.queries.statement_class` shows it.",
            Shape::Enum(STATEMENT_CLASSES.to_vec()),
        ),
        field(
            "explain",
            "How it explains, if it is an EXPLAIN.",
            nullable(Shape::Enum(EXPLAIN_KINDS.to_vec())),
        ),
        field(
            "partial",
            "Whether the statement's reads are not fully known (then it is not provably read-only).",
            Shape::Boolean,
        ),
        field(
            "securables",
            "What the statement needs, in first-use order: reads (with columns), then targets. Not listed: a \
             managed table's directory, which a run of DROP TABLE, TRUNCATE TABLE or INSERT also checks \
             only when it exists, on its resolved path.",
            array(securable),
        ),
        field(
            "output_schema",
            "The result columns: a query's analyzed output; a SHOW, DESCRIBE, EXPLAIN, SET or RESET command's \
             result (RESET and SET PATH: none); empty for writes, DDL and other commands.",
            array(output),
        ),
        field(
            "lineage",
            "Per output ordinal of a query, its sources and transform; else empty.",
            array(lineage),
        ),
        field(
            "filters",
            "The columns that decide which rows or groups, sorted by kind in the listed order, then by leaf.",
            array(leaves(&[filter_kind], &[])),
        ),
        field(
            "warnings",
            "Fixed codes, sorted; never text with user data.",
            array(Shape::Enum(WARNING_CODES.to_vec())),
        ),
    ])
}

fn privilege_names() -> Vec<&'static str> {
    PRIVILEGES.iter().map(|p| privilege_name(*p)).collect()
}

/// `Privilege`'s `Display`, as a `&'static str` (an exhaustive match).
#[must_use]
pub const fn privilege_name(privilege: Privilege) -> &'static str {
    match privilege {
        Privilege::Select => "SELECT",
        Privilege::Modify => "MODIFY",
        Privilege::Create => "CREATE",
        Privilege::Drop => "DROP",
        Privilege::Monitor => "MONITOR",
        Privilege::Usage => "USAGE",
        Privilege::Browse => "BROWSE",
        Privilege::Operate => "OPERATE",
        Privilege::Manage => "MANAGE",
        Privilege::Impersonate => "IMPERSONATE",
    }
}

/// The schema file's text: a JSON Schema (draft 2020-12) of [`analyze_v1`], two-space indented, with a
/// final newline.
#[must_use]
pub fn json_schema() -> String {
    let mut root = vec![
        (
            "$schema",
            J::Str("https://json-schema.org/draft/2020-12/schema".to_owned()),
        ),
        ("$id", J::Str(SCHEMA_ID.to_owned())),
        (
            "title",
            J::Str("Krishiv analyze result, krishiv.analyze.v1".to_owned()),
        ),
        (
            "description",
            J::Str(
                "What Session::analyze returns as JSON (Analysis::to_json): a statement's class, the securables it \
                 needs, its output schema, column lineage and filter columns. Generated by `cargo xtask docs-analyze` \
                 from krishiv_plan::analyzer::lineage::schema. Additive within v1: members and enum values may \
                 be added, so a reader ignores members it does not know and accepts enum values it does not \
                 know; new alternatives of a `oneOf` (such as a new lineage leaf form) may be added too, so a \
                 reader skips an element that matches no alternative it knows and does not validate unknown \
                 forms strictly; `required` lists the members v1 has had from the start. Not listed among the \
                 securables: a managed table's directory, which a run of DROP TABLE, TRUNCATE TABLE or INSERT \
                 also checks only when the directory exists, on its resolved path (not deterministic without \
                 file-system I/O)."
                    .to_owned(),
            ),
        ),
    ];
    if let J::Obj(members) = to_json(&analyze_v1(), None) {
        root.extend(members);
    }
    let mut out = String::new();
    J::Obj(root).write(&mut out, 0);
    out.push('\n');
    out
}

/// A JSON value with ordered members.
enum J {
    Str(String),
    Int(u64),
    Arr(Vec<J>),
    Obj(Vec<(&'static str, J)>),
}

impl J {
    fn write(&self, out: &mut String, indent: usize) {
        match self {
            Self::Str(s) => write_str(out, s),
            Self::Int(n) => {
                let _ = write!(out, "{n}");
            }
            Self::Arr(items) => {
                // Arrays of scalars stay on one line.
                if items
                    .iter()
                    .all(|i| matches!(i, Self::Str(_) | Self::Int(_)))
                {
                    out.push('[');
                    for (n, item) in items.iter().enumerate() {
                        if n > 0 {
                            out.push_str(", ");
                        }
                        item.write(out, indent);
                    }
                    out.push(']');
                    return;
                }
                out.push_str("[\n");
                for (n, item) in items.iter().enumerate() {
                    pad(out, indent + 1);
                    item.write(out, indent + 1);
                    out.push_str(if n + 1 < items.len() { ",\n" } else { "\n" });
                }
                pad(out, indent);
                out.push(']');
            }
            Self::Obj(members) => {
                out.push_str("{\n");
                for (n, (k, v)) in members.iter().enumerate() {
                    pad(out, indent + 1);
                    write_str(out, k);
                    out.push_str(": ");
                    v.write(out, indent + 1);
                    out.push_str(if n + 1 < members.len() { ",\n" } else { "\n" });
                }
                pad(out, indent);
                out.push('}');
            }
        }
    }
}

fn pad(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

/// A JSON string literal.
fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn strings(values: &[&'static str]) -> J {
    J::Arr(values.iter().map(|v| J::Str((*v).to_owned())).collect())
}

/// The JSON Schema of `shape`, with `description` first when given.
fn to_json(shape: &Shape, description: Option<&'static str>) -> J {
    let mut m: Vec<(&'static str, J)> = Vec::new();
    if let Some(d) = description {
        m.push(("description", J::Str(d.to_owned())));
    }
    match shape {
        Shape::String => m.push(("type", J::Str("string".to_owned()))),
        Shape::Boolean => m.push(("type", J::Str("boolean".to_owned()))),
        Shape::Count => {
            m.push(("type", J::Str("integer".to_owned())));
            m.push(("minimum", J::Int(0)));
        }
        Shape::Const(c) => m.push(("const", J::Str((*c).to_owned()))),
        Shape::Enum(values) => m.push(("enum", strings(values))),
        Shape::Nullable(inner) => m.push((
            "oneOf",
            J::Arr(vec![
                J::Obj(vec![("type", J::Str("null".to_owned()))]),
                to_json(inner, None),
            ]),
        )),
        Shape::Array(inner) => {
            m.push(("type", J::Str("array".to_owned())));
            m.push(("items", to_json(inner, None)));
        }
        Shape::Object(fields) => {
            m.push(("type", J::Str("object".to_owned())));
            let names: Vec<&'static str> = fields.iter().map(|f| f.name).collect();
            // Not closed (no `additionalProperties`): v1 may add members.
            m.push(("required", strings(&names)));
            m.push((
                "properties",
                J::Obj(
                    fields
                        .iter()
                        .map(|f| (f.name, to_json(&f.shape, Some(f.description))))
                        .collect(),
                ),
            ));
        }
        Shape::OneOf(shapes) => m.push((
            "oneOf",
            J::Arr(shapes.iter().map(|s| to_json(s, None)).collect()),
        )),
    }
    J::Obj(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::TableIdent;

    /// The lists hold every variant: an exhaustive match per enum (a new variant does not compile until it
    /// is listed).
    #[test]
    fn vocabularies_list_every_variant() {
        for p in PRIVILEGES {
            assert_eq!(privilege_name(p), p.to_string());
        }
        let privilege_listed = |p: Privilege| match p {
            Privilege::Select
            | Privilege::Modify
            | Privilege::Create
            | Privilege::Drop
            | Privilege::Monitor
            | Privilege::Usage
            | Privilege::Browse
            | Privilege::Operate
            | Privilege::Manage
            | Privilege::Impersonate => PRIVILEGES.contains(&p),
        };
        assert!(PRIVILEGES.iter().all(|p| privilege_listed(*p)));
        let transform_listed = |t: Transform| match t {
            Transform::Literal
            | Transform::Identity
            | Transform::Expression
            | Transform::Generated
            | Transform::Window
            | Transform::Aggregate
            | Transform::Opaque
            | Transform::Masked
            | Transform::View => TRANSFORMS.contains(&t),
        };
        assert!(TRANSFORMS.iter().all(|t| transform_listed(*t)));
        assert!(TRANSFORMS.windows(2).all(|w| w[0] < w[1]), "weakest first");
        let kind_listed = |k: FilterKind| match k {
            FilterKind::Filter
            | FilterKind::Join
            | FilterKind::GroupBy
            | FilterKind::Sort
            | FilterKind::Window
            | FilterKind::Conditional => FILTER_KINDS.contains(&k),
        };
        assert!(FILTER_KINDS.iter().all(|k| kind_listed(*k)));
        for w in LINEAGE_WARNINGS {
            let listed = match w {
                LineageWarning::LineageOpaqueUdf | LineageWarning::ViewNotExpanded => {
                    WARNING_CODES.contains(&w.code())
                }
            };
            assert!(listed, "{w}");
        }
        assert!(WARNING_CODES.windows(2).all(|w| w[0] < w[1]), "sorted");
        let t = TableIdent {
            catalog: "c".to_owned(),
            namespace: vec!["d".to_owned()],
            name: "t".to_owned(),
        };
        let all = [
            Securable::Table(t.clone()),
            Securable::View(t),
            Securable::Function(vec!["f".to_owned()]),
            Securable::Path(krishiv_common::secret::OptionValue::Plain("/p".to_owned())),
            Securable::Namespace(vec!["c".to_owned()]),
            Securable::System("cluster".to_owned()),
            Securable::Secret {
                scope: "s".to_owned(),
            },
            Securable::Connection("k".to_owned()),
        ];
        let kinds: Vec<&str> = all.iter().map(securable_kind).collect();
        assert_eq!(kinds, SECURABLE_KINDS);
        // The kind is the first word of the denial text.
        for s in &all {
            assert!(
                s.to_string()
                    .starts_with(&format!("{} ", securable_kind(s)))
            );
        }
        assert_eq!(securable_name(&all[0]), ["c", "d", "t"]);
        assert_eq!(securable_name(&all[3]), ["/p"]);
        assert_eq!(securable_name(&all[6]), ["s"]);
        let leaves = [
            Leaf::Table {
                name: vec!["t".to_owned()],
                column: "a".to_owned(),
            },
            Leaf::Path {
                url: "/p".to_owned(),
                column: "a".to_owned(),
            },
            Leaf::View {
                name: vec!["v".to_owned()],
                column: "a".to_owned(),
            },
        ];
        let keys: Vec<&str> = leaves.iter().map(leaf_key).collect();
        let forms: Vec<&str> = LEAF_FORMS.iter().map(|f| f.0).collect();
        assert_eq!(keys, forms);
        for (leaf, (_, _, text)) in leaves.iter().zip(LEAF_FORMS) {
            assert_eq!(matches!(leaf_name(leaf).0, LeafName::Text(_)), text);
        }
    }

    /// Every value a closed list of the description holds is one the session can emit, and the other way
    /// round: the `transform` and filter `kind` enums are the names of every variant.
    #[test]
    fn emitted_values_are_in_the_closed_lists() {
        fn enums(shape: &Shape, at: &str, out: &mut Vec<(String, Vec<&'static str>)>) {
            match shape {
                Shape::Enum(values) => out.push((at.to_owned(), values.clone())),
                Shape::Nullable(s) | Shape::Array(s) => enums(s, at, out),
                Shape::Object(fields) => {
                    for f in fields {
                        enums(&f.shape, &format!("{at}.{}", f.name), out);
                    }
                }
                Shape::OneOf(shapes) => {
                    for s in shapes {
                        enums(s, at, out);
                    }
                }
                Shape::String | Shape::Boolean | Shape::Count | Shape::Const(_) => {}
            }
        }
        let mut all = Vec::new();
        enums(&analyze_v1(), "$", &mut all);
        let of = |at: &str| {
            all.iter()
                .find(|(p, _)| p == at)
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        };
        let transforms: Vec<&str> = TRANSFORMS.iter().map(|t| t.name()).collect();
        assert_eq!(of("$.lineage.transform"), transforms);
        let kinds: Vec<&str> = FILTER_KINDS.iter().map(|k| k.name()).collect();
        assert_eq!(of("$.filters.kind"), kinds);
        assert_eq!(of("$.securables.kind"), SECURABLE_KINDS);
        assert_eq!(of("$.securables.privilege"), privilege_names());
        assert_eq!(of("$.warnings"), WARNING_CODES);
    }

    #[test]
    fn json_schema_is_valid_json_with_ordered_members() {
        let text = json_schema();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["$id"], SCHEMA_ID);
        assert_eq!(v["properties"]["schema_version"]["const"], SCHEMA_VERSION);
        // No object is closed, so v1 may grow.
        assert!(!text.contains("additionalProperties"));
        let required: Vec<&str> = v["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        assert_eq!(required[0], "schema_version");
        assert_eq!(required.len(), 9);
        // Members are written in description order, not sorted.
        assert!(text.find("\"statement_class\"") < text.find("\"explain\""));
        assert!(text.ends_with("}\n"));
        let mut escaped = String::new();
        write_str(&mut escaped, "a\"b\\c\nd\u{1}");
        assert_eq!(escaped, "\"a\\\"b\\\\c\\nd\\u0001\"");
    }
}
