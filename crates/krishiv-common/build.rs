// SPDX-License-Identifier: Apache-2.0
//! Generates the error-class registry from two JSON files of the same shape:
//! `spark-error-conditions.json` (Spark 4.3.0's `error-conditions.json`) and
//! `spark-error-conditions-4.2.1.json` (Spark 4.2.1's), both vendored verbatim
//! by `tests/scripts/gen_error_classes.py`, and `krishiv-error-classes.json`
//! (`KRISHIV_`-prefixed classes only). A Spark class is the union of the two
//! releases: its default text and parameters are the 4.2.1 ones (the default
//! profile is 4.2) or the 4.3.0 ones for a class only 4.3.0 has, and each
//! release's own text is kept for `message_template_for`. Output is
//! deterministic: `serde_json` without `preserve_order` keeps objects sorted
//! by key, and the merged list is sorted by class name.

use std::error::Error;
use std::fmt::Write as _;
use std::path::PathBuf;

use serde_json::{Map, Value};

#[path = "src/error/template.rs"]
#[expect(
    dead_code,
    reason = "shared with the library via #[path]; build.rs uses only `placeholders`"
)]
mod template;

type BoxError = Box<dyn Error>;

/// One generated leaf class (a top-level class, or one of its sub-classes).
struct Leaf {
    /// Spark name, e.g. `UNRESOLVED_COLUMN.WITH_SUGGESTION`.
    name: String,
    /// Rust identifier, e.g. `UnresolvedColumnWithSuggestion`.
    ident: String,
    /// `None` for Spark classes without a SQLSTATE (e.g. `_LEGACY_ERROR_TEMP_*`).
    sql_state: Option<String>,
    template: String,
    /// Spark parameter names in order of first appearance.
    params: Vec<String>,
    /// Whether only operators may see this class's details (`for_client`).
    operator_only: bool,
    /// The class's text and parameters in Spark 4.2.1, when it exists there.
    in_4_2: Option<(String, Vec<String>)>,
    /// The class's text and parameters in Spark 4.3.0, when it exists there.
    in_4_3: Option<(String, Vec<String>)>,
}

fn main() -> Result<(), BoxError> {
    println!("cargo::rerun-if-changed=vendored/spark-error-conditions.json");
    println!("cargo::rerun-if-changed=vendored/spark-error-conditions-4.2.1.json");
    println!("cargo::rerun-if-changed=krishiv-error-classes.json");
    println!("cargo::rerun-if-changed=error-audience.json");
    println!("cargo::rerun-if-changed=src/error/template.rs");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let spark_4_3 = std::fs::read_to_string(manifest.join("vendored/spark-error-conditions.json"))?;
    let spark_4_2 =
        std::fs::read_to_string(manifest.join("vendored/spark-error-conditions-4.2.1.json"))?;
    let krishiv = std::fs::read_to_string(manifest.join("krishiv-error-classes.json"))?;
    let mut leaves = merge_releases(parse(&spark_4_3, false)?, parse(&spark_4_2, false)?)?;
    leaves.extend(parse(&krishiv, true)?);
    leaves.sort_by(|a, b| a.name.cmp(&b.name));
    check_unique(&leaves)?;
    let audience = std::fs::read_to_string(manifest.join("error-audience.json"))?;
    apply_audience(&mut leaves, &audience)?;
    let out = PathBuf::from(std::env::var("OUT_DIR")?).join("error_classes.rs");
    std::fs::write(out, generate(&leaves)?)?;
    Ok(())
}

/// The union of Spark 4.3.0's and 4.2.1's classes. A class in both keeps the
/// 4.2.1 text and parameters as its default (the default profile is 4.2) and
/// records each release's own; the SQLSTATE must agree, so one accessor
/// serves both.
fn merge_releases(v4_3: Vec<Leaf>, v4_2: Vec<Leaf>) -> Result<Vec<Leaf>, BoxError> {
    let mut by_name: std::collections::BTreeMap<String, Leaf> = std::collections::BTreeMap::new();
    for mut l in v4_2 {
        l.in_4_2 = Some((l.template.clone(), l.params.clone()));
        by_name.insert(l.name.clone(), l);
    }
    for mut l in v4_3 {
        let entry = (l.template.clone(), l.params.clone());
        if let Some(existing) = by_name.get_mut(&l.name) {
            if existing.sql_state != l.sql_state {
                return Err(format!(
                    "{}: SQLSTATE differs between Spark 4.2.1 ({:?}) and 4.3.0 ({:?})",
                    l.name, existing.sql_state, l.sql_state
                )
                .into());
            }
            existing.in_4_3 = Some(entry);
        } else {
            l.in_4_3 = Some(entry);
            by_name.insert(l.name.clone(), l);
        }
    }
    Ok(by_name.into_values().collect())
}

/// Parses one registry file. Krishiv classes must be `KRISHIV_`-prefixed and
/// carry a SQLSTATE; Spark classes may omit it, as in Spark.
fn parse(json: &str, krishiv: bool) -> Result<Vec<Leaf>, BoxError> {
    let root: Map<String, Value> = serde_json::from_str(json)?;
    let mut leaves = Vec::new();
    for (name, info) in &root {
        check_name(name)?;
        if name.starts_with("KRISHIV_") != krishiv {
            return Err(format!("{name}: KRISHIV_ prefix must match the file it is in").into());
        }
        let sql_state = match info.get("sqlState") {
            None if !krishiv => None,
            None => return Err(format!("{name}: missing sqlState").into()),
            Some(v) => {
                let s = v
                    .as_str()
                    .ok_or_else(|| format!("{name}: sqlState must be a string"))?;
                let valid = s.len() == 5
                    && s.bytes()
                        .all(|b| b.is_ascii_digit() || b.is_ascii_uppercase());
                if !valid {
                    return Err(format!("{name}: invalid SQLSTATE {s:?}").into());
                }
                Some(s)
            }
        };
        let main = message(name, info)?;
        let operator_only = match info.get("audience").map(Value::as_str) {
            Some(_) if !krishiv => {
                return Err(format!(
                    "{name}: Spark classes get their audience in error-audience.json"
                )
                .into());
            }
            Some(Some("operator")) => true,
            None | Some(Some("user")) => false,
            Some(_) => {
                return Err(format!("{name}: audience must be \"user\" or \"operator\"").into());
            }
        };
        let first = leaves.len();
        match info.get("subClass") {
            None => leaves.push(leaf(name.clone(), sql_state, main)?),
            Some(subs) => {
                let subs = subs
                    .as_object()
                    .ok_or_else(|| format!("{name}: subClass must be an object"))?;
                if subs.is_empty() {
                    return Err(format!("{name}: empty subClass").into());
                }
                for (sub, sub_info) in subs {
                    check_name(sub)?;
                    let full = format!("{name}.{sub}");
                    // Spark: main template + " " + sub-class template.
                    let template = format!("{main} {}", message(&full, sub_info)?);
                    leaves.push(leaf(full, sql_state, template)?);
                }
            }
        }
        for l in &mut leaves[first..] {
            l.operator_only = operator_only;
        }
    }
    Ok(leaves)
}

/// Marks the Spark classes named in `error-audience.json` (`{"operator":
/// [...]}`, each a top-level name covering its sub-classes, a full
/// sub-class name, or a prefix ending in `*`) as operator-only. A name that matches no class is an
/// error, so a Spark refresh that renames one cannot silently expose it.
fn apply_audience(leaves: &mut [Leaf], json: &str) -> Result<(), BoxError> {
    let root: Map<String, Value> = serde_json::from_str(json)?;
    let names = root
        .get("operator")
        .and_then(Value::as_array)
        .ok_or("error-audience.json: missing \"operator\" array")?;
    for name in names {
        let name = name
            .as_str()
            .ok_or("error-audience.json: non-string name")?;
        if name.starts_with("KRISHIV_") {
            return Err(format!("error-audience.json: {name} sets its own audience").into());
        }
        // A trailing `*` names every class with that prefix.
        let prefix = name.strip_suffix('*');
        let mut hit = false;
        for l in leaves.iter_mut() {
            let matches = match prefix {
                Some(p) => l.name.starts_with(p),
                None => l.name == name || l.name.split('.').next() == Some(name),
            };
            if matches {
                l.operator_only = true;
                hit = true;
            }
        }
        if !hit {
            return Err(format!("error-audience.json: no error class {name}").into());
        }
    }
    Ok(())
}

fn check_unique(leaves: &[Leaf]) -> Result<(), BoxError> {
    let mut idents: Vec<&str> = leaves.iter().map(|l| l.ident.as_str()).collect();
    idents.sort_unstable();
    if let Some(w) = idents.windows(2).find(|w| w[0] == w[1]) {
        return Err(format!("two classes map to the Rust name {}", w[0]).into());
    }
    if idents.contains(&"KrishivError") || idents.contains(&"ErrorClass") {
        return Err("class name collides with KrishivError/ErrorClass".into());
    }
    Ok(())
}

fn check_name(name: &str) -> Result<(), BoxError> {
    let ok = name.starts_with(|c: char| c.is_ascii_uppercase() || c == '_')
        && name
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_');
    if ok {
        Ok(())
    } else {
        Err(format!("invalid error class name {name:?}").into())
    }
}

/// Spark joins the `message` lines with `\n`.
fn message(name: &str, info: &Value) -> Result<String, BoxError> {
    let lines = info
        .get("message")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{name}: missing message array"))?;
    let lines = lines
        .iter()
        .map(|l| {
            l.as_str()
                .ok_or_else(|| format!("{name}: non-string message line"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(lines.join("\n"))
}

fn leaf(name: String, sql_state: Option<&str>, template: String) -> Result<Leaf, BoxError> {
    let params: Vec<String> = template::placeholders(&template)
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut fields: Vec<String> = params.iter().map(|p| field_name(p)).collect();
    fields.sort_unstable();
    if fields.windows(2).any(|w| w[0] == w[1]) {
        return Err(format!("{name}: parameters collide after snake_case conversion").into());
    }
    Ok(Leaf {
        ident: name
            .split(['_', '.'])
            .map(|w| {
                let mut c = w.chars();
                c.next().map_or_else(String::new, |f| {
                    f.to_string() + &c.as_str().to_ascii_lowercase()
                })
            })
            .collect(),
        name,
        sql_state: sql_state.map(str::to_owned),
        template,
        params,
        operator_only: false,
        in_4_2: None,
        in_4_3: None,
    })
}

/// `objectName` -> `object_name`; Rust keywords become raw identifiers.
fn field_name(param: &str) -> String {
    let mut s = String::new();
    for c in param.chars() {
        if c.is_ascii_uppercase() {
            if !s.is_empty() && !s.ends_with('_') {
                s.push('_');
            }
            s.push(c.to_ascii_lowercase());
        } else if c == '-' {
            s.push('_');
        } else {
            s.push(c);
        }
    }
    if s.starts_with(|c: char| c.is_ascii_digit()) {
        s.insert(0, '_');
    }
    const KEYWORDS: &[&str] = &[
        "as", "async", "await", "box", "break", "const", "continue", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "gen", "if", "impl", "in", "let", "loop", "match", "mod",
        "move", "mut", "pub", "ref", "return", "static", "struct", "trait", "true", "try", "type",
        "unsafe", "use", "where", "while", "yield",
    ];
    if KEYWORDS.contains(&s.as_str()) {
        format!("r#{s}")
    } else {
        s
    }
}

fn state_doc(l: &Leaf) -> String {
    l.sql_state
        .as_ref()
        .map_or_else(String::new, |s| format!(" (SQLSTATE `{s}`)"))
}

/// (doc, method name, extra argument, return type, per-leaf match arm body).
type Accessor = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    fn(&Leaf) -> String,
);

/// `match release { .. }` over a class's per-release value; `None` for a
/// release the class is not in. A Krishiv class exists in every release.
fn per_release(l: &Leaf, value: fn(&(String, Vec<String>)) -> String) -> String {
    let arm = |r: &Option<(String, Vec<String>)>| {
        if l.name.starts_with("KRISHIV_") {
            format!("Some({})", value(&(l.template.clone(), l.params.clone())))
        } else {
            r.as_ref()
                .map_or_else(|| "None".to_owned(), |v| format!("Some({})", value(v)))
        }
    };
    format!(
        "match release {{ SparkRelease::V4_2 => {}, SparkRelease::V4_3 => {} }}",
        arm(&l.in_4_2),
        arm(&l.in_4_3)
    )
}

fn generate(leaves: &[Leaf]) -> Result<String, BoxError> {
    let mut o = String::new();
    writeln!(
        o,
        "// @generated by build.rs from the error-class JSON files. Do not edit."
    )?;
    writeln!(o)?;
    writeln!(
        o,
        "/// A Spark (or `KRISHIV_`-prefixed Krishiv) error class."
    )?;
    writeln!(o, "///")?;
    writeln!(
        o,
        "/// Classes with sub-classes appear only as their sub-classes."
    )?;
    writeln!(
        o,
        "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]"
    )?;
    writeln!(o, "#[non_exhaustive]")?;
    writeln!(o, "pub enum ErrorClass {{")?;
    for l in leaves {
        writeln!(o, "    /// `{}`{}.", l.name, state_doc(l))?;
        writeln!(o, "    {},", l.ident)?;
    }
    writeln!(o, "}}")?;
    writeln!(o)?;
    writeln!(o, "impl ErrorClass {{")?;
    writeln!(o, "    /// Every error class, in name order.")?;
    writeln!(o, "    pub const ALL: &'static [ErrorClass] = &[")?;
    for l in leaves {
        writeln!(o, "        ErrorClass::{},", l.ident)?;
    }
    writeln!(o, "    ];")?;
    let accessors: [Accessor; 7] = [
        (
            "Spark error class name, e.g. `UNRESOLVED_COLUMN.WITH_SUGGESTION`.",
            "name",
            "",
            "&'static str",
            |l| format!("{:?}", l.name),
        ),
        (
            "Five-character SQLSTATE; `None` for Spark classes that have none.",
            "sql_state",
            "",
            "Option<&'static str>",
            |l| format!("{:?}", l.sql_state),
        ),
        (
            "Message template with `<param>` placeholders.",
            "message_template",
            "",
            "&'static str",
            |l| format!("{:?}", l.template),
        ),
        (
            "Spark message parameter names, in template order.",
            "parameters",
            "",
            "&'static [&'static str]",
            |l| format!("&{:?}", l.params),
        ),
        (
            "Who may see this class's details: see [`Audience`].",
            "audience",
            "",
            "Audience",
            |l| {
                if l.operator_only {
                    "Audience::Operator".to_owned()
                } else {
                    "Audience::User".to_owned()
                }
            },
        ),
        (
            "The message template as `release` words it (the default view, \
             [`ErrorClass::message_template`], is Spark 4.2); `None` when the \
             class does not exist in that release.",
            "message_template_for",
            ", release: SparkRelease",
            "Option<&'static str>",
            |l| per_release(l, |(t, _)| format!("{t:?}")),
        ),
        (
            "The parameter names as `release` has them; `None` when the class \
             does not exist in that release.",
            "parameters_for",
            ", release: SparkRelease",
            "Option<&'static [&'static str]>",
            |l| per_release(l, |(_, p)| format!("&{p:?}")),
        ),
    ];
    for (doc, fname, args, ty, body) in accessors {
        writeln!(o)?;
        writeln!(o, "    /// {doc}")?;
        // `match_same_arms` fires only when two arms have the same body, so
        // the expectation is emitted exactly where it will be fulfilled.
        let mut seen = std::collections::HashSet::new();
        if !leaves.iter().all(|l| seen.insert(body(l))) {
            writeln!(
                o,
                "    #[expect(clippy::match_same_arms, reason = \"generated arms coincide for some classes\")]"
            )?;
        }
        writeln!(o, "    pub const fn {fname}(self{args}) -> {ty} {{")?;
        writeln!(o, "        match self {{")?;
        for l in leaves {
            writeln!(o, "            ErrorClass::{} => {},", l.ident, body(l))?;
        }
        writeln!(o, "        }}")?;
        writeln!(o, "    }}")?;
    }
    writeln!(o)?;
    writeln!(o, "    /// Looks up a class by its Spark name.")?;
    writeln!(
        o,
        "    pub fn from_name(name: &str) -> Option<ErrorClass> {{"
    )?;
    writeln!(o, "        match name {{")?;
    for l in leaves {
        writeln!(
            o,
            "            {:?} => Some(ErrorClass::{}),",
            l.name, l.ident
        )?;
    }
    writeln!(o, "            _ => None,")?;
    writeln!(o, "        }}")?;
    writeln!(o, "    }}")?;
    writeln!(o, "}}")?;
    for l in leaves {
        writeln!(o)?;
        writeln!(o, "/// Parameters of `{}`{}.", l.name, state_doc(l))?;
        writeln!(o, "///")?;
        writeln!(o, "/// ```text")?;
        for line in l.template.lines() {
            writeln!(o, "#[doc = {:?}]", line)?;
        }
        writeln!(o, "/// ```")?;
        writeln!(o, "#[derive(Debug, Clone, PartialEq, Eq)]")?;
        writeln!(o, "pub struct {} {{", l.ident)?;
        for p in &l.params {
            writeln!(o, "    /// Spark parameter `{p}`.")?;
            writeln!(o, "    pub {}: String,", field_name(p))?;
        }
        writeln!(o, "}}")?;
        writeln!(o)?;
        writeln!(o, "impl From<{}> for KrishivError {{", l.ident)?;
        let binding = if l.params.is_empty() {
            "_params"
        } else {
            "params"
        };
        writeln!(o, "    fn from({binding}: {}) -> Self {{", l.ident)?;
        writeln!(o, "        KrishivError::from_parts(")?;
        writeln!(o, "            ErrorClass::{},", l.ident)?;
        writeln!(o, "            vec![")?;
        for p in &l.params {
            writeln!(o, "                ({p:?}, params.{}),", field_name(p))?;
        }
        writeln!(o, "            ],")?;
        writeln!(o, "        )")?;
        writeln!(o, "    }}")?;
        writeln!(o, "}}")?;
    }
    Ok(o)
}
