// SPDX-License-Identifier: Apache-2.0
//! Generates the error-class registry from two JSON files of the same shape:
//! `spark-error-conditions.json` (Spark 4.2.0's `error-conditions.json`,
//! vendored verbatim by `tests/scripts/gen_error_classes.py`) and
//! `krishiv-error-classes.json` (`KRISHIV_`-prefixed classes only). Output is
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
}

fn main() -> Result<(), BoxError> {
    println!("cargo::rerun-if-changed=vendored/spark-error-conditions.json");
    println!("cargo::rerun-if-changed=krishiv-error-classes.json");
    println!("cargo::rerun-if-changed=src/error/template.rs");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let spark = std::fs::read_to_string(manifest.join("vendored/spark-error-conditions.json"))?;
    let krishiv = std::fs::read_to_string(manifest.join("krishiv-error-classes.json"))?;
    let mut leaves = parse(&spark, false)?;
    leaves.extend(parse(&krishiv, true)?);
    leaves.sort_by(|a, b| a.name.cmp(&b.name));
    check_unique(&leaves)?;
    let out = PathBuf::from(std::env::var("OUT_DIR")?).join("error_classes.rs");
    std::fs::write(out, generate(&leaves)?)?;
    Ok(())
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
    }
    Ok(leaves)
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

/// (doc, method name, return type, per-leaf match arm body).
type Accessor = (
    &'static str,
    &'static str,
    &'static str,
    fn(&Leaf) -> String,
);

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
    let accessors: [Accessor; 4] = [
        (
            "Spark error class name, e.g. `UNRESOLVED_COLUMN.WITH_SUGGESTION`.",
            "name",
            "&'static str",
            |l| format!("{:?}", l.name),
        ),
        (
            "Five-character SQLSTATE; `None` for Spark classes that have none.",
            "sql_state",
            "Option<&'static str>",
            |l| format!("{:?}", l.sql_state),
        ),
        (
            "Message template with `<param>` placeholders.",
            "message_template",
            "&'static str",
            |l| format!("{:?}", l.template),
        ),
        (
            "Spark message parameter names, in template order.",
            "parameters",
            "&'static [&'static str]",
            |l| format!("&{:?}", l.params),
        ),
    ];
    for (doc, fname, ty, body) in accessors {
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
        writeln!(o, "    pub const fn {fname}(self) -> {ty} {{")?;
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
