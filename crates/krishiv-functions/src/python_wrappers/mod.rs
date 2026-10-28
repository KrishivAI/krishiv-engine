// SPDX-License-Identifier: Apache-2.0
//! The Python wrappers of the registry (`python/krishiv/sql/_generated_functions.py` and `.pyi`).
//!
//! A wrapper exists for every registered function that `pyspark.sql.functions` also has, under PySpark's
//! name and parameters (`table.rs` is that signature table, as data). Registered functions PySpark lacks
//! are not exported; PySpark functions that are not registered are not generated. A registered function
//! that is in neither list makes [`generate`] fail, so a new registry entry cannot be forgotten.
//!
//! This is development tooling, compiled for the crate's tests only (not part of the library): the test
//! `the_checked_in_wrappers_match_the_registry` fails when the files are out of date, and
//! `KRISHIV_REGENERATE_PYTHON=1 cargo test -p krishiv-functions python_wrappers` rewrites them.

mod table;
#[cfg(test)]
mod tests;

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::path::Path;

/// The generated module, relative to the workspace root.
pub(crate) const PY_PATH: &str = "python/krishiv/sql/_generated_functions.py";
/// Its type stub, relative to the workspace root.
pub(crate) const PYI_PATH: &str = "python/krishiv/sql/_generated_functions.pyi";

/// The line length `ruff format` is configured with.
const MAX_LINE: usize = 120;

/// The support helpers a wrapper body can call (`krishiv/sql/_function_support.py`).
const HELPERS: &[&str] = &[
    "_at_least",
    "_call",
    "_cols",
    "_deprecated",
    "_int_or_col",
    "_lit",
    "_named",
    "_num_or_col",
    "_one_of",
    "_opt",
];

/// The generated files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Generated {
    /// `_generated_functions.py`.
    pub py: String,
    /// `_generated_functions.pyi`.
    pub pyi: String,
    /// The exported names, in `__all__` order.
    pub names: Vec<String>,
}

/// A check a wrapper makes before it builds the call.
#[derive(Debug, Clone, Copy)]
enum Check {
    None,
    /// At least this many positional columns (`greatest`, `least`).
    AtLeast(usize),
    /// The parameter is one of these values (`sha2`'s `numBits`).
    OneOf(&'static str, &'static [i64]),
}

/// One PySpark function and the registry function it calls.
#[derive(Debug, Clone)]
pub(crate) struct Row {
    py: &'static str,
    registry: &'static str,
    params: &'static str,
    check: Check,
    reversed: bool,
    deprecated: Option<&'static str>,
    /// A parameter with a `None` default that is sent even when `None` (as a SQL NULL) instead
    /// of being left out of the call: `lag`/`lead`'s `default`, which the parser fills with NULL,
    /// so the wrapper builds the same call the SQL form builds.
    always: Option<&'static str>,
}

impl Row {
    /// A function that calls the registry function of its own name.
    pub(crate) const fn new(py: &'static str, params: &'static str) -> Self {
        Self {
            py,
            registry: py,
            params,
            check: Check::None,
            reversed: false,
            deprecated: None,
            always: None,
        }
    }

    /// Calls this registry function instead.
    pub(crate) const fn to(mut self, registry: &'static str) -> Self {
        self.registry = registry;
        self
    }

    /// Sends the arguments to the registry in reverse order.
    pub(crate) const fn reversed(mut self) -> Self {
        self.reversed = true;
        self
    }

    /// Sends `param` even when it is `None` (as a SQL NULL).
    pub(crate) const fn always(mut self, param: &'static str) -> Self {
        self.always = Some(param);
        self
    }

    /// Requires at least `n` columns.
    pub(crate) const fn at_least(mut self, n: usize) -> Self {
        self.check = Check::AtLeast(n);
        self
    }

    /// Requires `param` to be one of `values`.
    pub(crate) const fn one_of(mut self, param: &'static str, values: &'static [i64]) -> Self {
        self.check = Check::OneOf(param, values);
        self
    }

    /// Warns that the function is deprecated, with PySpark's message.
    pub(crate) const fn deprecated(mut self, message: &'static str) -> Self {
        self.deprecated = Some(message);
        self
    }
}

/// How a parameter reaches the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Conv {
    /// A `Column`, or a `str` naming one.
    Column,
    /// As [`Conv::Column`], and a number is a literal (PySpark's `ColumnOrName | float`).
    Number,
    /// As [`Conv::Column`], and an `int` is a literal; a `float` is not (PySpark's `ColumnOrName | int`).
    Int,
    /// A `Column`, or any other value is a literal.
    Literal,
}

impl Conv {
    fn of_annotation(annotation: &str) -> Option<Self> {
        match annotation {
            "ColumnOrName" | "Column" => Some(Self::Column),
            "ColumnOrName | float" => Some(Self::Number),
            "ColumnOrName | int" => Some(Self::Int),
            "Column | int" | "Column | str" | "int | Column" | "str | Column" | "bool | Column"
            | "int" | "str" | "bool" | "Any" => Some(Self::Literal),
            _ => None,
        }
    }

    const fn helper(self) -> &'static str {
        match self {
            Self::Column => "_named",
            Self::Number => "_num_or_col",
            Self::Int => "_int_or_col",
            Self::Literal => "_lit",
        }
    }
}

#[derive(Debug)]
struct Param {
    name: String,
    annotation: String,
    conv: Conv,
    /// Python source of the default; `None` is an argument that is left out of the call.
    default: Option<String>,
    variadic: bool,
}

impl Param {
    fn declaration(&self) -> String {
        match (&self.default, self.variadic) {
            (_, true) => format!("*{}: {}", self.name, self.annotation),
            (Some(d), _) if d == "None" => {
                format!("{}: {} | None = None", self.name, self.annotation)
            }
            (Some(d), _) => format!("{}: {} = {d}", self.name, self.annotation),
            (None, _) => format!("{}: {}", self.name, self.annotation),
        }
    }

    fn argument(&self, always: bool) -> String {
        let (name, helper) = (&self.name, self.conv.helper());
        if self.variadic {
            format!("*_cols({name})")
        } else if !always && self.default.as_deref() == Some("None") {
            format!("*_opt({helper}, {name}, \"{name}\")")
        } else {
            format!("{helper}({name}, \"{name}\")")
        }
    }
}

/// Parses `name: annotation [= default]` items; `*name: ColumnOrName` is variadic.
fn parse_params(spec: &str) -> Result<Vec<Param>, String> {
    let mut out = Vec::new();
    for item in spec.split(',').map(str::trim).filter(|i| !i.is_empty()) {
        let (head, default) = match item.split_once('=') {
            Some((h, d)) => (h.trim(), Some(d.trim().to_owned())),
            None => (item, None),
        };
        let (name, annotation) = head
            .split_once(':')
            .map(|(n, a)| (n.trim(), a.trim()))
            .ok_or_else(|| format!("parameter `{item}` has no annotation"))?;
        let variadic = name.starts_with('*');
        let name = name.trim_start_matches('*');
        let conv = Conv::of_annotation(annotation)
            .ok_or_else(|| format!("parameter `{item}`: unknown annotation `{annotation}`"))?;
        if variadic && (annotation != "ColumnOrName" || default.is_some()) {
            return Err(format!(
                "`*{name}` must be `ColumnOrName` without a default"
            ));
        }
        out.push(Param {
            name: name.to_owned(),
            annotation: annotation.to_owned(),
            conv,
            default,
            variadic,
        });
    }
    Ok(out)
}

/// `head` + `args` + `tail`, laid out as `ruff format` does: on one line if it fits, else the
/// arguments on one indented line, else one per line with a trailing comma.
fn wrap(head: &str, args: &[String], tail: &str, indent: usize) -> String {
    let joined = args.join(", ");
    let one = format!("{head}{joined}{tail}");
    if args.is_empty() || one.len() <= MAX_LINE {
        return one;
    }
    let (outer, inner) = (" ".repeat(indent), " ".repeat(indent + 4));
    if inner.len() + joined.len() <= MAX_LINE {
        return format!("{head}\n{inner}{joined}\n{outer}{tail}");
    }
    let lines: String = args.iter().map(|a| format!("{inner}{a},\n")).collect();
    format!("{head}\n{lines}{outer}{tail}")
}

fn split_digits(s: &[u8]) -> (&[u8], &[u8]) {
    s.split_at(s.iter().take_while(|c| c.is_ascii_digit()).count())
}

/// The ordering `ruff` uses for `__all__`: digit runs compare as numbers (`log2` before `log10`).
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (mut a, mut b) = (a.as_bytes(), b.as_bytes());
    loop {
        match (a.split_first(), b.split_first()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some((x, _)), Some((y, _))) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let ((na, ra), (nb, rb)) = (split_digits(a), split_digits(b));
                let order = na.len().cmp(&nb.len()).then_with(|| na.cmp(nb));
                if order != Ordering::Equal {
                    return order;
                }
                (a, b) = (ra, rb);
            }
            (Some((x, ra)), Some((y, rb))) => {
                if x != y {
                    return x.cmp(y);
                }
                (a, b) = (ra, rb);
            }
        }
    }
}

/// One wrapper: the definition, and its stub.
fn render_function(row: &Row, params: &[Param]) -> (String, String) {
    let declarations: Vec<String> = params.iter().map(Param::declaration).collect();
    let head = format!("def {}(", row.py);
    let mut args: Vec<String> = params
        .iter()
        .map(|p| p.argument(row.always == Some(p.name.as_str())))
        .collect();
    if row.reversed {
        args.reverse();
    }
    args.insert(0, format!("\"{}\"", row.registry));
    let mut body = String::new();
    let doc = if row.deprecated.is_some() {
        format!(
            "Deprecated alias; calls the registry function ``{}``.",
            row.registry
        )
    } else {
        format!("Calls the registry function ``{}``.", row.registry)
    };
    body.push_str(&format!("    \"\"\"{doc}\"\"\"\n"));
    match row.check {
        Check::None => {}
        Check::AtLeast(n) => {
            let name = params.first().map_or("cols", |p| p.name.as_str());
            body.push_str(&format!("    _at_least(\"{}\", {name}, {n})\n", row.py));
        }
        Check::OneOf(param, values) => {
            let list: Vec<String> = values.iter().map(i64::to_string).collect();
            body.push_str(&format!(
                "    _one_of(\"{param}\", {param}, ({}))\n",
                list.join(", ")
            ));
        }
    }
    if let Some(message) = row.deprecated {
        body.push_str(&format!("    _deprecated(\"{message}\")\n"));
    }
    if row.py == "str_to_map" {
        // PySpark sends the default pair delimiter when only `keyValueDelim`
        // is given, so the third argument keeps its position.
        body.push_str("    if pairDelim is None and keyValueDelim is not None:\n");
        let defaulted = vec![
            format!("\"{}\"", row.registry),
            "_named(text, \"text\")".to_owned(),
            "_lit(\",\", \"pairDelim\")".to_owned(),
            "_named(keyValueDelim, \"keyValueDelim\")".to_owned(),
        ];
        body.push_str(&wrap("        return _call(", &defaulted, ")", 8));
        body.push('\n');
    }
    body.push_str(&wrap("    return _call(", &args, ")", 4));
    body.push('\n');
    let def = format!("{}\n{body}", wrap(&head, &declarations, ") -> Column:", 0));
    let stub = wrap(&head, &declarations, ") -> Column: ...", 0);
    (def, stub)
}

/// Generates the wrappers of the functions in `registered`.
///
/// # Errors
/// Returns a message when a registered function has neither a PySpark signature nor an entry in the
/// not-exported or hand-written lists, or when the table is malformed.
pub(crate) fn generate(registered: &BTreeSet<&str>) -> Result<Generated, String> {
    let rows = table::rows();
    let known: BTreeSet<&str> = rows
        .iter()
        .map(|r| r.registry)
        .chain(table::NOT_EXPORTED.iter().map(|(n, _)| *n))
        .chain(table::HANDWRITTEN.iter().map(|(n, _)| *n))
        .collect();
    let unaccounted: Vec<&str> = registered
        .iter()
        .copied()
        .filter(|n| !known.contains(n))
        .collect();
    if !unaccounted.is_empty() {
        return Err(format!(
            "registered functions without a PySpark signature: {}; add a row to \
             crates/krishiv-functions/src/python_wrappers/table.rs, or list them in NOT_EXPORTED",
            unaccounted.join(", ")
        ));
    }
    let mut chosen: Vec<(&Row, Vec<Param>)> = Vec::new();
    for row in rows.iter().filter(|r| registered.contains(r.registry)) {
        let params = parse_params(row.params).map_err(|e| format!("{}: {e}", row.py))?;
        chosen.push((row, params));
    }
    chosen.sort_by(|(a, _), (b, _)| natural_cmp(a.py, b.py));
    if let Some(w) = chosen.windows(2).find(|w| w[0].0.py == w[1].0.py) {
        return Err(format!("`{}` is in the table twice", w[0].0.py));
    }
    let names: Vec<String> = chosen.iter().map(|(r, _)| r.py.to_owned()).collect();
    let (defs, stubs): (Vec<String>, Vec<String>) = chosen
        .iter()
        .map(|(row, params)| render_function(row, params))
        .unzip();
    let body = defs.join("\n\n");
    let imports: String = HELPERS
        .iter()
        .filter(|h| body.contains(*h))
        .map(|h| format!("    {h},\n"))
        .collect();
    let all: String = names.iter().map(|n| format!("    \"{n}\",\n")).collect();
    // `Any` (a literal of any type, as `lag`'s default) needs its import.
    let typing = if body.contains(": Any") {
        "from typing import Any\n\n"
    } else {
        ""
    };
    let marker = "# @generated by `KRISHIV_REGENERATE_PYTHON=1 cargo test -p krishiv-functions python_wrappers`; do not edit.";
    let py = format!(
        "# SPDX-License-Identifier: Apache-2.0\n{marker}\n\
         \"\"\"The registered functions under their ``pyspark.sql.functions`` names and signatures.\n\n\
         Each wrapper calls the registry function by name; the registry resolves it when the query is analyzed.\n\
         \"\"\"\n\n\
         from __future__ import annotations\n\n{typing}\
         from krishiv.sql._function_support import (\n    ColumnOrName,\n{imports})\n\
         from krishiv.sql.column import Column\n\n\
         __all__ = [\n{all}]\n\n\n{body}"
    );
    let pyi = format!(
        "# SPDX-License-Identifier: Apache-2.0\n{marker}\n\
         {typing}from krishiv.sql._function_support import ColumnOrName\n\
         from krishiv.sql.column import Column\n\n\
         __all__ = [\n{all}]\n\n{}\n",
        stubs.join("\n")
    );
    Ok(Generated { py, pyi, names })
}

/// Generates the wrappers of every function the registry implements.
///
/// # Errors
/// As [`generate`].
pub(crate) fn generate_for_registry() -> Result<Generated, String> {
    let functions = crate::registry::builtins().functions();
    generate(&functions.iter().map(|d| d.name).collect())
}

/// The generated files under `root` (the workspace root) that differ from the registry's output.
///
/// # Errors
/// As [`generate`].
pub(crate) fn out_of_date(root: &Path) -> Result<Vec<&'static str>, String> {
    let generated = generate_for_registry()?;
    Ok([(PY_PATH, generated.py), (PYI_PATH, generated.pyi)]
        .into_iter()
        .filter(|(path, text)| {
            std::fs::read_to_string(root.join(path)).ok().as_deref() != Some(text)
        })
        .map(|(path, _)| path)
        .collect())
}
