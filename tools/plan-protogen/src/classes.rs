// SPDX-License-Identifier: Apache-2.0
//! Every element of `krishiv.plan.v1` added after the feature header (ADR
//! 0040) says which compatibility class it is, in a comment line:
//!
//! - `feature: <id>`: a reader without the feature would misread it; `<id>`
//!   must be in the codec's feature registry;
//! - `ignorable`: an older reader may skip it.
//!
//! The elements that predate the rule are listed in `baseline.txt`. A new
//! message or enum may carry the annotation once: its fields or values
//! inherit it. An element in the baseline that no longer exists is an error
//! too, so a removal (which must leave a `reserved` entry) is deliberate.

use std::collections::BTreeSet;

use prost_types::{DescriptorProto, EnumDescriptorProto, FileDescriptorProto, SourceCodeInfo};

// Field numbers of descriptor.proto, for `SourceCodeInfo` paths.
const FILE_MESSAGE: i32 = 4;
const FILE_ENUM: i32 = 5;
const MESSAGE_FIELD: i32 = 2;
const MESSAGE_NESTED: i32 = 3;
const MESSAGE_ENUM: i32 = 4;
const ENUM_VALUE: i32 = 2;

/// The class an element's comment declares.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Class {
    Feature(String),
    Ignorable,
}

/// A named element of the schema and the class its comments declare.
#[derive(Debug)]
struct Element {
    name: String,
    class: Option<Class>,
    /// The enclosing message or enum, by index into the element list.
    parent: Option<usize>,
}

fn class_of(info: Option<&SourceCodeInfo>, path: &[i32]) -> Option<Class> {
    let location = info?.location.iter().find(|l| l.path == path)?;
    let comments = [&location.leading_comments, &location.trailing_comments];
    comments
        .into_iter()
        .flatten()
        .flat_map(|c| c.lines())
        .map(str::trim)
        .find_map(|line| {
            if let Some(id) = line.strip_prefix("feature:") {
                let id = id.split_whitespace().next().unwrap_or_default();
                Some(Class::Feature(id.to_owned()))
            } else if line == "ignorable" || line.starts_with("ignorable:") {
                Some(Class::Ignorable)
            } else {
                None
            }
        })
}

struct Walker<'a> {
    info: Option<&'a SourceCodeInfo>,
    out: Vec<Element>,
}

impl Walker<'_> {
    fn push(&mut self, name: String, path: &[i32], parent: Option<usize>) -> usize {
        let class = class_of(self.info, path);
        self.out.push(Element {
            name,
            class,
            parent,
        });
        self.out.len() - 1
    }

    fn message(&mut self, m: &DescriptorProto, prefix: &str, path: &[i32], parent: Option<usize>) {
        let name = format!("{prefix}{}", m.name());
        let at = self.push(name.clone(), path, parent);
        for (i, f) in m.field.iter().enumerate() {
            let field_path = [path, &[MESSAGE_FIELD, index(i)]].concat();
            self.push(format!("{name}.{}", f.name()), &field_path, Some(at));
        }
        for (i, n) in m.nested_type.iter().enumerate() {
            let nested = [path, &[MESSAGE_NESTED, index(i)]].concat();
            self.message(n, &format!("{name}."), &nested, Some(at));
        }
        for (i, e) in m.enum_type.iter().enumerate() {
            let nested = [path, &[MESSAGE_ENUM, index(i)]].concat();
            self.enumeration(e, &format!("{name}."), &nested, Some(at));
        }
    }

    fn enumeration(
        &mut self,
        e: &EnumDescriptorProto,
        prefix: &str,
        path: &[i32],
        parent: Option<usize>,
    ) {
        let name = format!("{prefix}{}", e.name());
        let at = self.push(name.clone(), path, parent);
        for (i, v) in e.value.iter().enumerate() {
            let value_path = [path, &[ENUM_VALUE, index(i)]].concat();
            self.push(format!("{name}.{}", v.name()), &value_path, Some(at));
        }
    }
}

fn index(i: usize) -> i32 {
    i32::try_from(i).unwrap_or(i32::MAX)
}

fn elements(file: &FileDescriptorProto) -> Vec<Element> {
    let mut w = Walker {
        info: file.source_code_info.as_ref(),
        out: Vec::new(),
    };
    for (i, m) in file.message_type.iter().enumerate() {
        w.message(m, "", &[FILE_MESSAGE, index(i)], None);
    }
    for (i, e) in file.enum_type.iter().enumerate() {
        w.enumeration(e, "", &[FILE_ENUM, index(i)], None);
    }
    w.out
}

/// The names of every element of `file`, sorted (the baseline's form).
#[cfg(test)]
fn names(file: &FileDescriptorProto) -> Vec<String> {
    let mut names: Vec<_> = elements(file).into_iter().map(|e| e.name).collect();
    names.sort();
    names
}

/// The feature ids in the registry source `features_rs`: the first item of
/// each `(id, Variant, ...)` line inside the `registry! { ... }` invocation.
/// Lines elsewhere (the macro's definition, comments, other code) never
/// register an id.
pub fn registered_features(features_rs: &str) -> BTreeSet<String> {
    features_rs
        .lines()
        .skip_while(|l| l.trim() != "registry! {")
        .skip(1)
        .take_while(|l| l.trim() != "}")
        .filter_map(|l| l.trim().strip_prefix('('))
        .filter_map(|l| l.split_once(','))
        .map(|(id, _)| id.trim().to_owned())
        .filter(|id| {
            !id.is_empty()
                && id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        })
        .collect()
}

/// The problems with `file`'s annotations against `baseline` and the
/// registered `features`, one message each.
pub fn check(
    file: &FileDescriptorProto,
    baseline: &BTreeSet<String>,
    features: &BTreeSet<String>,
) -> Vec<String> {
    let elements = elements(file);
    let mut problems = Vec::new();
    for e in &elements {
        if let Some(Class::Feature(id)) = &e.class
            && !features.contains(id)
        {
            problems.push(format!(
                "{}: `feature: {id}` is not in the feature registry",
                e.name
            ));
        }
        if baseline.contains(&e.name) {
            continue;
        }
        // The element itself or an enclosing new element declares a class.
        let mut at = Some(e);
        let mut declared = false;
        while let Some(x) = at {
            if x.class.is_some() {
                declared = true;
                break;
            }
            at = x
                .parent
                .map(|p| &elements[p])
                .filter(|p| !baseline.contains(&p.name));
        }
        if !declared {
            problems.push(format!(
                "{}: a new element needs a `// feature: <id>` or `// ignorable` comment",
                e.name
            ));
        }
    }
    let present: BTreeSet<&str> = elements.iter().map(|e| e.name.as_str()).collect();
    for name in baseline {
        if !present.contains(name.as_str()) {
            problems.push(format!(
                "{name}: in baseline.txt but no longer in the schema (reserve its number and \
                 remove the line)"
            ));
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(source: &str) -> FileDescriptorProto {
        let dir = std::env::temp_dir().join(format!(
            "plan-protogen-test-{}-{:x}",
            std::process::id(),
            source.len() * 31 + source.bytes().map(usize::from).sum::<usize>()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("t.proto"), source).unwrap();
        let fds = protox::compile(["t.proto"], [&dir]);
        std::fs::remove_dir_all(&dir).unwrap();
        fds.unwrap().file.into_iter().next().unwrap()
    }

    const BASE: &str = "syntax = \"proto3\";\npackage t;\n\
        message A {\n  uint32 x = 1;\n}\nenum E {\n  E_UNSPECIFIED = 0;\n}\n";

    fn set(v: &[&str]) -> BTreeSet<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    fn baseline() -> BTreeSet<String> {
        set(&["A", "A.x", "E", "E.E_UNSPECIFIED"])
    }

    #[test]
    fn the_baseline_needs_no_annotation() {
        let file = compile(BASE);
        assert_eq!(names(&file), Vec::from_iter(baseline()));
        assert!(check(&file, &baseline(), &set(&[])).is_empty());
    }

    #[test]
    fn a_new_field_or_value_needs_a_class() {
        let file = compile(
            &BASE
                .replace("uint32 x = 1;", "uint32 x = 1;\n  uint32 y = 2;")
                .replace("E_UNSPECIFIED = 0;", "E_UNSPECIFIED = 0;\n  E_NEW = 1;"),
        );
        let problems = check(&file, &baseline(), &set(&[]));
        assert_eq!(problems.len(), 2, "{problems:?}");
        assert!(
            problems[0].starts_with("A.y: a new element needs"),
            "{problems:?}"
        );
        assert!(
            problems[1].starts_with("E.E_NEW: a new element needs"),
            "{problems:?}"
        );
    }

    #[test]
    fn annotated_elements_pass_and_new_messages_pass_it_on() {
        let source = BASE.replace(
            "uint32 x = 1;",
            "uint32 x = 1;\n  // feature: f\n  B y = 2;\n  uint32 z = 3; // ignorable: a hint\n",
        ) + "// feature: f\n// What B holds.\nmessage B {\n  uint32 w = 1;\n}\n";
        let file = compile(&source);
        assert!(check(&file, &baseline(), &set(&["f"])).is_empty());
        // The id must be registered.
        let problems = check(&file, &baseline(), &set(&["g"]));
        assert_eq!(problems.len(), 2, "{problems:?}");
        assert!(
            problems
                .iter()
                .all(|p| p.contains("`feature: f` is not in the feature registry"))
        );
    }

    #[test]
    fn a_field_of_a_baseline_message_does_not_inherit() {
        // `A` predates the rule, so a field added to it declares its own class.
        let source = BASE
            .replace("message A {", "// feature: f\nmessage A {")
            .replace("uint32 x = 1;", "uint32 x = 1;\n  uint32 y = 2;");
        let problems = check(&compile(&source), &baseline(), &set(&["f"]));
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].starts_with("A.y:"), "{problems:?}");
    }

    #[test]
    fn a_removed_baseline_element_is_reported() {
        let file = compile(&BASE.replace("uint32 x = 1;", "reserved 1;\n  reserved \"x\";"));
        let problems = check(&file, &baseline(), &set(&[]));
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].starts_with("A.x: in baseline.txt but no longer"),
            "{problems:?}"
        );
    }

    #[test]
    fn registry_lines_are_parsed() {
        let src = "registry! {\n    (alpha_one, AlphaOne, 3, Semantic, \"a\"),\n    \
                   (beta, Beta, 6, Optimization, \"b\"),\n}\n    (Not, an id)\n";
        assert_eq!(registered_features(src), set(&["alpha_one", "beta"]));
    }

    #[test]
    fn registry_like_lines_outside_the_invocation_are_ignored() {
        let src = "/*\n    (phantom, Phantom, 6, Semantic, \"p\"),\n*/\n\
                   macro_rules! registry {\n    (ghost, Ghost, 6, Semantic, \"g\"),\n}\n\
                   registry! {\n    (real, Real, 6, Semantic, \"r\"),\n}\n\
                   (after, After, 6, Semantic, \"a\"),\n";
        assert_eq!(registered_features(src), set(&["real"]));
        assert!(registered_features("(only, Only, 6, Semantic, \"o\"),\n").is_empty());
    }
}
