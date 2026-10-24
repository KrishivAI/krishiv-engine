// SPDX-License-Identifier: Apache-2.0
//! Tests of the wrapper generator.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::path::Path;

use super::{
    Generated, PY_PATH, PYI_PATH, generate, generate_for_registry, natural_cmp, out_of_date,
    parse_params, table, wrap,
};

/// The names in `table::NOT_EXPORTED` and `HANDWRITTEN` need no row; this is the registry in miniature.
fn generated_for(names: &[&str]) -> Generated {
    let registered: BTreeSet<&str> = names.iter().copied().collect();
    generate(&registered).expect("generates")
}

#[test]
fn parses_optional_defaulted_and_variadic_parameters() {
    let p = parse_params("col: ColumnOrName, scale: Column | int = None, flag: bool = False")
        .expect("parses");
    assert_eq!(
        p.iter().map(super::Param::declaration).collect::<Vec<_>>(),
        [
            "col: ColumnOrName",
            "scale: Column | int | None = None",
            "flag: bool = False"
        ]
    );
    assert_eq!(p[1].argument(), "*_opt(_lit, scale, \"scale\")");
    assert_eq!(p[2].argument(), "_lit(flag, \"flag\")");
    let v = parse_params("*cols: ColumnOrName").expect("parses");
    assert_eq!(v[0].declaration(), "*cols: ColumnOrName");
    assert_eq!(v[0].argument(), "*_cols(cols)");
    assert!(parse_params("").expect("parses").is_empty());
}

#[test]
fn rejects_malformed_parameters() {
    assert!(parse_params("col").is_err());
    assert!(parse_params("col: Sequence").is_err());
    assert!(parse_params("*cols: int").is_err());
}

#[test]
fn every_table_row_parses_and_names_are_unique() {
    let rows = table::rows();
    for row in &rows {
        parse_params(row.params).unwrap_or_else(|e| panic!("{}: {e}", row.py));
    }
    let names: BTreeSet<&str> = rows.iter().map(|r| r.py).collect();
    assert_eq!(names.len(), rows.len());
}

#[test]
fn wraps_like_ruff() {
    let (x, y, z) = ("x".repeat(56), "y".repeat(56), "z".repeat(56));
    let short = vec!["a".to_owned(), "b".to_owned()];
    assert_eq!(
        wrap("function_name(", &short, ")", 0),
        "function_name(a, b)"
    );
    // 129 columns on one line, but the arguments fit on one indented line.
    let two = vec![x.clone(), y.clone()];
    assert_eq!(
        wrap("function_name(", &two, ")", 0),
        format!("function_name(\n    {x}, {y}\n)")
    );
    // Three do not: one per line, with a trailing comma.
    let three = vec![x.clone(), y.clone(), z.clone()];
    assert_eq!(
        wrap("function_name(", &three, ")", 0),
        format!("function_name(\n    {x},\n    {y},\n    {z},\n)")
    );
}

#[test]
fn digit_runs_sort_as_numbers() {
    let mut names = [
        "log10",
        "log1p",
        "log2",
        "sha2",
        "sha1",
        "shiftRight",
        "shiftleft",
    ];
    names.sort_by(|a, b| natural_cmp(a, b));
    assert_eq!(
        names,
        [
            "log1p",
            "log2",
            "log10",
            "sha1",
            "sha2",
            "shiftRight",
            "shiftleft"
        ]
    );
    assert_eq!(natural_cmp("abs", "abs"), Ordering::Equal);
}

#[test]
fn a_pyspark_function_the_registry_lacks_is_not_generated() {
    let g = generated_for(&["abs"]);
    assert_eq!(g.names, ["abs"]);
    assert!(!g.py.contains("def acos"));
    assert!(g.py.contains("def abs(col: ColumnOrName) -> Column:"));
    assert!(g.pyi.contains("def abs(col: ColumnOrName) -> Column: ..."));
}

#[test]
fn a_registered_function_pyspark_lacks_is_not_exported() {
    let g = generated_for(&["abs", "&", "mod", "if", "count", "log"]);
    assert_eq!(g.names, ["abs"]);
}

#[test]
fn a_registered_function_without_a_signature_is_an_error() {
    let registered: BTreeSet<&str> = ["abs", "no_such_pyspark_function"].into();
    let e = generate(&registered).expect_err("unaccounted function");
    assert!(e.contains("no_such_pyspark_function"), "{e}");
    assert!(!e.contains("abs,"), "{e}");
}

#[test]
fn renders_optional_reversed_checked_and_deprecated_calls() {
    let g = generated_for(&[
        "ceil",
        "trim",
        "greatest",
        "sha2",
        "~",
        "shiftleft",
        "power",
    ]);
    let py = &g.py;
    assert!(py.contains(
        "def ceil(col: ColumnOrName, scale: Column | int | None = None) -> Column:\n    \
         \"\"\"Calls the registry function ``ceil``.\"\"\"\n    \
         return _call(\"ceil\", _named(col, \"col\"), *_opt(_lit, scale, \"scale\"))"
    ));
    // `trim(chars, str)`: the optional characters come first.
    assert!(
        py.contains("return _call(\"trim\", *_opt(_named, trim, \"trim\"), _named(col, \"col\"))")
    );
    assert!(py.contains(
        "    _at_least(\"greatest\", cols, 2)\n    return _call(\"greatest\", *_cols(cols))"
    ));
    assert!(py.contains("    _one_of(\"numBits\", numBits, (0, 224, 256, 384, 512))\n"));
    assert!(
        py.contains("def pow(col1: ColumnOrName | float, col2: ColumnOrName | float) -> Column:")
    );
    assert!(py.contains(
        "return _call(\"power\", _num_or_col(col1, \"col1\"), _num_or_col(col2, \"col2\"))"
    ));
    assert!(py.contains("def bitwise_not(col: ColumnOrName) -> Column:"));
    assert!(py.contains("return _call(\"~\", _named(col, \"col\"))"));
    assert!(py.contains("def shiftLeft("));
    assert!(py.contains("_deprecated(\"Deprecated in 3.2, use shiftleft instead.\")"));
}

#[test]
fn imports_only_the_helpers_that_are_used() {
    let g = generated_for(&["abs"]);
    assert!(
        g.py.contains("    ColumnOrName,\n    _call,\n    _named,\n)"),
        "{}",
        g.py
    );
}

#[test]
fn the_registry_is_fully_accounted_for_and_the_table_is_not_stale() {
    let registered: BTreeSet<&str> = crate::registry::builtins()
        .functions()
        .iter()
        .map(|d| d.name)
        .collect();
    for row in table::rows() {
        assert!(
            registered.contains(row.registry),
            "{} calls `{}`, which is not registered; remove the row",
            row.py,
            row.registry
        );
    }
    for (name, _) in table::NOT_EXPORTED.iter().chain(table::HANDWRITTEN) {
        assert!(
            registered.contains(name),
            "`{name}` is not registered; remove it from the table"
        );
    }
    let g = generate_for_registry().expect("every registered function is accounted for");
    assert!(g.names.len() > 150, "{}", g.names.len());
}

/// The checked-in files are the registry's output. With `KRISHIV_REGENERATE_PYTHON` set the test writes them
/// instead (the generated files are never edited by hand).
#[test]
fn the_checked_in_wrappers_match_the_registry() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if std::env::var_os("KRISHIV_REGENERATE_PYTHON").is_some() {
        let g = generate_for_registry().expect("generates");
        for (path, text) in [(PY_PATH, g.py), (PYI_PATH, g.pyi)] {
            std::fs::write(root.join(path), text).expect("writes the generated file");
        }
        return;
    }
    let stale = out_of_date(&root).expect("generates");
    assert!(
        stale.is_empty(),
        "{stale:?} out of date: run `KRISHIV_REGENERATE_PYTHON=1 cargo test -p krishiv-functions python_wrappers`"
    );
}

#[test]
fn an_int_parameter_takes_only_an_int_as_a_literal() {
    let g = generated_for(&["date_add", "atan2"]);
    assert!(g.py.contains("_int_or_col(days, \"days\")"), "{}", g.py);
    assert!(g.py.contains("_num_or_col(col1, \"col1\")"), "{}", g.py);
}
