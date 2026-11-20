// SPDX-License-Identifier: Apache-2.0
//! `variant_insert`, `variant_set`, `variant_delete`, `variant_array_append`, `variant_strip_nulls` and the `try_`
//! forms against a Spark 4.3.0 run (every expectation below is a row of that run).

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::{Array, ArrayRef, BooleanArray, StringArray};
use krishiv_common::error::ErrorClass;

use super::{Arg, BindArg, BindOptions, bind_scalar};

fn text_col(rows: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(rows.to_vec()))
}

fn arg(a: &ArrayRef) -> BindArg {
    BindArg::Column {
        ty: a.data_type().clone(),
        nullable: true,
    }
}

fn run(
    name: &str,
    cols: &[&ArrayRef],
    literal: Option<Option<bool>>,
) -> Result<ArrayRef, ErrorClass> {
    let mut binds: Vec<BindArg> = cols.iter().map(|c| arg(c)).collect();
    let mut args: Vec<Arg> = cols.iter().map(|c| Arg::Array(Arc::clone(c))).collect();
    if let Some(flag) = literal {
        let flag: ArrayRef = Arc::new(BooleanArray::from(vec![flag]));
        binds.push(BindArg::Literal(Arc::clone(&flag)));
        args.push(Arg::Scalar(flag));
    }
    let f = bind_scalar(name, &binds, &BindOptions::default()).map_err(|e| e.error_class())?;
    f.invoke(&args, cols[0].len()).map_err(|e| e.error_class())
}

fn variants(json: &[Option<&str>]) -> ArrayRef {
    run("parse_json", &[&text_col(json)], None).unwrap()
}

fn show(v: &ArrayRef) -> Vec<Option<String>> {
    let f = bind_scalar("to_json", &[arg(v)], &BindOptions::default()).unwrap();
    let out = f.invoke(&[Arg::Array(Arc::clone(v))], v.len()).unwrap();
    out.as_string_view()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect()
}

/// One row of `name(parse_json(v), path, parse_json(val)[, flag])` as JSON text.
fn edit(
    name: &str,
    v: &str,
    path: &str,
    val: &str,
    flag: Option<bool>,
) -> Result<Option<String>, ErrorClass> {
    let (v, p, val) = (
        variants(&[Some(v)]),
        text_col(&[Some(path)]),
        variants(&[Some(val)]),
    );
    run(name, &[&v, &p, &val], flag.map(Some)).map(|out| show(&out).remove(0))
}

fn ok(s: &str) -> Result<Option<String>, ErrorClass> {
    Ok(Some(s.to_owned()))
}

#[test]
fn insert_adds_fields_and_shifts_elements() {
    let i = |v, p, x| edit("variant_insert", v, p, x, None);
    assert_eq!(i(r#"{"a":1}"#, "$.b", "2"), ok(r#"{"a":1,"b":2}"#));
    assert_eq!(i("{}", "$.a.b", "1"), ok(r#"{"a":{"b":1}}"#));
    assert_eq!(
        i(r#"["a","b","c"]"#, "$[1]", r#""z""#),
        ok(r#"["a","z","b","c"]"#)
    );
    assert_eq!(
        i(r#"["a","b"]"#, "$[5]", r#""z""#),
        ok(r#"["a","b",null,null,null,"z"]"#)
    );
    assert_eq!(i("{}", "$.a", r#"{"x":1}"#), ok(r#"{"a":{"x":1}}"#));
    assert_eq!(i("{}", "$.a", "null"), ok(r#"{"a":null}"#));
    assert_eq!(i("{}", "$.a[2]", "2"), ok(r#"{"a":[null,null,2]}"#));
    assert_eq!(i("{}", "$.a.b[1]", "2"), ok(r#"{"a":{"b":[null,2]}}"#));
    assert_eq!(i("[1,2,3]", "$[3]", "9"), ok("[1,2,3,9]"));
    assert_eq!(i("[1,2,3]", "$[4]", "9"), ok("[1,2,3,null,9]"));
    assert_eq!(
        i(r#"{"a":[{"b":1}]}"#, "$.a[1].c", "2"),
        ok(r#"{"a":[{"b":1},{"c":2}]}"#)
    );
    assert_eq!(
        i(r#"{"a":[{"b":1}]}"#, "$.a[0].c", "2"),
        ok(r#"{"a":[{"b":1,"c":2}]}"#)
    );
}

#[test]
fn insert_errors() {
    let i = |v, p| edit("variant_insert", v, p, "2", None);
    assert_eq!(i(r#"{"a":1}"#, "$.a"), Err(ErrorClass::VariantDuplicateKey));
    assert_eq!(i("[1,2]", "$.a"), Err(ErrorClass::VariantPathTypeMismatch));
    assert_eq!(
        i(r#"{"a":1}"#, "$[0]"),
        Err(ErrorClass::VariantPathTypeMismatch)
    );
    assert_eq!(i("1", "$.a"), Err(ErrorClass::VariantPathTypeMismatch));
    assert_eq!(
        i(r#"{"a":null}"#, "$.a.b"),
        Err(ErrorClass::VariantPathTypeMismatch)
    );
    for bad in ["$", "a", "$."] {
        assert_eq!(
            i(r#"{"a":1}"#, bad),
            Err(ErrorClass::InvalidVariantPath),
            "{bad}"
        );
    }
}

#[test]
fn try_forms_turn_edit_errors_into_null_but_not_bad_paths() {
    let t = |name, v, p| edit(name, v, p, "2", None);
    assert_eq!(t("try_variant_insert", r#"{"a":1}"#, "$.a"), Ok(None));
    assert_eq!(t("try_variant_insert", "[1]", "$.a"), Ok(None));
    assert_eq!(
        t("try_variant_insert", r#"{"a":1}"#, "$"),
        Err(ErrorClass::InvalidVariantPath)
    );
    assert_eq!(t("try_variant_set", r#"{"a":1}"#, "$.a.b"), Ok(None));
    assert_eq!(t("try_variant_set", "1", "$.a"), Ok(None));
    assert_eq!(t("try_variant_array_append", r#"{"a":1}"#, "$"), Ok(None));
    assert_eq!(
        t("try_variant_array_append", r#"{"a":1}"#, "bad"),
        Err(ErrorClass::InvalidVariantPath)
    );
    assert_eq!(
        t("try_variant_array_append", r#"{"a":1}"#, "$.zz"),
        ok(r#"{"a":1}"#)
    );
}

#[test]
fn set_replaces_or_creates() {
    let s = |v, p, x, flag| edit("variant_set", v, p, x, flag);
    assert_eq!(s(r#"{"a":1}"#, "$.a", "2", None), ok(r#"{"a":2}"#));
    assert_eq!(s(r#"{"a":1}"#, "$.b", "2", None), ok(r#"{"a":1,"b":2}"#));
    assert_eq!(
        s(r#"{"a":1}"#, "$.b.c", "2", None),
        ok(r#"{"a":1,"b":{"c":2}}"#)
    );
    assert_eq!(s("[1,2]", "$[0]", "9", None), ok("[9,2]"));
    assert_eq!(s("[1,2]", "$[2]", "9", None), ok("[1,2,9]"));
    assert_eq!(s("[1,2]", "$[4]", "9", None), ok("[1,2,null,null,9]"));
    assert_eq!(s("[1,2]", "$[4]", "9", Some(false)), ok("[1,2]"));
    assert_eq!(s(r#"{"a":1}"#, "$.a", "5", Some(false)), ok(r#"{"a":5}"#));
    assert_eq!(s(r#"{"a":1}"#, "$.b.c", "5", Some(false)), ok(r#"{"a":1}"#));
    assert_eq!(
        s(r#"{"a":1}"#, "$.a.b", "9", None),
        Err(ErrorClass::VariantPathTypeMismatch)
    );
    assert_eq!(
        s(r#"{"a":1}"#, "$", "9", None),
        Err(ErrorClass::InvalidVariantPath)
    );
    // A NULL `create_if_missing` makes every row NULL.
    let (v, p, val) = (
        variants(&[Some("{}")]),
        text_col(&[Some("$.a")]),
        variants(&[Some("1")]),
    );
    let out = run("variant_set", &[&v, &p, &val], Some(None)).unwrap();
    assert_eq!(show(&out), vec![None]);
}

#[test]
fn array_append_needs_an_existing_array() {
    let a = |v, p| edit("variant_array_append", v, p, "4", None);
    assert_eq!(a("[1,2,3]", "$"), ok("[1,2,3,4]"));
    assert_eq!(a(r#"{"a":[1,2]}"#, "$.a"), ok(r#"{"a":[1,2,4]}"#));
    assert_eq!(a("[[1],[2]]", "$[1]"), ok("[[1],[2,4]]"));
    assert_eq!(a("[[1],[2]]", "$[2]"), ok("[[1],[2]]"));
    assert_eq!(a(r#"{"a":1}"#, "$.missing"), ok(r#"{"a":1}"#));
    assert_eq!(a(r#"{"a":1}"#, "$.b.c"), ok(r#"{"a":1}"#));
    for (v, p) in [
        (r#"{"a":1}"#, "$"),
        (r#"{"a":1}"#, "$.a"),
        ("1", "$[0]"),
        (r#"{"a":1}"#, "$[0]"),
        (r#"{"a":[1]}"#, "$.a[0]"),
    ] {
        assert_eq!(a(v, p), Err(ErrorClass::VariantPathTypeMismatch), "{v} {p}");
    }
    assert_eq!(a("[1]", "bad"), Err(ErrorClass::InvalidVariantPath));
}

#[test]
fn null_arguments_make_the_result_null() {
    let v = variants(&[
        Some(r#"{"a":1}"#),
        None,
        Some(r#"{"a":1}"#),
        Some(r#"{"a":1}"#),
    ]);
    let p = text_col(&[Some("$.b"), Some("$.b"), None, Some("$.b")]);
    let val = variants(&[Some("2"), Some("2"), Some("2"), None]);
    let out = run("variant_insert", &[&v, &p, &val], None).unwrap();
    assert_eq!(
        show(&out),
        vec![Some(r#"{"a":1,"b":2}"#.to_owned()), None, None, None]
    );
}

#[test]
fn delete_applies_paths_left_to_right_and_ignores_misses() {
    let d = |v: &str, paths: &[Option<&str>]| {
        let v = variants(&[Some(v)]);
        let cols: Vec<ArrayRef> = paths.iter().map(|p| text_col(&[*p])).collect();
        let mut all: Vec<&ArrayRef> = vec![&v];
        all.extend(cols.iter());
        run("variant_delete", &all, None).map(|o| show(&o).remove(0))
    };
    assert_eq!(
        d(r#"{"a":1,"b":2}"#, &[Some("$.a")]),
        Ok(Some(r#"{"b":2}"#.to_owned()))
    );
    assert_eq!(d("[1,2,3]", &[Some("$[1]")]), ok("[1,3]"));
    assert_eq!(d("[1,2,3]", &[Some("$[7]")]), ok("[1,2,3]"));
    assert_eq!(d(r#"{"a":{"b":1}}"#, &[Some("$.a.b")]), ok(r#"{"a":{}}"#));
    assert_eq!(d(r#"{"a":1}"#, &[Some("$.a.b")]), ok(r#"{"a":1}"#));
    assert_eq!(d(r#"{"a":1}"#, &[Some("$[0]")]), ok(r#"{"a":1}"#));
    assert_eq!(d("1", &[Some("$.a")]), ok("1"));
    assert_eq!(
        d(
            r#"{"a":1,"b":2,"c":3,"items":[1,2,3]}"#,
            &[None, Some("$.a"), Some("$.c")]
        ),
        ok(r#"{"b":2,"items":[1,2,3]}"#)
    );
    assert_eq!(
        d(r#"{"i":[1,2,3]}"#, &[Some("$.i[0]"), Some("$.i[0]")]),
        ok(r#"{"i":[3]}"#)
    );
    assert_eq!(d("{}", &[Some("$")]), Err(ErrorClass::InvalidVariantPath));
    assert_eq!(d("{}", &[Some("bad")]), Err(ErrorClass::InvalidVariantPath));
}

#[test]
fn strip_nulls_removes_object_nulls_and_by_default_array_nulls() {
    let s = |j: &str, flag: Option<bool>| {
        let v = variants(&[Some(j)]);
        run("variant_strip_nulls", &[&v], flag.map(Some)).map(|o| show(&o).remove(0))
    };
    assert_eq!(s(r#"{"a":1,"b":null,"c":3}"#, None), ok(r#"{"a":1,"c":3}"#));
    assert_eq!(s("[1,null,3]", None), ok("[1,3]"));
    assert_eq!(
        s(r#"{"a":{"b":null,"c":[1,null]}}"#, None),
        ok(r#"{"a":{"c":[1]}}"#)
    );
    assert_eq!(
        s(r#"{"a":[1,null],"b":null}"#, Some(false)),
        ok(r#"{"a":[1,null]}"#)
    );
    assert_eq!(s(r#"{"a":null}"#, None), ok("{}"));
    assert_eq!(s("null", None), ok("null"));
    assert_eq!(s("[null]", None), ok("[]"));
    assert_eq!(
        s(r#"{"a":[null,{"b":null}]}"#, Some(false)),
        ok(r#"{"a":[null,{}]}"#)
    );
    assert_eq!(
        s(r#"{"a":null,"b":[1,null,{"c":null}]}"#, None),
        ok(r#"{"b":[1,{}]}"#)
    );
    // A NULL flag, and a NULL variant, are NULL.
    let v = variants(&[Some("1"), None]);
    assert_eq!(
        show(&run("variant_strip_nulls", &[&v], Some(None)).unwrap()),
        vec![None, None]
    );
    assert_eq!(
        show(&run("variant_strip_nulls", &[&v], None).unwrap()),
        vec![Some("1".to_owned()), None]
    );
}

#[test]
fn an_untyped_null_argument_makes_every_row_null() {
    let null: ArrayRef = Arc::new(arrow_array::NullArray::new(1));
    let v = variants(&[Some(r#"{"a":1}"#)]);
    let p = text_col(&[Some("$.b")]);
    let val = variants(&[Some("2")]);
    // NULL as the variant, as the path and as the value; `variant_delete` and `variant_strip_nulls` of NULL.
    for cols in [[&null, &p, &val], [&v, &null, &val], [&v, &p, &null]] {
        for name in ["variant_insert", "try_variant_set", "variant_array_append"] {
            let out = run(name, &cols, None).unwrap();
            assert_eq!(show(&out), vec![None], "{name}");
        }
    }
    assert_eq!(
        show(&run("variant_delete", &[&null, &p], None).unwrap()),
        vec![None]
    );
    assert_eq!(
        show(&run("variant_delete", &[&v, &null], None).unwrap()),
        vec![Some(r#"{"a":1}"#.to_owned())]
    );
    assert_eq!(
        show(&run("variant_strip_nulls", &[&null], None).unwrap()),
        vec![None]
    );
}

#[test]
fn a_value_of_another_type_is_cast_to_variant() {
    let v = variants(&[Some(r#"{"a":1}"#)]);
    let p = text_col(&[Some("$.b")]);
    let int: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![7]));
    let text: ArrayRef = text_col(&[Some("x")]);
    let out = run("variant_insert", &[&v, &p, &int], None).unwrap();
    assert_eq!(show(&out), vec![Some(r#"{"a":1,"b":7}"#.to_owned())]);
    let out = run("variant_insert", &[&v, &p, &text], None).unwrap();
    assert_eq!(show(&out), vec![Some(r#"{"a":1,"b":"x"}"#.to_owned())]);
}

/// Spark checks the paths of `variant_delete` even for a NULL variant; `variant_insert` returns NULL first.
#[test]
fn delete_validates_its_paths_for_a_null_variant() {
    let v = variants(&[None]);
    let bad = text_col(&[Some("$[-1]")]);
    assert_eq!(
        run("variant_delete", &[&v, &bad], None).map(|o| show(&o).remove(0)),
        Err(ErrorClass::InvalidVariantPath)
    );
    let good = text_col(&[Some("$.a")]);
    assert_eq!(
        run("variant_delete", &[&v, &good], None).map(|o| show(&o).remove(0)),
        Ok(None)
    );
    let val = variants(&[Some("1")]);
    assert_eq!(
        run("variant_insert", &[&v, &bad, &val], None).map(|o| show(&o).remove(0)),
        Ok(None)
    );
}

/// A path that creates more levels than the variant parser's depth bound, or an index past the size limit, is
/// refused before anything is built.
#[test]
fn a_created_value_past_the_size_limit_is_refused() {
    let deep = format!("${}", ".a".repeat(1002));
    let r = edit("variant_set", "{}", &deep, "1", None);
    assert_eq!(r, Err(ErrorClass::VariantSizeLimit), "deep");
    assert_eq!(
        edit("variant_set", "[1]", "$[100000000]", "1", None),
        Err(ErrorClass::VariantSizeLimit)
    );
}
