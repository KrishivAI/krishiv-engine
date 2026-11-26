// SPDX-License-Identifier: Apache-2.0
//! The KLL scalars over the three element types: scalar and list ranks and quantiles, NULL rows, merge, size, text.

use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array};

use super::{bind, bytes, col};
use crate::sketch::kll::{Kll, KllType};

fn sketch(t: KllType, n: i64) -> Vec<u8> {
    let mut s = Kll::new(t, 200, "kll_test").unwrap();
    for v in 1..=n {
        match t {
            KllType::Bigint => s.update_long(v),
            #[expect(clippy::cast_precision_loss, reason = "small test values")]
            KllType::Float => s.update_f32(v as f32),
            #[expect(clippy::cast_precision_loss, reason = "small test values")]
            KllType::Double => s.update_f64(v as f64),
        }
    }
    s.to_bytes()
}

fn f64_list(rows: &[Option<&[f64]>]) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new());
    for r in rows {
        match r {
            Some(v) => {
                for x in *v {
                    b.values().append_value(*x);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

fn run(name: &str, args: &[&ArrayRef]) -> ArrayRef {
    let rows = args[0].len();
    bind(name, args)
        .invoke(&args.iter().map(|a| col(a)).collect::<Vec<_>>(), rows)
        .unwrap()
}

#[test]
fn size_text_and_merge_keep_null_rows() {
    for (t, suffix) in [
        (KllType::Bigint, "bigint"),
        (KllType::Float, "float"),
        (KllType::Double, "double"),
    ] {
        let a = sketch(t, 10);
        let b = sketch(t, 5);
        let x = bytes(&[Some(&a), None]);
        let y = bytes(&[Some(&b), Some(&b)]);
        let n = run(&format!("kll_sketch_get_n_{suffix}"), &[&x]);
        let got: Vec<Option<i64>> = n.as_primitive::<Int64Type>().iter().collect();
        assert_eq!(got, vec![Some(10), None]);
        let s = run(&format!("kll_sketch_to_string_{suffix}"), &[&x]);
        assert!(s.as_string::<i32>().value(0).contains("Kll"), "{s:?}");
        assert!(s.is_null(1));
        let m = run(&format!("kll_sketch_merge_{suffix}"), &[&x, &y]);
        assert!(m.is_valid(0) && m.is_null(1));
        let merged = bytes(&[Some(m.as_binary::<i32>().value(0))]);
        let n = run(&format!("kll_sketch_get_n_{suffix}"), &[&merged]);
        assert_eq!(n.as_primitive::<Int64Type>().value(0), 15);
    }
}

#[test]
fn quantiles_and_ranks_scalar_and_list() {
    let s = sketch(KllType::Double, 100);
    let sk = bytes(&[Some(&s), None, Some(&s)]);
    let r: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.5), Some(0.5), None]));
    let q = run("kll_sketch_get_quantile_double", &[&sk, &r]);
    assert!(q.is_valid(0) && q.is_null(1) && q.is_null(2));
    let ranks = f64_list(&[Some(&[0.0, 1.0][..]), None, Some(&[0.5][..])]);
    let ql = run("kll_sketch_get_quantile_double", &[&sk, &ranks]);
    let l = ql.as_list::<i32>();
    assert_eq!(l.value(0).as_primitive::<Float64Type>().len(), 2);
    assert!(l.is_null(1));
    assert_eq!(l.value(2).len(), 1);
    let v: ArrayRef = Arc::new(Float64Array::from(vec![Some(50.0), Some(1.0), None]));
    let rk = run("kll_sketch_get_rank_double", &[&sk, &v]);
    assert!(rk.is_valid(0) && rk.is_null(1) && rk.is_null(2));
    let vals = f64_list(&[Some(&[1.0, 100.0][..]), None, None]);
    let rl = run("kll_sketch_get_rank_double", &[&sk, &vals]);
    let l = rl.as_list::<i32>();
    assert_eq!(l.value(0).len(), 2);
    assert!(l.is_null(1) && l.is_null(2));
}

#[test]
fn bigint_and_float_quantiles_and_ranks() {
    let s = sketch(KllType::Bigint, 20);
    let sk = bytes(&[Some(&s)]);
    let r: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.5)]));
    let q = run("kll_sketch_get_quantile_bigint", &[&sk, &r]);
    assert!(q.as_primitive::<Int64Type>().value(0) > 0);
    let ranks = f64_list(&[Some(&[0.1, 0.9][..])]);
    let ql = run("kll_sketch_get_quantile_bigint", &[&sk, &ranks]);
    assert_eq!(ql.as_list::<i32>().value(0).len(), 2);
    let v: ArrayRef = Arc::new(Int64Array::from(vec![Some(10)]));
    let rk = run("kll_sketch_get_rank_bigint", &[&sk, &v]);
    assert!(rk.as_primitive::<Float64Type>().value(0) > 0.0);

    let s = sketch(KllType::Float, 20);
    let sk = bytes(&[Some(&s)]);
    let q = run("kll_sketch_get_quantile_float", &[&sk, &r]);
    assert!(q.as_primitive::<Float32Type>().value(0) > 0.0);
    let ql = run("kll_sketch_get_quantile_float", &[&sk, &ranks]);
    assert_eq!(ql.as_list::<i32>().value(0).len(), 2);
    let v: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![Some(10.0_f32)]));
    let rk = run("kll_sketch_get_rank_float", &[&sk, &v]);
    assert!(rk.as_primitive::<Float64Type>().value(0) > 0.0);
}
