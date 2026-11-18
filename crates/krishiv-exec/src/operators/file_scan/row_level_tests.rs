// SPDX-License-Identifier: Apache-2.0
//! Tests of the row-level (`_krishiv_row_file`) scan of a `FileScan`.

use arrow_array::BooleanArray;

use super::*;
use crate::operators::expr::FnPredicate;

fn filter(rows: RowLevel) -> PushedFilter {
    let predicate: Arc<dyn super::super::expr::Predicate> =
        Arc::new(FnPredicate(|b: &RecordBatch| {
            Ok(BooleanArray::from(vec![true; b.num_rows()]))
        }));
    PushedFilter {
        rows,
        ..PushedFilter::typed(predicate, PushdownExpr::Constant(true))
    }
}

#[test]
fn a_filter_left_out_that_may_raise_keeps_runtime_filters_off() {
    let safe = filter(RowLevel::Full);
    let raising = filter(RowLevel::Prefix(None));
    // Nothing left out: as `row_level`.
    assert!(matches!(
        scan_row_level(std::slice::from_ref(&safe), 1),
        RowLevel::Full
    ));
    // A safe filter left out changes nothing.
    assert!(matches!(
        scan_row_level(&[safe.clone(), safe.clone()], 1),
        RowLevel::Full
    ));
    // A raising filter left out: only the usable prefix may drop rows.
    let level = scan_row_level(&[safe.clone(), raising.clone()], 1);
    assert!(
        matches!(&level, RowLevel::Prefix(Some(PushdownExpr::Constant(true)))),
        "{level:?}"
    );
    assert!(matches!(
        scan_row_level(std::slice::from_ref(&raising), 0),
        RowLevel::Prefix(None)
    ));
    // A later raising filter counts too.
    assert!(matches!(
        scan_row_level(&[safe.clone(), safe, raising], 1),
        RowLevel::Prefix(Some(_))
    ));
}
