// SPDX-License-Identifier: Apache-2.0
//! Every feature a case's header lists is one a reader must have: a reader with every other
//! feature refuses the plan naming exactly that one. This covers each new case with no list to
//! extend; a feature's own file keeps only the refusals that need more (several features, a
//! subquery, a construct outside the inventory).

use super::*;

#[test]
fn a_reader_missing_one_listed_feature_refuses_the_plan_naming_it() {
    let mut checked = 0;
    for (name, plan) in cases() {
        let bytes = encode(&plan).unwrap();
        for feature in plan_features(&plan).iter() {
            let reader = ReaderFeatures::only(Feature::ALL.iter().copied().filter(|f| f != feature));
            assert_eq!(
                decode_with(&bytes, &Limits::default(), &reader).unwrap_err(),
                CodecError::UnsupportedFeatures {
                    missing: vec![feature.to_string()]
                },
                "{name} without {feature}"
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}
