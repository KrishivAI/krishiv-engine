// SPDX-License-Identifier: Apache-2.0
//! `fixed_size_list_carrier` (design P3-T075): the vector carrier's codec tests.

use super::*;

/// A vector carrier (design P3-T075) needs its own feature, and a reader without it refuses the plan naming it.
#[test]
fn a_vector_carrier_needs_its_feature() {
    use std::sync::Arc;
    let vector = arrow_schema::DataType::FixedSizeList(
        Arc::new(arrow_schema::Field::new(
            "item",
            arrow_schema::DataType::Float32,
            true,
        )),
        768,
    );
    assert_eq!(
        feature_of(Construct::ArrowType(&vector)),
        Some(Feature::FixedSizeListCarrier)
    );
    assert_eq!(
        message("vector_carrier_aggregate").features,
        ["fixed_size_list_carrier"]
    );
    let (_, plan) = cases()
        .into_iter()
        .find(|(n, _)| n == "vector_carrier_aggregate")
        .unwrap();
    let bytes = encode(&plan).unwrap();
    let reader = ReaderFeatures::only([Feature::NestedTypes]);
    let err = decode_with(&bytes, &Limits::default(), &reader).unwrap_err();
    assert_eq!(
        err,
        CodecError::UnsupportedFeatures {
            missing: vec!["fixed_size_list_carrier".to_owned()]
        }
    );
}
