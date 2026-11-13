// SPDX-License-Identifier: Apache-2.0
//! The allow-list on protocols: every section 3.5 row on an edited protocol (the rows the
//! oracle cannot create: `catalogManaged`, coordinated commits, `geospatial`, unknown), and
//! the class and parameters of delta-spark's own refusal of an unknown reader feature.

use std::path::{Path, PathBuf};

use krishiv_common::error::ErrorClass;

use super::*;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/delta-4.4.1")
}

fn reader(features: &[&str]) -> ReadProtocol {
    let names = || Some(features.iter().map(|f| (*f).to_owned()).collect());
    ReadProtocol {
        min_reader_version: 3,
        reader_features: names(),
        writer_features: names(),
        configuration: HashMap::new(),
    }
}

fn class(p: &ReadProtocol) -> Option<ErrorClass> {
    check_read("t", p).err().map(|e| e.error_class())
}

#[test]
fn delta_unsupported_features_refused() {
    // The "yes" rows read; their oracle tables are read back in `table::oracle_tests`.
    for f in READABLE {
        assert_eq!(class(&reader(&[f])), None, "{f}");
    }
    for f in [
        "variantShredding",
        "variantShredding-preview",
        "geospatial",
        "adaptiveMetadata-preview",
        "krishivProbeFeature",
    ] {
        assert_eq!(
            class(&reader(&["deletionVectors", f])),
            Some(ErrorClass::DeltaUnsupportedFeaturesForRead),
            "{f}"
        );
    }
    // Every unsupported feature is named, in the protocol's order.
    let e = check_read(
        "t",
        &reader(&["geospatial", "deletionVectors", "variantShredding-preview"]),
    )
    .unwrap_err();
    assert!(
        e.message()
            .contains(": geospatial, variantShredding-preview."),
        "{}",
        e.message()
    );
    // Legacy readers (version 1 and 2) carry no reader-feature list.
    let legacy = ReadProtocol {
        min_reader_version: 2,
        ..ReadProtocol::default()
    };
    assert_eq!(class(&legacy), None);
    // Writer-only features never block a read.
    let writer_only = ReadProtocol {
        writer_features: Some(vec!["changeDataFeed".to_owned(), "rowTracking".to_owned()]),
        ..reader(&["deletionVectors"])
    };
    assert_eq!(class(&writer_only), None);
}

#[test]
fn coordinated_commits_refused() {
    let feature = ReadProtocol {
        writer_features: Some(vec![COORDINATED_COMMITS.to_owned()]),
        ..ReadProtocol::default()
    };
    let config = ReadProtocol {
        configuration: HashMap::from([(
            "delta.coordinatedCommits.commitCoordinator-preview".to_owned(),
            "dynamodb".to_owned(),
        )]),
        ..ReadProtocol::default()
    };
    for p in [feature, config] {
        let e = check_read("db.t", &p).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation
        );
        assert!(
            e.message().contains("coordinated commits"),
            "{}",
            e.message()
        );
    }
}

#[test]
fn catalog_managed_read_refused() {
    for f in CATALOG_MANAGED {
        let e = check_read("db.t", &reader(&["deletionVectors", f])).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation,
            "{f}"
        );
        assert!(e.message().contains("catalog-managed"), "{}", e.message());
    }
    // The kernel's own refusal (no catalog version given) maps to the same class.
    let e = snapshot_error(&KernelError::MaxCatalogVersion("m".to_owned()), "db.t");
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation
    );
    assert!(e.message().contains("catalog-managed"), "{}", e.message());
}

#[test]
fn row_tracking_columns_refused() {
    let e = row_tracking_refused("db.t");
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation
    );
    assert!(e.message().contains("row-tracking"), "{}", e.message());
}

#[test]
fn the_kernels_feature_refusal_is_delta_unsupported_features_for_read() {
    let e = snapshot_error(
        &KernelError::unsupported("Feature 'geospatial' is not supported"),
        "t",
    );
    assert_eq!(e.error_class(), ErrorClass::DeltaUnsupportedFeaturesForRead);
    assert!(e.message().contains(": geospatial."), "{}", e.message());
    let other = snapshot_error(&KernelError::unsupported("copy_atomic"), "t");
    assert_eq!(
        other.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation
    );
}

/// The refusal has delta-spark's class, SQLSTATE and parameters (its probe of an unknown
/// reader feature), with Krishiv's `<deltaVersion>`.
#[test]
fn the_refusal_matches_delta_sparks() {
    let probe: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixtures().join("PROBE-unknown-reader-feature.json")).unwrap(),
    )
    .unwrap();
    let features: Vec<String> = probe["protocol"]["readerFeatures"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f.as_str().unwrap().to_owned())
        .collect();
    let p = ReadProtocol {
        min_reader_version: 3,
        reader_features: Some(features),
        ..ReadProtocol::default()
    };
    let e = check_read("file:<path>", &p).unwrap_err();
    assert_eq!(e.error_class().name(), probe["condition"].as_str().unwrap());
    assert_eq!(e.sql_state(), probe["sqlstate"].as_str());
    let mut params: Vec<(String, String)> = e
        .message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    params.sort();
    let mut want: Vec<(String, String)> = probe["parameters"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            let v = v.as_str().unwrap();
            // Krishiv names its own version where delta-spark names its.
            let v = if k == "deltaVersion" {
                delta_version()
            } else {
                v.to_owned()
            };
            (k.clone(), v)
        })
        .collect();
    want.sort();
    assert_eq!(params, want);
}
