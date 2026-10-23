// SPDX-License-Identifier: Apache-2.0
//! Golden `krishiv.plan.v1` bytes (compatibility fixture, ADR 0040).
//!
//! The plans are the hand-built cases of `common/inventory.rs`, so they
//! change only when the format does. The directories under `tests/golden/`:
//!
//! - `base/` holds the cases that use no feature, and `<feature>/` the cases
//!   that use that feature (a case that uses several sits in one of their
//!   directories). These files are byte for byte what the encoder writes
//!   today, forever: a plan that does not use a feature encodes the same
//!   before and after the feature exists, so a change to them is either an
//!   undeclared feature or an epoch change.
//! - `v1` to `v4` are the frozen files of the sequential format versions.
//!   They are never written again and must keep decoding to their plans.
//!
//! A new feature adds its directory and edits no existing file: run
//! `KRISHIV_BLESS_GOLDEN=1 cargo test -p krishiv-plan-codec --test golden`,
//! which writes a file for every case that has none.
#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

#[path = "common/inventory.rs"]
mod inventory;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use krishiv_exec::physical::PhysicalPlan;
use krishiv_plan_codec::features::RETIRED;
use krishiv_plan_codec::proto as pb;
use krishiv_plan_codec::{
    CodecError, Feature, Limits, ReaderFeatures, decode, decode_with, encode, features_of,
    plan_features,
};
use prost::Message as _;

/// The parsed message of golden bytes (flat tables: bounded nesting).
#[expect(
    clippy::disallowed_methods,
    reason = "test code over the codec's own goldens"
)]
fn message(bytes: &[u8]) -> pb::PhysicalPlan {
    pb::PhysicalPlan::decode(bytes).unwrap()
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// What a golden directory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    Base,
    Feature(Feature),
    Legacy(u32),
}

fn dir_kind(name: &str) -> Dir {
    if name == "base" {
        return Dir::Base;
    }
    if let Some(f) = Feature::from_id(name) {
        return Dir::Feature(f);
    }
    let n = name.strip_prefix('v').and_then(|n| n.parse().ok());
    Dir::Legacy(n.unwrap_or_else(|| {
        panic!("tests/golden/{name} is not `base`, a registered feature or a legacy `v<N>`")
    }))
}

/// Every golden file: its directory, case name, path and bytes.
fn goldens() -> Vec<(Dir, String, PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root()).unwrap() {
        let dir = entry.unwrap().path();
        let kind = dir_kind(dir.file_name().unwrap().to_str().unwrap());
        for file in std::fs::read_dir(&dir).unwrap() {
            let path = file.unwrap().path();
            let name = path.file_stem().unwrap().to_str().unwrap().to_owned();
            let bytes = std::fs::read(&path).unwrap();
            out.push((kind, name, path, bytes));
        }
    }
    out.sort_by(|a, b| a.2.cmp(&b.2));
    out
}

fn case(name: &str, path: &Path) -> PhysicalPlan {
    inventory::cases()
        .into_iter()
        .find(|(c, _)| c == name)
        .unwrap_or_else(|| panic!("{}: no such case in the inventory", path.display()))
        .1
}

/// Writes a file for every inventory case that has none outside the legacy
/// directories: in `base/`, or in the directory of its last feature.
fn bless() {
    let have: BTreeSet<String> = goldens()
        .into_iter()
        .filter(|(d, ..)| !matches!(d, Dir::Legacy(_)))
        .map(|(_, name, ..)| name)
        .collect();
    for (name, plan) in inventory::cases() {
        if have.contains(&name) {
            continue;
        }
        let dir = plan_features(&plan)
            .last()
            .map_or("base", |f| f.id())
            .to_owned();
        std::fs::create_dir_all(root().join(&dir)).unwrap();
        let path = root().join(dir).join(format!("{name}.bin"));
        std::fs::write(path, encode(&plan).unwrap()).unwrap();
    }
}

#[test]
fn every_golden_decodes_to_its_plan_and_current_ones_are_exact() {
    if std::env::var_os("KRISHIV_BLESS_GOLDEN").is_some() {
        bless();
    }
    let mut current = BTreeSet::new();
    let mut legacy = BTreeSet::new();
    for (dir, name, path, bytes) in goldens() {
        let plan = case(&name, &path);
        let decoded = decode(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(
            decoded == plan,
            "{}: decodes to another plan",
            path.display()
        );
        let features = plan_features(&plan);
        match dir {
            Dir::Legacy(n) => {
                legacy.insert(n);
                continue;
            }
            Dir::Base => assert!(features.is_empty(), "{}: uses features", path.display()),
            Dir::Feature(f) => assert!(
                features.contains(&f),
                "{}: does not use {f}",
                path.display()
            ),
        }
        assert!(
            encode(&plan).unwrap() == bytes,
            "{}: the encoder no longer writes these bytes; a format change needs a feature",
            path.display()
        );
        let header = features_of(&bytes).unwrap().features;
        let expected: Vec<String> = features.iter().map(|f| f.id().to_owned()).collect();
        assert_eq!(header, expected, "{}", path.display());
        assert!(
            current.insert(name.clone()),
            "{name}: two current golden files"
        );
    }
    for (name, _) in inventory::cases() {
        assert!(current.contains(&name), "{name}: no golden file (bless it)");
    }
    assert_eq!(
        legacy,
        BTreeSet::from([1, 2, 3, 4]),
        "the legacy directories"
    );
}

#[test]
fn the_registry_is_sorted_and_every_feature_has_goldens() {
    let ids: Vec<&str> = Feature::ALL.iter().map(|f| f.id()).collect();
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "not sorted: {ids:?}");
    for f in Feature::ALL {
        let id = f.id();
        assert!(
            id.starts_with(|c: char| c.is_ascii_lowercase())
                && id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "{id} is not snake_case"
        );
        assert!(!RETIRED.contains(&id), "{id} is retired");
        assert_eq!(Feature::from_id(id), Some(*f));
        assert!(
            (3..=krishiv_plan_codec::POST_MIGRATION_READER_VERSION).contains(&f.legacy_level()),
            "{id}: legacy level {}",
            f.legacy_level()
        );
        assert!(!f.meaning().is_empty(), "{id}: no meaning");
        let dir = root().join(id);
        let files = std::fs::read_dir(&dir).map_or(0, Iterator::count);
        assert!(files > 0, "{id}: no golden files in {}", dir.display());
    }
    // `goldens` panics on a directory that is none of the three kinds.
    assert!(!goldens().is_empty());
}

/// The reader feature sets to simulate: every subset of the registry while
/// it is small, then the empty and full sets, each feature alone and each
/// feature left out.
fn reader_sets() -> Vec<BTreeSet<Feature>> {
    let all = Feature::ALL;
    if all.len() <= 8 {
        return (0..1_u32 << all.len())
            .map(|mask| {
                (0..all.len())
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| all[i])
                    .collect()
            })
            .collect();
    }
    let full: BTreeSet<Feature> = all.iter().copied().collect();
    let mut sets = vec![BTreeSet::new(), full.clone()];
    for f in all {
        sets.push(BTreeSet::from([*f]));
        let mut without = full.clone();
        without.remove(f);
        sets.push(without);
    }
    sets
}

#[test]
fn an_older_reader_reads_exactly_the_plans_whose_features_it_has() {
    for (_, name, path, bytes) in goldens() {
        let plan = case(&name, &path);
        let needs = plan_features(&plan);
        for set in reader_sets() {
            let reader = ReaderFeatures::only(set.iter().copied());
            let result = decode_with(&bytes, &Limits::default(), &reader);
            let missing: Vec<String> = needs.difference(&set).map(|f| f.id().to_owned()).collect();
            if missing.is_empty() {
                assert!(result.unwrap() == plan, "{}", path.display());
            } else {
                assert_eq!(
                    result.unwrap_err(),
                    CodecError::UnsupportedFeatures { missing },
                    "{} with {set:?}",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn a_header_that_hides_or_invents_a_feature_is_corrupt() {
    let mut checked = 0;
    for (dir, _, path, bytes) in goldens() {
        if matches!(dir, Dir::Legacy(_)) {
            continue;
        }
        let m = message(&bytes);
        let mut edited = Vec::new();
        // Each declared feature stripped.
        for i in 0..m.features.len() {
            let mut e = m.clone();
            e.features.remove(i);
            edited.push(e);
        }
        // Each unused feature added at its sorted place.
        for f in Feature::ALL {
            if !m.features.iter().any(|id| id == f.id()) {
                let mut e = m.clone();
                e.features.push(f.id().to_owned());
                e.features.sort();
                edited.push(e);
            }
        }
        for e in edited {
            let err = decode(&e.encode_to_vec()).unwrap_err();
            assert_eq!(
                err,
                CodecError::Corrupt(
                    "the feature header does not match the plan's constructs".to_owned()
                ),
                "{} with {:?}",
                path.display(),
                e.features
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}
