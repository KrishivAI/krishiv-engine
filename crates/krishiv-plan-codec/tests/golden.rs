// SPDX-License-Identifier: Apache-2.0
//! Golden `krishiv.plan.v1` bytes (compatibility fixture, ADR 0026).
//!
//! `tests/golden/v<N>/<case>.bin` holds the bytes the codec wrote for each
//! plan of `common/inventory.rs` while format version N was current. The
//! plans are built by hand, not by the planner, so they change only when
//! the format does. Two rules keep plans decodable across releases:
//!
//! - every golden file of every version still decodes, with the current
//!   decoder, to the plan its case builds (N reads N-1);
//! - the current version's files are byte-for-byte what the encoder writes
//!   today, so a format change cannot slip in without a new version.
//!
//! When the format changes (a new field, a new node kind), keep the old
//! `v<N>` directory untouched, raise `FORMAT_VERSION`, and add `v<N+1>`
//! with `KRISHIV_BLESS_GOLDEN=1 cargo test -p krishiv-plan-codec --test
//! golden`. A case added to the inventory later gets files in the current
//! version only.
#![expect(clippy::unwrap_used, reason = "test code")]

#[path = "common/inventory.rs"]
mod inventory;

use std::path::PathBuf;

use krishiv_plan_codec::{FORMAT_VERSION, decode, encode};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn version_dir(n: u32) -> PathBuf {
    root().join(format!("v{n}"))
}

#[test]
fn golden_files_of_every_version_decode_to_their_plans() {
    if std::env::var_os("KRISHIV_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(version_dir(FORMAT_VERSION)).unwrap();
        for (name, plan) in inventory::cases() {
            let path = version_dir(FORMAT_VERSION).join(format!("{name}.bin"));
            std::fs::write(path, encode(&plan).unwrap()).unwrap();
        }
    }
    let cases = inventory::cases();
    let mut versions = Vec::new();
    for entry in std::fs::read_dir(root()).unwrap() {
        let dir = entry.unwrap().path();
        let n: u32 = dir
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix('v'))
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("{} is not a v<N> directory", dir.display()));
        versions.push(n);
        let mut files = 0;
        for file in std::fs::read_dir(&dir).unwrap() {
            let path = file.unwrap().path();
            let name = path.file_stem().unwrap().to_str().unwrap().to_owned();
            let bytes = std::fs::read(&path).unwrap();
            let (_, plan) = cases
                .iter()
                .find(|(c, _)| *c == name)
                .unwrap_or_else(|| panic!("{}: no such case in the inventory", path.display()));
            let decoded = decode(&bytes).unwrap_or_else(|e| panic!("v{n} {name}: {e}"));
            assert!(decoded == *plan, "v{n} {name}: decodes to another plan");
            if n == FORMAT_VERSION {
                assert!(
                    encode(plan).unwrap() == bytes,
                    "{name}: the encoder no longer writes the v{n} bytes; a format change needs a new version"
                );
            }
            files += 1;
        }
        if n == FORMAT_VERSION {
            assert_eq!(files, cases.len(), "a case without a golden file in v{n}");
        }
    }
    assert!(
        versions.contains(&FORMAT_VERSION),
        "no golden files for the current format version"
    );
}
