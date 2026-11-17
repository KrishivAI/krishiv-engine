// SPDX-License-Identifier: Apache-2.0
//! Spark Connect artifact names (`AddArtifacts`, design P3-T011a §3.9) on
//! arbitrary text. Validation must return a name or an error, never panic;
//! an accepted name must be a relative path that stays below the directory
//! it is joined to: every segment non-empty, not `.` or `..`, without a
//! separator, NUL or other control character, and the path one of the four
//! stored prefixes.
#![no_main]

use std::path::{Component, Path};

use krishiv_connect::artifact::{ArtifactKind, MAX_NAME_BYTES, parse_artifact_name};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(name) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(parsed) = parse_artifact_name(name) else {
        return;
    };
    let path = parsed.path();
    assert!(path.len() <= MAX_NAME_BYTES);
    assert!(name.starts_with(path), "{name:?} -> {path:?}");
    for segment in parsed.segments() {
        assert!(!segment.is_empty() && segment != "." && segment != "..");
        assert!(!segment.contains(['/', '\\', '\0']));
        assert!(!segment.chars().any(char::is_control));
    }
    // Joined to a base, the path only descends.
    let joined = Path::new("/base").join(path);
    assert!(joined.starts_with("/base"));
    assert!(
        Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_))),
        "{path:?}"
    );
    let prefix = match parsed.kind() {
        ArtifactKind::Cache => "cache/",
        ArtifactKind::PyFiles => "pyfiles/",
        ArtifactKind::Archives => "archives/",
        ArtifactKind::Files => "files/",
    };
    assert!(path.starts_with(prefix));
    if let Some(fragment) = parsed.fragment() {
        assert_eq!(parsed.kind(), ArtifactKind::Archives);
        assert!(!fragment.is_empty() && !fragment.contains(['/', '#', '\0']));
    }
});
