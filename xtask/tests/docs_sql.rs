// SPDX-License-Identifier: Apache-2.0
//! The checked-in SQL function reference is what `cargo xtask docs-sql` generates.

use std::path::Path;

use xtask::docs_sql::{render, run, slug};

#[test]
fn the_checked_in_pages_match_the_generator() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    run(root, true).unwrap();
}

#[test]
fn slugs_are_file_safe_and_pages_unique() {
    assert_eq!(slug("array_contains"), "array_contains");
    assert_eq!(slug("<=>"), "op-lt-eq-gt");
    let pages = render().unwrap();
    assert!(pages.len() > 500);
    for path in pages.keys() {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        assert!(
            stem.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-'),
            "{stem}"
        );
    }
    let page = &pages[Path::new("docs/reference/sql/array_contains.md")];
    assert!(page.contains("SELECT array_contains(array(1, 2, 3), 2);"));
}
