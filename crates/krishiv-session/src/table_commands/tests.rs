// SPDX-License-Identifier: Apache-2.0
//! Unit tests of `table_commands.rs` (kept apart so the file stays under its size baseline).

use super::*;
use krishiv_types::{ArrayType, DecimalType};

#[test]
fn a_namespace_location_is_qualified_against_the_warehouse() {
    let wh = std::path::Path::new("/w/warehouse");
    assert_eq!(
        qualified_namespace_location("someloc", wh),
        "file:/w/warehouse/someloc"
    );
    assert_eq!(qualified_namespace_location("/abs/x", wh), "file:/abs/x");
    assert_eq!(
        qualified_namespace_location("file:/abs/x", wh),
        "file:/abs/x"
    );
    assert_eq!(
        qualified_namespace_location("s3a://bucket/x", wh),
        "s3a://bucket/x"
    );
    // A colon after a path character is not a scheme.
    assert_eq!(
        qualified_namespace_location("a/b:c", wh),
        "file:/w/warehouse/a/b:c"
    );
}

#[test]
fn an_owned_directory_is_exclusive_and_reports_why_it_cannot_be_made() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("a").join("t");
    let owned = OwnedDirectory::create(path.clone()).unwrap();
    assert!(matches!(
        OwnedDirectory::create(path.clone()),
        Err(OwnedDirectoryError::Exists)
    ));
    drop(owned);
    assert!(!path.exists(), "a directory not kept is removed again");
    // A file where the parent should be.
    let file = tmp.path().join("f");
    std::fs::write(&file, b"x").unwrap();
    assert!(matches!(
        OwnedDirectory::create(file.join("t")),
        Err(OwnedDirectoryError::Io(m)) if m.contains("cannot create")
    ));
    // A parent that exists but takes no new entries.
    #[cfg(target_os = "linux")]
    assert!(matches!(
        OwnedDirectory::create(PathBuf::from("/proc/krishiv-owned-directory")),
        Err(OwnedDirectoryError::Io(_))
    ));
}

#[test]
fn ansi_store_assignment() {
    use DataType as T;
    assert!(can_store(&T::Integer, &T::Integer));
    assert!(can_store(&T::Null, &T::Date));
    assert!(can_store(&T::Integer, &T::Byte));
    assert!(can_store(
        &T::Long,
        &T::Decimal(DecimalType::try_new(5, 2).unwrap())
    ));
    assert!(can_store(&T::Integer, &T::String));
    assert!(can_store(&T::Date, &T::String));
    assert!(can_store(&T::Date, &T::Timestamp));
    assert!(!can_store(&T::String, &T::Binary));
    assert!(!can_store(&T::String, &T::Integer));
    assert!(!can_store(&T::String, &T::Date));
    assert!(!can_store(&T::Boolean, &T::Integer));
    assert!(can_store(&T::Timestamp, &T::Date));
    let a = |t: T| T::Array(Arc::new(ArrayType::new(t, true)));
    assert!(can_store(&a(T::Integer), &a(T::Long)));
    assert!(!can_store(&a(T::String), &a(T::Long)));
}

#[test]
fn literal_text_strips_one_pair_of_quotes() {
    assert_eq!(literal_text("'a b'"), "a b");
    assert_eq!(literal_text("\"x\""), "x");
    assert_eq!(literal_text(" true "), "true");
    assert_eq!(literal_text("'"), "'");
}
