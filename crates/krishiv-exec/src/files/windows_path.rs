// SPDX-License-Identifier: Apache-2.0
//! Windows local paths in the forms [`super::LocalPaths`] reads (the Windows port, 0.3.0).
//!
//! On Windows a local path may be written `C:\dir\f`, `C:/dir/f`, `\\?\C:\dir\f` (what
//! `std::fs::canonicalize` returns), `file:///C:/dir/f` or, as `format!("file://{}", dir)`
//! builds it, `file://C:\dir\f`. [`normalize`] turns each into the Unix-like form the locator
//! already handles: `/` separators and the drive as the first segment (`/C:/dir/f`,
//! `file:///C:/dir/f`), so the object path is `C:/dir/f` (which `object_store`'s
//! `LocalFileSystem` rooted at `file:///` opens as `C:\dir\f`) and the display is
//! `file:/C:/dir/f`, as Hadoop qualifies a Windows path. The function is pure and compiled on
//! every platform so its tests run everywhere; only Windows builds call it, since `\` is an
//! ordinary file-name character on Unix.

/// `path` with `\` as `/`, the verbatim prefix `\\?\` dropped and a leading drive (`X:`) made
/// the first segment of an absolute path. Other text (cloud URLs, UNC shares, relative paths)
/// keeps its form apart from the separators.
#[must_use]
pub(crate) fn normalize(path: &str) -> String {
    let slashed = path.replace('\\', "/");
    let text = slashed.strip_prefix("//?/").unwrap_or(&slashed);
    if is_drive(text) {
        return format!("/{text}");
    }
    if let Some(rest) = text.strip_prefix("file://")
        && is_drive(rest)
    {
        return format!("file:///{rest}");
    }
    if let Some(rest) = text.strip_prefix("file:")
        && is_drive(rest)
    {
        return format!("file:/{rest}");
    }
    text.to_owned()
}

/// Whether `text` starts with a drive (`X:`) followed by `/`, `\` or nothing.
pub(crate) fn is_drive(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() >= 2
        && b[0].is_ascii_alphabetic()
        && b[1] == b':'
        && (b.len() == 2 || matches!(b[2], b'/' | b'\\'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drive_paths_become_absolute() {
        assert_eq!(normalize(r"C:\Users\a\t"), "/C:/Users/a/t");
        assert_eq!(normalize("C:/Users/a/t"), "/C:/Users/a/t");
        assert_eq!(normalize("d:"), "/d:");
        assert_eq!(normalize(r"\\?\D:\a\b.parquet"), "/D:/a/b.parquet");
    }

    #[test]
    fn file_urls_with_a_drive_keep_it_in_the_path() {
        assert_eq!(
            normalize(r"file://C:\Temp\.tmp1\t"),
            "file:///C:/Temp/.tmp1/t"
        );
        assert_eq!(normalize("file:///C:/Temp/t"), "file:///C:/Temp/t");
        assert_eq!(normalize("file:/C:/Temp/t"), "file:/C:/Temp/t");
        assert_eq!(normalize("file:C:/Temp/t"), "file:/C:/Temp/t");
    }

    #[test]
    fn other_text_keeps_its_form() {
        assert_eq!(normalize("s3://b/k"), "s3://b/k");
        assert_eq!(normalize("file://host/a"), "file://host/a");
        assert_eq!(normalize(r"rel\dir"), "rel/dir");
        assert_eq!(normalize("/abs/x"), "/abs/x");
        // A scheme longer than one letter is not a drive.
        assert_eq!(normalize("hf://d/x"), "hf://d/x");
        assert!(!is_drive("C:x"));
        assert!(!is_drive("1:/x"));
        assert!(!is_drive("C"));
        assert!(is_drive(r"C:\x"));
    }
}
