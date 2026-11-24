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

/// `path` with `\` as `/`, the verbatim prefix `\\?\` dropped before a drive and a leading drive
/// (`X:`) made the first segment of an absolute path. A verbatim UNC path `\\?\UNC\srv\share`
/// becomes the share `//srv/share`; any other verbatim or device text keeps its `//?/` form, so it
/// never turns into a relative path. Other text (cloud URLs, UNC shares, relative paths) keeps its
/// form apart from the separators.
#[must_use]
pub(crate) fn normalize(path: &str) -> String {
    let slashed = path.replace('\\', "/");
    if let Some(share) = slashed.strip_prefix("//?/UNC/") {
        return format!("//{share}");
    }
    let text = match slashed.strip_prefix("//?/") {
        Some(rest) if is_drive(rest) => rest,
        _ => &slashed,
    };
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

/// The drive-led text of a local path in the locator's form (`/C:/x` gives `C:/x`, as do `//C:/x`
/// and `///C:/x`); `None` when no drive follows the leading slashes.
pub(crate) fn drive_led(local: &str) -> Option<&str> {
    let text = local.trim_start_matches('/');
    is_drive(text).then_some(text)
}

/// Whether `text` starts with a drive (`X:`) followed by `/`, `\` or nothing.
pub(crate) fn is_drive(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() >= 2
        && b[0].is_ascii_alphabetic()
        && b[1] == b':'
        && (b.len() == 2 || matches!(b[2], b'/' | b'\\'))
}

/// The object-store segments of a root-relative local path on Windows (`/data/x`, which names
/// `\data\x` on the current drive, as `java.io.File` reads it): `parts` with the drive of `cwd`
/// (the working directory in the locator's form, `/C:/..`) first, when `parts` has no drive of its
/// own. `object_store`'s `LocalFileSystem` cannot open a `file:///data/x` URL on Windows. `None`
/// when nothing changes.
pub(crate) fn on_drive_of<'a>(parts: &[&'a str], cwd: &'a str) -> Option<Vec<&'a str>> {
    if parts.first().is_some_and(|p| is_drive(p)) {
        return None;
    }
    let drive = drive_led(cwd)?.get(..2)?;
    Some(
        std::iter::once(drive)
            .chain(parts.iter().copied())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_drive_led_path_loses_its_leading_slashes() {
        assert_eq!(drive_led("/C:/Users/a"), Some("C:/Users/a"));
        assert_eq!(drive_led("///d:/a"), Some("d:/a"));
        assert_eq!(drive_led("/C:"), Some("C:"));
        assert_eq!(drive_led("/tmp/a"), None);
        assert_eq!(drive_led("/ab:/x"), None);
    }

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
        // A verbatim UNC path is the share, never the relative `UNC/srv/share/x`.
        assert_eq!(normalize(r"\\?\UNC\srv\share\x"), "//srv/share/x");
        assert_eq!(normalize(r"\\srv\share\x"), "//srv/share/x");
        // Other verbatim and device forms keep their prefix.
        assert_eq!(normalize(r"\\?\Volume{0a}\x"), "//?/Volume{0a}/x");
        assert_eq!(normalize(r"\\.\COM1"), "//./COM1");
        // A scheme longer than one letter is not a drive.
        assert_eq!(normalize("hf://d/x"), "hf://d/x");
        assert!(!is_drive("C:x"));
        assert!(!is_drive("1:/x"));
        assert!(!is_drive("C"));
        assert!(is_drive(r"C:\x"));
    }

    #[test]
    fn a_root_relative_path_takes_the_working_directory_drive() {
        assert_eq!(
            on_drive_of(&["data", "x"], "/D:/a/b"),
            Some(vec!["D:", "data", "x"])
        );
        assert_eq!(on_drive_of(&[], "/c:"), Some(vec!["c:"]));
        assert_eq!(on_drive_of(&["C:", "x"], "/D:/a"), None);
        // A working directory without a drive (never on Windows) changes nothing.
        assert_eq!(on_drive_of(&["data"], "/home/a"), None);
    }
}
