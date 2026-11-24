// SPDX-License-Identifier: Apache-2.0
//! Native paths and `file://` URLs of local paths, in the forms [`super::LocalPaths`] reads and
//! displays (re-exported from [`crate::files`]).

use super::windows_path;

/// The native file-system path of a local path written as [`super::LocalPaths`] displays it (`file:/a/b`,
/// `file:///a/b` or `/a/b`): the `file:` prefix and the extra slashes after it go and, on Windows, a drive-led path loses its
/// leading slashes (`file:/C:/x` is `C:/x`; `/C:/x` would name a directory `C:` on the current
/// drive). Other text is returned as it is.
#[must_use]
pub fn native_path(text: &str) -> std::path::PathBuf {
    // `file:///a` and `file:/a` name the same path; the slashes before the first segment fold to one.
    let local = match text.strip_prefix("file:") {
        Some(rest) if rest.starts_with('/') => {
            &rest[rest.len() - rest.trim_start_matches('/').len() - 1..]
        }
        Some(rest) => rest,
        None => text,
    };
    if cfg!(windows)
        && let Some(led) = windows_path::drive_led(local)
    {
        return std::path::PathBuf::from(led);
    }
    std::path::PathBuf::from(local)
}

/// The `file://` URL of the absolute local path `path`: `file:///a/b` on Unix and, on Windows,
/// `file:///C:/a/b` for `C:\a\b` (separators as `/`, the drive as the first segment), the form
/// [`super::LocalPaths`] and URL parsers read.
#[must_use]
pub fn file_url(path: &std::path::Path) -> String {
    let text = path.to_string_lossy();
    let slashed = if cfg!(windows) {
        text.replace('\\', "/")
    } else {
        text.into_owned()
    };
    if slashed.starts_with('/') {
        format!("file://{slashed}")
    } else {
        format!("file:///{slashed}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn native_paths_and_file_urls_keep_unix_paths() {
        assert_eq!(native_path("file:/a/b"), std::path::PathBuf::from("/a/b"));
        assert_eq!(native_path("/a/b"), std::path::PathBuf::from("/a/b"));
        assert_eq!(native_path("file:///a").as_os_str(), "/a");
        assert_eq!(native_path("file:/").as_os_str(), "/");
        assert_eq!(file_url(std::path::Path::new("/a/b")), "file:///a/b");
    }

    #[cfg(windows)]
    #[test]
    fn native_paths_and_file_urls_carry_the_drive() {
        assert_eq!(
            native_path("file:/C:/a/b"),
            std::path::PathBuf::from("C:/a/b")
        );
        assert_eq!(
            native_path("file:///C:/a"),
            std::path::PathBuf::from("C:/a")
        );
        assert_eq!(native_path("/C:/a"), std::path::PathBuf::from("C:/a"));
        assert_eq!(file_url(std::path::Path::new(r"C:\a\b")), "file:///C:/a/b");
    }
}
