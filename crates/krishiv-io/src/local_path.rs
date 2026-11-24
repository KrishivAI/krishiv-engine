// SPDX-License-Identifier: Apache-2.0
//! Object paths of local files, with each file name kept as it is.

use object_store::path::Path;

use crate::error::{Result, unsupported};

/// The object path of the local file whose names, from the root, are `parts`.
///
/// The names are kept as they are (`Path::parse`), never percent-encoded as
/// `Path::from_iter` and `Path::from` encode `~`, `%`, `#`, `[` or non-ASCII
/// text: the local stores map a path's text to the file name unchanged and list
/// a file under its own name, so an encoded path names another file (an 8.3
/// Windows temp directory `RUNNER~1` read as `RUNNER%7E1`).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a name no object path holds: a control
/// character, an empty name, `.` or `..`.
pub fn local_object_path<S: AsRef<str>>(parts: &[S]) -> Result<Path> {
    let joined = parts
        .iter()
        .map(AsRef::as_ref)
        .collect::<Vec<_>>()
        .join("/");
    Path::parse(&joined).map_err(|e| unsupported(format!("local paths with this file name ({e})")))
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    #[test]
    fn names_are_kept_as_written() {
        for name in [
            "RUNNER~1",
            "a%7Eb",
            "p=a%3Ab",
            "jos\u{e9}",
            "a#b",
            "[x]{y}^|`\"<>*?",
            "a b",
        ] {
            let path = local_object_path(&["tmp", name, "f"]).unwrap();
            assert_eq!(path.as_ref(), format!("tmp/{name}/f"));
            assert_eq!(path.parts().nth(1).unwrap().as_ref(), name);
        }
        assert_eq!(local_object_path::<&str>(&[]).unwrap().as_ref(), "");
    }

    #[test]
    fn names_no_object_path_holds_are_refused() {
        for name in ["a\u{1}b", "", ".", ".."] {
            let e = local_object_path(&["tmp", name, "f"]).unwrap_err();
            assert_eq!(
                e.error_class(),
                ErrorClass::KrishivUnsupportedFeature,
                "{name:?}"
            );
        }
    }
}
