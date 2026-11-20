// SPDX-License-Identifier: Apache-2.0
//! The VARIANT edit functions of Spark 4.3: `variant_insert`, `variant_set`, `variant_delete`,
//! `variant_array_append` and `variant_strip_nulls` (and the `try_` forms, which turn the two edit errors into NULL).
//!
//! An edit walks a parsed path (`$.a[1]['b']`, see [`parse_path`](super::parse_path)) through the value and writes
//! the edited copy with a [`VariantWriter`]. A value the edit does not change is returned as the original bytes.
//!
//! - `variant_insert`: an absent object key is added (a present one is `VARIANT_DUPLICATE_KEY`); at an array index
//!   `i <= len` the value goes in and later elements shift right; past the end the gap is filled with variant nulls.
//!   Missing intermediate keys and indexes are created (a key step makes an object, an index step an array).
//! - `variant_set`: as insert, but an existing field or element is replaced; `create_if_missing = false` leaves the
//!   value unchanged where the target is absent.
//! - `variant_delete`: removes an existing field or element; an absent or unreachable target changes nothing.
//! - `variant_array_append`: the target must be an array, which gains the value at its end; an absent key or index
//!   changes nothing.
//! - A path step against a value of the other kind (a key on an array, any step on a scalar, an append to a
//!   non-array) is `VARIANT_PATH_TYPE_MISMATCH` naming the path to that value, except for `variant_delete`.
//!
//! Ground truth: a Spark 4.3.0 run (`variant_insert` ... `variant_strip_nulls`).

use krishiv_common::error::{
    InvalidVariantPath, KrishivError, VariantDuplicateKey, VariantPathTypeMismatch,
    VariantSizeLimit,
};
use krishiv_types::variant::{VARIANT_SIZE_LIMIT, VariantKind, VariantRef, VariantWriter};

use super::{PathSegment, parse_path};

/// What an edit does at the end of the path.
#[derive(Debug, Clone, Copy)]
pub enum Edit<'a> {
    /// `variant_insert(v, path, value)`.
    Insert(VariantRef<'a>),
    /// `variant_set(v, path, value, create_if_missing)`.
    Set(VariantRef<'a>, bool),
    /// One path of `variant_delete`.
    Delete,
    /// `variant_array_append(v, path, value)`.
    Append(VariantRef<'a>),
}

/// Why an edit failed: both become NULL in the `try_` forms.
#[derive(Debug)]
pub enum EditFailure {
    /// `VARIANT_PATH_TYPE_MISMATCH` at the given path to the value.
    TypeMismatch(String),
    /// `VARIANT_DUPLICATE_KEY` of an insert.
    DuplicateKey(String),
}

impl EditFailure {
    /// The Spark error of the failure, in function `name`, for the call's `path` text.
    #[must_use]
    pub fn into_error(self, name: &str, path: &str) -> KrishivError {
        match self {
            Self::TypeMismatch(failed_at) => VariantPathTypeMismatch {
                path: path.to_owned(),
                failed_at,
                function_name: format!("`{name}`"),
            }
            .into(),
            Self::DuplicateKey(key) => VariantDuplicateKey { key }.into(),
        }
    }
}

/// The edit of a value: its new `(metadata, value)` bytes, or `None` when nothing changed.
pub type Edited = Option<(Vec<u8>, Vec<u8>)>;

/// `INVALID_VARIANT_PATH`: the path is no `function` target.
#[must_use]
pub fn invalid_path(path: &str, function: &str) -> KrishivError {
    InvalidVariantPath {
        path: path.to_owned(),
        function_name: format!("`{function}`"),
    }
    .into()
}

/// The steps of the edit path `path`: [`parse_path`], with `INVALID_VARIANT_PATH` for text that does not parse and
/// for the bare root `$` unless `allow_root` (only `variant_array_append` takes it).
///
/// # Errors
/// `INVALID_VARIANT_PATH`.
pub fn edit_path(
    path: &str,
    function: &str,
    allow_root: bool,
) -> Result<Vec<PathSegment>, KrishivError> {
    match parse_path(path) {
        Some(p) if allow_root || !p.is_empty() => Ok(p),
        _ => Err(invalid_path(path, function)),
    }
}

/// The text of the path to a value, as Spark prints it in the error (`$.a[0]`).
fn show(at: &[PathSegment]) -> String {
    let mut s = String::from("$");
    for step in at {
        match step {
            PathSegment::Key(k) => {
                s.push('.');
                s.push_str(k);
            }
            PathSegment::Index(i) => s.push_str(&format!("[{i}]")),
        }
    }
    s
}

/// Applies `edit` at `path` of the value `root`.
///
/// # Errors
/// [`EditFailure`] as an `Ok(Err(..))`; `MALFORMED_VARIANT` and the writer's limits as the outer error.
pub fn apply(
    root: VariantRef<'_>,
    path: &[PathSegment],
    edit: Edit<'_>,
) -> Result<Result<Edited, EditFailure>, KrishivError> {
    let mut w = VariantWriter::new();
    let mut at = Vec::new();
    match rec(&mut w, root, path, edit, &mut at)? {
        Ok(false) => Ok(Ok(None)),
        Ok(true) => Ok(Ok(Some(w.finish()?))),
        Err(f) => Ok(Err(f)),
    }
}

type Step = Result<Result<bool, EditFailure>, KrishivError>;

/// Writes `v` with `edit` applied at `path` below it; whether anything changed.
fn rec(
    w: &mut VariantWriter,
    v: VariantRef<'_>,
    path: &[PathSegment],
    edit: Edit<'_>,
    at: &mut Vec<PathSegment>,
) -> Step {
    let mismatch = |at: &[PathSegment]| Ok(Err(EditFailure::TypeMismatch(show(at))));
    let Some((step, rest)) = path.split_first() else {
        // The end of the path: only `variant_array_append` gets here.
        return match (edit, v.kind()) {
            (Edit::Append(val), VariantKind::Array) => {
                let start = w.position();
                let n = v.len()?;
                let mut offsets = Vec::with_capacity(n.saturating_add(1));
                for i in 0..n {
                    offsets.push(w.position().saturating_sub(start));
                    w.append_variant(&v.get_index(i)?)?;
                }
                offsets.push(w.position().saturating_sub(start));
                w.append_variant(&val)?;
                w.finish_array(start, &offsets)?;
                Ok(Ok(true))
            }
            _ => {
                w.append_variant(&v)?;
                mismatch(at)
            }
        };
    };
    let last = rest.is_empty();
    match (step, v.kind()) {
        (PathSegment::Key(key), VariantKind::Object) => {
            let start = w.position();
            let n = v.len()?;
            let mut fields = Vec::with_capacity(n.saturating_add(1));
            let mut changed = false;
            let mut found = false;
            for i in 0..n {
                let k = v.field_key_at(i)?;
                let child = v.field_at(i)?;
                if k != key {
                    fields.push(w.field(start, k)?);
                    w.append_variant(&child)?;
                    continue;
                }
                found = true;
                let append = matches!(edit, Edit::Append(_));
                if last && !append {
                    match edit {
                        Edit::Insert(_) => {
                            return Ok(Err(EditFailure::DuplicateKey(key.clone())));
                        }
                        Edit::Set(val, _) => {
                            fields.push(w.field(start, k)?);
                            w.append_variant(&val)?;
                            changed = true;
                        }
                        Edit::Delete => changed = true,
                        Edit::Append(_) => {}
                    }
                    continue;
                }
                fields.push(w.field(start, k)?);
                at.push(step.clone());
                let inner = rec(w, child, rest, edit, at)?;
                at.pop();
                match inner {
                    Ok(c) => changed |= c,
                    Err(f) => return Ok(Err(f)),
                }
            }
            if !found && let Some(val) = created(edit) {
                fields.push(w.field(start, key)?);
                write_new(w, rest, val)?;
                changed = true;
            }
            w.finish_object(start, fields)?;
            Ok(Ok(changed))
        }
        (PathSegment::Index(index), VariantKind::Array) => {
            let n = v.len()?;
            let start = w.position();
            let mut offsets = Vec::with_capacity(n.saturating_add(1));
            let mut changed = false;
            let append = matches!(edit, Edit::Append(_));
            for i in 0..n {
                let child = v.get_index(i)?;
                if i != *index {
                    offsets.push(w.position().saturating_sub(start));
                    w.append_variant(&child)?;
                    continue;
                }
                if last && !append {
                    match edit {
                        Edit::Insert(val) => {
                            offsets.push(w.position().saturating_sub(start));
                            w.append_variant(&val)?;
                            offsets.push(w.position().saturating_sub(start));
                            w.append_variant(&child)?;
                            changed = true;
                        }
                        Edit::Set(val, _) => {
                            offsets.push(w.position().saturating_sub(start));
                            w.append_variant(&val)?;
                            changed = true;
                        }
                        Edit::Delete => changed = true,
                        Edit::Append(_) => {}
                    }
                    continue;
                }
                offsets.push(w.position().saturating_sub(start));
                at.push(step.clone());
                let inner = rec(w, child, rest, edit, at)?;
                at.pop();
                match inner {
                    Ok(c) => changed |= c,
                    Err(f) => return Ok(Err(f)),
                }
            }
            if *index >= n
                && let Some(val) = created(edit)
            {
                for _ in n..*index {
                    offsets.push(w.position().saturating_sub(start));
                    w.null();
                }
                offsets.push(w.position().saturating_sub(start));
                write_new(w, rest, val)?;
                changed = true;
            }
            w.finish_array(start, &offsets)?;
            Ok(Ok(changed))
        }
        // A step of the other kind, or onto a scalar: `variant_delete` ignores it.
        _ => {
            w.append_variant(&v)?;
            if matches!(edit, Edit::Delete) {
                Ok(Ok(false))
            } else {
                mismatch(at)
            }
        }
    }
}

/// The value an insert or a creating set writes where the path leaves the existing value.
fn created<'a>(edit: Edit<'a>) -> Option<VariantRef<'a>> {
    match edit {
        Edit::Insert(v) | Edit::Set(v, true) => Some(v),
        Edit::Set(_, false) | Edit::Delete | Edit::Append(_) => None,
    }
}

/// The most levels one edit creates below an existing value (the variant JSON parser's depth bound).
const MAX_CREATED_DEPTH: usize = 1000;

fn size_limit() -> KrishivError {
    VariantSizeLimit {
        size_limit: "128.0 MiB".to_owned(),
        function_name: "`variant`".to_owned(),
    }
    .into()
}

/// Writes the value an edit creates where the path leaves the existing value: the nested objects and arrays of the
/// remaining steps around `val`.
fn write_new(
    w: &mut VariantWriter,
    rest: &[PathSegment],
    val: VariantRef<'_>,
) -> Result<(), KrishivError> {
    let Some((step, tail)) = rest.split_first() else {
        return w.append_variant(&val);
    };
    // The nested value must fit the variant size limit (each level and each filler null costs bytes), so a path
    // longer than the limit, or an index past it, is refused before any recursion or allocation.
    if rest.len() > MAX_CREATED_DEPTH {
        return Err(size_limit());
    }
    let start = w.position();
    match step {
        PathSegment::Key(k) => {
            let field = w.field(start, k)?;
            write_new(w, tail, val)?;
            w.finish_object(start, vec![field])
        }
        PathSegment::Index(i) => {
            if *i >= VARIANT_SIZE_LIMIT / 5 {
                return Err(size_limit());
            }
            let mut offsets = Vec::with_capacity(i.saturating_add(1));
            for _ in 0..*i {
                offsets.push(w.position().saturating_sub(start));
                w.null();
            }
            offsets.push(w.position().saturating_sub(start));
            write_new(w, tail, val)?;
            w.finish_array(start, &offsets)
        }
    }
}

/// `variant_strip_nulls(v, include_arrays)`: removes the object fields whose value is a variant null, and with
/// `include_arrays` the array elements too, at every depth (an object left empty stays as `{}`). A bare variant
/// null is returned as it is.
///
/// # Errors
/// `MALFORMED_VARIANT` and the writer's limits.
pub fn strip_nulls(
    v: VariantRef<'_>,
    include_arrays: bool,
) -> Result<(Vec<u8>, Vec<u8>), KrishivError> {
    let mut w = VariantWriter::new();
    strip(&mut w, v, include_arrays)?;
    w.finish()
}

fn strip(
    w: &mut VariantWriter,
    v: VariantRef<'_>,
    include_arrays: bool,
) -> Result<(), KrishivError> {
    match v.kind() {
        VariantKind::Object => {
            let start = w.position();
            let mut fields = Vec::new();
            for i in 0..v.len()? {
                let child = v.field_at(i)?;
                if child.kind() == VariantKind::Null {
                    continue;
                }
                fields.push(w.field(start, v.field_key_at(i)?)?);
                strip(w, child, include_arrays)?;
            }
            w.finish_object(start, fields)
        }
        VariantKind::Array => {
            let start = w.position();
            let mut offsets = Vec::new();
            for i in 0..v.len()? {
                let child = v.get_index(i)?;
                if include_arrays && child.kind() == VariantKind::Null {
                    continue;
                }
                offsets.push(w.position().saturating_sub(start));
                strip(w, child, include_arrays)?;
            }
            w.finish_array(start, &offsets)
        }
        _ => w.append_variant(&v),
    }
}
