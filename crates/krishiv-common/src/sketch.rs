// SPDX-License-Identifier: Apache-2.0
//! A sketch reference carried in column statistics (P2-T182): the version
//! of the sketch format and its bytes. Formats fill `None` for now; a later
//! statistics task interprets the bytes.

/// A serialized distinct-count sketch (e.g. HLL) of one column.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SketchRef {
    /// The sketch format version.
    pub version: u32,
    /// The sketch bytes.
    pub bytes: Vec<u8>,
}

impl SketchRef {
    /// A sketch reference of `version` with `bytes`.
    #[must_use]
    pub fn new(version: u32, bytes: Vec<u8>) -> Self {
        Self { version, bytes }
    }
}
