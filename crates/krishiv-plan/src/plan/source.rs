// SPDX-License-Identifier: Apache-2.0
//! What a relation reads: [`ReadSource`] (an unresolved read), [`Scan`] (a resolved one) and
//! [`TableVersion`]. Split out of `plan.rs` to keep it under 2,000 lines.

use std::collections::BTreeMap;

use krishiv_common::secret::OptionValue;

use crate::BoundColumn;

/// What an unresolved read reads.
///
/// Option values are typed ([`OptionValue`], ADR 0038): front ends build
/// them with [`OptionValue::from_user`], so secrets never print, not even
/// through `Debug`, and never enter the canonical encoding in plaintext.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReadSource {
    /// A table, view or CTE by name (`spark.read.table`, `FROM t`).
    Table {
        /// Name parts, e.g. `["cat", "db", "t"]`.
        name: Vec<String>,
        /// Data source options.
        options: BTreeMap<String, OptionValue>,
    },
    /// Files of a format (`spark.read.parquet(path)`).
    #[non_exhaustive]
    Files {
        /// Format name, e.g. `parquet`, as written (compared
        /// case-insensitively; a resolved [`Scan`] holds it in lower case).
        format: String,
        /// Paths or globs, typed like options (built with
        /// [`OptionValue::from_user`] under the key `path`, as Spark keeps
        /// them as the `path` option): a signed URL's credentials are secret
        /// parts.
        paths: Vec<OptionValue>,
        /// Data source options.
        options: BTreeMap<String, OptionValue>,
        /// The schema the user declared (`spark.read.schema(..)`); `None`
        /// infers it from the files. Build with [`ReadSource::files`] and
        /// [`ReadSource::with_schema`].
        schema: Option<krishiv_types::StructType>,
    },
    /// Arrow data held in the session's relation store (Spark's
    /// `LocalRelation`). Built only from a lease; the analyzer resolves it
    /// to a [`Scan`] with the Arrow schema's columns.
    Local(crate::LocalRelation),
}

impl ReadSource {
    /// A read of files of `format` at `paths` whose schema is inferred.
    #[must_use]
    pub fn files(
        format: impl Into<String>,
        paths: Vec<OptionValue>,
        options: BTreeMap<String, OptionValue>,
    ) -> Self {
        Self::Files {
            format: format.into(),
            paths,
            options,
            schema: None,
        }
    }

    /// This read with the user-declared `schema`; a table read is returned
    /// unchanged (Spark's `NamedTable` has no schema).
    #[must_use]
    pub fn with_schema(self, schema: krishiv_types::StructType) -> Self {
        match self {
            Self::Files {
                format,
                paths,
                options,
                ..
            } => Self::Files {
                format,
                paths,
                options,
                schema: Some(schema),
            },
            other @ (Self::Table { .. } | Self::Local(_)) => other,
        }
    }
}

/// A resolved read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scan {
    /// What was read.
    pub source: ReadSource,
    /// Output attributes.
    pub output: Vec<BoundColumn>,
    /// The table version pinned for the whole query (ADR 0037); `None` for
    /// sources without versions.
    pub version: Option<TableVersion>,
}

/// Field-metadata key the analyzer puts on every field of the generated
/// `_metadata` struct, so a table column of that name and shape is never
/// taken for it.
pub(crate) const GENERATED_METADATA_KEY: &str = "krishiv.generated_file_metadata";

/// `t` without the analyzer's provenance marker on the fields of a generated
/// `_metadata` struct: what a stored column keeps when a query's output is
/// materialized (`CREATE TABLE .. AS SELECT _metadata`), so the stored
/// values are never mistaken for the generated column.
#[must_use]
pub fn without_generated_marker(t: &krishiv_types::DataType) -> krishiv_types::DataType {
    use krishiv_types::{DataType, StructField, StructType};
    let DataType::Struct(s) = t else {
        return t.clone();
    };
    DataType::Struct(StructType::new(s.fields().iter().map(|f| {
        let mut map = f.metadata().as_map().clone();
        map.remove(GENERATED_METADATA_KEY);
        StructField::new(f.name(), f.data_type().clone(), f.nullable())
            .with_metadata(krishiv_types::Metadata::from_map(map))
    })))
}

impl Scan {
    /// The hidden `_metadata` column of a file source this scan outputs
    /// (always last), if it does: a struct of the source's own fields.
    #[must_use]
    pub fn metadata_column(&self) -> Option<&BoundColumn> {
        let c = self.output.last()?;
        let krishiv_types::DataType::Struct(s) = &c.data_type else {
            return None;
        };
        // Every field is one of the source's own (a pruned scan keeps only
        // the fields read), so a user column of this name that holds anything
        // else is a table column.
        const FIELDS: [&str; 7] = [
            "file_path",
            "file_name",
            "file_size",
            "file_block_start",
            "file_block_length",
            "file_modification_time",
            "row_index",
        ];
        (c.name == "_metadata"
            && !s.fields().is_empty()
            && s.fields().iter().all(|f| {
                FIELDS.iter().any(|n| *n == f.name())
                    && f.metadata().as_map().contains_key(GENERATED_METADATA_KEY)
            }))
        .then_some(c)
    }
}

/// A table version: an Iceberg snapshot id or a Delta table version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableVersion(pub i64);
