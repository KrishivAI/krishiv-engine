// SPDX-License-Identifier: Apache-2.0
//! How a write reaches a table (design P3-lakehouse §4.1): the four write
//! modes every Spark write resolves to, and the capabilities a table declares
//! for them, mirroring Spark's `TableCapability` write values. The session
//! checks the capability before planning, so a format never receives a mode it
//! did not declare.

use krishiv_common::error::UnsupportedFeatureTableOperation;
use krishiv_plan::Expr;

use crate::error::Result;

/// What a write does to the rows already in the table.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum WriteMode {
    /// Adds rows: `INSERT INTO`, `append`, `insertInto`, V2 `append`.
    Append,
    /// Replaces every row: `INSERT OVERWRITE` without a partition spec, the
    /// `overwrite` save mode, V2 `overwrite(lit(true))`.
    Truncate,
    /// Replaces the rows `filter` selects: a static partition spec, V2
    /// `overwrite(cond)`, `INSERT ... REPLACE WHERE`.
    OverwriteByFilter(Expr),
    /// Replaces the partitions the new rows fall in: `partitionOverwriteMode =
    /// dynamic`, V2 `overwritePartitions`.
    OverwriteDynamic,
}

impl WriteMode {
    /// Spark's name for the operation, as the unsupported-operation error
    /// prints it.
    #[must_use]
    pub fn operation(&self) -> &'static str {
        match self {
            Self::Append => "append",
            Self::Truncate => "truncate",
            Self::OverwriteByFilter(_) => "overwrite by filter",
            Self::OverwriteDynamic => "dynamic partition overwrite",
        }
    }
}

/// The write capabilities of a table; each is Spark's `TableCapability` of the
/// same name. A table declares none by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per Spark TableCapability value, as Spark lists them"
)]
pub struct TableCapabilities {
    /// `BATCH_WRITE`: the table accepts a batch write ([`WriteMode::Append`]).
    pub batch_write: bool,
    /// `TRUNCATE`: [`WriteMode::Truncate`].
    pub truncate: bool,
    /// `OVERWRITE_BY_FILTER`: [`WriteMode::OverwriteByFilter`].
    pub overwrite_by_filter: bool,
    /// `OVERWRITE_DYNAMIC`: [`WriteMode::OverwriteDynamic`].
    pub overwrite_dynamic: bool,
    /// `ACCEPT_ANY_SCHEMA`: the table takes rows of any schema.
    pub accept_any_schema: bool,
    /// `AUTOMATIC_SCHEMA_EVOLUTION`: the table evolves its schema to fit a
    /// write's.
    pub automatic_schema_evolution: bool,
}

impl TableCapabilities {
    /// A table that appends, truncates and overwrites by filter or
    /// dynamically, as a directory-backed table does.
    #[must_use]
    pub fn all_write_modes() -> Self {
        Self {
            batch_write: true,
            truncate: true,
            overwrite_by_filter: true,
            overwrite_dynamic: true,
            ..Self::default()
        }
    }

    /// Whether the table declared what `mode` needs.
    #[must_use]
    pub fn allows(&self, mode: &WriteMode) -> bool {
        match mode {
            WriteMode::Append => self.batch_write,
            WriteMode::Truncate => self.truncate,
            WriteMode::OverwriteByFilter(_) => self.overwrite_by_filter,
            WriteMode::OverwriteDynamic => self.overwrite_dynamic,
        }
    }

    /// Passes when the table declared what `mode` needs.
    ///
    /// # Errors
    /// `UNSUPPORTED_FEATURE.TABLE_OPERATION` (0A000) naming `table` and the
    /// operation, before anything is written.
    pub fn check(&self, mode: &WriteMode, table: &str) -> Result<()> {
        if self.allows(mode) {
            Ok(())
        } else {
            Err(UnsupportedFeatureTableOperation {
                table_name: table.to_owned(),
                operation: mode.operation().to_owned(),
            }
            .into())
        }
    }
}

#[cfg(test)]
mod tests {
    use krishiv_plan::Literal;

    use super::*;

    fn modes() -> Vec<WriteMode> {
        vec![
            WriteMode::Append,
            WriteMode::Truncate,
            WriteMode::OverwriteByFilter(Expr::literal(Literal::Boolean(true))),
            WriteMode::OverwriteDynamic,
        ]
    }

    #[test]
    fn a_table_declares_nothing_by_default() {
        let none = TableCapabilities::default();
        for mode in modes() {
            assert!(!none.allows(&mode), "{mode:?}");
            let err = none.check(&mode, "`t`").unwrap_err();
            assert!(err.to_string().contains("`t`"), "{err}");
        }
    }

    #[test]
    fn each_mode_needs_its_own_capability() {
        let only = |f: fn(&mut TableCapabilities)| {
            let mut c = TableCapabilities::default();
            f(&mut c);
            c
        };
        let cases = [
            (only(|c| c.batch_write = true), 0),
            (only(|c| c.truncate = true), 1),
            (only(|c| c.overwrite_by_filter = true), 2),
            (only(|c| c.overwrite_dynamic = true), 3),
        ];
        let modes = modes();
        for (caps, allowed) in cases {
            for (i, mode) in modes.iter().enumerate() {
                assert_eq!(caps.allows(mode), i == allowed, "{caps:?} / {mode:?}");
            }
        }
    }

    #[test]
    fn a_directory_table_takes_every_write_mode() {
        let all = TableCapabilities::all_write_modes();
        for mode in modes() {
            all.check(&mode, "`t`").unwrap();
        }
    }
}
