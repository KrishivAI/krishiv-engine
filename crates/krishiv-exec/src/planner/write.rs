// SPDX-License-Identifier: Apache-2.0
//! How a write reaches a table (design P3-lakehouse §4.1): the four write
//! modes every Spark write resolves to, what a write asks of a table
//! ([`WriteRequest`]), and the capabilities a table declares for the modes,
//! mirroring Spark's `TableCapability` write values. The session checks the
//! capability before planning, so a format never receives a mode it did not
//! declare. The rules and the error text are Spark v4.0.0's
//! `TableCapabilityCheck` and `QueryCompilationErrors`.

use std::collections::BTreeMap;

use arrow_schema::SchemaRef;
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
    /// The `<operation>` Spark's `UNSUPPORTED_FEATURE.TABLE_OPERATION` prints
    /// when the table cannot do this mode.
    #[must_use]
    pub fn operation(&self) -> &'static str {
        match self {
            Self::Append => "append in batch mode",
            Self::Truncate => "truncate in batch mode",
            Self::OverwriteByFilter(_) => "overwrite by filter in batch mode",
            Self::OverwriteDynamic => "dynamic overwrite in batch mode",
        }
    }
}

/// What one write asks of a table: everything a format's writer needs, so the
/// trait does not change when the Iceberg and Delta writes arrive.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WriteRequest {
    /// What the write does to existing rows.
    pub mode: WriteMode,
    /// The target as Spark's errors print it (`` `cat`.`db`.`t` ``).
    pub table_name: String,
    /// The schema of the rows the sink receives.
    pub input_schema: SchemaRef,
    /// The write options (`mergeSchema`, `overwriteSchema`, the per-write
    /// `partitionOverwriteMode`, ...), keys as written.
    pub options: BTreeMap<String, String>,
    /// The idempotency key of the commit (ADR 0014, commit semantics item
    /// 5), so a retried write commits once.
    pub write_id: Option<String>,
}

impl WriteRequest {
    /// A write of `mode` into `table_name` of rows of `input_schema`, with no
    /// options and no write id.
    #[must_use]
    pub fn new(mode: WriteMode, table_name: impl Into<String>, input_schema: SchemaRef) -> Self {
        Self {
            mode,
            table_name: table_name.into(),
            input_schema,
            options: BTreeMap::new(),
            write_id: None,
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
    /// `BATCH_WRITE`: the table accepts a batch write.
    pub batch_write: bool,
    /// `V1_BATCH_WRITE`: the table accepts a batch write through the V1
    /// `InsertableRelation` path; it stands in for `BATCH_WRITE` for every
    /// mode but dynamic overwrite.
    pub v1_batch_write: bool,
    /// `TRUNCATE`: the table can drop every row.
    pub truncate: bool,
    /// `OVERWRITE_BY_FILTER`: the table can replace the rows a filter selects
    /// (and, for a truncate, stands in for `TRUNCATE`).
    pub overwrite_by_filter: bool,
    /// `OVERWRITE_DYNAMIC`: the table can replace the partitions it receives.
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

    /// Whether the table declared what `mode` needs (Spark's
    /// `TableCapabilityCheck`): every mode needs a batch write, a truncate
    /// also `TRUNCATE` or `OVERWRITE_BY_FILTER`, a filter overwrite
    /// `OVERWRITE_BY_FILTER`, a dynamic overwrite `BATCH_WRITE` itself and
    /// `OVERWRITE_DYNAMIC`.
    #[must_use]
    pub fn allows(&self, mode: &WriteMode) -> bool {
        let batch = self.batch_write || self.v1_batch_write;
        match mode {
            WriteMode::Append => batch,
            WriteMode::Truncate => batch && (self.truncate || self.overwrite_by_filter),
            WriteMode::OverwriteByFilter(_) => batch && self.overwrite_by_filter,
            WriteMode::OverwriteDynamic => self.batch_write && self.overwrite_dynamic,
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
    use std::sync::Arc;

    use arrow_schema::Schema;
    use krishiv_common::error::ErrorClass;
    use krishiv_plan::Literal;

    use super::*;

    fn modes() -> [WriteMode; 4] {
        [
            WriteMode::Append,
            WriteMode::Truncate,
            WriteMode::OverwriteByFilter(Expr::literal(Literal::Boolean(true))),
            WriteMode::OverwriteDynamic,
        ]
    }

    fn allowed(caps: TableCapabilities) -> [bool; 4] {
        modes().map(|m| caps.allows(&m))
    }

    #[test]
    fn a_table_declares_nothing_by_default() {
        let none = TableCapabilities::default();
        assert_eq!(allowed(none), [false; 4]);
        for mode in modes() {
            let err = none.check(&mode, "`c`.`db`.`t`").unwrap_err();
            assert_eq!(
                err.error_class(),
                ErrorClass::UnsupportedFeatureTableOperation
            );
            assert_eq!(err.sql_state(), Some("0A000"));
            let text = err.to_string();
            assert!(
                text.contains("`c`.`db`.`t`") && text.contains(mode.operation()),
                "{text}"
            );
        }
    }

    #[test]
    fn the_operation_strings_are_sparks() {
        let ops = modes().map(|m| m.operation());
        assert_eq!(
            ops,
            [
                "append in batch mode",
                "truncate in batch mode",
                "overwrite by filter in batch mode",
                "dynamic overwrite in batch mode"
            ]
        );
    }

    #[test]
    fn every_mode_needs_a_batch_write_first() {
        let without = |f: fn(&mut TableCapabilities)| {
            let mut c = TableCapabilities::all_write_modes();
            c.batch_write = false;
            f(&mut c);
            c
        };
        assert_eq!(allowed(without(|_| {})), [false; 4]);
        // The V1 path stands in for a batch write, but not for dynamic overwrite.
        assert_eq!(
            allowed(without(|c| c.v1_batch_write = true)),
            [true, true, true, false]
        );
    }

    #[test]
    fn each_mode_needs_its_own_capability_on_top() {
        let batch = TableCapabilities {
            batch_write: true,
            ..TableCapabilities::default()
        };
        let with = |f: fn(&mut TableCapabilities)| {
            let mut c = batch;
            f(&mut c);
            c
        };
        assert_eq!(allowed(batch), [true, false, false, false]);
        assert_eq!(
            allowed(with(|c| c.truncate = true)),
            [true, true, false, false]
        );
        assert_eq!(
            allowed(with(|c| c.overwrite_by_filter = true)),
            [true, true, true, false],
            "overwrite-by-filter also covers a truncate"
        );
        assert_eq!(
            allowed(with(|c| c.overwrite_dynamic = true)),
            [true, false, false, true]
        );
    }

    #[test]
    fn a_directory_table_takes_every_write_mode() {
        let all = TableCapabilities::all_write_modes();
        assert_eq!(allowed(all), [true; 4]);
        for mode in modes() {
            all.check(&mode, "`t`").unwrap();
        }
    }

    #[derive(Debug)]
    struct Bare(TableCapabilities);

    impl crate::planner::TableProvider for Bare {
        fn scan(&self) -> Result<Arc<dyn crate::operators::ScanPushdown>> {
            Err(krishiv_common::error::KrishivUnsupportedFeature {
                feature: "a scan".to_owned(),
            }
            .into())
        }
        fn capabilities(&self) -> TableCapabilities {
            self.0
        }
    }

    #[test]
    fn the_default_writer_refuses_what_the_table_did_not_declare_and_names_a_forgotten_one() {
        use crate::planner::TableProvider as _;
        let request = |mode| WriteRequest::new(mode, "`c`.`db`.`t`", Arc::new(Schema::empty()));
        let refused = Bare(TableCapabilities::default())
            .writer(&request(WriteMode::Append))
            .err()
            .unwrap();
        assert_eq!(
            refused.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation
        );
        assert!(refused.to_string().contains("`c`.`db`.`t`"), "{refused}");
        // Declared, but the provider forgot to implement `writer`.
        let forgotten = Bare(TableCapabilities::all_write_modes())
            .writer(&request(WriteMode::Append))
            .err()
            .unwrap();
        assert_eq!(
            forgotten.error_class(),
            ErrorClass::KrishivUnsupportedFeature
        );
        assert!(forgotten.to_string().contains("Bare"), "{forgotten}");
    }

    #[test]
    fn a_request_starts_without_options_or_a_write_id() {
        let schema = Arc::new(Schema::empty());
        let req = WriteRequest::new(WriteMode::Append, "`t`", Arc::clone(&schema));
        assert!(req.options.is_empty() && req.write_id.is_none());
        assert_eq!(req.table_name, "`t`");
        assert_eq!(req.input_schema, schema);
    }
}
