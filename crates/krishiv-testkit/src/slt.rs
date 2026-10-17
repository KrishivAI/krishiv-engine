// SPDX-License-Identifier: Apache-2.0
//! sqllogictest runner stub (P0-T23). The adapter answers every statement with `NotImplemented`
//! until Krishiv can execute SQL (P2).

use std::fmt;
use std::path::PathBuf;

use sqllogictest::{DB, DBOutput, DefaultColumnType, Runner};

/// Error returned by the stub adapter.
#[derive(Debug)]
pub struct NotImplemented(pub String);

impl fmt::Display for NotImplemented {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "krishiv SQL execution not implemented: {}", self.0)
    }
}

impl std::error::Error for NotImplemented {}

/// Stub database: echoes nothing, fails every statement with [`NotImplemented`].
#[derive(Debug, Default)]
pub struct EchoDb;

impl DB for EchoDb {
    type Error = NotImplemented;
    type ColumnType = DefaultColumnType;

    fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        Err(NotImplemented(sql.to_owned()))
    }

    fn engine_name(&self) -> &str {
        "krishiv-stub"
    }
}

/// Directory of `.slt` files.
#[must_use]
pub fn slt_dir() -> PathBuf {
    crate::tests_dir().join("sqllogictest")
}

/// Run one `.slt` file against the stub adapter.
pub fn run_file(path: &std::path::Path) -> Result<(), sqllogictest::TestError> {
    let mut runner = Runner::new(|| async { Ok::<_, NotImplemented>(EchoDb) });
    runner.run_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqllogictest::Record;

    type R = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn smoke_slt_parses() -> R {
        let records = sqllogictest::parse_file::<DefaultColumnType>(slt_dir().join("smoke.slt"))?;
        let statements = records
            .iter()
            .filter(|r| matches!(r, Record::Statement { .. }))
            .count();
        assert_eq!(statements, 1);
        Ok(())
    }

    #[test]
    #[ignore = "P2: SQL execution not implemented"]
    fn smoke_slt_runs() -> R {
        run_file(&slt_dir().join("smoke.slt"))?;
        Ok(())
    }
}
