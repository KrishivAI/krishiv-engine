// SPDX-License-Identifier: Apache-2.0
//! `GetSqlInfo`: what the server tells a Flight SQL client about itself (design P3-T101 §5).
//! ADBC reads `FLIGHT_SQL_SERVER_TRANSACTION` when it connects (no transactions, so it refuses
//! autocommit off); JDBC reads the rest lazily. Built once per server; each `GetSqlInfo` call
//! selects its ids from it.

use arrow_flight::sql::metadata::{SqlInfoData, SqlInfoDataBuilder};
use arrow_flight::sql::{SqlInfo, SqlSupportedCaseSensitivity, SqlSupportedTransaction};
use krishiv_common::error::{InternalError, KrishivError};

/// The server name reported as `FLIGHT_SQL_SERVER_NAME`.
pub const SERVER_NAME: &str = "Krishiv";

/// The identifier quote reported as `SQL_IDENTIFIER_QUOTE_CHAR` (Spark SQL quotes with a
/// backtick).
pub const IDENTIFIER_QUOTE: &str = "`";

/// What the server reports beyond the fixed answers: the parts the caller knows (versions, the
/// parser's keywords, the function registry). The function lists are by Flight SQL's four
/// groups; a function may be in none of them.
#[derive(Debug, Clone, Default)]
pub struct ServerInfo {
    /// `FLIGHT_SQL_SERVER_VERSION`: the Krishiv version.
    pub version: String,
    /// `FLIGHT_SQL_SERVER_ARROW_VERSION`: the Arrow version the server encodes with.
    pub arrow_version: String,
    /// `SQL_KEYWORDS`: the parser's keywords.
    pub keywords: Vec<String>,
    /// `SQL_NUMERIC_FUNCTIONS`.
    pub numeric_functions: Vec<String>,
    /// `SQL_STRING_FUNCTIONS`.
    pub string_functions: Vec<String>,
    /// `SQL_SYSTEM_FUNCTIONS`.
    pub system_functions: Vec<String>,
    /// `SQL_DATETIME_FUNCTIONS`.
    pub datetime_functions: Vec<String>,
}

/// The `GetSqlInfo` table of `info`: name `Krishiv`, `READ_ONLY = false`, SQL yes, Substrait
/// no, transactions none, cancel yes, backtick quotes, case-insensitive identifiers (quoted
/// ones too: Spark resolves both ignoring case by default), and `info`'s versions, keywords and
/// functions.
///
/// # Errors
/// `INTERNAL_ERROR` if arrow-flight cannot build the table (a bug: the values are fixed types).
pub fn sql_info_data(info: &ServerInfo) -> Result<SqlInfoData, KrishivError> {
    let mut b = SqlInfoDataBuilder::new();
    b.append(SqlInfo::FlightSqlServerName, SERVER_NAME);
    b.append(SqlInfo::FlightSqlServerVersion, info.version.as_str());
    b.append(
        SqlInfo::FlightSqlServerArrowVersion,
        info.arrow_version.as_str(),
    );
    b.append(SqlInfo::FlightSqlServerReadOnly, false);
    b.append(SqlInfo::FlightSqlServerSql, true);
    b.append(SqlInfo::FlightSqlServerSubstrait, false);
    b.append(
        SqlInfo::FlightSqlServerTransaction,
        SqlSupportedTransaction::None as i32,
    );
    b.append(SqlInfo::FlightSqlServerCancel, true);
    b.append(SqlInfo::FlightSqlServerBulkIngestion, false);
    b.append(SqlInfo::SqlDdlCatalog, false);
    b.append(SqlInfo::SqlIdentifierQuoteChar, IDENTIFIER_QUOTE);
    let insensitive = SqlSupportedCaseSensitivity::SqlCaseSensitivityCaseInsensitive as i32;
    b.append(SqlInfo::SqlIdentifierCase, insensitive);
    b.append(SqlInfo::SqlQuotedIdentifierCase, insensitive);
    b.append(SqlInfo::SqlKeywords, info.keywords.clone());
    b.append(
        SqlInfo::SqlNumericFunctions,
        function_names(&info.numeric_functions),
    );
    b.append(
        SqlInfo::SqlStringFunctions,
        function_names(&info.string_functions),
    );
    b.append(
        SqlInfo::SqlSystemFunctions,
        function_names(&info.system_functions),
    );
    b.append(
        SqlInfo::SqlDatetimeFunctions,
        function_names(&info.datetime_functions),
    );
    b.build().map_err(|e| {
        KrishivError::from(InternalError {
            message: format!("Flight SQL GetSqlInfo table: {e}"),
        })
    })
}

/// The names of a function list JDBC's `getNumericFunctions` and its siblings can use: operators
/// (`%`, `+`, `||`) are left out (review of 9b66ee1fc, nit).
fn function_names(names: &[String]) -> Vec<String> {
    names
        .iter()
        .filter(|n| {
            n.chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, RecordBatch, UInt32Array, UnionArray};

    fn info() -> ServerInfo {
        ServerInfo {
            version: "0.2.1".into(),
            arrow_version: "60".into(),
            keywords: vec!["SELECT".into(), "FROM".into()],
            numeric_functions: vec!["abs".into()],
            string_functions: vec!["upper".into()],
            system_functions: vec![],
            datetime_functions: vec!["current_date".into()],
        }
    }

    fn select(ids: &[SqlInfo]) -> RecordBatch {
        let data = sql_info_data(&info()).expect("table");
        data.record_batch(ids.iter().map(|i| *i as u32))
            .expect("selection")
    }

    /// The union child of row `row`: its type id names the value kind.
    fn kind(batch: &RecordBatch, row: usize) -> i8 {
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<UnionArray>()
            .expect("union");
        values.type_id(row)
    }

    #[test]
    fn function_lists_hold_names_only() {
        let names: Vec<String> = ["abs", "%", "+", "||", "log10", "_x", "1x", ""]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(function_names(&names), ["abs", "log10", "_x"]);
    }

    #[test]
    fn reports_no_transactions_for_adbc() {
        let batch = select(&[SqlInfo::FlightSqlServerTransaction]);
        assert_eq!(batch.num_rows(), 1);
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("ids");
        assert_eq!(ids.value(0), SqlInfo::FlightSqlServerTransaction as u32);
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<UnionArray>()
            .expect("union");
        let child = values.child(kind(&batch, 0));
        let ints = child
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .expect("int32 child");
        assert_eq!(
            ints.value(values.value_offset(0)),
            SqlSupportedTransaction::None as i32
        );
    }

    #[test]
    fn reports_name_quote_and_lists() {
        let batch = select(&[
            SqlInfo::FlightSqlServerName,
            SqlInfo::SqlIdentifierQuoteChar,
            SqlInfo::SqlKeywords,
            SqlInfo::SqlDatetimeFunctions,
        ]);
        assert_eq!(batch.num_rows(), 4);
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<UnionArray>()
            .expect("union");
        let strings = |row: usize| {
            let child = values.child(values.type_id(row));
            child
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .map(|s| s.value(values.value_offset(row)).to_owned())
        };
        let names: Vec<_> = (0..2).filter_map(strings).collect();
        assert!(names.contains(&SERVER_NAME.to_owned()));
        assert!(names.contains(&IDENTIFIER_QUOTE.to_owned()));
        // Keywords and functions are string lists, a different union child than the names.
        assert_ne!(kind(&batch, 2), kind(&batch, 0));
        assert_eq!(kind(&batch, 2), kind(&batch, 3));
    }

    #[test]
    fn unknown_ids_are_left_out() {
        let data = sql_info_data(&info()).expect("table");
        let batch = data.record_batch([9_999_u32]).expect("selection");
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn every_reported_id_is_selectable() {
        let data = sql_info_data(&info()).expect("table");
        let all = data.record_batch(std::iter::empty()).expect("all");
        // Empty selection means every id: the 18 answers `sql_info_data` appends.
        assert_eq!(all.num_rows(), 18);
        assert!(all.column(1).null_count() == 0);
    }
}
