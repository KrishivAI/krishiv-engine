// SPDX-License-Identifier: Apache-2.0
//! `krishiv_flight_sql_requests_total{command, outcome}` (design P3-T101 §7): requests that
//! passed authentication, by command and by how they ended. Both labels are fixed enums, so
//! the series are bounded; `krishiv server` merges [`Requests::counts`] into its scrape as it
//! does the authentication counters.

use std::sync::atomic::{AtomicU64, Ordering};

use tonic::Code;

/// The catalogue name of the counter (`krishiv-exec`'s metrics registry).
pub const REQUESTS_METRIC: &str = "krishiv.flight_sql.requests";

/// The `command` label: the Flight SQL RPC or command of a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Command {
    /// `Handshake`.
    Handshake,
    /// `GetFlightInfo(CommandStatementQuery)`.
    StatementQuery,
    /// `DoPut(CommandStatementUpdate)`.
    StatementUpdate,
    /// `DoGet` of a statement ticket.
    Fetch,
    /// `CreatePreparedStatement`.
    PreparedCreate,
    /// `GetFlightInfo` or `DoPut` (bind) of a prepared query.
    PreparedQuery,
    /// `DoPut(CommandPreparedStatementUpdate)`.
    PreparedUpdate,
    /// `ClosePreparedStatement`.
    PreparedClose,
    /// `GetCatalogs`.
    Catalogs,
    /// `GetDbSchemas`.
    DbSchemas,
    /// `GetTables`.
    Tables,
    /// `GetTableTypes`.
    TableTypes,
    /// `GetPrimaryKeys`, `GetExportedKeys`, `GetImportedKeys`, `GetCrossReference`.
    Keys,
    /// `GetSqlInfo`.
    SqlInfo,
    /// `GetXdbcTypeInfo`.
    XdbcTypeInfo,
    /// `CancelFlightInfo`, `CancelQuery`.
    Cancel,
    /// `SetSessionOptions`, `GetSessionOptions`.
    SessionOptions,
    /// `CloseSession`.
    SessionClose,
    /// `ListActions` (no session; the actions this server serves).
    ListActions,
    /// Everything design §4.6 refuses.
    Unsupported,
}

impl Command {
    /// Every command, in label order.
    pub const ALL: [Self; 20] = [
        Self::Handshake,
        Self::StatementQuery,
        Self::StatementUpdate,
        Self::Fetch,
        Self::PreparedCreate,
        Self::PreparedQuery,
        Self::PreparedUpdate,
        Self::PreparedClose,
        Self::Catalogs,
        Self::DbSchemas,
        Self::Tables,
        Self::TableTypes,
        Self::Keys,
        Self::SqlInfo,
        Self::XdbcTypeInfo,
        Self::Cancel,
        Self::SessionOptions,
        Self::SessionClose,
        Self::ListActions,
        Self::Unsupported,
    ];

    /// The label value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Handshake => "handshake",
            Self::StatementQuery => "statement_query",
            Self::StatementUpdate => "statement_update",
            Self::Fetch => "fetch",
            Self::PreparedCreate => "prepared_create",
            Self::PreparedQuery => "prepared_query",
            Self::PreparedUpdate => "prepared_update",
            Self::PreparedClose => "prepared_close",
            Self::Catalogs => "catalogs",
            Self::DbSchemas => "db_schemas",
            Self::Tables => "tables",
            Self::TableTypes => "table_types",
            Self::Keys => "keys",
            Self::SqlInfo => "sql_info",
            Self::XdbcTypeInfo => "xdbc_type_info",
            Self::Cancel => "cancel",
            Self::SessionOptions => "session_options",
            Self::SessionClose => "session_close",
            Self::ListActions => "list_actions",
            Self::Unsupported => "unsupported",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// The `outcome` label: a request's gRPC status, grouped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Outcome {
    /// `OK`.
    Ok,
    /// `INVALID_ARGUMENT`, `FAILED_PRECONDITION`, `OUT_OF_RANGE`, `ALREADY_EXISTS`.
    Invalid,
    /// `NOT_FOUND`.
    NotFound,
    /// `PERMISSION_DENIED`, `UNAUTHENTICATED`.
    Denied,
    /// `RESOURCE_EXHAUSTED`.
    Exhausted,
    /// `CANCELLED`, `DEADLINE_EXCEEDED`, `ABORTED`.
    Cancelled,
    /// `UNAVAILABLE` (a draining server).
    Unavailable,
    /// `UNIMPLEMENTED`.
    Unimplemented,
    /// `INTERNAL`, `UNKNOWN`, `DATA_LOSS`.
    Failed,
}

impl Outcome {
    /// Every outcome, in label order.
    pub const ALL: [Self; 9] = [
        Self::Ok,
        Self::Invalid,
        Self::NotFound,
        Self::Denied,
        Self::Exhausted,
        Self::Cancelled,
        Self::Unavailable,
        Self::Unimplemented,
        Self::Failed,
    ];

    /// The outcome of a request that ended with `code`.
    #[must_use]
    pub const fn of(code: Code) -> Self {
        match code {
            Code::Ok => Self::Ok,
            Code::InvalidArgument
            | Code::FailedPrecondition
            | Code::OutOfRange
            | Code::AlreadyExists => Self::Invalid,
            Code::NotFound => Self::NotFound,
            Code::PermissionDenied | Code::Unauthenticated => Self::Denied,
            Code::ResourceExhausted => Self::Exhausted,
            Code::Cancelled | Code::DeadlineExceeded | Code::Aborted => Self::Cancelled,
            Code::Unavailable => Self::Unavailable,
            Code::Unimplemented => Self::Unimplemented,
            Code::Internal | Code::Unknown | Code::DataLoss => Self::Failed,
        }
    }

    /// The label value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Invalid => "invalid",
            Self::NotFound => "not_found",
            Self::Denied => "denied",
            Self::Exhausted => "exhausted",
            Self::Cancelled => "cancelled",
            Self::Unavailable => "unavailable",
            Self::Unimplemented => "unimplemented",
            Self::Failed => "failed",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// One series of the counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestCount {
    /// The `command` label.
    pub command: Command,
    /// The `outcome` label.
    pub outcome: Outcome,
    /// Requests so far.
    pub count: u64,
}

/// The counter: one cell per (command, outcome), lock-free.
#[derive(Debug)]
pub struct Requests {
    cells: [[AtomicU64; Outcome::ALL.len()]; Command::ALL.len()],
}

impl Default for Requests {
    fn default() -> Self {
        Self {
            cells: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU64::new(0))),
        }
    }
}

impl Requests {
    /// Counts one request of `command` that ended with `code`.
    pub fn record(&self, command: Command, code: Code) {
        self.cells[command.index()][Outcome::of(code).index()].fetch_add(1, Ordering::Relaxed);
    }

    /// The series with a non-zero count, in label order.
    #[must_use]
    pub fn counts(&self) -> Vec<RequestCount> {
        let mut out = Vec::new();
        for command in Command::ALL {
            for outcome in Outcome::ALL {
                let count = self.cells[command.index()][outcome.index()].load(Ordering::Relaxed);
                if count > 0 {
                    out.push(RequestCount {
                        command,
                        outcome,
                        count,
                    });
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_exec::{CATALOGUE, FLIGHT_SQL_COMMAND_LABEL, OUTCOME_LABEL};

    #[test]
    fn labels_are_the_registry_s() {
        let commands: Vec<_> = Command::ALL.iter().map(|c| c.as_str()).collect();
        assert_eq!(commands, FLIGHT_SQL_COMMAND_LABEL.values);
        let outcomes: Vec<_> = Outcome::ALL.iter().map(|o| o.as_str()).collect();
        assert_eq!(outcomes, OUTCOME_LABEL.values);
        let def = CATALOGUE
            .iter()
            .find(|d| d.name == REQUESTS_METRIC)
            .expect("registered");
        let names: Vec<_> = def.labels.iter().map(|l| l.name).collect();
        assert_eq!(names, ["command", "outcome"]);
    }

    #[test]
    fn index_is_label_order() {
        for (i, c) in Command::ALL.iter().enumerate() {
            assert_eq!(c.index(), i);
        }
        for (i, o) in Outcome::ALL.iter().enumerate() {
            assert_eq!(o.index(), i);
        }
    }

    #[test]
    fn every_code_has_an_outcome() {
        use Code::*;
        let all = [
            Ok,
            Cancelled,
            Unknown,
            InvalidArgument,
            DeadlineExceeded,
            NotFound,
            AlreadyExists,
            PermissionDenied,
            ResourceExhausted,
            FailedPrecondition,
            Aborted,
            OutOfRange,
            Unimplemented,
            Internal,
            Unavailable,
            DataLoss,
            Unauthenticated,
        ];
        let seen: std::collections::HashSet<_> = all.iter().map(|c| Outcome::of(*c)).collect();
        assert_eq!(seen.len(), Outcome::ALL.len(), "every outcome reachable");
        assert_eq!(Outcome::of(Unavailable), Outcome::Unavailable);
        assert_eq!(Outcome::of(FailedPrecondition), Outcome::Invalid);
    }

    #[test]
    fn counts_only_seen_series_in_order() {
        let r = Requests::default();
        assert!(r.counts().is_empty());
        r.record(Command::Fetch, Code::Ok);
        r.record(Command::Fetch, Code::Ok);
        r.record(Command::Handshake, Code::Unauthenticated);
        r.record(Command::Fetch, Code::Cancelled);
        assert_eq!(
            r.counts(),
            [
                RequestCount {
                    command: Command::Handshake,
                    outcome: Outcome::Denied,
                    count: 1
                },
                RequestCount {
                    command: Command::Fetch,
                    outcome: Outcome::Ok,
                    count: 2
                },
                RequestCount {
                    command: Command::Fetch,
                    outcome: Outcome::Cancelled,
                    count: 1
                },
            ]
        );
    }
}
