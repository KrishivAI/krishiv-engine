// SPDX-License-Identifier: Apache-2.0
//! The arrow-rs `FlightSqlService` implementation (design P3-T101 §2, §4, §5): Handshake,
//! statement queries (`GetFlightInfo` / `DoGet`), `GetSqlInfo`, `GetXdbcTypeInfo`, the metadata listings (§5), `CancelFlightInfo`, `CancelQuery`. Every RPC has the
//! caller `AuthLayer` attached (fail closed), and an RPC that starts new work (via `admit_new`) first
//! passes the drain check; what 65a does
//! not serve never reaches it: [`crate::gate`] answers `UNIMPLEMENTED` first (§16 m5).

use std::pin::Pin;

use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{
    ActionCancelQueryRequest, ActionCancelQueryResult, ActionClosePreparedStatementRequest,
    ActionCreatePreparedStatementRequest, Any, Command as FlightCommand, CommandGetCatalogs,
    CommandGetCrossReference, CommandGetDbSchemas, CommandGetExportedKeys, CommandGetImportedKeys,
    CommandGetPrimaryKeys, CommandGetSqlInfo, CommandGetTableTypes, CommandGetTables,
    CommandGetXdbcTypeInfo, CommandPreparedStatementQuery, CommandPreparedStatementUpdate,
    CommandStatementQuery, CommandStatementUpdate, ProstMessageExt, TicketStatementQuery,
};
use arrow_flight::{
    Action, CancelFlightInfoRequest, CancelFlightInfoResult, CancelStatus, FlightData,
    FlightDescriptor, FlightInfo, HandshakeRequest, HandshakeResponse, Ticket,
};
use futures::stream::{self, Peekable, Stream};
use krishiv_common::error::{InvalidHandleFormat, KrishivError};
use krishiv_session::auth::{Authenticated, BasicHeader, basic_header};
use prost::Message;
use tonic::{Request, Response, Status, Streaming};

use crate::call::draining;
use crate::gate::CANCEL_FLIGHT_INFO;
use crate::metadata::{Listing, batch_stream, fixed_info};
use crate::metrics::Command;
use crate::prepared::ActionStream;
use crate::server::FlightSqlServer;
use crate::session_options::SetSessionOptionsRequest;
use crate::statement::{DoGetStream, internal};
use crate::status::flight_status;
use crate::update::DoPutStream;

/// `ActionCancelQueryResult.CancelResult.CANCEL_RESULT_CANCELLED` (Flight SQL's proto; arrow-rs
/// does not export the enum).
const CANCEL_RESULT_CANCELLED: i32 = 1;

type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>;

impl FlightSqlServer {
    /// The caller of a call that works on something that already exists: a fetch of a ticket
    /// issued before a drain, a cancel, `ClosePreparedStatement` (design P3-T066 §5.1: these are
    /// allowed while draining). Binding rows to a prepared handle starts an execution, so it is
    /// refused (`admit_new`).
    fn admit<T>(&self, request: &Request<T>) -> Result<Authenticated, Status> {
        self.inner.auth.caller(request)
    }

    /// The caller of a call that starts new work (handshake, statement, update, prepared-statement create/execute/bind,
    /// metadata command, session options): refused with `KRISHIV_SERVER_DRAINING` while the server drains (design P3-T066 §5.1), a
    /// non-retryable `FAILED_PRECONDITION` and not `UNAVAILABLE`.
    fn admit_new<T>(&self, request: &Request<T>) -> Result<Authenticated, Status> {
        if self.draining() {
            return Err(draining());
        }
        self.admit(request)
    }

    /// Counts `result` under `command` (`krishiv_flight_sql_requests_total`).
    pub(crate) fn counted<T>(
        &self,
        command: Command,
        result: Result<T, Status>,
    ) -> Result<T, Status> {
        let code = match &result {
            Ok(_) => tonic::Code::Ok,
            Err(s) => s.code(),
        };
        self.inner.requests.record(command, code);
        result
    }

    /// `GetFlightInfo` of a listing (design §5): no session.
    fn listing_info_call(
        &self,
        listing: &Listing,
        request: &Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let command = listing.command();
        let result = self
            .admit_new(request)
            .and_then(|_| Self::listing_info(listing, request.get_ref().clone()));
        self.counted(command, result)
    }

    /// `DoGet` of a listing (design §5).
    async fn listing_call(
        &self,
        listing: Listing,
        request: &Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        let command = listing.command();
        let result = match self.admit_new(request) {
            Ok(caller) => self.listing(listing, request.metadata(), &caller).await,
            Err(s) => Err(s),
        };
        self.counted(command, result)
    }

    /// `DoPut(CommandStatementUpdate)` after the gate (design §4.3).
    pub(crate) async fn do_put_update(
        &self,
        update: CommandStatementUpdate,
        request: &Request<()>,
    ) -> Result<Response<DoPutStream>, Status> {
        let caller = self.admit_new(request)?;
        self.statement_update(update, request.metadata(), &caller)
            .await
    }

    /// `DoPut(CommandPreparedStatementQuery)` after the gate (design §4.4).
    pub(crate) async fn do_put_bind(
        &self,
        bind: CommandPreparedStatementQuery,
        stream: Peekable<Streaming<FlightData>>,
        request: &Request<()>,
    ) -> Result<Response<DoPutStream>, Status> {
        let caller = self.admit_new(request)?;
        self.prepared_bind(bind, stream, request.metadata(), &caller)
            .await
    }

    /// `DoPut(CommandPreparedStatementUpdate)` after the gate (design §4.4).
    pub(crate) async fn do_put_prepared_update(
        &self,
        update: CommandPreparedStatementUpdate,
        stream: Peekable<Streaming<FlightData>>,
        request: &Request<()>,
    ) -> Result<Response<DoPutStream>, Status> {
        let caller = self.admit_new(request)?;
        self.prepared_update(&update, stream, request.metadata(), &caller)
            .await
    }

    /// `CreatePreparedStatement` after the gate (design §4.4).
    pub(crate) async fn do_action_create(
        &self,
        create: ActionCreatePreparedStatementRequest,
        request: &Request<Action>,
    ) -> Result<Response<ActionStream>, Status> {
        let caller = self.admit_new(request)?;
        self.prepared_create(create, request.metadata(), &caller)
            .await
    }

    /// `ClosePreparedStatement` after the gate (design §4.4).
    pub(crate) fn do_action_close(
        &self,
        close: &ActionClosePreparedStatementRequest,
        request: &Request<Action>,
    ) -> Result<Response<ActionStream>, Status> {
        let caller = self.admit(request)?;
        self.prepared_close(close, &caller)
    }

    /// `SetSessionOptions` after the gate (design §3.2).
    pub(crate) async fn do_action_set_options(
        &self,
        set: SetSessionOptionsRequest,
        request: &Request<Action>,
    ) -> Result<Response<ActionStream>, Status> {
        let caller = self.admit_new(request)?;
        self.set_session_options(set, request.metadata(), &caller)
            .await
    }

    /// `GetSessionOptions` after the gate (design §3.2).
    pub(crate) async fn do_action_get_options(
        &self,
        request: &Request<Action>,
    ) -> Result<Response<ActionStream>, Status> {
        let caller = self.admit_new(request)?;
        self.get_session_options(request.metadata(), &caller).await
    }

    /// `CloseSession` after the gate (design §3.2).
    pub(crate) fn do_action_close_session(
        &self,
        request: &Request<Action>,
    ) -> Result<Response<ActionStream>, Status> {
        let caller = self.admit(request)?;
        Ok(self.close_session(request.metadata(), &caller))
    }

    /// Cancels the statement whose ticket is `info`'s first endpoint's (design §4.5): a missing
    /// or malformed ticket is `INVALID_HANDLE.FORMAT` (`INVALID_ARGUMENT`, §6); another
    /// principal's handle and an unknown one get one `NOT_FOUND` answer.
    fn cancel_info(&self, caller: &Authenticated, info: Option<&FlightInfo>) -> Result<(), Status> {
        let handle = info
            .and_then(|info| info.endpoint.first())
            .and_then(|endpoint| endpoint.ticket.as_ref())
            .and_then(|ticket| Any::decode(ticket.ticket.as_ref()).ok())
            .and_then(|any| FlightCommand::try_from(any).ok())
            .and_then(|command| match command {
                FlightCommand::TicketStatementQuery(t) => Some(t.statement_handle),
                _ => None,
            })
            .ok_or_else(|| flight_status(&malformed_ticket()))?;
        self.cancel_statement(caller, &handle)
            .map_err(|e| flight_status(&e))
    }
}

/// A cancel whose ticket is not a statement ticket of this server.
fn malformed_ticket() -> KrishivError {
    InvalidHandleFormat {
        handle: "<ticket>".to_owned(),
    }
    .into()
}

#[tonic::async_trait]
impl FlightSqlService for FlightSqlServer {
    type FlightService = Self;

    /// Authenticates like any RPC (`AuthLayer`) and answers with the caller's bearer token in
    /// the `authorization` header, which both drivers send back (design §3.1; for a `Basic`
    /// login, its password). Creates no session.
    async fn do_handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<HandshakeStream>, Status> {
        let result = self.admit_new(&request).map(|_| {
            let header = request
                .metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok());
            // A `Basic` login answers its password as the bearer token (P3-T101 note §3.1):
            // Arrow's clients send that from then on (`authenticateBasicToken`).
            let bearer = header.and_then(|v| match basic_header(v) {
                BasicHeader::Credentials(b) => {
                    let mut text = format!("Bearer {}", b.password());
                    let value = text.parse().ok();
                    krishiv_common::secret::wipe(&mut text);
                    value
                }
                _ => v
                    .get(..7)
                    .is_some_and(|s| s.eq_ignore_ascii_case("bearer "))
                    .then(|| v.parse().ok())
                    .flatten(),
            });
            let out: HandshakeStream = Box::pin(stream::iter([Ok(HandshakeResponse {
                protocol_version: 0,
                payload: Default::default(),
            })]));
            let mut response = Response::new(out);
            if let Some(value) = bearer {
                response.metadata_mut().insert("authorization", value);
            }
            response
        });
        self.counted(Command::Handshake, result)
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let result = match self.admit_new(&request) {
            Ok(caller) => {
                let descriptor = request.get_ref().clone();
                self.statement_info(query, request.metadata(), descriptor, &caller)
                    .await
            }
            Err(s) => Err(s),
        };
        self.counted(Command::StatementQuery, result)
    }

    async fn get_flight_info_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let result = match self.admit_new(&request) {
            Ok(caller) => {
                let descriptor = request.get_ref().clone();
                self.prepared_info(&query, request.metadata(), descriptor, &caller)
                    .await
            }
            Err(s) => Err(s),
        };
        self.counted(Command::PreparedQuery, result)
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        let result = match self.admit(&request) {
            Ok(caller) => {
                self.statement_get(ticket, request.metadata(), &caller)
                    .await
            }
            Err(s) => Err(s),
        };
        self.counted(Command::Fetch, result)
    }

    /// `GetSqlInfo` reads no session state (design §3.2, v3 review m6): no session.
    async fn get_flight_info_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let result = self.admit_new(&request).and_then(|_| {
            let schema = query.clone().into_builder(&self.inner.sql_info).schema();
            fixed_info(&schema, &query.as_any(), request.get_ref().clone()).map(Response::new)
        });
        self.counted(Command::SqlInfo, result)
    }

    async fn do_get_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        let result = self.admit_new(&request).and_then(|_| {
            query
                .into_builder(&self.inner.sql_info)
                .build()
                .map(|batch| Response::new(batch_stream(batch)))
                .map_err(|e| internal(format!("the GetSqlInfo table: {e}")))
        });
        self.counted(Command::SqlInfo, result)
    }

    /// `GetXdbcTypeInfo` (design §5) reads no session state either: no session.
    async fn get_flight_info_xdbc_type_info(
        &self,
        query: CommandGetXdbcTypeInfo,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let result = self.admit_new(&request).and_then(|_| {
            let schema = query.into_builder(&self.inner.xdbc_type_info).schema();
            fixed_info(&schema, &query.as_any(), request.get_ref().clone()).map(Response::new)
        });
        self.counted(Command::XdbcTypeInfo, result)
    }

    async fn do_get_xdbc_type_info(
        &self,
        query: CommandGetXdbcTypeInfo,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        let result = self.admit_new(&request).and_then(|_| {
            query
                .into_builder(&self.inner.xdbc_type_info)
                .build()
                .map(|batch| Response::new(batch_stream(batch)))
                .map_err(|e| internal(format!("the GetXdbcTypeInfo table: {e}")))
        });
        self.counted(Command::XdbcTypeInfo, result)
    }

    async fn get_flight_info_catalogs(
        &self,
        query: CommandGetCatalogs,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.listing_info_call(&Listing::Catalogs(query), &request)
    }

    async fn do_get_catalogs(
        &self,
        query: CommandGetCatalogs,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.listing_call(Listing::Catalogs(query), &request).await
    }

    async fn get_flight_info_schemas(
        &self,
        query: CommandGetDbSchemas,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.listing_info_call(&Listing::DbSchemas(query), &request)
    }

    async fn do_get_schemas(
        &self,
        query: CommandGetDbSchemas,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.listing_call(Listing::DbSchemas(query), &request).await
    }

    async fn get_flight_info_tables(
        &self,
        query: CommandGetTables,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.listing_info_call(&Listing::Tables(query), &request)
    }

    async fn do_get_tables(
        &self,
        query: CommandGetTables,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.listing_call(Listing::Tables(query), &request).await
    }

    async fn get_flight_info_table_types(
        &self,
        query: CommandGetTableTypes,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.listing_info_call(&Listing::TableTypes(query), &request)
    }

    async fn do_get_table_types(
        &self,
        query: CommandGetTableTypes,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.listing_call(Listing::TableTypes(query), &request)
            .await
    }

    async fn get_flight_info_primary_keys(
        &self,
        query: CommandGetPrimaryKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.listing_info_call(&Listing::PrimaryKeys(query), &request)
    }

    async fn do_get_primary_keys(
        &self,
        query: CommandGetPrimaryKeys,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.listing_call(Listing::PrimaryKeys(query), &request)
            .await
    }

    async fn get_flight_info_exported_keys(
        &self,
        query: CommandGetExportedKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.listing_info_call(&Listing::ExportedKeys(query), &request)
    }

    async fn do_get_exported_keys(
        &self,
        query: CommandGetExportedKeys,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.listing_call(Listing::ExportedKeys(query), &request)
            .await
    }

    async fn get_flight_info_imported_keys(
        &self,
        query: CommandGetImportedKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.listing_info_call(&Listing::ImportedKeys(query), &request)
    }

    async fn do_get_imported_keys(
        &self,
        query: CommandGetImportedKeys,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.listing_call(Listing::ImportedKeys(query), &request)
            .await
    }

    async fn get_flight_info_cross_reference(
        &self,
        query: CommandGetCrossReference,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.listing_info_call(&Listing::CrossReference(query), &request)
    }

    async fn do_get_cross_reference(
        &self,
        query: CommandGetCrossReference,
        request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.listing_call(Listing::CrossReference(query), &request)
            .await
    }

    /// `CancelFlightInfo` (design §4.5). The crate's call gate lets no other action reach here.
    async fn do_action_fallback(
        &self,
        request: Request<Action>,
    ) -> Result<Response<ActionStream>, Status> {
        if request.get_ref().r#type != CANCEL_FLIGHT_INFO {
            return self.counted(
                Command::Unsupported,
                Err(Status::unimplemented(format!(
                    "the Flight action {:?} is not supported",
                    request.get_ref().r#type
                ))),
            );
        }
        let result = self.admit(&request).and_then(|caller| {
            let cancel = CancelFlightInfoRequest::decode(request.get_ref().body.as_ref())
                .map_err(|_| flight_status(&malformed_ticket()))?;
            self.cancel_info(&caller, cancel.info.as_ref())?;
            let body = CancelFlightInfoResult::new(CancelStatus::Cancelled).encode_to_vec();
            let out: ActionStream = Box::pin(stream::iter([Ok(arrow_flight::Result {
                body: body.into(),
            })]));
            Ok(Response::new(out))
        });
        self.counted(Command::Cancel, result)
    }

    /// The deprecated `CancelQuery` (design §4.5): `CancelFlightInfo`'s rules, its own answer.
    async fn do_action_cancel_query(
        &self,
        query: ActionCancelQueryRequest,
        request: Request<Action>,
    ) -> Result<ActionCancelQueryResult, Status> {
        let result = self.admit(&request).and_then(|caller| {
            let info = FlightInfo::decode(query.info.as_ref())
                .map_err(|_| flight_status(&malformed_ticket()))?;
            self.cancel_info(&caller, Some(&info))?;
            Ok(ActionCancelQueryResult {
                result: CANCEL_RESULT_CANCELLED,
            })
        });
        self.counted(Command::Cancel, result)
    }

    async fn register_sql_info(&self, _id: i32, _result: &arrow_flight::sql::SqlInfo) {}
}
