// SPDX-License-Identifier: Apache-2.0
//! The arrow-rs `FlightSqlService` implementation (design P3-T101 §2, §4, §5): Handshake,
//! statement queries (`GetFlightInfo` / `DoGet`), `GetSqlInfo`, the metadata listings (§5), `CancelFlightInfo`, `CancelQuery`. Every RPC
//! starts with the drain check and the caller `AuthLayer` attached (fail closed); what 65a does
//! not serve never reaches it: [`crate::gate`] answers `UNIMPLEMENTED` first (§16 m5).

use std::pin::Pin;

use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{
    ActionCancelQueryRequest, ActionCancelQueryResult, ActionClosePreparedStatementRequest,
    ActionCreatePreparedStatementRequest, Any, Command as FlightCommand, CommandGetCatalogs,
    CommandGetCrossReference, CommandGetDbSchemas, CommandGetExportedKeys, CommandGetImportedKeys,
    CommandGetPrimaryKeys, CommandGetSqlInfo, CommandGetTableTypes, CommandGetTables,
    CommandPreparedStatementQuery, CommandPreparedStatementUpdate, CommandStatementQuery,
    CommandStatementUpdate, ProstMessageExt, TicketStatementQuery,
};
use arrow_flight::{
    Action, CancelFlightInfoRequest, CancelFlightInfoResult, CancelStatus, FlightData,
    FlightDescriptor, FlightInfo, HandshakeRequest, HandshakeResponse, Ticket,
};
use futures::stream::{self, Peekable, Stream};
use krishiv_common::error::{InvalidHandleFormat, KrishivError};
use krishiv_session::auth::Authenticated;
use prost::Message;
use tonic::{Request, Response, Status, Streaming};

use crate::call::shutting_down;
use crate::gate::CANCEL_FLIGHT_INFO;
use crate::metadata::{Listing, batch_stream, fixed_info};
use crate::metrics::Command;
use crate::prepared::ActionStream;
use crate::server::FlightSqlServer;
use crate::statement::{DoGetStream, internal};
use crate::status::flight_status;
use crate::update::DoPutStream;

/// `ActionCancelQueryResult.CancelResult.CANCEL_RESULT_CANCELLED` (Flight SQL's proto; arrow-rs
/// does not export the enum).
const CANCEL_RESULT_CANCELLED: i32 = 1;

type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>;

impl FlightSqlServer {
    /// The drain check and the caller (design §2 "Drain", §3.1).
    fn admit<T>(&self, request: &Request<T>) -> Result<Authenticated, Status> {
        if self.draining() {
            return Err(shutting_down());
        }
        self.inner.auth.caller(request)
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
            .admit(request)
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
        let result = match self.admit(request) {
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
        let caller = self.admit(request)?;
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
        let caller = self.admit(request)?;
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
        let caller = self.admit(request)?;
        self.prepared_update(&update, stream, request.metadata(), &caller)
            .await
    }

    /// `CreatePreparedStatement` after the gate (design §4.4).
    pub(crate) async fn do_action_create(
        &self,
        create: ActionCreatePreparedStatementRequest,
        request: &Request<Action>,
    ) -> Result<Response<ActionStream>, Status> {
        let caller = self.admit(request)?;
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
    /// the `authorization` header, which both drivers send back (design §3.1). Creates no session.
    async fn do_handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<HandshakeStream>, Status> {
        let result = self.admit(&request).map(|_| {
            let bearer = request
                .metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .filter(|v| {
                    v.get(..7)
                        .is_some_and(|s| s.eq_ignore_ascii_case("bearer "))
                })
                .and_then(|v| v.parse().ok());
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
        let result = match self.admit(&request) {
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
        let result = match self.admit(&request) {
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
        let result = self.admit(&request).and_then(|_| {
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
        let result = self.admit(&request).and_then(|_| {
            query
                .into_builder(&self.inner.sql_info)
                .build()
                .map(|batch| Response::new(batch_stream(batch)))
                .map_err(|e| internal(format!("the GetSqlInfo table: {e}")))
        });
        self.counted(Command::SqlInfo, result)
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
