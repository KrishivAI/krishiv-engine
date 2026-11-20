// SPDX-License-Identifier: Apache-2.0
//! The fail-closed front of the Flight service (design P3-T101 §4.6, §16 m5): every gRPC call is
//! classified as a [`FlightRpc`] from its method and its command's `Any` type URL (or its action
//! type) before arrow-rs decodes a body, and [`check_flight`] answers whatever 65a does not serve
//! `UNIMPLEMENTED`, counted as `unsupported`, before any authentication or session work. The
//! match in `check_flight` has no wildcard, so a new variant must be decided there.

use arrow_flight::flight_service_server::FlightService;
use arrow_flight::sql::{
    ActionClosePreparedStatementRequest, ActionCreatePreparedStatementRequest, Any,
    CommandGetCatalogs, CommandGetCrossReference, CommandGetDbSchemas, CommandGetExportedKeys,
    CommandGetImportedKeys, CommandGetPrimaryKeys, CommandGetSqlInfo, CommandGetTableTypes,
    CommandGetTables, CommandGetXdbcTypeInfo, CommandPreparedStatementQuery,
    CommandPreparedStatementUpdate, CommandStatementQuery, CommandStatementUpdate, ProstMessageExt,
    TicketStatementQuery,
};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, PollInfo, SchemaResult, Ticket,
};
use futures::StreamExt;
use futures::stream::Peekable;
use prost::Message;
use tonic::{Request, Response, Status, Streaming};

use crate::metrics::Command;
use crate::server::FlightSqlServer;
use crate::session_options::{
    CLOSE_SESSION, GET_SESSION_OPTIONS, SET_SESSION_OPTIONS, SetSessionOptionsRequest,
};

/// The Flight action `CancelFlightInfo`.
pub(crate) const CANCEL_FLIGHT_INFO: &str = "CancelFlightInfo";
/// The deprecated Flight SQL action `CancelQuery` (design §4.5).
pub(crate) const CANCEL_QUERY: &str = "CancelQuery";
/// The Flight SQL action `CreatePreparedStatement` (design §4.4).
pub(crate) const CREATE_PREPARED_STATEMENT: &str = "CreatePreparedStatement";
/// The Flight SQL action `ClosePreparedStatement` (design §4.4).
pub(crate) const CLOSE_PREPARED_STATEMENT: &str = "ClosePreparedStatement";

/// One gRPC call as 65a sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FlightRpc {
    /// `Handshake`.
    Handshake,
    /// `GetFlightInfo(CommandStatementQuery)`.
    StatementInfo,
    /// `DoGet(TicketStatementQuery)`.
    Fetch,
    /// `GetFlightInfo` or `DoGet` of `CommandGetSqlInfo`.
    SqlInfo,
    /// `GetFlightInfo` or `DoGet` of `CommandGetXdbcTypeInfo`.
    XdbcTypeInfo,
    /// `GetFlightInfo` or `DoGet` of a catalog listing or a key command (design §5).
    Metadata,
    /// The `CancelFlightInfo` action.
    CancelFlightInfo,
    /// The deprecated `CancelQuery` action.
    CancelQuery,
    /// `DoPut(CommandStatementUpdate)`.
    Update,
    /// The `CreatePreparedStatement` action.
    CreatePrepared,
    /// The `ClosePreparedStatement` action.
    ClosePrepared,
    /// `GetFlightInfo(CommandPreparedStatementQuery)`.
    PreparedInfo,
    /// `DoPut(CommandPreparedStatementQuery)`: a binding.
    PreparedBind,
    /// `DoPut(CommandPreparedStatementUpdate)`.
    PreparedUpdate,
    /// The `SetSessionOptions` action.
    SetSessionOptions,
    /// The `GetSessionOptions` action.
    GetSessionOptions,
    /// The `CloseSession` action.
    CloseSession,
    /// A command whose `Any` could not be decoded.
    Malformed(&'static str),
    /// Anything else: the RPC, and the command type or action named in the answer.
    Unsupported(String),
}

/// Whether `type_url` is one of the metadata commands of design §5.
fn is_metadata(type_url: &str) -> bool {
    [
        CommandGetCatalogs::type_url(),
        CommandGetDbSchemas::type_url(),
        CommandGetTables::type_url(),
        CommandGetTableTypes::type_url(),
        CommandGetPrimaryKeys::type_url(),
        CommandGetExportedKeys::type_url(),
        CommandGetImportedKeys::type_url(),
        CommandGetCrossReference::type_url(),
    ]
    .contains(&type_url)
}

impl FlightRpc {
    /// A `GetFlightInfo` descriptor's command.
    fn of_descriptor(descriptor: &FlightDescriptor) -> Self {
        match Any::decode(descriptor.cmd.as_ref()) {
            Ok(any) if any.type_url == CommandStatementQuery::type_url() => Self::StatementInfo,
            Ok(any) if any.type_url == CommandGetSqlInfo::type_url() => Self::SqlInfo,
            Ok(any) if any.type_url == CommandGetXdbcTypeInfo::type_url() => Self::XdbcTypeInfo,
            Ok(any) if is_metadata(&any.type_url) => Self::Metadata,
            Ok(any) if any.type_url == CommandPreparedStatementQuery::type_url() => {
                Self::PreparedInfo
            }
            Ok(any) => Self::Unsupported(format!("GetFlightInfo of {:?}", any.type_url)),
            Err(_) => Self::Malformed("a FlightDescriptor command"),
        }
    }

    /// A `DoGet` ticket.
    fn of_ticket(ticket: &Ticket) -> Self {
        match Any::decode(ticket.ticket.as_ref()) {
            Ok(any) if any.type_url == TicketStatementQuery::type_url() => Self::Fetch,
            Ok(any) if any.type_url == CommandGetSqlInfo::type_url() => Self::SqlInfo,
            Ok(any) if any.type_url == CommandGetXdbcTypeInfo::type_url() => Self::XdbcTypeInfo,
            Ok(any) if is_metadata(&any.type_url) => Self::Metadata,
            Ok(any) => Self::Unsupported(format!("DoGet of {:?}", any.type_url)),
            Err(_) => Self::Malformed("a ticket"),
        }
    }

    /// A `DoPut`, from the descriptor of its first message (`None`: the stream sent none).
    fn of_put(first: Option<&FlightData>) -> Self {
        let Some(descriptor) = first.and_then(|d| d.flight_descriptor.as_ref()) else {
            return Self::Malformed("a DoPut without a FlightDescriptor");
        };
        match Any::decode(descriptor.cmd.as_ref()) {
            Ok(any) if any.type_url == CommandStatementUpdate::type_url() => Self::Update,
            Ok(any) if any.type_url == CommandPreparedStatementQuery::type_url() => {
                Self::PreparedBind
            }
            Ok(any) if any.type_url == CommandPreparedStatementUpdate::type_url() => {
                Self::PreparedUpdate
            }
            Ok(any) => Self::Unsupported(format!("DoPut of {:?}", any.type_url)),
            Err(_) => Self::Malformed("a DoPut command"),
        }
    }

    /// A `DoAction` action.
    fn of_action(action: &Action) -> Self {
        match action.r#type.as_str() {
            CANCEL_FLIGHT_INFO => Self::CancelFlightInfo,
            CANCEL_QUERY => Self::CancelQuery,
            CREATE_PREPARED_STATEMENT => Self::CreatePrepared,
            CLOSE_PREPARED_STATEMENT => Self::ClosePrepared,
            SET_SESSION_OPTIONS => Self::SetSessionOptions,
            GET_SESSION_OPTIONS => Self::GetSessionOptions,
            CLOSE_SESSION => Self::CloseSession,
            other => Self::Unsupported(format!("the Flight action {other:?}")),
        }
    }
}

/// `Ok` for what 65a serves; `UNIMPLEMENTED` (or `INVALID_ARGUMENT` for an undecodable command)
/// for the rest, before any session work (design §16 m5).
pub(crate) fn check_flight(rpc: &FlightRpc) -> Result<(), Status> {
    match rpc {
        FlightRpc::Handshake
        | FlightRpc::StatementInfo
        | FlightRpc::Fetch
        | FlightRpc::SqlInfo
        | FlightRpc::XdbcTypeInfo
        | FlightRpc::Metadata
        | FlightRpc::CancelFlightInfo
        | FlightRpc::CancelQuery
        | FlightRpc::Update
        | FlightRpc::CreatePrepared
        | FlightRpc::ClosePrepared
        | FlightRpc::PreparedInfo
        | FlightRpc::PreparedBind
        | FlightRpc::PreparedUpdate
        | FlightRpc::SetSessionOptions
        | FlightRpc::GetSessionOptions
        | FlightRpc::CloseSession => Ok(()),
        FlightRpc::Malformed(what) => Err(Status::invalid_argument(format!(
            "{what} is not a Flight SQL command"
        ))),
        FlightRpc::Unsupported(what) => Err(Status::unimplemented(format!(
            "{what} is not supported by this server"
        ))),
    }
}

/// What `ListActions` answers: the actions the gate lets through, no more (design §4.6, §9; review
/// of 9b66ee1fc, finding 5). arrow-rs's fixed list would name actions 65a refuses.
pub(crate) fn served_actions() -> Vec<ActionType> {
    vec![
        ActionType {
            r#type: CANCEL_FLIGHT_INFO.to_owned(),
            description: "Cancel the running fetches of a FlightInfo's ticket.".to_owned(),
        },
        ActionType {
            r#type: CANCEL_QUERY.to_owned(),
            description: "Cancel a query (deprecated; use CancelFlightInfo).".to_owned(),
        },
        ActionType {
            r#type: CREATE_PREPARED_STATEMENT.to_owned(),
            description: "Create a prepared statement.".to_owned(),
        },
        ActionType {
            r#type: CLOSE_PREPARED_STATEMENT.to_owned(),
            description: "Close a prepared statement.".to_owned(),
        },
        ActionType {
            r#type: SET_SESSION_OPTIONS.to_owned(),
            description: "Set session options (catalog, schema, SQL configuration keys)."
                .to_owned(),
        },
        ActionType {
            r#type: GET_SESSION_OPTIONS.to_owned(),
            description: "Get the session's options.".to_owned(),
        },
        ActionType {
            r#type: CLOSE_SESSION.to_owned(),
            description: "Close the cookie's session.".to_owned(),
        },
    ]
}

/// The tonic `FlightService` in front of [`FlightSqlServer`]'s arrow-rs dispatch.
#[derive(Clone)]
pub struct FlightGate(pub(crate) FlightSqlServer);

impl std::fmt::Debug for FlightGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlightGate").finish_non_exhaustive()
    }
}

type Inner = FlightSqlServer;

impl FlightGate {
    /// [`check_flight`], with a refusal counted as `unsupported`.
    fn gate(&self, rpc: &FlightRpc) -> Result<(), Status> {
        match check_flight(rpc) {
            Ok(()) => Ok(()),
            Err(s) => self.0.counted(Command::Unsupported, Err(s)),
        }
    }

    /// An RPC 65a never serves.
    fn refuse<T>(&self, rpc: &str) -> Result<T, Status> {
        self.0.counted(
            Command::Unsupported,
            Err(Status::unimplemented(format!(
                "{rpc} is not supported by this server"
            ))),
        )
    }
}

#[tonic::async_trait]
impl FlightService for FlightGate {
    type HandshakeStream = <Inner as FlightService>::HandshakeStream;
    type ListFlightsStream = <Inner as FlightService>::ListFlightsStream;
    type DoGetStream = <Inner as FlightService>::DoGetStream;
    type DoPutStream = <Inner as FlightService>::DoPutStream;
    type DoExchangeStream = <Inner as FlightService>::DoExchangeStream;
    type DoActionStream = <Inner as FlightService>::DoActionStream;
    type ListActionsStream = <Inner as FlightService>::ListActionsStream;

    async fn handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        self.gate(&FlightRpc::Handshake)?;
        FlightService::handshake(&self.0, request).await
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        self.refuse("ListFlights")
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.gate(&FlightRpc::of_descriptor(request.get_ref()))?;
        FlightService::get_flight_info(&self.0, request).await
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        self.refuse("PollFlightInfo")
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        self.refuse("GetSchema")
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        self.gate(&FlightRpc::of_ticket(request.get_ref()))?;
        FlightService::do_get(&self.0, request).await
    }

    /// `DoPut` is dispatched here, not by arrow-rs (whose dispatch takes the stream and decodes
    /// the command before any gate could see it): the first message is peeked and classified, a
    /// refused command answers before any session work, and the served ones go to their handlers
    /// with the stream, its first message still unread (design §4.3, §4.4, §16 m5).
    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let (metadata, extensions, stream) = request.into_parts();
        let mut stream: Peekable<Streaming<FlightData>> = stream.peekable();
        let first = match std::pin::Pin::new(&mut stream).peek().await {
            Some(Ok(data)) => Some(data.clone()),
            Some(Err(s)) => return Err(s.clone()),
            None => None,
        };
        let rpc = FlightRpc::of_put(first.as_ref());
        self.gate(&rpc)?;
        let request = Request::from_parts(metadata, extensions, ());
        let command = first
            .and_then(|d| d.flight_descriptor)
            .and_then(|d| Any::decode(d.cmd.as_ref()).ok());
        match rpc {
            FlightRpc::Update => {
                let update = command
                    .and_then(|any| any.unpack::<CommandStatementUpdate>().ok().flatten())
                    .ok_or_else(|| Status::invalid_argument("a malformed CommandStatementUpdate"));
                let result = match update {
                    Ok(update) => self.0.do_put_update(update, &request).await,
                    Err(s) => Err(s),
                };
                self.0.counted(Command::StatementUpdate, result)
            }
            FlightRpc::PreparedBind => {
                let bind = command
                    .and_then(|any| any.unpack::<CommandPreparedStatementQuery>().ok().flatten())
                    .ok_or_else(|| {
                        Status::invalid_argument("a malformed CommandPreparedStatementQuery")
                    });
                let result = match bind {
                    Ok(bind) => self.0.do_put_bind(bind, stream, &request).await,
                    Err(s) => Err(s),
                };
                self.0.counted(Command::PreparedQuery, result)
            }
            FlightRpc::PreparedUpdate => {
                let update = command
                    .and_then(|any| {
                        any.unpack::<CommandPreparedStatementUpdate>()
                            .ok()
                            .flatten()
                    })
                    .ok_or_else(|| {
                        Status::invalid_argument("a malformed CommandPreparedStatementUpdate")
                    });
                let result = match update {
                    Ok(update) => {
                        self.0
                            .do_put_prepared_update(update, stream, &request)
                            .await
                    }
                    Err(s) => Err(s),
                };
                self.0.counted(Command::PreparedUpdate, result)
            }
            _ => self.refuse("DoPut"),
        }
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        self.refuse("DoExchange")
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let rpc = FlightRpc::of_action(request.get_ref());
        self.gate(&rpc)?;
        match rpc {
            // Dispatched here, not by arrow-rs, whose answer could not carry the cookie of a
            // session the create made (design §3.2).
            FlightRpc::CreatePrepared => {
                let create = Any::decode(request.get_ref().body.as_ref())
                    .ok()
                    .and_then(|any| {
                        any.unpack::<ActionCreatePreparedStatementRequest>()
                            .ok()
                            .flatten()
                    })
                    .ok_or_else(|| {
                        Status::invalid_argument("a malformed ActionCreatePreparedStatementRequest")
                    });
                let result = match create {
                    Ok(create) => self.0.do_action_create(create, &request).await,
                    Err(s) => Err(s),
                };
                self.0.counted(Command::PreparedCreate, result)
            }
            FlightRpc::ClosePrepared => {
                let close = Any::decode(request.get_ref().body.as_ref())
                    .ok()
                    .and_then(|any| {
                        any.unpack::<ActionClosePreparedStatementRequest>()
                            .ok()
                            .flatten()
                    })
                    .ok_or_else(|| {
                        Status::invalid_argument("a malformed ActionClosePreparedStatementRequest")
                    });
                let result = close.and_then(|close| self.0.do_action_close(&close, &request));
                self.0.counted(Command::PreparedClose, result)
            }
            // Not in arrow-rs's trait at all (facts A1): their bodies are plain messages.
            FlightRpc::SetSessionOptions => {
                let result = match SetSessionOptionsRequest::decode(request.get_ref().body.as_ref())
                {
                    Ok(set) => self.0.do_action_set_options(set, &request).await,
                    Err(_) => Err(Status::invalid_argument(
                        "a malformed SetSessionOptionsRequest",
                    )),
                };
                self.0.counted(Command::SessionOptions, result)
            }
            FlightRpc::GetSessionOptions => {
                let result = self.0.do_action_get_options(&request).await;
                self.0.counted(Command::SessionOptions, result)
            }
            FlightRpc::CloseSession => {
                let result = self.0.do_action_close_session(&request);
                self.0.counted(Command::SessionClose, result)
            }
            _ => FlightService::do_action(&self.0, request).await,
        }
    }

    /// Served without a session, for every role (§9: arrow-rs's blanket impl needs none either),
    /// and while the server drains: it starts no work (design P3-T066 §5.1).
    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = futures::stream::iter(served_actions().into_iter().map(Ok));
        let stream: Self::ListActionsStream = Box::pin(actions);
        self.0
            .counted(Command::ListActions, Ok(Response::new(stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn any_of<M: ProstMessageExt>(m: &M) -> Vec<u8> {
        m.as_any().encode_to_vec()
    }

    #[test]
    fn served_commands_pass_the_check() {
        let query = FlightDescriptor::new_cmd(any_of(&CommandStatementQuery::default()));
        assert_eq!(FlightRpc::of_descriptor(&query), FlightRpc::StatementInfo);
        let info = FlightDescriptor::new_cmd(any_of(&CommandGetSqlInfo::default()));
        assert_eq!(FlightRpc::of_descriptor(&info), FlightRpc::SqlInfo);
        let fetch = Ticket::new(any_of(&TicketStatementQuery::default()));
        assert_eq!(FlightRpc::of_ticket(&fetch), FlightRpc::Fetch);
        let types = CommandGetXdbcTypeInfo::default();
        let types_info = FlightDescriptor::new_cmd(any_of(&types));
        assert_eq!(
            FlightRpc::of_descriptor(&types_info),
            FlightRpc::XdbcTypeInfo
        );
        let types_ticket = Ticket::new(any_of(&types));
        assert_eq!(FlightRpc::of_ticket(&types_ticket), FlightRpc::XdbcTypeInfo);
        for rpc in [
            FlightRpc::Handshake,
            FlightRpc::StatementInfo,
            FlightRpc::Fetch,
            FlightRpc::SqlInfo,
            FlightRpc::XdbcTypeInfo,
            FlightRpc::CancelFlightInfo,
            FlightRpc::CancelQuery,
        ] {
            assert!(check_flight(&rpc).is_ok(), "{rpc:?}");
        }
    }

    #[test]
    fn unknown_commands_and_actions_are_unimplemented() {
        let catalogs = FlightDescriptor::new_cmd(any_of(&arrow_flight::sql::CommandGetCatalogs {}));
        assert_eq!(FlightRpc::of_descriptor(&catalogs), FlightRpc::Metadata);
        let substrait = FlightDescriptor::new_cmd(any_of(
            &arrow_flight::sql::CommandStatementSubstraitPlan::default(),
        ));
        let rpc = FlightRpc::of_descriptor(&substrait);
        assert!(matches!(rpc, FlightRpc::Unsupported(_)), "{rpc:?}");
        assert_eq!(
            check_flight(&rpc).map_err(|s| s.code()),
            Err(tonic::Code::Unimplemented)
        );

        // arrow-rs decodes these bodies before its own UNIMPLEMENTED default; the gate does not.
        for kind in [
            "BeginTransaction",
            "EndTransaction",
            "CreatePreparedSubstraitPlan",
        ] {
            let rpc = FlightRpc::of_action(&Action::new(kind, b"not a protobuf".to_vec()));
            assert_eq!(
                check_flight(&rpc).map_err(|s| s.code()),
                Err(tonic::Code::Unimplemented)
            );
        }
        assert_eq!(
            FlightRpc::of_action(&Action::new(CANCEL_QUERY, Vec::new())),
            FlightRpc::CancelQuery
        );
    }

    /// Every action `ListActions` names passes the gate.
    #[test]
    fn listed_actions_are_the_served_ones() {
        let listed: Vec<String> = served_actions().into_iter().map(|a| a.r#type).collect();
        assert_eq!(
            listed,
            [
                CANCEL_FLIGHT_INFO,
                CANCEL_QUERY,
                CREATE_PREPARED_STATEMENT,
                CLOSE_PREPARED_STATEMENT,
                SET_SESSION_OPTIONS,
                GET_SESSION_OPTIONS,
                CLOSE_SESSION
            ]
        );
        for kind in &listed {
            let rpc = FlightRpc::of_action(&Action::new(kind.clone(), Vec::new()));
            assert!(check_flight(&rpc).is_ok(), "{kind}");
        }
    }

    #[test]
    fn doput_is_classified_by_its_first_message() {
        let put = |m: Vec<u8>| FlightData::new().with_descriptor(FlightDescriptor::new_cmd(m));
        let update = put(any_of(&CommandStatementUpdate::default()));
        assert_eq!(FlightRpc::of_put(Some(&update)), FlightRpc::Update);
        let bind = put(any_of(&CommandPreparedStatementQuery::default()));
        assert_eq!(FlightRpc::of_put(Some(&bind)), FlightRpc::PreparedBind);
        let prepared = put(any_of(&CommandPreparedStatementUpdate::default()));
        assert_eq!(
            FlightRpc::of_put(Some(&prepared)),
            FlightRpc::PreparedUpdate
        );
        let info = FlightDescriptor::new_cmd(any_of(&CommandPreparedStatementQuery::default()));
        assert_eq!(FlightRpc::of_descriptor(&info), FlightRpc::PreparedInfo);
        let ingest = put(any_of(&arrow_flight::sql::CommandStatementIngest::default()));
        assert!(matches!(
            FlightRpc::of_put(Some(&ingest)),
            FlightRpc::Unsupported(_)
        ));
        assert!(matches!(
            FlightRpc::of_put(Some(&FlightData::new())),
            FlightRpc::Malformed(_)
        ));
        assert!(matches!(FlightRpc::of_put(None), FlightRpc::Malformed(_)));
        assert!(matches!(
            FlightRpc::of_put(Some(&put(vec![0xff, 0xff]))),
            FlightRpc::Malformed(_)
        ));
    }

    #[test]
    fn an_undecodable_command_is_invalid_argument() {
        let rpc = FlightRpc::of_ticket(&Ticket::new(vec![0xff, 0xff, 0xff]));
        assert_eq!(rpc, FlightRpc::Malformed("a ticket"));
        assert_eq!(
            check_flight(&rpc).map_err(|s| s.code()),
            Err(tonic::Code::InvalidArgument)
        );
    }
}
