// SPDX-License-Identifier: Apache-2.0
//! The fail-closed front of the Flight service (design P3-T101 §4.6, §16 m5): every gRPC call is
//! classified as a [`FlightRpc`] from its method and its command's `Any` type URL (or its action
//! type) before arrow-rs decodes a body, and [`check_flight`] answers whatever 65a does not serve
//! `UNIMPLEMENTED`, counted as `unsupported`, before any authentication or session work. The
//! match in `check_flight` has no wildcard, so a new variant must be decided there.

use arrow_flight::flight_service_server::FlightService;
use arrow_flight::sql::{
    Any, CommandGetSqlInfo, CommandStatementQuery, ProstMessageExt, TicketStatementQuery,
};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, PollInfo, SchemaResult, Ticket,
};
use prost::Message;
use tonic::{Request, Response, Status, Streaming};

use crate::metrics::Command;
use crate::server::FlightSqlServer;

/// The Flight action `CancelFlightInfo`.
pub(crate) const CANCEL_FLIGHT_INFO: &str = "CancelFlightInfo";
/// The deprecated Flight SQL action `CancelQuery` (design §4.5).
pub(crate) const CANCEL_QUERY: &str = "CancelQuery";

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
    /// The `CancelFlightInfo` action.
    CancelFlightInfo,
    /// The deprecated `CancelQuery` action.
    CancelQuery,
    /// A command whose `Any` could not be decoded.
    Malformed(&'static str),
    /// Anything else: the RPC, and the command type or action named in the answer.
    Unsupported(String),
}

impl FlightRpc {
    /// A `GetFlightInfo` descriptor's command.
    fn of_descriptor(descriptor: &FlightDescriptor) -> Self {
        match Any::decode(descriptor.cmd.as_ref()) {
            Ok(any) if any.type_url == CommandStatementQuery::type_url() => Self::StatementInfo,
            Ok(any) if any.type_url == CommandGetSqlInfo::type_url() => Self::SqlInfo,
            Ok(any) => Self::Unsupported(format!("GetFlightInfo of {:?}", any.type_url)),
            Err(_) => Self::Malformed("a FlightDescriptor command"),
        }
    }

    /// A `DoGet` ticket.
    fn of_ticket(ticket: &Ticket) -> Self {
        match Any::decode(ticket.ticket.as_ref()) {
            Ok(any) if any.type_url == TicketStatementQuery::type_url() => Self::Fetch,
            Ok(any) if any.type_url == CommandGetSqlInfo::type_url() => Self::SqlInfo,
            Ok(any) => Self::Unsupported(format!("DoGet of {:?}", any.type_url)),
            Err(_) => Self::Malformed("a ticket"),
        }
    }

    /// A `DoAction` action.
    fn of_action(action: &Action) -> Self {
        match action.r#type.as_str() {
            CANCEL_FLIGHT_INFO => Self::CancelFlightInfo,
            CANCEL_QUERY => Self::CancelQuery,
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
        | FlightRpc::CancelFlightInfo
        | FlightRpc::CancelQuery => Ok(()),
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

    /// `DoPut` (updates, prepared binds, ingest) is 65b (design §4.3).
    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        self.refuse("DoPut")
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
        self.gate(&FlightRpc::of_action(request.get_ref()))?;
        FlightService::do_action(&self.0, request).await
    }

    /// Served without a session, for every role (§9: arrow-rs's blanket impl needs none either).
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
        for rpc in [
            FlightRpc::Handshake,
            FlightRpc::StatementInfo,
            FlightRpc::Fetch,
            FlightRpc::SqlInfo,
            FlightRpc::CancelFlightInfo,
            FlightRpc::CancelQuery,
        ] {
            assert!(check_flight(&rpc).is_ok(), "{rpc:?}");
        }
    }

    #[test]
    fn unknown_commands_and_actions_are_unimplemented() {
        let catalogs = FlightDescriptor::new_cmd(any_of(&arrow_flight::sql::CommandGetCatalogs {}));
        let rpc = FlightRpc::of_descriptor(&catalogs);
        assert!(matches!(rpc, FlightRpc::Unsupported(_)), "{rpc:?}");
        assert_eq!(
            check_flight(&rpc).map_err(|s| s.code()),
            Err(tonic::Code::Unimplemented)
        );

        // arrow-rs decodes these bodies before its own UNIMPLEMENTED default; the gate does not.
        for kind in [
            "ClosePreparedStatement",
            "BeginTransaction",
            "CreatePreparedStatement",
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
        assert_eq!(listed, [CANCEL_FLIGHT_INFO, CANCEL_QUERY]);
        for kind in &listed {
            let rpc = FlightRpc::of_action(&Action::new(kind.clone(), Vec::new()));
            assert!(check_flight(&rpc).is_ok(), "{kind}");
        }
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
