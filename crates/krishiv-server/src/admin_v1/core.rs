// SPDX-License-Identifier: Apache-2.0
//! What both transports share: decode a request, run it through the idempotency store, call the
//! [`AdminApi`] and encode the response. REST and gRPC are thin skins over [`Core::call`], so they
//! answer the same.

use std::sync::Arc;
use std::time::Duration;

use prost::Message;

use super::action::{self, Action};
use super::idempotency::{Idempotency, MAX_ID_LEN, Run};
use super::{
    AdminApi, AdminError, AdminResult, Call, CancelQueriesByTagRequest, CancelQueryRequest,
    DrainRequest, ErrorKind, GetCapabilitiesRequest, GetConfigRequest, GetDrainStatusRequest,
    GetReadinessRequest, GetResourceGroupConfigRequest, GetVersionRequest, KillSessionRequest,
    ListQueriesRequest, ListResourceGroupsRequest, ListSessionsRequest,
    ReloadResourceGroupConfigRequest, ReplaceResourceGroupConfigRequest, ResumeServingRequest,
    SetConfigDefaultsRequest, ValidateResourceGroupConfigRequest,
};

/// The RPCs that change something: they take a `request_id`, are remembered by the idempotency
/// store and write an action record.
const MUTATING: &[&str] = &[
    "KillSession",
    "CancelQuery",
    "CancelQueriesByTag",
    "Drain",
    "ResumeServing",
    "ReplaceResourceGroupConfig",
    "ReloadResourceGroupConfig",
    "SetConfigDefaults",
];

pub(super) struct Core {
    api: Arc<dyn AdminApi>,
    idempotency: Arc<Idempotency>,
}

impl Core {
    pub(super) fn new(api: Arc<dyn AdminApi>, ttl: Duration) -> Self {
        Self {
            api,
            idempotency: Arc::new(Idempotency::new(ttl)),
        }
    }

    /// Whether `rpc` changes something.
    pub(super) fn is_mutating(rpc: &str) -> bool {
        MUTATING.contains(&rpc)
    }

    /// Runs `rpc` on an encoded request. `request_id` is the idempotency key the REST header
    /// carried, used when the message has none; a call with neither gets `generated`.
    pub(super) async fn call(
        &self,
        rpc: &str,
        mut call: Call,
        request: &[u8],
        header_id: Option<&str>,
    ) -> AdminResult<Vec<u8>> {
        if !Self::is_mutating(rpc) {
            return Self::invoke(self.api.as_ref(), rpc, &call, request).await;
        }
        let refuse = |call: &Call, error: AdminError| {
            // A refused mutation still leaves its action record.
            action::record(&Action {
                principal: &call.caller.principal,
                role: call.caller.role.as_deref().unwrap_or("none"),
                auth_method: &call.caller.method,
                rpc,
                transport: call.transport,
                target: "",
                own: false,
                request_id: &call.request_id,
                result: "failed",
                error_class: &error.error_class,
                replay: false,
            });
            error
        };
        let message_id = request_id_in(rpc, request).map_err(|e| refuse(&call, e))?;
        let sent = if message_id.is_empty() {
            header_id.unwrap_or_default().to_owned()
        } else {
            message_id
        };
        if sent.len() > MAX_ID_LEN || !sent.is_ascii() {
            return Err(refuse(
                &call,
                AdminError::invalid_argument(format!(
                    "the request id must be ASCII and at most {MAX_ID_LEN} bytes"
                )),
            ));
        }
        if !sent.is_empty() {
            sent.clone_into(&mut call.request_id);
        }
        // What makes two calls "the same request": the RPC and the message without its id.
        let mut fingerprint = rpc.as_bytes().to_vec();
        fingerprint
            .extend_from_slice(&without_request_id(rpc, request).map_err(|e| refuse(&call, e))?);
        // The mutation and the storing of its result run on their own task: a request dropped
        // mid-flight (a client disconnect) must not leave the work done with no result stored, or
        // a retry with the same id would run it again.
        let api = Arc::clone(&self.api);
        let idempotency = Arc::clone(&self.idempotency);
        let scope = api.idempotency_scope(&call);
        let (rpc_owned, request_owned, call_owned) =
            (rpc.to_owned(), request.to_vec(), call.clone());
        let sent_owned = sent.clone();
        let task = tokio::spawn(async move {
            idempotency
                .run(&scope, &sent_owned, &fingerprint, || {
                    Self::invoke(api.as_ref(), &rpc_owned, &call_owned, &request_owned)
                })
                .await
        });
        let (result, how) = task.await.unwrap_or_else(|_| {
            (
                Err(AdminError::new(
                    ErrorKind::Internal,
                    "INTERNAL_ERROR",
                    "XX000",
                    "The request could not be completed.",
                )),
                Run::Ran,
            )
        });
        if how != Run::Ran {
            let class = result.as_ref().err().map_or("", |e| e.error_class.as_str());
            let outcome = match &result {
                Ok(_) => "ok",
                Err(e) if e.denied => "denied",
                Err(_) => "failed",
            };
            action::record(&Action {
                principal: &call.caller.principal,
                role: call.caller.role.as_deref().unwrap_or("none"),
                auth_method: &call.caller.method,
                rpc,
                transport: call.transport,
                target: "",
                own: false,
                request_id: &call.request_id,
                result: outcome,
                error_class: class,
                // A refused reuse is not a replay: nothing was returned from the store.
                replay: how == Run::Replayed,
            });
        }
        result
    }

    async fn invoke(
        api: &dyn AdminApi,
        rpc: &str,
        call: &Call,
        request: &[u8],
    ) -> AdminResult<Vec<u8>> {
        let bad = |e: prost::DecodeError| AdminError::invalid_argument(e.to_string());
        match rpc {
            "GetVersion" => {
                GetVersionRequest::decode(request).map_err(bad)?;
                Ok(api.get_version(call).await?.encode_to_vec())
            }
            "GetCapabilities" => {
                GetCapabilitiesRequest::decode(request).map_err(bad)?;
                Ok(api.get_capabilities(call).await?.encode_to_vec())
            }
            "ListSessions" => {
                let r = ListSessionsRequest::decode(request).map_err(bad)?;
                Ok(api.list_sessions(call, r).await?.encode_to_vec())
            }
            "KillSession" => {
                let r = KillSessionRequest::decode(request).map_err(bad)?;
                Ok(api.kill_session(call, r).await?.encode_to_vec())
            }
            "ListQueries" => {
                let r = ListQueriesRequest::decode(request).map_err(bad)?;
                Ok(api.list_queries(call, r).await?.encode_to_vec())
            }
            "CancelQuery" => {
                let r = CancelQueryRequest::decode(request).map_err(bad)?;
                Ok(api.cancel_query(call, r).await?.encode_to_vec())
            }
            "CancelQueriesByTag" => {
                let r = CancelQueriesByTagRequest::decode(request).map_err(bad)?;
                Ok(api.cancel_queries_by_tag(call, r).await?.encode_to_vec())
            }
            "GetReadiness" => {
                GetReadinessRequest::decode(request).map_err(bad)?;
                Ok(api.get_readiness(call).await?.encode_to_vec())
            }
            "GetDrainStatus" => {
                GetDrainStatusRequest::decode(request).map_err(bad)?;
                Ok(api.get_drain_status(call).await?.encode_to_vec())
            }
            "Drain" => {
                let r = DrainRequest::decode(request).map_err(bad)?;
                Ok(api.drain(call, r).await?.encode_to_vec())
            }
            "ResumeServing" => {
                let r = ResumeServingRequest::decode(request).map_err(bad)?;
                Ok(api.resume_serving(call, r).await?.encode_to_vec())
            }
            "ListResourceGroups" => {
                let r = ListResourceGroupsRequest::decode(request).map_err(bad)?;
                Ok(api.list_resource_groups(call, r).await?.encode_to_vec())
            }
            "GetResourceGroupConfig" => {
                let r = GetResourceGroupConfigRequest::decode(request).map_err(bad)?;
                Ok(api
                    .get_resource_group_config(call, r)
                    .await?
                    .encode_to_vec())
            }
            "ValidateResourceGroupConfig" => {
                let r = ValidateResourceGroupConfigRequest::decode(request).map_err(bad)?;
                Ok(api
                    .validate_resource_group_config(call, r)
                    .await?
                    .encode_to_vec())
            }
            "ReplaceResourceGroupConfig" => {
                let r = ReplaceResourceGroupConfigRequest::decode(request).map_err(bad)?;
                Ok(api
                    .replace_resource_group_config(call, r)
                    .await?
                    .encode_to_vec())
            }
            "ReloadResourceGroupConfig" => {
                let r = ReloadResourceGroupConfigRequest::decode(request).map_err(bad)?;
                Ok(api
                    .reload_resource_group_config(call, r)
                    .await?
                    .encode_to_vec())
            }
            "GetConfig" => {
                let r = GetConfigRequest::decode(request).map_err(bad)?;
                Ok(api.get_config(call, r).await?.encode_to_vec())
            }
            "SetConfigDefaults" => {
                let r = SetConfigDefaultsRequest::decode(request).map_err(bad)?;
                Ok(api.set_config_defaults(call, r).await?.encode_to_vec())
            }
            other => Err(AdminError::internal(format!("no handler for {other}"))),
        }
    }
}

/// The `request_id` field of a mutating request.
fn request_id_in(rpc: &str, request: &[u8]) -> AdminResult<String> {
    let bad = |e: prost::DecodeError| AdminError::invalid_argument(e.to_string());
    Ok(match rpc {
        "KillSession" => KillSessionRequest::decode(request).map_err(bad)?.request_id,
        "CancelQuery" => CancelQueryRequest::decode(request).map_err(bad)?.request_id,
        "CancelQueriesByTag" => {
            CancelQueriesByTagRequest::decode(request)
                .map_err(bad)?
                .request_id
        }
        "Drain" => DrainRequest::decode(request).map_err(bad)?.request_id,
        "ResumeServing" => {
            ResumeServingRequest::decode(request)
                .map_err(bad)?
                .request_id
        }
        "ReplaceResourceGroupConfig" => {
            ReplaceResourceGroupConfigRequest::decode(request)
                .map_err(bad)?
                .request_id
        }
        "ReloadResourceGroupConfig" => {
            ReloadResourceGroupConfigRequest::decode(request)
                .map_err(bad)?
                .request_id
        }
        "SetConfigDefaults" => {
            SetConfigDefaultsRequest::decode(request)
                .map_err(bad)?
                .request_id
        }
        _ => String::new(),
    })
}

/// The request with its `request_id` cleared, re-encoded: two requests that differ only in the id
/// are the same request.
fn without_request_id(rpc: &str, request: &[u8]) -> AdminResult<Vec<u8>> {
    let bad = |e: prost::DecodeError| AdminError::invalid_argument(e.to_string());
    Ok(match rpc {
        "KillSession" => {
            let mut m = KillSessionRequest::decode(request).map_err(bad)?;
            m.request_id.clear();
            m.encode_to_vec()
        }
        "CancelQuery" => {
            let mut m = CancelQueryRequest::decode(request).map_err(bad)?;
            m.request_id.clear();
            m.encode_to_vec()
        }
        "CancelQueriesByTag" => {
            let mut m = CancelQueriesByTagRequest::decode(request).map_err(bad)?;
            m.request_id.clear();
            m.encode_to_vec()
        }
        "Drain" => {
            let mut m = DrainRequest::decode(request).map_err(bad)?;
            m.request_id.clear();
            m.encode_to_vec()
        }
        "ResumeServing" => {
            let mut m = ResumeServingRequest::decode(request).map_err(bad)?;
            m.request_id.clear();
            m.encode_to_vec()
        }
        "ReplaceResourceGroupConfig" => {
            let mut m = ReplaceResourceGroupConfigRequest::decode(request).map_err(bad)?;
            m.request_id.clear();
            m.encode_to_vec()
        }
        "ReloadResourceGroupConfig" => {
            let mut m = ReloadResourceGroupConfigRequest::decode(request).map_err(bad)?;
            m.request_id.clear();
            m.encode_to_vec()
        }
        "SetConfigDefaults" => {
            let mut m = SetConfigDefaultsRequest::decode(request).map_err(bad)?;
            m.request_id.clear();
            m.encode_to_vec()
        }
        _ => request.to_vec(),
    })
}
