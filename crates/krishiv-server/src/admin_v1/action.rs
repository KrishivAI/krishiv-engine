// SPDX-License-Identifier: Apache-2.0
//! The admin action log (design §5.6): one structured record at `info` on the target
//! `krishiv::admin_action` for every mutating call, allowed or refused. The target is exempt from
//! the server's level filter by being the operator's record of who changed what; a replay of a
//! stored response writes a record with `replay = true`.

/// What a mutating call did, for [`record`].
#[derive(Debug, Clone)]
pub struct Action<'a> {
    /// The caller's principal name.
    pub principal: &'a str,
    /// The caller's role, if any.
    pub role: &'a str,
    /// How the caller authenticated.
    pub auth_method: &'a str,
    /// The RPC (`CancelQuery`).
    pub rpc: &'a str,
    /// `grpc`, `rest` or `ui`.
    pub transport: &'a str,
    /// The target: session or query ids, the tag selector and the ids it matched.
    pub target: &'a str,
    /// Whether the target was the caller's own work.
    pub own: bool,
    /// The idempotency key, or the generated request id.
    pub request_id: &'a str,
    /// `ok`, `denied` or `failed`.
    pub result: &'a str,
    /// The error class of a refused or failed call (empty when `ok`).
    pub error_class: &'a str,
    /// A replay of a stored response.
    pub replay: bool,
}

/// Writes the record. Never takes a secret: every field is an id, a name or a class.
pub fn record(a: &Action<'_>) {
    tracing::info!(
        target: "krishiv::admin_action",
        principal = a.principal,
        role = a.role,
        auth_method = a.auth_method,
        rpc = a.rpc,
        transport = a.transport,
        target_ids = a.target,
        own = a.own,
        request_id = a.request_id,
        result = a.result,
        error_class = a.error_class,
        replay = a.replay,
        "admin action"
    );
}
