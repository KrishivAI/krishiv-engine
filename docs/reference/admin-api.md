---
description: "The krishiv.admin.v1 control API of krishiv server: inspect and end sessions and queries, drain the server, and manage resource groups over gRPC or REST."
---

# Admin API (`krishiv.admin.v1`)

Stability: stable since 0.3. Changes to `krishiv.admin.v1` are additive (new methods, new fields); a breaking change
ships as `krishiv.admin.v2` next to it.

`krishiv server` serves the admin API on its admin listener (port `8080` by default), over gRPC and over REST under
`/api/admin/v1`. With authentication configured, every call needs a token, and the caller's role decides what it may do
(viewer, operator or admin). See [Securing a single-node server](../guides/single-node-server-security.md).

| Method | REST | What it does |
|---|---|---|
| `GetVersion` | `GET /version` | the server version and the API revision |
| `GetCapabilities` | `GET /capabilities` | the same rows as `system.capabilities` ([System tables](system-tables.md)) |
| `ListSessions` | `GET /sessions` | the open client sessions |
| `KillSession` | `POST /sessions/{session_id}:kill` | end a session and cancel its queries |
| `ListQueries` | `GET /queries` | running and recent queries |
| `CancelQuery` | `POST /queries/{query_id}:cancel` | cancel one query |
| `CancelQueriesByTag` | `POST /queries:cancelByTag` | cancel every query that carries a tag |
| `GetReadiness` | `GET /readiness` | whether the server accepts new queries |
| `GetDrainStatus`, `Drain`, `ResumeServing` | `GET`, `POST`, `DELETE /drain` | stop taking new queries, wait for running ones, resume |
| `ListResourceGroups` | `GET /resourceGroups` | the resource groups and their current load |
| `GetResourceGroupConfig`, `ValidateResourceGroupConfig`, `ReplaceResourceGroupConfig`, `ReloadResourceGroupConfig` | `/resourceGroups/config` | read, check, replace or reload the resource-group file ([Resource groups](../guides/resource-groups.md)) |
| `GetConfig`, `SetConfigDefaults` | `GET /config`, `PATCH /config/defaults` | read the server's settings and change session defaults |

The machine-readable definitions are `descriptor.json` (the gRPC service), `openapi.json` and `routes.json` in
`crates/krishiv-server/admin-v1/` of the repository. The API revision that a server speaks is in its
`system.capabilities` rows.
