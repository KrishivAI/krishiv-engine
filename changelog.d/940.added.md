Flight SQL queries and updates, and statements started through the server's HTTP SQL route, join the W3C trace in the request's `traceparent` and `baggage` headers, shown in `system.queries.trace_id`.

A session says who its client is with `SessionBuilder::source`, and a resource group's `source` and `queryType` selectors now see the session's client type and the statement's class (P3-T003).

`krishiv.admin.v1` lists, reads, validates, replaces and reloads the resource-group configuration, and reads and sets run-time config defaults, over gRPC and REST (P3-T066 PR-52c). A duration in the OpenAPI document is now a ProtoJSON string such as `"5s"`.
