# krishiv-access-proto

The `krishiv.access.v1` protobuf messages of Krishiv's external access-policy service (ADR 0049). A
policy service (an OPA, Ranger or Cedar adapter, or your own) implements `AccessPolicy.Check` and
`AccessPolicy.Version`; Krishiv's `external` provider (`krishiv.access.provider = external`) calls it.
Messages only; the gRPC stubs are in `krishiv-server` (feature `access-external`).
