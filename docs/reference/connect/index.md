---
description: "How much of the Spark Connect protocol Krishiv serves, arm by arm, for the vendored protocol version."
---

# Spark Connect

Spark Connect lets a Spark client talk to an engine over gRPC. Krishiv serves a subset of the protocol today: the
[coverage page](coverage.md) lists every relation, command and expression the translator dispatches on and whether
Krishiv handles it, partly handles it, does not handle it yet or leaves it out of scope. A request that uses an arm
Krishiv does not handle fails with `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SUPPORTED`.
