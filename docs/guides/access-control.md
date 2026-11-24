---
description: "Who may read or change what: the role matrix and the external policy service of `krishiv server`, and the grants provider (GRANT and REVOKE) of a session built with it."
---

# Access control with grants

Who can read or change what is decided by an **access-policy provider**. Which providers you can use depends on
how Krishiv runs.

> **`krishiv server` does not apply `GRANT` and `REVOKE` yet.** On a server, the role matrix or an external policy
> service decides. The grants provider described from [Grant and revoke](#grant-and-revoke) on works only in a
> session that a Rust program builds with it. On a server, `GRANT`, `REVOKE`, `SHOW GRANTS` and the role statements
> are refused: the error class is `KRISHIV_UNSUPPORTED_FEATURE`, unless the role matrix refuses the statement first.

## On `krishiv server`

The provider is set by `krishiv.access.provider` (startup only):

| Value | Decides |
|---|---|
| unset or `roles` | everything, by the role matrix (`viewer`, `operator`, `admin`) |
| `external` | data objects (tables, views, columns, namespaces, functions, paths, secrets, connections), by the policy service at `krishiv.access.external.endpoint`; the role matrix still decides the server's own objects, so a `viewer` may write when the service allows it |
| `roles+external` | data objects, by the role matrix and the policy service together: both must allow |
| `grants`, or a list that names it | nothing: the server refuses to start |
| `catalog` | nothing: refused until it is available |

Under the role matrix a `viewer` sees metadata and its own rows of `system.queries` and `system.sessions`, an
`operator` also reads tables, views, functions and paths, and only an `admin` writes, creates or drops.
[Securing a single-node server](single-node-server-security.md) describes the tokens and their roles; the
`krishiv.access.*` rows of the [configuration reference](../reference/config/index.md) describe the policy service's
settings.

A server without authentication has no access policy at all: it allows everything, and it refuses to start with
`external` or `grants`. Configure authentication before you rely on a provider.

On a server with authentication, path confinement (`krishiv.io.local.allowedRoots`,
`krishiv.io.objectStore.allowedPrefixes`) applies first under every provider; a server without authentication does not
confine paths.

## In a session with the grants provider

A session opened in Python or with `Session::builder()` has no access policy by default: it allows everything. A Rust
program gets the grants provider by building its session with `SessionBuilder::grants` and the same policy as
`SessionBuilder::access_policy` (crate `krishiv-session`, module `grants`). In such a session `GRANT` and `REVOKE`
decide data objects (tables, views, columns, namespaces, functions, paths, secrets, connections), the policy the
provider was built over (the role matrix, for example) decides everything else, and an admin passes every data check.
The rest of this page describes that provider.

## Grant and revoke

```sql
GRANT SELECT ON TABLE sales.orders TO user:ana;
GRANT MODIFY ON sales.orders TO user:bob;
GRANT CREATE ON SCHEMA sales TO user:ana;
GRANT SELECT ON FUNCTION sales.mask TO user:ana;
GRANT ALL PRIVILEGES ON PATH 's3://bucket/landing/' TO user:ana;
REVOKE SELECT ON TABLE sales.orders FROM user:ana;
```

A principal is `user:<name>`, `group:<name>` or `role:<name>`. Roles group privileges:

```sql
CREATE ROLE analysts;
GRANT SELECT ON TABLE sales.orders TO role:analysts;
GRANT ROLE analysts TO user:ana;
```

`WITH GRANT OPTION` lets the grantee pass a privilege on; `REVOKE GRANT OPTION FOR` takes that right away and keeps
the privilege. Revoking it on a table clears the flag on the table's columns too. The grant option alone does not let a principal
issue `GRANT`: when the provider is built over the role matrix, the matrix must also allow the statement, so a
`viewer` who holds a privilege with grant option cannot pass it on.

`SHOW GRANTS` (a Krishiv extension, 7 columns) lists grants; `SHOW GRANTS ON TABLE t` narrows it to one object. Without the `Monitor` privilege both return only the caller's own
effective grants, so an empty result is not an audit that nobody else has access. A listing of tables, views or functions shows only the ones the caller may see.

## Views and functions

A view runs with its **invoker's** rights by default. `SQL SECURITY DEFINER` makes it run with its definer's rights, so
a view can expose some columns or rows of a table its readers cannot read:

```sql
CREATE OR REPLACE VIEW sales.my_orders SQL SECURITY DEFINER AS
SELECT id, total FROM sales.orders WHERE owner = 'ana';
GRANT SELECT ON VIEW sales.my_orders TO user:ana;
```

The definer is whoever last created or altered the view: `ALTER VIEW ... AS` makes the alterer the definer, even when
another principal owns the view. Grant and revoke on that principal.

Dropping a view (`DROP VIEW`) needs the `DROP` privilege on it, as `DROP FUNCTION` does on a function.

## Check it

Run the statement as the principal; a denied one fails with `KRISHIV_INSUFFICIENT_PERMISSIONS`. For the server's
authentication settings see [Securing a single-node server](single-node-server-security.md).
