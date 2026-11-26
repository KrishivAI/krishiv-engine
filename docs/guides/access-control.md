---
description: "Who may read or change what in server mode (token roles and an external policy service), and GRANT in a library session."
---

# Access control

In server mode, who can read or change what is decided by an **access-policy provider**
(`krishiv.access.provider`, startup only). `krishiv server` accepts these:

| Setting | Decides |
|---|---|
| unset, or `roles` | everything, by the role matrix |
| `external` | data objects, by your policy service (`krishiv.access.external.endpoint`); the role matrix still decides the server's own objects, so a `viewer` may write if the service allows it |
| `roles+external` | both must allow |

`external` and `roles+external` need authentication. The policy service can only allow or deny.

**`krishiv server` does not apply object grants.** A provider list that names `grants` is refused at startup, and
`GRANT`, `REVOKE`, `SHOW GRANTS` and role statements are refused. `GRANT` works only in a
[library session](#grants-in-a-library-session).

**Row filters and column masks are planned and cannot be used yet.** No SQL statement or policy-service field declares
one, and a provider decision that carries one is refused with `KRISHIV_UNSUPPORTED_FEATURE`.

Without authentication a session has no access policy at all: it allows everything. Configure authentication before you
rely on any provider.

On a server with authentication, path confinement (`krishiv.io.local.allowedRoots`,
`krishiv.io.objectStore.allowedPrefixes`) applies first under every provider; a server without authentication does not
confine paths. `catalog` is refused for now.

## The role matrix

| A caller with the role | May |
|---|---|
| `viewer` | see metadata; read `system.queries`, `system.sessions` (own rows), `system.capabilities` and `system.metrics`. It cannot read your tables. |
| `operator` | also read every table, view, function and path, use secrets and connections, see others' queries and stop them |
| `admin` | also write, create and drop, change cluster-wide settings, and run client code such as Python UDFs |

A principal with no role is denied everything. The matrix has no per-table rules: for allow or deny per object on a
server, use the `external` provider. For authentication and token roles see
[Securing a single-node server](single-node-server-security.md).

## Grants in a library session

A Rust program that embeds the `krishiv-session` crate can enforce grants itself. Give the session builder the same
`GrantsPolicy` as both `access_policy` and `grants`; a builder with grants but no access policy is refused. The
statements below need such a session. There is no Python or server setting for it.

### Grant and revoke

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
issue `GRANT`: the policy that decides the session's system objects must also allow the statement.

`SHOW GRANTS` (a Krishiv extension, 7 columns) lists grants; `SHOW GRANTS ON TABLE t` narrows it to one object. Without the `Monitor` privilege both return only the caller's own
effective grants, so an empty result is not an audit that nobody else has access. A listing of tables, views or functions shows only the ones the caller may see.

### Views and functions

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
