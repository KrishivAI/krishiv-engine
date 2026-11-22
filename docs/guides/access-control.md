---
title: Access control with grants
---

# Access control with grants

In server mode, who can read or change what is decided by an **access-policy provider**
(`krishiv.access.provider`, startup only):

| Provider | Decides | Default when |
|---|---|---|
| `grants` | data objects (tables, views, columns, namespaces, functions, paths, secrets, connections) by `GRANT` and `REVOKE`; the role matrix still decides the server's own objects; an admin passes every data check | authentication is configured |
| `roles` | everything, by the role matrix (`viewer`, `operator`, `admin`) | no authentication |

Path confinement (`krishiv.io.local.allowedRoots`, `krishiv.io.objectStore.allowedPrefixes`) applies first under
every provider. `catalog`, `external` and provider lists are refused for now.

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
the privilege. Revoking it on a table clears the flag on the table's columns too.

`SHOW GRANTS` (a Krishiv extension, 7 columns) lists the grants you may see; `SHOW GRANTS ON TABLE t` narrows it to
one object. A listing of tables, views or functions shows only the ones the caller may see.

## Views and functions

A view runs with its **invoker's** rights by default. `SQL SECURITY DEFINER` makes it run with its owner's rights, so
a view can expose some columns or rows of a table its readers cannot read:

```sql
CREATE OR REPLACE VIEW sales.my_orders SQL SECURITY DEFINER AS
SELECT id, total FROM sales.orders WHERE owner = 'ana';
GRANT SELECT ON VIEW sales.my_orders TO user:ana;
```

Dropping a view (`DROP VIEW`) needs the `DROP` privilege on it, as `DROP FUNCTION` does on a function.

## Check it

Connect as the principal and run the statement; a denied one fails with an error whose message names the object and
the missing privilege. For the server's authentication settings see
[Securing a single-node server](single-node-server-security.md).
