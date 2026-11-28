`ANALYZE TABLES [FROM | IN schema] COMPUTE STATISTICS [NOSCAN]` now runs instead of failing to parse. Namespace
commands report a missing, nested or unnamed schema as Spark does (`SCHEMA_NOT_FOUND` with the name as written,
`NESTED_DATABASE_UNSUPPORTED_BY_V1_SESSION_CATALOG`, `MISSING_DATABASE_FOR_V1_SESSION_CATALOG`).
A table, schema or function name with more namespace parts than the session catalog holds is
`REQUIRES_SINGLE_PART_NAMESPACE`, naming the parts after `spark_catalog`, in `CREATE SCHEMA`, `REFRESH` and
`UNCACHE TABLE`, `SHOW COLUMNS` and the function statements too. `SHOW COLUMNS IN table FROM schema` looks the
table up in that schema, then raises `SHOW_COLUMNS_WITH_CONFLICT_NAMESPACE` unless the clause is exactly the
table's schema (`FROM spark_catalog.default` now conflicts, as in Spark).
