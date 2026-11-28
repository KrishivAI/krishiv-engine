`ANALYZE TABLES [FROM | IN schema] COMPUTE STATISTICS [NOSCAN]` now runs instead of failing to parse. Namespace
commands report a missing, nested or unnamed schema as Spark does (`SCHEMA_NOT_FOUND` with the name as written,
`NESTED_DATABASE_UNSUPPORTED_BY_V1_SESSION_CATALOG`, `MISSING_DATABASE_FOR_V1_SESSION_CATALOG`).
