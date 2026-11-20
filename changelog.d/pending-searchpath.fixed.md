A command's missing-table error (DESCRIBE, DROP VIEW, UNCACHE and the other commands) now ends its search path at the current database, as Spark does after USE.
