The Connect server and `spark.conf.get` now know `spark.sql.timestampType` (default `TIMESTAMP_LTZ`), so a PySpark client can read it; it was answered with SQL_CONF_NOT_FOUND.
