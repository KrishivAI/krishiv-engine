JSON and CSV files now follow `spark.sql.legacy.timeParserPolicy=LEGACY` as Spark does: dates and timestamps are read
and written with Spark's lenient legacy patterns (and their defaults when none is named), with the fallback to the
default grammar on, and CSV schema inference infers no dates there. A trailing zone such as `Etc/GMT+1` in a file
timestamp is now read as written (it was read as NULL), and ` GMT` is dropped only when the first reading fails.
