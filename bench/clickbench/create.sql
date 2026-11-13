-- ClickBench table for Krishiv: a view over the single Parquet file, read in place (nothing is loaded).
-- `query.py` runs this file after setting spark.sql.parquet.binaryAsString=true (the file's strings are unannotated
-- BINARY) and replaces `hits.parquet` by its absolute path.
CREATE OR REPLACE TEMPORARY VIEW hits AS SELECT * FROM parquet.`hits.parquet`;
