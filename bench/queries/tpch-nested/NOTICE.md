# Nested TPC-H queries

`q01.sql` to `q22.sql` are copied unchanged from Apache DataFusion Comet,
`benchmarks/nested/queries/` (https://github.com/apache/datafusion-comet, pull request #7044), licensed under the
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Each file keeps Comet's header: the query is
derived from TPC-H under the terms of the TPC Fair Use Policy; TPC-H queries are Copyright 1993-2022 Transaction
Processing Performance Council.

The queries read the nested tables written by `krishiv_bench/nested_tpch.py`; see `bench/README.md`.
