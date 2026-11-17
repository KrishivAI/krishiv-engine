-- query 24: nope(1)
SELECT python_worker_logs(a, b) AS c1 FROM t24 WHERE x IN (1, 2) AND y = 'x(python_worker_logs)';
SELECT tuple_sketch_estimate_integer(a, b) AS c2 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_estimate_integer)';
SELECT legacy_fn_13(a, b) AS c3 FROM t24 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT to_timestamp_ltz(a, b) AS c4 FROM t24 WHERE x IN (1, 2) AND y = 'x(to_timestamp_ltz)';
SELECT legacy_fn_14(a, b) AS c5 FROM t24 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT kll_sketch_get_rank_double(a, b) AS c6 FROM t24 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_rank_double)';
