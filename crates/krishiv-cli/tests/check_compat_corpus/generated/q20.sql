-- query 20: nope(1)
SELECT python_worker_logs(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(python_worker_logs)';
SELECT kll_sketch_to_string_bigint(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_bigint)';
SELECT xpath_float(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(xpath_float)';
SELECT legacy_fn_21(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_21)';
SELECT legacy_fn_26(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
