-- query 16: nope(1)
SELECT hll_sketch_agg(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(hll_sketch_agg)';
SELECT python_worker_logs(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(python_worker_logs)';
SELECT legacy_fn_0(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT date_add(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(date_add)';
