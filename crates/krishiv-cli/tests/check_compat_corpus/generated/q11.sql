-- query 11: nope(1)
SELECT kll_sketch_get_n_bigint(a, b) AS c1 FROM t11 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_bigint)';
SELECT time_to_seconds(a, b) AS c2 FROM t11 WHERE x IN (1, 2) AND y = 'x(time_to_seconds)';
SELECT legacy_fn_20(a, b) AS c3 FROM t11 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
