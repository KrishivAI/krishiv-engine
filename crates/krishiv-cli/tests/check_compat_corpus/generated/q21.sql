-- query 21: nope(1)
SELECT unix_millis(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(unix_millis)';
SELECT string(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(string)';
SELECT kll_sketch_get_quantile_bigint(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_bigint)';
SELECT legacy_fn_11(a, b) AS c4 FROM t21 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT tuple_sketch_summary_double(a, b) AS c5 FROM t21 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_summary_double)';
