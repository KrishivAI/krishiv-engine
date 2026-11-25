-- query 22: nope(1)
SELECT kll_sketch_get_n_bigint(a, b) AS c1 FROM t22 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_bigint)';
SELECT xpath_double(a, b) AS c2 FROM t22 WHERE x IN (1, 2) AND y = 'x(xpath_double)';
SELECT legacy_fn_12(a, b) AS c3 FROM t22 WHERE x IN (1, 2) AND y = 'x(legacy_fn_12)';
SELECT legacy_fn_15(a, b) AS c4 FROM t22 WHERE x IN (1, 2) AND y = 'x(legacy_fn_15)';
