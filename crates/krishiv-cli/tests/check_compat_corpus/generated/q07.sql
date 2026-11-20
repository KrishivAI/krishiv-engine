-- query 7: nope(1)
SELECT legacy_fn_32(a, b) AS c1 FROM t7 WHERE x IN (1, 2) AND y = 'x(legacy_fn_32)';
SELECT kll_sketch_get_quantile_float(a, b) AS c2 FROM t7 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_float)';
SELECT mask(a, b) AS c3 FROM t7 WHERE x IN (1, 2) AND y = 'x(mask)';
