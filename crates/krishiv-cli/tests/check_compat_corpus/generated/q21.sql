-- query 21: nope(1)
SELECT inline_outer(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(inline_outer)';
SELECT legacy_fn_27(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(legacy_fn_27)';
SELECT kll_sketch_get_quantile_bigint(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_bigint)';
