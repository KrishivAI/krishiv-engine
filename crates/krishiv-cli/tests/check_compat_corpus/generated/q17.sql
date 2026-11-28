-- query 17: nope(1)
SELECT kll_sketch_agg_bigint(a, b) AS c1 FROM t17 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_bigint)';
SELECT kll_sketch_agg_bigint(a, b) AS c2 FROM t17 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_bigint)';
SELECT legacy_fn_6(a, b) AS c3 FROM t17 WHERE x IN (1, 2) AND y = 'x(legacy_fn_6)';
