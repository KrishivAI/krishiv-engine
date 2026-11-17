-- query 15: nope(1)
SELECT kll_sketch_agg_double(a, b) AS c1 FROM t15 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_double)';
SELECT theta_sketch_agg(a, b) AS c2 FROM t15 WHERE x IN (1, 2) AND y = 'x(theta_sketch_agg)';
SELECT legacy_fn_0(a, b) AS c3 FROM t15 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT current_schema(a, b) AS c4 FROM t15 WHERE x IN (1, 2) AND y = 'x(current_schema)';
