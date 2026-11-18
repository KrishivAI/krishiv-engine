-- query 15: nope(1)
SELECT kll_sketch_agg_float(a, b) AS c1 FROM t15 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_float)';
SELECT theta_union(a, b) AS c2 FROM t15 WHERE x IN (1, 2) AND y = 'x(theta_union)';
SELECT legacy_fn_0(a, b) AS c3 FROM t15 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT current_schema(a, b) AS c4 FROM t15 WHERE x IN (1, 2) AND y = 'x(current_schema)';
