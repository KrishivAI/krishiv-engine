-- query 16: nope(1)
SELECT kll_sketch_get_n_bigint(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_bigint)';
SELECT theta_union_agg(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(theta_union_agg)';
SELECT legacy_fn_0(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT current_schema(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(current_schema)';
