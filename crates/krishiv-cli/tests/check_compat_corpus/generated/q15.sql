-- query 15: nope(1)
SELECT kll_sketch_get_n_bigint(a, b) AS c1 FROM t15 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_bigint)';
SELECT tuple_intersection_double(a, b) AS c2 FROM t15 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_double)';
SELECT legacy_fn_0(a, b) AS c3 FROM t15 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT current_schema(a, b) AS c4 FROM t15 WHERE x IN (1, 2) AND y = 'x(current_schema)';
