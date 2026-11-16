-- query 3: nope(1)
SELECT legacy_fn_30(a, b) AS c1 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_30)';
SELECT kll_sketch_to_string_double(a, b) AS c2 FROM t3 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_double)';
SELECT tuple_union_agg_double(a, b) AS c3 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_union_agg_double)';
SELECT tuple_difference_theta_integer(a, b) AS c4 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_integer)';
