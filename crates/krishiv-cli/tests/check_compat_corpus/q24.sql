-- query 24: nope(1)
SELECT kll_sketch_to_string_double(a, b) AS c1 FROM t24 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_double)';
SELECT tuple_intersection_theta_double(a, b) AS c2 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_double)';
SELECT legacy_fn_13(a, b) AS c3 FROM t24 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT try_make_timestamp_ntz(a, b) AS c4 FROM t24 WHERE x IN (1, 2) AND y = 'x(try_make_timestamp_ntz)';
SELECT legacy_fn_14(a, b) AS c5 FROM t24 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT kll_sketch_get_quantile_double(a, b) AS c6 FROM t24 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_double)';
