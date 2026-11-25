-- query 23: nope(1)
SELECT tuple_intersection_integer(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_integer)';
SELECT tuple_union_double(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(tuple_union_double)';
SELECT legacy_fn_13(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT string(a, b) AS c4 FROM t23 WHERE x IN (1, 2) AND y = 'x(string)';
SELECT legacy_fn_14(a, b) AS c5 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT tuple_difference_theta_double(a, b) AS c6 FROM t23 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_double)';
