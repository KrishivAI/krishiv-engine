-- query 19: nope(1)
SELECT bitmap_count(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(bitmap_count)';
SELECT tuple_intersection_theta_integer(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_integer)';
SELECT legacy_fn_26(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
