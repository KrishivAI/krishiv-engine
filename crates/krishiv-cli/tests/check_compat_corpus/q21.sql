-- query 21: nope(1)
SELECT to_char(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(to_char)';
SELECT collect_list(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(collect_list)';
SELECT tuple_intersection_theta_integer(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_integer)';
SELECT regr_count(a, b) AS c4 FROM t21 WHERE x IN (1, 2) AND y = 'x(regr_count)';
SELECT legacy_fn_11(a, b) AS c5 FROM t21 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
