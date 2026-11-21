-- query 20: nope(1)
SELECT theta_union_agg(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(theta_union_agg)';
SELECT theta_intersection_agg(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(theta_intersection_agg)';
SELECT tuple_difference_integer(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(tuple_difference_integer)';
SELECT legacy_fn_26(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT some(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(some)';
