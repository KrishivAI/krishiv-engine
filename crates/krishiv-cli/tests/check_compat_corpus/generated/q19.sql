-- query 19: nope(1)
SELECT theta_union(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(theta_union)';
SELECT theta_intersection(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(theta_intersection)';
SELECT try_reflect(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(try_reflect)';
SELECT legacy_fn_26(a, b) AS c4 FROM t19 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT sort_array(a, b) AS c5 FROM t19 WHERE x IN (1, 2) AND y = 'x(sort_array)';
