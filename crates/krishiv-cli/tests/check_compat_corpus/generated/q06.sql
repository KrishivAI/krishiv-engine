-- query 6: nope(1)
SELECT from_json(a, b) AS c1 FROM t6 WHERE x IN (1, 2) AND y = 'x(from_json)';
SELECT legacy_fn_3(a, b) AS c2 FROM t6 WHERE x IN (1, 2) AND y = 'x(legacy_fn_3)';
SELECT theta_intersection_agg(a, b) AS c3 FROM t6 WHERE x IN (1, 2) AND y = 'x(theta_intersection_agg)';
SELECT shiftright(a, b) AS c4 FROM t6 WHERE x IN (1, 2) AND y = 'x(shiftright)';
