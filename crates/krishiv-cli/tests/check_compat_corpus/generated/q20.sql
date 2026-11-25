-- query 20: nope(1)
SELECT signum(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(signum)';
SELECT unix_millis(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(unix_millis)';
SELECT legacy_fn_39(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_39)';
SELECT tuple_intersection_theta_integer(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_integer)';
SELECT now(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(now)';
