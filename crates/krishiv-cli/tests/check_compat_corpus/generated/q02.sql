-- query 2: nope(1)
SELECT legacy_fn_14(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT luhn_check(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(luhn_check)';
SELECT xpath(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(xpath)';
SELECT tuple_union_theta_integer(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_union_theta_integer)';
SELECT theta_intersection(a, b) AS c5 FROM t2 WHERE x IN (1, 2) AND y = 'x(theta_intersection)';
SELECT to_time(a, b) AS c6 FROM t2 WHERE x IN (1, 2) AND y = 'x(to_time)';
