-- query 28: nope(1)
SELECT power(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(power)';
SELECT tuple_difference_theta_double(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_double)';
SELECT xpath_long(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(xpath_long)';
