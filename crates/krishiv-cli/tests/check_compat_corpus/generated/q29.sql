-- query 29: nope(1)
SELECT printf(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(printf)';
SELECT tuple_difference_double(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(tuple_difference_double)';
SELECT xpath_float(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(xpath_float)';
