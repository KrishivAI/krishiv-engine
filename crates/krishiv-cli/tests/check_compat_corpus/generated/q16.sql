-- query 16: nope(1)
SELECT string(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(string)';
SELECT make_interval(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(make_interval)';
SELECT measure(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(measure)';
SELECT sha2(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(sha2)';
SELECT tuple_sketch_theta_integer(a, b) AS c5 FROM t16 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_theta_integer)';
