-- query 27: nope(1)
SELECT percentile_disc(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(percentile_disc)';
SELECT tuple_intersection_theta_double(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_double)';
SELECT xpath_long(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(xpath_long)';
