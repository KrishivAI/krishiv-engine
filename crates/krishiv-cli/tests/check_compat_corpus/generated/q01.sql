-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT min_by(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(min_by)';
SELECT theta_difference(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(theta_difference)';
SELECT pi(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(pi)';
SELECT sha1(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(sha1)';
SELECT kll_sketch_get_n_float(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_float)';
