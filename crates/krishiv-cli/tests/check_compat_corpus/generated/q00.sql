-- query 0: nope(1)
SELECT var_pop(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(var_pop)';
SELECT count_min_sketch(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT theta_difference(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(theta_difference)';
