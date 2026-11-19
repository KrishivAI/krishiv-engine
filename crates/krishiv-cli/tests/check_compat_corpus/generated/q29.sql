-- query 29: nope(1)
SELECT xpath_float(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(xpath_float)';
SELECT kll_sketch_get_n_float(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_float)';
SELECT sqrt(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(sqrt)';
