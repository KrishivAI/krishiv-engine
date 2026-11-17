-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT minute(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(minute)';
SELECT sentences(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(sentences)';
SELECT pmod(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(pmod)';
SELECT sha2(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(sha2)';
SELECT kll_sketch_get_n_double(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_double)';
