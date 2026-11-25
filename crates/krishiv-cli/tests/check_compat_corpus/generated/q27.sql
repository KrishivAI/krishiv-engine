-- query 27: nope(1)
SELECT measure(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(measure)';
SELECT approx_top_k(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(approx_top_k)';
SELECT approx_top_k_combine(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(approx_top_k_combine)';
SELECT pi(a, b) AS c4 FROM t27 WHERE x IN (1, 2) AND y = 'x(pi)';
SELECT approx_count_distinct(a, b) AS c5 FROM t27 WHERE x IN (1, 2) AND y = 'x(approx_count_distinct)';
