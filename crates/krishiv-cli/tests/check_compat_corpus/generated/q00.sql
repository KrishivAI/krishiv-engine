-- query 0: nope(1)
SELECT variance(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(variance)';
SELECT count_min_sketch(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT sentences(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(sentences)';
