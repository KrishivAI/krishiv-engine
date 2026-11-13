-- query 0: nope(1)
SELECT vector_cosine_similarity(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(vector_cosine_similarity)';
SELECT count_min_sketch(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT reflect(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(reflect)';
