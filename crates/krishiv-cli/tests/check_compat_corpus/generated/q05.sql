-- query 5: nope(1)
SELECT count_min_sketch(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT try_element_at(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(try_element_at)';
SELECT hash(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(hash)';
