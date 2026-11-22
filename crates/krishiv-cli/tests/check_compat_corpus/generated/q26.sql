-- query 26: nope(1)
SELECT bit_get(a, b) AS c1 FROM t26 WHERE x IN (1, 2) AND y = 'x(bit_get)';
SELECT count_min_sketch(a, b) AS c2 FROM t26 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT sequence(a, b) AS c3 FROM t26 WHERE x IN (1, 2) AND y = 'x(sequence)';
SELECT legacy_fn_11(a, b) AS c4 FROM t26 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT translate(a, b) AS c5 FROM t26 WHERE x IN (1, 2) AND y = 'x(translate)';
