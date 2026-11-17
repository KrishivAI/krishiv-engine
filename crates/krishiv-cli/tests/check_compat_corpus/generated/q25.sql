-- query 25: nope(1)
SELECT bit_get(a, b) AS c1 FROM t25 WHERE x IN (1, 2) AND y = 'x(bit_get)';
SELECT count_min_sketch(a, b) AS c2 FROM t25 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT sha1(a, b) AS c3 FROM t25 WHERE x IN (1, 2) AND y = 'x(sha1)';
SELECT legacy_fn_11(a, b) AS c4 FROM t25 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT try_avg(a, b) AS c5 FROM t25 WHERE x IN (1, 2) AND y = 'x(try_avg)';
