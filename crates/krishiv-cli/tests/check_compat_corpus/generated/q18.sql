-- query 18: nope(1)
SELECT kll_sketch_to_string_double(a, b) AS c1 FROM t18 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_double)';
SELECT array_join(a, b) AS c2 FROM t18 WHERE x IN (1, 2) AND y = 'x(array_join)';
SELECT zeroifnull(a, b) AS c3 FROM t18 WHERE x IN (1, 2) AND y = 'x(zeroifnull)';
