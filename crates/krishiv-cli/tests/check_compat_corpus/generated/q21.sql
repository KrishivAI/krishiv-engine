-- query 21: nope(1)
SELECT tuple_sketch_summary_integer(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_summary_integer)';
SELECT zeroifnull(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(zeroifnull)';
SELECT locate(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(locate)';
SELECT tuple_difference_double(a, b) AS c4 FROM t21 WHERE x IN (1, 2) AND y = 'x(tuple_difference_double)';
