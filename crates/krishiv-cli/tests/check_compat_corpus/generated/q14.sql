-- query 14: nope(1)
SELECT tuple_sketch_summary_integer(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_summary_integer)';
SELECT size(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(size)';
SELECT tuple_difference_integer(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(tuple_difference_integer)';
SELECT array_distinct(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
