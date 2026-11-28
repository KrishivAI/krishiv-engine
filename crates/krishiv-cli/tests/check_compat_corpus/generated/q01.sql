-- query 1: nope(1)
SELECT tuple_difference_double(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(tuple_difference_double)';
SELECT sentences(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(sentences)';
SELECT weekday(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(weekday)';
