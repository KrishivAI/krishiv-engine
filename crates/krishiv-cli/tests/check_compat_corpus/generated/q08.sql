-- query 8: nope(1)
SELECT array_compact(a, b) AS c1 FROM t8 WHERE x IN (1, 2) AND y = 'x(array_compact)';
SELECT tuple_difference_theta_integer(a, b) AS c2 FROM t8 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_integer)';
SELECT ceil(a, b) AS c3 FROM t8 WHERE x IN (1, 2) AND y = 'x(ceil)';
SELECT string_agg(a, b) AS c4 FROM t8 WHERE x IN (1, 2) AND y = 'x(string_agg)';
SELECT array_except(a, b) AS c5 FROM t8 WHERE x IN (1, 2) AND y = 'x(array_except)';
