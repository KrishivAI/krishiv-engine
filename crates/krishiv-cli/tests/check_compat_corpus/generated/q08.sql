-- query 8: nope(1)
SELECT array_compact(a, b) AS c1 FROM t8 WHERE x IN (1, 2) AND y = 'x(array_compact)';
SELECT try_reflect(a, b) AS c2 FROM t8 WHERE x IN (1, 2) AND y = 'x(try_reflect)';
SELECT ceil(a, b) AS c3 FROM t8 WHERE x IN (1, 2) AND y = 'x(ceil)';
SELECT time_from_seconds(a, b) AS c4 FROM t8 WHERE x IN (1, 2) AND y = 'x(time_from_seconds)';
SELECT array_except(a, b) AS c5 FROM t8 WHERE x IN (1, 2) AND y = 'x(array_except)';
