-- query 17: nope(1)
SELECT stddev_pop(a, b) AS c1 FROM t17 WHERE x IN (1, 2) AND y = 'x(stddev_pop)';
SELECT repeat(a, b) AS c2 FROM t17 WHERE x IN (1, 2) AND y = 'x(repeat)';
SELECT map_filter(a, b) AS c3 FROM t17 WHERE x IN (1, 2) AND y = 'x(map_filter)';
SELECT degrees(a, b) AS c4 FROM t17 WHERE x IN (1, 2) AND y = 'x(degrees)';
