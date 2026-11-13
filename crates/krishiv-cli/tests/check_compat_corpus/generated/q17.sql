-- query 17: nope(1)
SELECT split_part(a, b) AS c1 FROM t17 WHERE x IN (1, 2) AND y = 'x(split_part)';
SELECT regexp_substr(a, b) AS c2 FROM t17 WHERE x IN (1, 2) AND y = 'x(regexp_substr)';
SELECT make_time(a, b) AS c3 FROM t17 WHERE x IN (1, 2) AND y = 'x(make_time)';
SELECT dayname(a, b) AS c4 FROM t17 WHERE x IN (1, 2) AND y = 'x(dayname)';
