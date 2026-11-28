-- query 25: nope(1)
SELECT count_min_sketch(a, b) AS c1 FROM t25 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT grouping_id(a, b) AS c2 FROM t25 WHERE x IN (1, 2) AND y = 'x(grouping_id)';
SELECT to_date(a, b) AS c3 FROM t25 WHERE x IN (1, 2) AND y = 'x(to_date)';
SELECT acosh(a, b) AS c4 FROM t25 WHERE x IN (1, 2) AND y = 'x(acosh)';
