-- query 29: nope(1)
SELECT regexp_count(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(regexp_count)';
SELECT xpath_long(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(xpath_long)';
SELECT date_sub(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(date_sub)';
