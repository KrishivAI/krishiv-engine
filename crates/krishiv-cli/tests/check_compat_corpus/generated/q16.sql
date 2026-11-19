-- query 16: nope(1)
SELECT sort_array(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(sort_array)';
SELECT regexp_extract_all(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(regexp_extract_all)';
SELECT luhn_check(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(luhn_check)';
SELECT dayname(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(dayname)';
