-- query 17: nope(1)
SELECT soundex(a, b) AS c1 FROM t17 WHERE x IN (1, 2) AND y = 'x(soundex)';
SELECT regexp_instr(a, b) AS c2 FROM t17 WHERE x IN (1, 2) AND y = 'x(regexp_instr)';
SELECT make_date(a, b) AS c3 FROM t17 WHERE x IN (1, 2) AND y = 'x(make_date)';
SELECT dayname(a, b) AS c4 FROM t17 WHERE x IN (1, 2) AND y = 'x(dayname)';
