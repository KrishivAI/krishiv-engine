-- query 27: nope(1)
SELECT legacy_fn_34(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(legacy_fn_34)';
SELECT bool_or(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(bool_or)';
SELECT convert_timezone(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(convert_timezone)';
SELECT minute(a, b) AS c4 FROM t27 WHERE x IN (1, 2) AND y = 'x(minute)';
SELECT last(a, b) AS c5 FROM t27 WHERE x IN (1, 2) AND y = 'x(last)';
