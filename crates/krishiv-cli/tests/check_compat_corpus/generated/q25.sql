-- query 25: nope(1)
SELECT legacy_fn_37(a, b) AS c1 FROM t25 WHERE x IN (1, 2) AND y = 'x(legacy_fn_37)';
SELECT xpath_int(a, b) AS c2 FROM t25 WHERE x IN (1, 2) AND y = 'x(xpath_int)';
SELECT regexp_extract_all(a, b) AS c3 FROM t25 WHERE x IN (1, 2) AND y = 'x(regexp_extract_all)';
SELECT collations(a, b) AS c4 FROM t25 WHERE x IN (1, 2) AND y = 'x(collations)';
SELECT shiftright(a, b) AS c5 FROM t25 WHERE x IN (1, 2) AND y = 'x(shiftright)';
