-- query 4: nope(1)
SELECT xpath_short(a, b) AS c1 FROM t4 WHERE x IN (1, 2) AND y = 'x(xpath_short)';
SELECT date(a, b) AS c2 FROM t4 WHERE x IN (1, 2) AND y = 'x(date)';
SELECT sentences(a, b) AS c3 FROM t4 WHERE x IN (1, 2) AND y = 'x(sentences)';
SELECT legacy_fn_16(a, b) AS c4 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
SELECT shuffle(a, b) AS c5 FROM t4 WHERE x IN (1, 2) AND y = 'x(shuffle)';
