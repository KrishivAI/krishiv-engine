-- query 9: nope(1)
SELECT legacy_fn_12(a, b) AS c1 FROM t9 WHERE x IN (1, 2) AND y = 'x(legacy_fn_12)';
SELECT upper(a, b) AS c2 FROM t9 WHERE x IN (1, 2) AND y = 'x(upper)';
SELECT legacy_fn_15(a, b) AS c3 FROM t9 WHERE x IN (1, 2) AND y = 'x(legacy_fn_15)';
SELECT legacy_fn_14(a, b) AS c4 FROM t9 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT legacy_fn_0(a, b) AS c5 FROM t9 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT sentences(a, b) AS c6 FROM t9 WHERE x IN (1, 2) AND y = 'x(sentences)';
