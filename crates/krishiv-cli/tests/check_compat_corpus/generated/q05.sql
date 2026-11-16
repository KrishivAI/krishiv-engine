-- query 5: nope(1)
SELECT sin(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(sin)';
SELECT measure(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(measure)';
SELECT xpath_string(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(xpath_string)';
SELECT legacy_fn_13(a, b) AS c4 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT approx_top_k_accumulate(a, b) AS c5 FROM t5 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT legacy_fn_1(a, b) AS c6 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_1)';
