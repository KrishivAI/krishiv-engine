-- query 6: nope(1)
SELECT sqrt(a, b) AS c1 FROM t6 WHERE x IN (1, 2) AND y = 'x(sqrt)';
SELECT kll_sketch_to_string_float(a, b) AS c2 FROM t6 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_float)';
SELECT xpath_int(a, b) AS c3 FROM t6 WHERE x IN (1, 2) AND y = 'x(xpath_int)';
SELECT legacy_fn_13(a, b) AS c4 FROM t6 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT approx_top_k_accumulate(a, b) AS c5 FROM t6 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT legacy_fn_1(a, b) AS c6 FROM t6 WHERE x IN (1, 2) AND y = 'x(legacy_fn_1)';
