-- query 5: nope(1)
SELECT url_encode(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(url_encode)';
SELECT legacy_fn_13(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT approx_top_k_accumulate(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT legacy_fn_1(a, b) AS c4 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_1)';
