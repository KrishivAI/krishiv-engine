-- query 11: nope(1)
SELECT to_variant_object(a, b) AS c1 FROM t11 WHERE x IN (1, 2) AND y = 'x(to_variant_object)';
SELECT approx_top_k_accumulate(a, b) AS c2 FROM t11 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT legacy_fn_26(a, b) AS c3 FROM t11 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT hypot(a, b) AS c4 FROM t11 WHERE x IN (1, 2) AND y = 'x(hypot)';
SELECT decode(a, b) AS c5 FROM t11 WHERE x IN (1, 2) AND y = 'x(decode)';
