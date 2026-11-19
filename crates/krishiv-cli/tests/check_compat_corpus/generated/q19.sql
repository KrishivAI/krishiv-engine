-- query 19: nope(1)
SELECT reflect(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(reflect)';
SELECT kll_sketch_to_string_double(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_double)';
SELECT xpath_long(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(xpath_long)';
SELECT legacy_fn_21(a, b) AS c4 FROM t19 WHERE x IN (1, 2) AND y = 'x(legacy_fn_21)';
SELECT legacy_fn_26(a, b) AS c5 FROM t19 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
