-- query 8: nope(1)
SELECT legacy_fn_32(a, b) AS c1 FROM t8 WHERE x IN (1, 2) AND y = 'x(legacy_fn_32)';
SELECT is_valid_variant(a, b) AS c2 FROM t8 WHERE x IN (1, 2) AND y = 'x(is_valid_variant)';
SELECT nvl(a, b) AS c3 FROM t8 WHERE x IN (1, 2) AND y = 'x(nvl)';
