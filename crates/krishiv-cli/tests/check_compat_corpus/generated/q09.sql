-- query 9: nope(1)
SELECT array_compact(a, b) AS c1 FROM t9 WHERE x IN (1, 2) AND y = 'x(array_compact)';
SELECT kll_sketch_get_quantile_float(a, b) AS c2 FROM t9 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_float)';
SELECT coalesce(a, b) AS c3 FROM t9 WHERE x IN (1, 2) AND y = 'x(coalesce)';
SELECT to_xml(a, b) AS c4 FROM t9 WHERE x IN (1, 2) AND y = 'x(to_xml)';
SELECT array_except(a, b) AS c5 FROM t9 WHERE x IN (1, 2) AND y = 'x(array_except)';
