-- query 27: nope(1)
SELECT legacy_fn_1(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(legacy_fn_1)';
SELECT log2(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(log2)';
SELECT acos(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(acos)';
SELECT percentile(a, b) AS c4 FROM t27 WHERE x IN (1, 2) AND y = 'x(percentile)';
SELECT tuple_union_double(a, b) AS c5 FROM t27 WHERE x IN (1, 2) AND y = 'x(tuple_union_double)';
