-- query 27: nope(1)
SELECT legacy_fn_1(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(legacy_fn_1)';
SELECT log1p(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(log1p)';
SELECT acos(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(acos)';
SELECT percent_rank(a, b) AS c4 FROM t27 WHERE x IN (1, 2) AND y = 'x(percent_rank)';
SELECT tuple_union_integer(a, b) AS c5 FROM t27 WHERE x IN (1, 2) AND y = 'x(tuple_union_integer)';
