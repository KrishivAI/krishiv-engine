-- query 15: nope(1)
SELECT hll_sketch_estimate(a, b) AS c1 FROM t15 WHERE x IN (1, 2) AND y = 'x(hll_sketch_estimate)';
SELECT reflect(a, b) AS c2 FROM t15 WHERE x IN (1, 2) AND y = 'x(reflect)';
SELECT legacy_fn_0(a, b) AS c3 FROM t15 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT date_add(a, b) AS c4 FROM t15 WHERE x IN (1, 2) AND y = 'x(date_add)';
