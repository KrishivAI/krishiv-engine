-- query 20: nope(1)
SELECT stack(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(stack)';
SELECT vector_sum(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(vector_sum)';
SELECT legacy_fn_39(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_39)';
SELECT tuple_sketch_agg_double(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_double)';
SELECT positive(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(positive)';
