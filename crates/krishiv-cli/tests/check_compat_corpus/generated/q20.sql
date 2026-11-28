-- query 20: nope(1)
SELECT st_srid(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(st_srid)';
SELECT vector_normalize(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(vector_normalize)';
SELECT legacy_fn_39(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_39)';
SELECT tuple_sketch_agg_integer(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_integer)';
SELECT position(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(position)';
