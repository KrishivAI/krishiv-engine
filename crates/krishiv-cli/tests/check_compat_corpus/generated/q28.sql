-- query 28: nope(1)
SELECT tuple_intersection_double(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_double)';
SELECT theta_union(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(theta_union)';
SELECT flatten(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(flatten)';
SELECT kll_sketch_agg_bigint(a, b) AS c4 FROM t28 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_bigint)';
