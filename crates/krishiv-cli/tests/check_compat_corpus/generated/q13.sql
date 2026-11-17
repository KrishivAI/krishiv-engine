-- query 13: nope(1)
SELECT tuple_sketch_agg_integer(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_integer)';
SELECT smallint(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(smallint)';
SELECT theta_union_agg(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(theta_union_agg)';
SELECT array_distinct(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
