-- query 13: nope(1)
SELECT xpath(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(xpath)';
SELECT sin(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(sin)';
SELECT tuple_sketch_agg_double(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_double)';
SELECT array_distinct(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
