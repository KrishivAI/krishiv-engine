-- query 14: nope(1)
SELECT tuple_difference_theta_integer(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_integer)';
SELECT sql_keywords(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(sql_keywords)';
SELECT schema_of_variant_agg(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(schema_of_variant_agg)';
SELECT array_distinct(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
