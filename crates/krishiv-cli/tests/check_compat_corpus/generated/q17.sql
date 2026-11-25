-- query 17: nope(1)
SELECT schema_of_variant_agg(a, b) AS c1 FROM t17 WHERE x IN (1, 2) AND y = 'x(schema_of_variant_agg)';
SELECT tuple_union_theta_integer(a, b) AS c2 FROM t17 WHERE x IN (1, 2) AND y = 'x(tuple_union_theta_integer)';
SELECT explode(a, b) AS c3 FROM t17 WHERE x IN (1, 2) AND y = 'x(explode)';
SELECT to_json(a, b) AS c4 FROM t17 WHERE x IN (1, 2) AND y = 'x(to_json)';
