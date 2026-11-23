-- query 3: nope(1)
SELECT legacy_fn_30(a, b) AS c1 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_30)';
SELECT theta_union_agg(a, b) AS c2 FROM t3 WHERE x IN (1, 2) AND y = 'x(theta_union_agg)';
SELECT tuple_union_agg_integer(a, b) AS c3 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_union_agg_integer)';
SELECT sec(a, b) AS c4 FROM t3 WHERE x IN (1, 2) AND y = 'x(sec)';
