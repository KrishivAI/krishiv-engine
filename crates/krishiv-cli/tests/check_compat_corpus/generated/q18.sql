-- query 18: nope(1)
SELECT array_join(a, b) AS c1 FROM t18 WHERE x IN (1, 2) AND y = 'x(array_join)';
SELECT vector_avg(a, b) AS c2 FROM t18 WHERE x IN (1, 2) AND y = 'x(vector_avg)';
SELECT legacy_fn_23(a, b) AS c3 FROM t18 WHERE x IN (1, 2) AND y = 'x(legacy_fn_23)';
SELECT tuple_intersection_integer(a, b) AS c4 FROM t18 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_integer)';
