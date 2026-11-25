-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT log2(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(log2)';
SELECT tuple_intersection_theta_integer(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_integer)';
SELECT md5(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(md5)';
SELECT regexp_like(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(regexp_like)';
SELECT try_reflect(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(try_reflect)';
