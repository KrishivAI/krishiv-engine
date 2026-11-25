-- query 16: nope(1)
SELECT vector_cosine_similarity(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(vector_cosine_similarity)';
SELECT tuple_difference_theta_integer(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_integer)';
SELECT tuple_intersection_integer(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_integer)';
SELECT from_xml(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(from_xml)';
