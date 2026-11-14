-- query 2: nope(1)
SELECT current_database(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(current_database)';
SELECT try_to_binary(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(try_to_binary)';
SELECT try_to_timestamp(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(try_to_timestamp)';
SELECT tuple_intersection_theta_double(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_double)';
