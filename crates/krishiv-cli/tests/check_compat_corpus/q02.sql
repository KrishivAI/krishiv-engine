-- query 2: nope(1)
SELECT current_time(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(current_time)';
SELECT unix_seconds(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(unix_seconds)';
SELECT url_encode(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(url_encode)';
SELECT tuple_intersection_theta_double(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_double)';
