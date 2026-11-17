-- query 29: nope(1)
SELECT window_time(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(window_time)';
SELECT kll_sketch_get_n_double(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_double)';
SELECT st_geogfromwkb(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(st_geogfromwkb)';
