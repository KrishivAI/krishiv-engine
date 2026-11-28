-- query 16: nope(1)
SELECT shiftright(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(shiftright)';
SELECT radians(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(radians)';
SELECT listagg(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(listagg)';
SELECT date_part(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(date_part)';
