WITH totals(k, v) AS (SELECT 1, 2)
SELECT CAST(k AS DECIMAL(10, 2)), int(v), date('2020-01-01'), try_cast(v AS VARCHAR(10))
FROM totals LEFT JOIN VALUES (1, 2) AS t(a, b) ON a = k
WHERE k LIKE 'a%' AND v BETWEEN 1 AND 5;
ALTER TABLE s ADD COLUMNS (c INT);
SELECT IDENTIFIER('t'), lower(name), size(array(1, 2)) FROM s TABLESAMPLE (10 PERCENT);
