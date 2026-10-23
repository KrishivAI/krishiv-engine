-- Source: ClickHouse/ClickHouse tests/benchmarks/tpc-h/queries/query_17.sql at a79b7c063486 (Apache-2.0).
-- Adaptations: none (verbatim).
SELECT
    sum(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem, part
WHERE (p_partkey = l_partkey)
    AND (p_brand = 'Brand#23')
    AND (p_container = 'MED BOX')
    AND (l_quantity < (
        SELECT 0.2 * avg(l_quantity)
        FROM lineitem
        WHERE (l_partkey = p_partkey)
    ));
