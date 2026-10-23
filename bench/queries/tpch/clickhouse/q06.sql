-- Source: ClickHouse/ClickHouse tests/benchmarks/tpc-h/queries/query_06.sql at a79b7c063486 (Apache-2.0).
-- Adaptations: the alternative formulation from the upstream README (known problem Q6, ClickHouse issue 70136).
SELECT
    sum(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE (l_shipdate >= date '1994-01-01')
    AND (l_shipdate < date '1994-01-01' + INTERVAL 1 YEAR)
    AND (l_discount BETWEEN 0.06::Decimal(12,2) - 0.01::Decimal(12,2) AND 0.06::Decimal(12,2) + 0.01::Decimal(12,2))
    AND (l_quantity < 24);
