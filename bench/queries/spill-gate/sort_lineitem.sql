-- Memory gate, sort spill on TPC-H data: a full sort of lineitem (the LIMIT is above the
-- top-N operator's bound, so every row is sorted). See bench/README.md, "Spill coverage".
SELECT l_orderkey, l_linenumber, l_extendedprice
FROM lineitem
ORDER BY l_extendedprice DESC, l_orderkey, l_linenumber
LIMIT 1000000
