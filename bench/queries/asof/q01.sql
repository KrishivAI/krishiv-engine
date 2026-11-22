SELECT count(*) AS n, sum(q.bid_cents) AS bid, sum(t.qty) AS qty
FROM trades t ASOF JOIN quotes q
  MATCH_CONDITION (t.ts >= q.ts)
  ON t.symbol = q.symbol
