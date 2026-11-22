SELECT count(*) AS n, sum(q.ask_cents) AS ask, sum(t.price_cents) AS price
FROM trades t ASOF JOIN quotes q
  MATCH_CONDITION (t.ts < q.ts)
  ON t.symbol = q.symbol
