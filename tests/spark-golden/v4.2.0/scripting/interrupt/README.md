# Connect Interrupt during a SQL script, Spark 4.2.0 (P2b row 15, R3)

Produced by `tests/scripts/gen_scripting_interrupt.py` against a real Spark 4.2.0 Connect server
(`apache/spark:4.2.0`, `local[2]`, ANSI on) driven by `pyspark-client==4.2.0`. Output: `interrupt.out`.

Script shape: `BEGIN DECLARE i INT DEFAULT 0; [DECLARE EXIT HANDLER FOR SQLEXCEPTION BEGIN INSERT INTO r3_log VALUES ('handler'); END;] <body> INSERT INTO r3_log VALUES ('after'); END`.
Body (a) `WHILE true DO SET i = i + 1; END WHILE;` (no job), body (b) a long `SELECT count(*)` join. Interrupt sent 4 s in,
by interruptAll and by tag (both return 1 operation id). Not probed: interrupt by operation id (the python client cannot
set an operation id; it is the same server path as tag, `SparkConnectService` interrupt of the execute holder).

## Observed (all 8 cases identical)
- The client call ends in ~4.0 s with `OPERATION_CANCELED`, SQLSTATE `HY008` (`org.apache.spark.SparkSQLException`), for (a) and (b),
  with and without handler, for interruptAll and interruptTag. So the interrupt does stop a script even between leaves
  (a SET-only loop): Spark checks cancellation inside the script loop, not only in a running job.
- The handler never fires (`r3_log` has no `handler` row): OPERATION_CANCELED is not caught by `EXIT HANDLER FOR SQLEXCEPTION`.
- No later leaf runs (`r3_log` empty, no `after` row): the script ends at once.
- The session stays usable afterwards (`SELECT 1` works).

## Time limit
No script-level execution limit was found in Spark's SQLConf (v4.2.0): only `spark.sql.thriftServer.queryTimeout`
(Thrift Server only, per statement) and `spark.sql.broadcastTimeout` (unrelated). Only the first 100,000 of ~410,000 characters of
SQLConf.scala were read, so "no Connect execution timeout config exists" is **inconclusive**; nothing was run to test it.
Per-script vs per-leaf for a Connect limit therefore stays unverified.
