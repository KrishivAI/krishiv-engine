// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";
import { Session, KrishivError } from "../src/index.js";

// Runs against a real `krishiv server` built with the flight-sql feature.
// KRISHIV_TEST_SERVER=host:port, KRISHIV_TEST_TOKEN=<bearer token> (omit when the server has no auth).
const target = process.env["KRISHIV_TEST_SERVER"];
const token = process.env["KRISHIV_TEST_TOKEN"];

describe.skipIf(!target)("a real krishiv server", () => {
  const open = () => {
    const b = Session.builder().remote(`krishiv://${target}`);
    return (token ? b.token(token) : b).getOrCreate();
  };

  it("returns rows, types and Arrow", async () => {
    const s = await open();
    const df = s.sql("SELECT id, id * 2 AS twice, CAST(id AS STRING) AS label, id > 2 AS big FROM range(5)");
    expect(await df.collect()).toHaveLength(5);
    expect((await df.toArrow()).numRows).toBe(5);
    expect(await df.count()).toBe(5);
    await s.close();
  });

  it("keeps session state between calls", async () => {
    const s = await open();
    await s.conf.set("spark.sql.session.timeZone", "UTC");
    expect(await s.conf.get("spark.sql.session.timeZone")).toBe("UTC");
    await s.sql("CREATE OR REPLACE TEMP VIEW js_t AS SELECT 1 AS x").collect();
    expect(await s.sql("SELECT x FROM js_t").collect()).toEqual([{ x: 1 }]);
    await s.close();
  });

  it("reports the Spark error class and SQLSTATE", async () => {
    const s = await open();
    const err = await s.sql("SELECT 1/0").collect().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(KrishivError);
    expect((err as KrishivError).errorClass).toBe("DIVIDE_BY_ZERO");
    expect((err as KrishivError).sqlState).toBe("22012");
    await s.close();
  });

  it("refuses a wrong token", async () => {
    if (!token) return;
    const s = await Session.builder().remote(`krishiv://${target}`).token("wrong").getOrCreate();
    await expect(s.sql("SELECT 1").collect()).rejects.toBeInstanceOf(KrishivError);
    await s.close();
  });
});
