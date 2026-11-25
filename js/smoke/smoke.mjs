// SPDX-License-Identifier: Apache-2.0
// The installed-package smoke of publish-js.yml: run from a scratch project where the packed `krishiv` and one
// `@krishiv/engine-<platform>` tarball are installed, under Node, Bun or Deno. It checks that the loader finds the
// platform package with no override, that SQL runs in the embedded engine, and that a failure carries its Spark error
// class. Exits non-zero on any failure.
import { KrishivError, Session } from "krishiv";

function check(ok, what) {
  if (!ok) throw new Error(`smoke: ${what}`);
}

if (process.env.KRISHIV_NATIVE_ENGINE) throw new Error("smoke: unset KRISHIV_NATIVE_ENGINE, the platform package is under test");
const s = await Session.builder().config("spark.sql.session.timeZone", "UTC").getOrCreate();
try {
  check((await s.conf.get("spark.sql.session.timeZone")) === "UTC", "conf round trip");
  const t = await s.sql("SELECT id, CAST(id AS STRING) AS s FROM range(5)").toArrow();
  check(t.numRows === 5, `5 rows, got ${t.numRows}`);
  const last = t.get(4);
  check(last?.id === 4n && last?.s === "4", `row 4 is {id: 4n, s: "4"}, got ${String(last)}`);
  const e = await s
    .sql("SELECT * FROM no_such_table")
    .toArrow()
    .then(() => undefined, (x) => x);
  check(e instanceof KrishivError, `a KrishivError, got ${String(e)}`);
  check(e.errorClass === "TABLE_OR_VIEW_NOT_FOUND", `TABLE_OR_VIEW_NOT_FOUND, got ${e.errorClass}`);
} finally {
  await s.close();
}
console.log("smoke: ok");
