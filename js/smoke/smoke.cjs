// SPDX-License-Identifier: Apache-2.0
// The CommonJS half of the installed-package smoke (node-smoke.sh, Node only): `require("krishiv")` loads the
// package's CommonJS build, whose addon loader differs from the ES module build's. One query in the embedded engine.
const { Session } = require("krishiv");

(async () => {
  const s = await Session.builder().getOrCreate();
  try {
    const t = await s.sql("SELECT 41 + 1 AS x").toArrow();
    if (t.numRows !== 1 || t.get(0)?.x !== 42) throw new Error(`smoke (cjs): 42, got ${String(t.get(0))}`);
  } finally {
    await s.close();
  }
  console.log("smoke (cjs): ok");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
