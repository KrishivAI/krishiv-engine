// SPDX-License-Identifier: Apache-2.0
// Packs the npm packages of `krishiv` (design/node-engine-napi.md; publish-js.yml). Nothing is published here.
//
//   node scripts/npm-packages.mjs platform <platform> <krishiv.node> <out dir>
//       packs `@krishiv/engine-<platform>` (js/npm/<platform>) with the given addon binary
//   node scripts/npm-packages.mjs main <out dir>
//       packs `krishiv` itself (run `npm run build` first), its optionalDependencies naming every platform package
//
// The version of every package is the one of js/package.json (kept by `cargo xtask release-bump`): the platform
// manifests carry 0.0.0 and are stamped here, so the packages always go out in lockstep, which the loader checks.
import { execFileSync } from "node:child_process";
import { copyFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const js = resolve(fileURLToPath(new URL("..", import.meta.url)));
const readJson = (p) => JSON.parse(readFileSync(p, "utf8"));
const writeJson = (p, v) => writeFileSync(p, `${JSON.stringify(v, null, 2)}\n`);
const version = readJson(join(js, "package.json")).version;

/** The platforms, one directory each under js/npm. */
function platforms() {
  return readdirSync(join(js, "npm"), { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort();
}

/** `npm pack` of `dir` into `out`; prints and returns the tarball's path. */
function pack(dir, out) {
  mkdirSync(out, { recursive: true });
  const name = execFileSync("npm", ["pack", "--pack-destination", resolve(out)], { cwd: dir, encoding: "utf8" })
    .trim()
    .split("\n")
    .pop();
  const path = join(resolve(out), name);
  console.log(path);
  return path;
}

function packPlatform(platform, binary, out) {
  const src = join(js, "npm", platform);
  if (!existsSync(join(src, "package.json"))) throw new Error(`unknown platform ${platform} (no js/npm/${platform})`);
  if (!existsSync(binary)) throw new Error(`no addon binary at ${binary}`);
  const dir = mkdtempSync(join(tmpdir(), `krishiv-${platform}-`));
  cpSync(src, dir, { recursive: true });
  const pkg = readJson(join(dir, "package.json"));
  pkg.version = version;
  writeJson(join(dir, "package.json"), pkg);
  copyFileSync(binary, join(dir, "krishiv.node"));
  return pack(dir, out);
}

function packMain(out) {
  if (!existsSync(join(js, "dist", "index.js"))) throw new Error("no js/dist: run `npm run build` first");
  const dir = mkdtempSync(join(tmpdir(), "krishiv-main-"));
  for (const f of ["package.json", "README.md", "dist"]) cpSync(join(js, f), join(dir, f), { recursive: true });
  const pkg = readJson(join(dir, "package.json"));
  // Added only at pack time: the platform packages are not on the registry before the first publish, so a
  // committed entry would break `npm ci` in the repository.
  pkg.optionalDependencies = {
    ...pkg.optionalDependencies,
    ...Object.fromEntries(platforms().map((p) => [`@krishiv/engine-${p}`, version])),
  };
  writeJson(join(dir, "package.json"), pkg);
  return pack(dir, out);
}

const [cmd, ...args] = process.argv.slice(2);
if (cmd === "platform" && args.length === 3) packPlatform(args[0], resolve(args[1]), args[2]);
else if (cmd === "main" && args.length === 1) packMain(args[0]);
else if (cmd === "list" && args.length === 0) console.log(platforms().join("\n"));
else {
  console.error("usage: npm-packages.mjs platform <platform> <krishiv.node> <out dir> | main <out dir> | list");
  process.exit(2);
}
