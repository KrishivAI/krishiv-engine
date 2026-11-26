#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Installs the packed `krishiv` tarball and one `@krishiv/engine-<platform>` tarball from IN into a scratch project
# outside the repository and runs js/smoke/smoke.mjs under each RUNTIME (publish-js.yml); under Node also
# js/smoke/smoke.cjs, the CommonJS build.
#
#   node-smoke.sh IN RUNTIME...   (RUNTIME: node | bun | deno; bun and deno are installed from npm into a scratch dir)
#
# `--omit=optional`: before the first publish the other platform packages that `krishiv` names are not on the
# registry; the one under test is a direct dependency from its tarball, so it is installed.
set -euo pipefail
in=$(cd "$1" && pwd)
shift
repo=$(cd "$(dirname "$0")/../.." && pwd)
app=$(mktemp -d)
cp "$repo/js/smoke/smoke.mjs" "$repo/js/smoke/smoke.cjs" "$app/"
cd "$app"
echo '{"name": "krishiv-smoke", "private": true, "type": "module"}' > package.json
node --version
npm install --omit=optional --no-audit --no-fund "$in"/*.tgz
unset KRISHIV_NATIVE_ENGINE
for rt in "$@"; do
  case "$rt" in
    node) node smoke.mjs && node smoke.cjs ;;
    bun | deno)
      tools=$(mktemp -d)
      # Major versions only: these are the runtimes a user installs; a new release breaking the addon is news.
      if [ "$rt" = bun ]; then pkg=bun@1; else pkg=deno@2; fi
      npm install --no-save --no-audit --no-fund --prefix "$tools" "$pkg"
      "$tools/node_modules/.bin/$rt" --version
      if [ "$rt" = bun ]; then
        "$tools/node_modules/.bin/bun" smoke.mjs
      else
        "$tools/node_modules/.bin/deno" run --allow-all smoke.mjs
      fi
      ;;
    *)
      echo "unknown runtime $rt" >&2
      exit 2
      ;;
  esac
  echo "smoke ($rt): ok"
done
