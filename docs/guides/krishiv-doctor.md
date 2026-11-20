---
description: "Check a krishiv server configuration before you start it: settings, authentication, TLS files, ports, directories and path allow lists."
---

# Check a server configuration with krishiv doctor

> `krishiv doctor` is not part of the 0.2.0 release; this page describes the command as it is being built.

`krishiv doctor` takes the same `--conf KEY=VALUE`, `--host`, `--port`, `--connect-port` and `--allow-root` options as `krishiv server`, checks the configuration they make, and says what to change. It does not start a server and does not use the network. Run it before you start the server, and in the pipeline that deploys it.

```bash
krishiv doctor --host 0.0.0.0 \
  --conf krishiv.auth.tokens.ops.secret=file:/etc/krishiv/ops.token \
  --conf krishiv.auth.tokens.ops.principal=ops \
  --conf krishiv.auth.tokens.ops.role=admin
```

```text
ok      config          3 settings parsed
FAIL    listener        refusing to serve on 0.0.0.0: ... bearer tokens on a non-loopback address need TLS ...
                        fix: bind a loopback address, or configure authentication and TLS as the message says
ok      authentication  configured, secrets resolve
ok      tls             off (plaintext listeners)
...
1 check(s) failed, 0 warning(s)
```

The exit status is 0 when no check failed (warnings are allowed) and 1 when one did. `--json` prints `{"ok": bool, "checks": [{"name", "status", "detail", "fix"}]}` for a script.

## What it checks

`config`, `listener`, `authentication` and `tls` run the code `krishiv server` runs at startup, so a configuration the doctor passes is one the server accepts, and the other way round.

| Check | Fails when |
| --- | --- |
| `config` | a setting does not parse, or names a Spark Connect setting Krishiv does not read |
| `listener` | the address is not loopback and there is no authentication, a bearer token or OIDC has no TLS, or client certificates have no TLS |
| `authentication` | a secret file or variable cannot be read (an OIDC or assertion issuer is not contacted), or a secret file is readable by group or others. A warning when it is off on a loopback address |
| `tls` | the certificate or key does not load, or the key is readable by group or others |
| `ports` | the Spark Connect or admin port is taken |
| `user` | the process is root without `--allow-root` |
| `directories` | `krishiv.spill.dir`, `krishiv.io.cache.path` or `krishiv.checkpoint.dir` is set and is not a directory the server can write |
| `paths` | `krishiv.io.local.allowedRoots` holds a path that is not an existing absolute directory, or `krishiv.io.objectStore.allowedPrefixes` holds something that is not an object-store prefix. A warning when authentication is on and neither list is set |

Two checks are `skipped`, with the reason: the `object store` (credentials and reachability need the network) and `catalogs` (declared catalogs are not in this release).
