# Security Policy

Krishiv is **pre-alpha** and has no released versions. Only the `main` branch is supported; fixes land there.

## Reporting a vulnerability

Please do not open a public issue for security problems. Report privately through GitHub Security Advisories:
open the repository's **Security** tab and choose **Report a vulnerability**.

Include affected commit or version, a description, and reproduction steps if you have them. We aim to acknowledge
a report within 7 days and to share an assessment or fix plan within 30 days. This is a best-effort project, so
these are goals, not guarantees. We will credit reporters in the advisory unless you ask us not to.

## Scope

The Python UDF process pool provides process isolation, not a sandbox. Do not run untrusted code in it; use WASM
UDFs for untrusted code.
