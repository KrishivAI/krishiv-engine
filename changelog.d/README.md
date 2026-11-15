# Changelog fragments

Every pull request adds one file here, `changelog.d/<pr number>.<type>.md`, with one or two lines for a user of
Krishiv. At release, `cargo xtask changelog assemble <version>` writes them into `CHANGELOG.md` and deletes them;
`cargo xtask changelog pending` shows what the next release would list.

Types, in the order a release lists them: `breaking`, `added`, `performance`, `compat` (Spark compatibility),
`fixed`, `deprecated`, `removed`, `security`.

Write what a user can now do or what changed for them. Leave out internal function names, internal plans and
competitor names. For `performance`, give the measured number and where it was measured.

A pull request that only changes CI or docs carries the `no-changelog` label instead.
