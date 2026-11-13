# SPDX-License-Identifier: Apache-2.0
"""PySpark Connect test-suite runner and pass-rate dashboard (P3-T030).

Design: krishiv-internal `design/P3-pyspark-suite-runner.md`. The modules are stdlib-only so the
runner works in a bare venv next to `pyspark-client`; the Spark test sources are never imported
by anything here except through pytest in the CI sandbox.
"""
