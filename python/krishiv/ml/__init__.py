# SPDX-License-Identifier: Apache-2.0
"""Not supported: importing this raises ``KRISHIV_UNSUPPORTED_FEATURE`` naming what to use instead."""

from krishiv.errors import unsupported_module

raise unsupported_module("krishiv.ml, the MLlib API (there is no machine-learning library)")
