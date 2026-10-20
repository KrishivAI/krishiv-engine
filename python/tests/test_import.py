# SPDX-License-Identifier: Apache-2.0
import krishiv


def test_version_is_str():
    assert isinstance(krishiv.__version__, str)
    assert krishiv.__version__


def test_build_info_reports_mimalloc():
    # Every shipped wheel (abi3 and free-threaded) links mimalloc.
    assert krishiv.build_info() == {"version": krishiv.__version__, "allocator": "mimalloc"}


def _env_after_import(env: dict[str, str]) -> dict[str, str]:
    import json
    import os
    import subprocess
    import sys

    code = (
        "import json, os, krishiv; "
        "print(json.dumps({k: os.environ.get(k) for k in "
        "('MIMALLOC_PURGE_DELAY', 'MIMALLOC_ARENA_EAGER_COMMIT')}))"
    )
    base = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}
    out = subprocess.run([sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_allocator_settings_match_the_engine_and_user_values_win():
    import pathlib
    import re

    # The engine's constant (krishiv_common::alloc::ALLOCATOR_ENV) is what the package sets.
    alloc_rs = pathlib.Path(__file__).resolve().parents[2] / "crates/krishiv-common/src/alloc.rs"
    block = alloc_rs.read_text().split("pub const ALLOCATOR_ENV", 1)[1].split("];", 1)[0]
    engine = dict(re.findall(r'\("(MIMALLOC_[A-Z_]+)", "([^"]*)"\)', block))
    assert engine == {"MIMALLOC_PURGE_DELAY": "10", "MIMALLOC_ARENA_EAGER_COMMIT": "0"}
    assert _env_after_import({}) == engine
    assert _env_after_import({"MIMALLOC_PURGE_DELAY": "3"}) == {**engine, "MIMALLOC_PURGE_DELAY": "3"}
