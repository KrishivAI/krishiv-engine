#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Import the built abi3 wheel under user-mode QEMU with a pre-AVX2 CPU (Nehalem) and require a
# clean ImportError that names the missing features and the compat package, not SIGILL.
#
# Everything the emulated process loads must itself run on a Nehalem: the guest runs in the PyPA
# manylinux_2_28 image (baseline glibc and CPython) with the static host qemu mounted in. It must
# not borrow the runner's userland (`qemu-x86_64 -L /`): the runner's own libraries can be built
# for a newer CPU level, which dies with SIGILL before krishiv is even reached.
#
# usage: oldcpu-import.sh <qemu-x86_64 (static)> <wheel> <manylinux image>
set -euo pipefail

qemu=$1
wheel=$2
image=$3
py=/opt/python/cp311-cp311/bin/python

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
install -m 0755 "$qemu" "$work/qemu-x86_64"
cp "$wheel" "$work/"
# qemu-user shows the host's /proc/cpuinfo, so the check is pointed at a Nehalem-class one;
# what is proved is that the ImportError comes before the native module is loaded (which would
# die with SIGILL under -cpu Nehalem).
printf 'flags\t\t: fpu sse sse2 pni ssse3 cx16 sse4_1 sse4_2 popcnt\n' > "$work/nehalem-cpuinfo"

cat > "$work/inner.sh" <<INNER
set -euo pipefail
$py -m pip install --quiet --no-deps --no-index --target /w/site /w/$(basename "$wheel")
# The interpreter and libc must start under the emulated CPU, or the next step proves nothing.
/w/qemu-x86_64 -cpu Nehalem $py -c 'import sys; print(sys.version)'
set +e
KRISHIV_CPUINFO=/w/nehalem-cpuinfo PYTHONPATH=/w/site \
  /w/qemu-x86_64 -cpu Nehalem $py -c 'import krishiv' >/w/old.out 2>&1
echo \$? >/w/old.code
INNER

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$work:/w" "$image" bash /w/inner.sh
cat "$work/old.out"
code=$(cat "$work/old.code")
test "$code" -eq 1
grep -q 'ImportError' "$work/old.out"
if grep -qi 'illegal instruction' "$work/old.out"; then
  echo "::error::import hit SIGILL under QEMU Nehalem"
  exit 1
fi
grep -q 'avx2' "$work/old.out"
grep -q 'krishiv-compat' "$work/old.out"
