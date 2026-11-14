#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reference port of Spark 4.2 XORShiftRandom + SampleExec (BernoulliCellSampler), used by gen_tablesample.py to self-check the fixture (SQL-14)."""

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1

def _rotl(x, r): return ((x << r) | (x >> (32 - r))) & M32
def _mix_last(h, k):
    k = (k * 0xCC9E2D51) & M32; k = _rotl(k, 15); k = (k * 0x1B873593) & M32
    return h ^ k
def _mix(h, k):
    h = _mix_last(h, k); h = _rotl(h, 13); return (h * 5 + 0xE6546B64) & M32
def _fin(h, n):
    h ^= n; h ^= h >> 16; h = (h * 0x85EBCA6B) & M32; h ^= h >> 13; h = (h * 0xC2B2AE35) & M32; h ^= h >> 16
    return h
def bytes_hash(data, seed):  # scala.util.hashing.MurmurHash3.bytesHash
    h = seed & M32; i = 0; n = len(data)
    while n - i >= 4:
        k = data[i] | data[i+1] << 8 | data[i+2] << 16 | data[i+3] << 24
        h = _mix(h, k); i += 4
    rem = n - i; k = 0
    if rem == 3: k ^= data[i+2] << 16
    if rem >= 2: k ^= data[i+1] << 8
    if rem >= 1:
        k ^= data[i]; h = _mix_last(h, k)
    return _fin(h, n)

def hash_seed(seed):
    b = (seed & M64).to_bytes(8, "big")
    low = bytes_hash(b, 1007110753)
    high = bytes_hash(b, low)
    return ((high << 32) | low) & M64

class XORShiftRandom:
    def __init__(self, seed): self.s = hash_seed(seed)
    def next(self, bits):
        x = self.s
        x ^= (x << 21) & M64; x ^= x >> 35; x ^= (x << 4) & M64
        self.s = x
        return x & ((1 << bits) - 1)
    def next_double(self):  # java.util.Random.nextDouble
        return ((self.next(26) << 27) + self.next(27)) * (1.0 / (1 << 53))

def bernoulli_cell(rows_by_partition, lb, ub, seed):
    """SampleExec without replacement: partition i uses XORShiftRandom(seed + i); keep iff lb <= x < ub."""
    out = []
    for i, part in enumerate(rows_by_partition):
        rng = XORShiftRandom((seed + i))  # seed + i wraps as a signed 64-bit long
        for r in part:
            if ub - lb <= 0.0: continue
            x = rng.next_double()
            if lb <= x < ub: out.append(r)
    return out

if __name__ == "__main__":
    import struct
    r = XORShiftRandom(7)
    print(struct.pack(">d", r.next_double()).hex(), "3fd20a5781c87c7c")
    print(hash_seed(7) - (1 << 64) if hash_seed(7) >> 63 else hash_seed(7), -2714775767673150655)
    got = bernoulli_cell([list(range(1000))], 0.0, 0.1, 7)
    print(len(got), got[:7])
