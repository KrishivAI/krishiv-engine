// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";
import { bytes, count, elapsed, nanos, oneLine, percent, seconds } from "./format";

describe("bytes", () => {
  it("keeps whole bytes and gives one decimal above them", () => {
    expect(bytes(0)).toBe("0 B");
    expect(bytes(1023)).toBe("1023 B");
    expect(bytes(1024)).toBe("1.0 KiB");
    expect(bytes(1536)).toBe("1.5 KiB");
    expect(bytes(5 * 1024 ** 3)).toBe("5.0 GiB");
  });
  it("stops at TiB and keeps the sign", () => {
    expect(bytes(2048 * 1024 ** 4)).toBe("2048.0 TiB");
    expect(bytes(-2048)).toBe("-2.0 KiB");
  });
  it("shows a dash for a missing figure", () => {
    expect(bytes(null)).toBe("–");
    expect(bytes(undefined)).toBe("–");
  });
});

describe("nanos", () => {
  it("walks the units", () => {
    expect(nanos(999)).toBe("999 ns");
    expect(nanos(1_500)).toBe("1.5 µs");
    expect(nanos(999_960)).toBe("1 ms");
    expect(nanos(820_000_000)).toBe("820 ms");
    expect(nanos(4_200_000_000)).toBe("4.2 s");
    expect(nanos(252_000_000_000)).toBe("4m 12s");
    expect(nanos(8_940_000_000_000)).toBe("2h 29m");
  });
  it("keeps the sign of a negative duration", () => {
    expect(nanos(-820_000_000)).toBe("−820 ms");
  });
  it("shows a dash for a missing or infinite figure", () => {
    expect(nanos(null)).toBe("–");
    expect(nanos(Number.NaN)).toBe("–");
    expect(nanos(Number.POSITIVE_INFINITY)).toBe("–");
  });
});

describe("count, percent, seconds, elapsed, oneLine", () => {
  it("count groups thousands", () => {
    expect(count(1234567)).toBe("1,234,567");
    expect(count(null)).toBe("–");
  });
  it("percent has one decimal", () => {
    expect(percent(0.1234)).toBe("12.3%");
    expect(percent(undefined)).toBe("–");
  });
  it("seconds under one is 0 s", () => {
    expect(seconds(0.4)).toBe("0 s");
    expect(seconds(90)).toBe("1m 30s");
  });
  it("elapsed never goes below zero", () => {
    expect(elapsed(2_000_000, 3_000)).toBe("1.0 s");
    expect(elapsed(5_000_000, 3_000)).toBe("0 ns");
    expect(elapsed(undefined, 3_000)).toBe("–");
  });
  it("oneLine folds white space and cuts at max", () => {
    expect(oneLine("SELECT\n  1,\t2")).toBe("SELECT 1, 2");
    expect(oneLine("abcdef", 3)).toBe("abc…");
    expect(oneLine("")).toBe("–");
    expect(oneLine(null)).toBe("–");
  });
});
