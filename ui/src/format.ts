export function bytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return "–";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let v = n;
  let i = 0;
  while (Math.abs(v) >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${i === 0 ? v : v.toFixed(1)} ${units[i]}`;
}

/** A duration given in nanoseconds: `820 ms`, `4.2 s`, `4m 12s`, `2h 29m`; negative values keep their sign. */
export function nanos(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "–";
  if (n < 0) return `−${nanos(-n)}`;
  if (n < 1_000) return `${Math.round(n)} ns`;
  if (n < 1_000_000) {
    const us = Math.round(n / 100) / 10;
    return us >= 1000 ? "1 ms" : `${us.toFixed(1)} µs`;
  }
  const ms = Math.round(n / 1_000_000);
  if (ms < 1000) return `${ms} ms`;
  const tenths = Math.round(n / 100_000_000);
  if (tenths < 600) return `${(tenths / 10).toFixed(1)} s`;
  const total = Math.round(n / 1_000_000_000);
  if (total < 3600) return `${Math.floor(total / 60)}m ${total % 60}s`;
  return `${Math.floor(total / 3600)}h ${Math.floor((total % 3600) / 60)}m`;
}

export function count(n: number | null | undefined): string {
  return n === null || n === undefined ? "–" : n.toLocaleString("en-US");
}

export function percent(r: number | null | undefined): string {
  return r === null || r === undefined ? "–" : `${(r * 100).toFixed(1)}%`;
}

export function seconds(s: number): string {
  return s < 1 ? "0 s" : nanos(s * 1_000_000_000);
}

/** Microseconds since the epoch, as the system tables hold them. */
export function when(micros: number | null | undefined): string {
  if (micros === null || micros === undefined) return "–";
  return new Date(micros / 1000).toLocaleString();
}

/** How long a running query has run, from its submission time. */
export function elapsed(micros: number | undefined, nowMs: number): string {
  if (micros === undefined) return "–";
  return nanos(Math.max(0, nowMs * 1000 - micros) * 1000);
}

export function oneLine(text: string | null | undefined, max = 120): string {
  if (!text) return "–";
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > max ? `${t.slice(0, max)}…` : t;
}
