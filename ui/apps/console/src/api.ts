// SPDX-License-Identifier: Apache-2.0
// The token comes from the URL the server printed (`/?token=...`) once: the page trades it for an HttpOnly,
// SameSite=Strict session cookie (`POST /api/internal/session`) and removes it from the address bar. No script keeps
// the token, no request carries it in an address (the event stream sends the cookie), and a request that changes
// something carries the page's own `X-Krishiv-Csrf` header, which a page on another origin cannot add.

/** Opens the session from the token in the address, if any, and says whether the server accepts this browser. */
export async function initSession(): Promise<boolean> {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("token");
  if (fromUrl) {
    url.searchParams.delete("token");
    history.replaceState(null, "", url.pathname + url.search + url.hash);
    const res = await fetch("/api/internal/session", {
      method: "POST",
      headers: { Authorization: `Bearer ${fromUrl}`, "X-Krishiv-Csrf": "1" },
    });
    return res.ok;
  }
  const res = await fetch("/api/internal/health", { headers: { "X-Krishiv-Csrf": "1" } });
  return res.ok;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, { ...init, headers: { ...(init.headers as Record<string, string>), "X-Krishiv-Csrf": "1" } });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(res.status, body.error ?? "error", body.message ?? `HTTP ${res.status}`);
  }
  return body as T;
}

export type * from "@krishiv/ui/data/types";
