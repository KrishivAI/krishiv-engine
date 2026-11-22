// The embedded view (P3-T071): `?embed=1&exp=..&sig=..[&session=..][&theme=..]` on `/queries/{id}/profile`. The
// signed link is the only credential (one query, of the session it names if any, read-only, bounded life); the
// page keeps no session. An embedding console may theme it with a `theme` parameter or a `postMessage` from an
// allowed origin; both pass `validateTheme`, which
// mirrors krishiv-server's `embed::validate_theme`: colours, font names and radii only, never a URL or CSS text.

const COLOURS = [
  "accent", "bg", "bg-subtle", "border", "border-strong", "code-bg", "error", "error-bg", "focus", "info", "info-bg",
  "muted", "primary", "primary-hover", "on-primary", "success", "success-bg", "surface", "surface-2", "text", "text-2",
  "warning", "warning-bg",
];
const FONTS = ["font-display", "font-mono", "font-text"];
const RADII = ["radius-sm", "radius-md", "radius-lg", "radius-xl"];

export type Theme = { mode?: "light" | "dark" | "auto"; tokens: Record<string, string> };

const isColour = (s: string) => /^#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/.test(s);
const isFont = (s: string) => /^[A-Za-z0-9-][A-Za-z0-9 -]{0,63}$/.test(s);
const isLength = (s: string) => {
  const m = /^(\d{1,2}(?:\.\d{1,3})?)(px|rem)$/.exec(s);
  return m !== null && Number(m[1]) <= 64;
};

/** The theme if it is valid, else the reason it is not (the page ignores it and logs the reason). */
export function validateTheme(v: unknown): { ok: true; theme: Theme } | { ok: false; reason: string } {
  if (typeof v !== "object" || v === null || Array.isArray(v)) return { ok: false, reason: "the theme is not an object" };
  const theme: Theme = { tokens: {} };
  for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
    if (k === "mode") {
      if (val !== "light" && val !== "dark" && val !== "auto") return { ok: false, reason: "mode is light, dark or auto" };
      theme.mode = val;
    } else if (k === "tokens") {
      if (typeof val !== "object" || val === null || Array.isArray(val)) return { ok: false, reason: "tokens is not an object" };
      for (const [name, value] of Object.entries(val as Record<string, unknown>)) {
        if (typeof value !== "string") return { ok: false, reason: `token ${name}: the value is not a string` };
        const ok = COLOURS.includes(name) ? isColour(value) : FONTS.includes(name) ? isFont(value) : RADII.includes(name) ? isLength(value) : null;
        if (ok === null) return { ok: false, reason: `token ${name}: not a theme token` };
        if (!ok) return { ok: false, reason: `token ${name}: the value is not allowed` };
        theme.tokens[name] = value;
      }
    } else {
      return { ok: false, reason: `${k}: not a theme field` };
    }
  }
  return { ok: true, theme };
}

/** Applies a checked theme to the page: the brand tokens are CSS custom properties, `mode` is `data-theme`. */
export function applyTheme(theme: Theme, root: HTMLElement = document.documentElement): void {
  if (theme.mode === "light" || theme.mode === "dark") root.setAttribute("data-theme", theme.mode);
  else if (theme.mode === "auto") root.removeAttribute("data-theme");
  for (const [name, value] of Object.entries(theme.tokens)) root.style.setProperty(`--k-${name}`, value);
}

export type EmbedLink = { exp: string; sig: string; theme: string | null };

/** The embed parameters of the address, or null when this is not the embedded view. */
export function embedLink(search: string): EmbedLink | null {
  const p = new URLSearchParams(search);
  // Exactly one `embed`, as the server decides framing (a repeated parameter is not the embedded view).
  const all = p.getAll("embed");
  if (all.length !== 1 || all[0] !== "1") return null;
  return { exp: p.get("exp") ?? "", sig: p.get("sig") ?? "", theme: p.get("theme") };
}

/** The origins as the browser reports them (`MessageEvent.origin` has no default port). */
function canonicalOrigins(origins: string[]): Set<string> {
  return new Set(
    origins.map((o) => {
      try {
        return new URL(o).origin;
      } catch {
        return o;
      }
    }),
  );
}

/**
 * Takes theme messages from the embedding page: only from an allowed origin, only valid ones; the rest is logged.
 * The listener is installed at once; messages from the parent that arrive before the allowed origins are known
 * (`origins` still pending) are held and checked then, so a theme sent when the frame loads is not lost.
 */
export function listenForTheme(
  origins: string[] | Promise<string[]>,
  parentWindow: Window = window,
): () => void {
  let allowed: Set<string> | null = null;
  let held: MessageEvent[] = [];
  const handle = (e: MessageEvent) => {
    if (allowed === null || !allowed.has(e.origin)) {
      console.warn(`krishiv: ignored a theme message from ${e.origin} (not in krishiv.ui.allowedOrigins)`);
      return;
    }
    const data = e.data as { type?: unknown; theme?: unknown } | null;
    if (data === null || typeof data !== "object" || data.type !== "krishiv.theme") return;
    const r = validateTheme(data.theme);
    if (!r.ok) {
      console.warn(`krishiv: ignored an invalid theme message: ${r.reason}`);
      return;
    }
    applyTheme(r.theme);
  };
  const on = (e: MessageEvent) => {
    if (e.source !== parentWindow.parent) return;
    if (allowed === null) held.push(e);
    else handle(e);
  };
  parentWindow.addEventListener("message", on);
  const ready = (list: string[]) => {
    allowed = canonicalOrigins(list);
    const queued = held;
    held = [];
    for (const e of queued) handle(e);
  };
  if (Array.isArray(origins)) ready(origins);
  // If the origins never arrive, every message stays held (and is never applied).
  else origins.then(ready, () => {});
  return () => parentWindow.removeEventListener("message", on);
}
