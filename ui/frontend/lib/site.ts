/**
 * GitHub project pages serve the app under /<repo-name>. NEXT_PUBLIC_BASE_PATH
 * must match that prefix (leading slash, no trailing slash), or be empty for
 * apex / custom-domain deploys.
 */
export function getSiteBasePath(): string {
  const raw = process.env.NEXT_PUBLIC_BASE_PATH?.trim();
  if (!raw || raw === "/") return "";
  return raw.startsWith("/") ? raw : `/${raw}`;
}

export const siteBasePath = getSiteBasePath();

export function withBasePath(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  if (!siteBasePath) return p;
  return `${siteBasePath}${p}`;
}

export const isStaticExport = process.env.NEXT_PUBLIC_STATIC_EXPORT === "true";
