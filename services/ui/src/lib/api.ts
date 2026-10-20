export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(),
    options.body instanceof FormData ? 120_000 : 60_000,
  );
  try {
    const response = await fetch(`/api${path}`, {
      ...options,
      signal: options.signal ?? controller.signal,
      headers: {
        ...(options.body && !(options.body instanceof FormData)
          ? { "Content-Type": "application/json" }
          : {}),
        ...options.headers,
      },
    });
    const text = await response.text();
    let payload: unknown;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch {
      throw new ApiError(
        `Server returned an unexpected response (${response.status}). Check that the API is available.`,
        response.status,
      );
    }
    if (!response.ok) {
      const detail =
        payload && typeof payload === "object" && "detail" in payload
          ? payload.detail
          : payload;
      throw new ApiError(
        typeof detail === "string" ? detail : JSON.stringify(detail),
        response.status,
      );
    }
    return payload as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new Error(
      error instanceof Error && error.name === "AbortError"
        ? "Request timed out. Check the API and try again."
        : "Could not reach the API. Check the connection and try again.",
    );
  } finally {
    clearTimeout(timer);
  }
}
export const json = (method: string, body: unknown): RequestInit => ({
  method,
  body: JSON.stringify(body),
});
export const activeRun = (status?: string) =>
  ["queued", "running"].includes((status ?? "").toLowerCase());
