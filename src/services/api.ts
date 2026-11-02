import type {
  AnalysisResult,
  BackendHealth,
  DatasetPreview,
  DatasetSummary,
  ModelEvaluation,
  ModelId,
  ModelStatus,
} from "../types";

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/+$/, "");
const healthBaseUrl = apiBaseUrl.endsWith("/api")
  ? apiBaseUrl.slice(0, -4)
  : apiBaseUrl;
const requestTimeoutMs = 30_000;

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  baseUrl = apiBaseUrl,
  timeoutMs = requestTimeoutMs,
): Promise<T> {
  let response: Response;
  let body: string;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    response = await fetch(`${baseUrl}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
    body = await response.text();
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error("The SentiScope API request timed out. Check backend status and try again.");
    }
    throw new Error("Cannot reach the SentiScope API. Start the backend and try again.", {
      cause: error,
    });
  } finally {
    window.clearTimeout(timeout);
  }

  const contentType = response.headers.get("content-type") ?? "unknown content type";
  let payload: unknown;
  try {
    payload = JSON.parse(body);
  } catch {
    const status = `HTTP ${response.status}`;
    if (!response.ok) {
      throw new Error(`The SentiScope API request failed (${status}; ${contentType}). Check the backend response and API proxy.`);
    }
    throw new Error(`The SentiScope API returned a non-JSON response (${status}; ${contentType}). Check the API URL and development proxy.`);
  }

  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof payload.detail === "string"
        ? payload.detail
        : typeof payload === "object" &&
            payload !== null &&
            "detail" in payload &&
            Array.isArray(payload.detail)
          ? payload.detail
              .map((item) =>
                typeof item === "object" && item !== null && "msg" in item && typeof item.msg === "string"
                  ? item.msg
                  : "",
              )
              .filter(Boolean)
              .join("; ")
        : `The API request failed (HTTP ${response.status}).`;
    throw new Error(detail || `The API request failed (HTTP ${response.status}).`);
  }

  return payload as T;
}

export function getHealth(): Promise<BackendHealth> {
  return requestJson("/health", undefined, healthBaseUrl);
}

export function getDatasetSummary(): Promise<DatasetSummary> {
  return requestJson("/dataset/summary");
}

export function getDatasetPreview(params: {
  q?: string;
  label?: string;
  split?: string;
  limit?: number;
  offset?: number;
}): Promise<DatasetPreview> {
  const query = new URLSearchParams();
  if (params.q) query.set("q", params.q);
  if (params.label && params.label !== "all") query.set("label", params.label);
  query.set("split", params.split ?? "train");
  query.set("limit", String(params.limit ?? 8));
  query.set("offset", String(params.offset ?? 0));
  return requestJson(`/dataset/preview?${query.toString()}`);
}

export function analyzeText(text: string, model: ModelId): Promise<AnalysisResult> {
  return requestJson("/analyze", {
    method: "POST",
    body: JSON.stringify({ text, model }),
  });
}

export function trainModels(force = false): Promise<{ models: ModelStatus[]; messages: string[] }> {
  return requestJson("/models/train", {
    method: "POST",
    body: JSON.stringify({ force }),
  }, apiBaseUrl, 10 * 60 * 1000);
}

export function getModelEvaluation(): Promise<ModelEvaluation> {
  return requestJson("/models/evaluation");
}

export function reloadDataset(): Promise<BackendHealth["dataset"]> {
  return requestJson("/dataset/reload", { method: "POST" }, apiBaseUrl, 10 * 60 * 1000);
}
