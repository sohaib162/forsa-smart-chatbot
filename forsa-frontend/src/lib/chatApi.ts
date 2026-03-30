/**
 * Chat API Service
 * ================
 * Handles API calls to the chatbot backend.
 *
 * Environment variable: VITE_CHAT_API_BASE_URL (default: http://localhost:8001)
 */

const CHAT_API_BASE_URL =
  import.meta.env.VITE_CHAT_API_BASE_URL || "http://localhost:8001";

/** Response shape from the chatbot API. */
export interface ChatResponse {
  status: string;
  category: string;
  answer: string;
  sources: Array<{
    s3_key: string;
    filename: string;
    category: string;
    ext: string;
    lang: "AR" | "FR";
  }>;
  /** Citation faithfulness score (0.0–1.0), returned by conventions pipeline. */
  faithfulness?: number;
  /** Number of knowledge graph nodes used in retrieval. */
  kg_nodes_used?: number;
  /** List of argument types found (PENALTY, OBLIGATION, etc.). */
  argument_types?: string[];
  /** End-to-end latency in milliseconds. */
  latency_ms?: number;
}

export interface ChatRequestPayload {
  equipe: string;
  question: {
    categorie_id: Record<string, string>;
  };
}

/**
 * Process a question through the chatbot API.
 *
 * Includes a request timeout of 2 minutes to match the backend pipeline timeout.
 */
export async function processQuestion(
  payload: ChatRequestPayload
): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000); // 2 min

  try {
    const response = await fetch(`${CHAT_API_BASE_URL}/process-question`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      throw new Error(
        `Chat API error: ${response.status} ${response.statusText}${errorText ? ` — ${errorText}` : ""}`
      );
    }

    return response.json();
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(
        "La requête a expiré. Le serveur met trop de temps à répondre."
      );
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}