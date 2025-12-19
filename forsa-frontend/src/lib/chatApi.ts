/**
 * Chat API Service
 * Handles API calls to the chatbot backend
 */

const CHAT_API_BASE_URL = import.meta.env.VITE_CHAT_API_BASE_URL || 'http://localhost:8001';

export interface ChatResponse {
  status: string;
  category: string;
  answer: string;
  sources: Array<{
    s3_key: string;
    filename: string;
    category: string;
    ext: string;
    lang: 'AR' | 'FR';
  }>;
}

export interface ChatRequestPayload {
  equipe: string;
  question: {
    categorie_id: Record<string, string>;
  };
}

/**
 * Process a question through the chatbot API
 */
export async function processQuestion(payload: ChatRequestPayload): Promise<ChatResponse> {
  const response = await fetch(`${CHAT_API_BASE_URL}/process-question`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Chat API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}