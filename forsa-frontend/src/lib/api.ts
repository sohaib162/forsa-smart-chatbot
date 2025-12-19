/**
 * API Service for Document Retrieval
 * ==================================
 * Handles all API calls to the FastAPI backend for document management.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface Document {
  s3_key: string;
  filename: string;
  category: string;
  ext: string;
  lang: 'AR' | 'FR';
}

export interface DocumentsResponse {
  total: number;
  documents: Document[];
  error?: string;
  message?: string;
}

export interface DocumentFilters {
  category?: string;
  lang?: 'AR' | 'FR';
  q?: string;
}

/**
 * Fetch all documents with optional filtering
 */
export async function fetchDocuments(filters?: DocumentFilters): Promise<DocumentsResponse> {
  const params = new URLSearchParams();

  if (filters?.category) params.append('category', filters.category);
  if (filters?.lang) params.append('lang', filters.lang);
  if (filters?.q) params.append('q', filters.q);

  const url = `${API_BASE_URL}/documents${params.toString() ? `?${params.toString()}` : ''}`;

  // ✅ SIMPLE GET: no custom headers => no preflight
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to fetch documents: ${response.status} ${response.statusText}`);
  }

  return response.json();
}


/**
 * Get the URL for viewing/downloading a document
 * @param s3_key The S3 key of the document
 * @returns The full URL to access the document via API proxy
 */
export function getDocumentUrl(s3_key: string): string {
  // URL encode the s3_key to handle special characters and spaces
  const encodedKey = encodeURIComponent(s3_key);
  return `${API_BASE_URL}/document/${encodedKey}`;
}

/**
 * Open a document in a new tab
 */
export function openDocument(s3_key: string): void {
  const url = getDocumentUrl(s3_key);
  window.open(url, '_blank');
}

/**
 * Download a document with proper filename
 */
export async function downloadDocument(s3_key: string, filename: string): Promise<void> {
  const url = getDocumentUrl(s3_key);

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to download: ${response.statusText}`);
    }

    const blob = await response.blob();
    const blobUrl = window.URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Clean up
    window.URL.revokeObjectURL(blobUrl);
  } catch (error) {
    console.error('Download failed:', error);
    throw error;
  }
}

/**
 * Check if API is healthy
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
