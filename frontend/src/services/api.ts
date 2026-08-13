import { mockService } from "./mockService";
import { 
  ChatResponse, 
  DocumentMetadataResponse, 
  DocumentPageResponse, 
  DocumentHighlightResponse 
} from "../types/api";
import type { LLMModelId, LLMModelOption } from "../types/api";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Keep model selection usable while the dynamic backend catalog is loading or
// temporarily unreachable. The backend remains authoritative and validates the
// selected model when /chat is submitted.
export const FALLBACK_LLM_MODELS: LLMModelOption[] = [
  {
    id: "gemma",
    label: "Gemma 3 4B (lokal)",
    available: true,
    loaded: false,
    provider: "gemma",
    model_id: "gemma-4-E4B-it-Q4_K_M.gguf",
    device: "local",
    is_default: true,
  },
  {
    id: "groq_gpt_oss_20b",
    label: "Groq GPT-OSS 20B (sürətli)",
    available: true,
    loaded: false,
    provider: "groq",
    model_id: "openai/gpt-oss-20b",
    device: "remote:groq",
    is_default: false,
  },
  {
    id: "groq_gpt_oss_120b",
    label: "Groq GPT-OSS 120B",
    available: true,
    loaded: false,
    provider: "groq",
    model_id: "openai/gpt-oss-120b",
    device: "remote:groq",
    is_default: false,
  },
  {
    id: "nvidia_gpt_oss",
    label: "NVIDIA GPT-OSS 120B",
    available: true,
    loaded: false,
    provider: "nvidia_gpt_oss",
    model_id: "openai/gpt-oss-120b",
    device: "remote:nvidia",
    is_default: false,
  },
];

// Helper to determine if we should use mock API
const useMockApi = (): boolean => {
  const envMockSetting = import.meta.env.VITE_USE_MOCK_API;
  
  // Allow toggling in localStorage so the user can change it dynamically in Settings
  const localOverride = localStorage.getItem("reguaz-use-mock-api");
  if (localOverride !== null) {
    return localOverride === "true";
  }

  return envMockSetting === "true";
};

// Response helper
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const message = errorData.error?.message || `API error: ${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const apiService = {
  // Toggle check
  isMockEnabled: () => useMockApi(),
  
  // Set toggle dynamically
  setMockEnabled: (enabled: boolean) => {
    localStorage.setItem("reguaz-use-mock-api", enabled ? "true" : "false");
  },

  // 1. GET /health
  getHealth: async (): Promise<{ status: string; app: string }> => {
    if (useMockApi()) {
      return mockService.getHealth();
    }
    
    try {
      const res = await fetch(`${API_URL}/health`);
      return handleResponse<{ status: string; app: string }>(res);
    } catch (error) {
      console.warn("Failed to connect to backend, falling back to Mock API:", error);
      return mockService.getHealth();
    }
  },

  // 2. POST /chat
  postChat: async (
    question: string,
    sessionId?: string | null,
    llmModel?: LLMModelId | null,
  ): Promise<ChatResponse> => {
    if (useMockApi()) {
      return mockService.postChat(question, sessionId, llmModel);
    }

    const res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        session_id: sessionId,
        llm_model: llmModel,
      }),
    });
    return handleResponse<ChatResponse>(res);
  },

  getLlmModels: async (): Promise<LLMModelOption[]> => {
    if (useMockApi()) {
      return mockService.getLlmModels();
    }
    try {
      const res = await fetch(`${API_URL}/llm-models`);
      return handleResponse<LLMModelOption[]>(res);
    } catch (error) {
      console.warn(
        "Backend model catalog is unavailable; using the local selection catalog:",
        error,
      );
      return FALLBACK_LLM_MODELS;
    }
  },

  // 3. GET /documents
  getDocuments: async (): Promise<DocumentMetadataResponse[]> => {
    if (useMockApi()) {
      return mockService.getDocuments();
    }

    try {
      const res = await fetch(`${API_URL}/documents`);
      return handleResponse<DocumentMetadataResponse[]>(res);
    } catch (e) {
      // If endpoint is not found or not connected, return mock adapter list
      console.warn("Real /documents endpoint failed, using mock data adapter:", e);
      return mockService.getDocuments();
    }
  },

  // 4. GET /documents/:id
  getDocumentMetadata: async (documentId: string): Promise<DocumentMetadataResponse | null> => {
    if (useMockApi()) {
      return mockService.getDocumentMetadata(documentId);
    }

    const res = await fetch(`${API_URL}/documents/${encodeURIComponent(documentId)}`);
    return handleResponse<DocumentMetadataResponse>(res);
  },

  // 5. GET /documents/:id/page/:page_number
  getDocumentPage: async (documentId: string, pageNumber: number): Promise<DocumentPageResponse | null> => {
    if (useMockApi()) {
      return mockService.getDocumentPage(documentId, pageNumber);
    }

    const res = await fetch(`${API_URL}/documents/${encodeURIComponent(documentId)}/page/${pageNumber}`);
    return handleResponse<DocumentPageResponse>(res);
  },

  // 6. GET /documents/highlight
  getHighlight: async (documentId: string, chunkId: string): Promise<DocumentHighlightResponse | null> => {
    if (useMockApi()) {
      return mockService.getHighlight(documentId, chunkId);
    }

    const res = await fetch(
      `${API_URL}/documents/highlight?document_id=${encodeURIComponent(documentId)}&chunk_id=${encodeURIComponent(chunkId)}`
    );
    return handleResponse<DocumentHighlightResponse>(res);
  },

  // 7. Conversation History Hooks placeholder (GET, POST, DELETE /history)
  getHistory: async (): Promise<any> => {
    // History is managed in Zustand and local mock storage, representing Auth / session states
    await new Promise(r => setTimeout(r, 100));
    return [];
  },

  postHistory: async (session: any): Promise<any> => {
    await new Promise(r => setTimeout(r, 100));
    return session;
  },

  deleteHistory: async (_sessionId: string): Promise<boolean> => {
    await new Promise(r => setTimeout(r, 100));
    return true;
  }
};
