import { create } from "zustand";
import { Citation, SourceDocument, MetricsResponse, LLMModelId } from "../types/api";

export interface UIHistoryItem {
  sessionId: string;
  title: string;
  updatedAt: string;
}

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceDocument[];
  metrics?: MetricsResponse;
  citations?: Citation[];
  pipelineVersion?: "v1" | "v2";
  warnings?: Array<Record<string, unknown>>;
  model?: Record<string, unknown> | null;
  isStreaming?: boolean;
}

const SELECTABLE_MODELS: LLMModelId[] = [
  "gemma",
  "nvidia_gpt_oss",
  "groq_gpt_oss_20b",
  "groq_gpt_oss_120b",
];

const browserStorage = typeof window === "undefined" ? null : window.localStorage;
const storedModel = browserStorage?.getItem("reguaz-llm-model") as LLMModelId | null;
const initialModel = storedModel && SELECTABLE_MODELS.includes(storedModel)
  ? storedModel
  : "gemma";

interface ChatState {
  history: UIHistoryItem[];
  activeSessionId: string | null;
  messages: UIMessage[];
  isGenerating: boolean;
  selectedModel: LLMModelId;
  
  // Actions
  setHistory: (history: UIHistoryItem[]) => void;
  setActiveSessionId: (sessionId: string | null) => void;
  setMessages: (messages: UIMessage[]) => void;
  setIsGenerating: (isGenerating: boolean) => void;
  setSelectedModel: (model: LLMModelId) => void;
  
  // Custom Mutation Helpers
  createNewSession: () => string;
  deleteSession: (sessionId: string) => void;
  addUserMessage: (content: string) => UIMessage;
  appendStreamChunk: (messageId: string, chunk: string) => void;
  finalizeMessage: (
    messageId: string, 
    finalContent: string, 
    sources?: SourceDocument[], 
    metrics?: MetricsResponse,
    citations?: Citation[],
    pipelineVersion?: "v1" | "v2",
    warnings?: Array<Record<string, unknown>>,
    model?: Record<string, unknown> | null,
  ) => void;
}

export const useChatStore = create<ChatState>((set) => {
  return {
    history: [
      {
        sessionId: "session-1",
        title: "Minimum nizamnamə kapitalı",
        updatedAt: "2026-07-08T18:30:00Z",
      },
      {
        sessionId: "session-2",
        title: "Kredit risklərinin tənzimlənməsi",
        updatedAt: "2026-07-08T14:15:00Z",
      }
    ],
    activeSessionId: "session-1",
    messages: [
      {
        id: "msg-1",
        role: "user",
        content: "Bankların minimum nizamnamə kapitalı ilə bağlı tələb hansıdır?",
      },
      {
        id: "msg-2",
        role: "assistant",
        content: "Azərbaycan Respublikasının Mərkəzi Bankının normativ aktlarına əsasən, yeni fəaliyyətə başlayan banklar və mövcud banklar üçün minimum nizamnamə kapitalı tələbi **50,000,000 AZN** (əlli milyon manat) məbləğində müəyyən edilmişdir [1]. Bu normativ tələb bankın maliyyə dayanıqlığını təmin etmək məqsədi daşıyır. Kapitalın formalaşdırılması yalnız nağdsız pul vəsaitləri şəklində olmalıdır [2].",
        sources: [
          {
            citation: 1,
            chunk_id: "802-IIQ_Art4_p1",
            document_id: "802-IIQ-Azərbaycan Respublikasının Mərkəzi Bankı haqqında",
            document_name: "Azərbaycan Respublikasının Mərkəzi Bankı haqqında Qanun",
            category: "laws",
            chapter: "I Fəsil",
            article: "Maddə 4. Nizamnamə kapitalı",
            page: 12,
            chunk_preview: "Bankların minimum nizamnamə kapitalının məbləği 50 milyon manat olmalıdır. Nizamnamə kapitalı yalnız pul vəsaitləri ilə ödənilə bilər..."
          },
          {
            citation: 2,
            chunk_id: "590-IIQ_Art12_p2",
            document_id: "590-IIQ-Banklar haqqında",
            document_name: "Banklar haqqında Azərbaycan Respublikasının Qanunu",
            category: "laws",
            chapter: "III Fəsil",
            article: "Maddə 12. Nizamnamə kapitalının ödənilməsi",
            page: 8,
            chunk_preview: "Bankın nizamnamə kapitalının formalaşdırılması zamanı borc və ya girov götürülmüş vəsaitlərdən, habelə digər qeyri-qanuni mənbələrdən istifadə oluna bilməz..."
          }
        ],
        metrics: {
          retrieval_time: 0.142,
          generation_time: 1.120,
          total_time: 1.262,
        }
      }
    ],
    isGenerating: false,
    selectedModel: initialModel,

    setHistory: (history) => set({ history }),
    setActiveSessionId: (activeSessionId) => set({ activeSessionId }),
    setMessages: (messages) => set({ messages }),
    setIsGenerating: (isGenerating) => set({ isGenerating }),
    setSelectedModel: (selectedModel) => {
      browserStorage?.setItem("reguaz-llm-model", selectedModel);
      set({ selectedModel });
    },

    createNewSession: () => {
      const newId = "session-" + Date.now();
      const newSession: UIHistoryItem = {
        sessionId: newId,
        title: "Yeni Söhbət",
        updatedAt: new Date().toISOString(),
      };
      set((state) => ({
        history: [newSession, ...state.history],
        activeSessionId: newId,
        messages: [],
      }));
      return newId;
    },

    deleteSession: (sessionId) => {
      set((state) => {
        const nextHistory = state.history.filter((h) => h.sessionId !== sessionId);
        const isActiveDeleted = state.activeSessionId === sessionId;
        const nextActiveId = isActiveDeleted 
          ? (nextHistory.length > 0 ? nextHistory[0].sessionId : null) 
          : state.activeSessionId;

        return {
          history: nextHistory,
          activeSessionId: nextActiveId,
          messages: nextActiveId === state.activeSessionId ? state.messages : [],
        };
      });
    },

    addUserMessage: (content) => {
      const newUserMsg: UIMessage = {
        id: "msg-user-" + Date.now(),
        role: "user",
        content,
      };

      set((state) => {
        // Auto-update conversation title if it is "Yeni Söhbət"
        const nextHistory = state.history.map((h) => {
          if (h.sessionId === state.activeSessionId && h.title === "Yeni Söhbət") {
            return {
              ...h,
              title: content.length > 28 ? content.slice(0, 28) + "..." : content,
              updatedAt: new Date().toISOString(),
            };
          }
          return h;
        });

        return {
          messages: [...state.messages, newUserMsg],
          history: nextHistory,
        };
      });

      return newUserMsg;
    },

    appendStreamChunk: (messageId, chunk) => {
      set((state) => {
        const existingMsg = state.messages.find((m) => m.id === messageId);
        
        if (existingMsg) {
          return {
            messages: state.messages.map((m) =>
              m.id === messageId 
                ? { ...m, content: m.content + chunk } 
                : m
            ),
          };
        } else {
          const newAssistantMsg: UIMessage = {
            id: messageId,
            role: "assistant",
            content: chunk,
            isStreaming: true,
          };
          return {
            messages: [...state.messages, newAssistantMsg],
          };
        }
      });
    },

    finalizeMessage: (messageId, finalContent, sources, metrics, citations, pipelineVersion, warnings, model) => {
      set((state) => ({
        messages: state.messages.map((m) =>
          m.id === messageId
            ? {
                ...m,
                content: finalContent,
                sources,
                metrics,
                citations,
                pipelineVersion,
                warnings,
                model,
                isStreaming: false,
              }
            : m
        ),
      }));
    },
  };
});
