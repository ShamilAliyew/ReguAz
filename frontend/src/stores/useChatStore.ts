import { create } from "zustand";

import type {
  Citation,
  LLMModelId,
  MetricsResponse,
  SourceDocument,
} from "../types/api";

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

type MessagesBySession = Record<string, UIMessage[]>;

const CHAT_STORAGE_KEY_PREFIX = "reguaz-chat-history-v2";
const CHAT_STORAGE_VERSION = 2;
const CHAT_RETENTION_MS = 7 * 24 * 60 * 60 * 1_000;
const MAX_PERSISTED_SESSIONS = 30;
const MAX_MESSAGES_PER_SESSION = 100;

const SELECTABLE_MODELS: LLMModelId[] = [
  "gemma",
  "nvidia_gpt_oss",
  "groq_gpt_oss_20b",
  "groq_gpt_oss_120b",
];

interface PersistedChatState {
  version: number;
  savedAt: number;
  expiresAt: number;
  history: UIHistoryItem[];
  activeSessionId: string | null;
  messagesBySession: MessagesBySession;
}

interface RestoredChatState {
  history: UIHistoryItem[];
  activeSessionId: string | null;
  messagesBySession: MessagesBySession;
}

interface ChatState {
  storageOwnerId: string | null;
  history: UIHistoryItem[];
  activeSessionId: string | null;
  messages: UIMessage[];
  messagesBySession: MessagesBySession;
  isGenerating: boolean;
  selectedModel: LLMModelId;

  setStorageOwner: (ownerId: string | null) => void;
  setHistory: (history: UIHistoryItem[]) => void;
  setActiveSessionId: (sessionId: string | null) => void;
  setMessages: (messages: UIMessage[]) => void;
  setIsGenerating: (isGenerating: boolean) => void;
  setSelectedModel: (model: LLMModelId) => void;
  createNewSession: () => string;
  deleteSession: (sessionId: string) => void;
  addUserMessage: (content: string) => {
    message: UIMessage;
    sessionId: string;
  };
  appendStreamChunk: (
    messageId: string,
    chunk: string,
    sessionId?: string,
  ) => void;
  finalizeMessage: (
    messageId: string,
    finalContent: string,
    sources?: SourceDocument[],
    metrics?: MetricsResponse,
    citations?: Citation[],
    pipelineVersion?: "v1" | "v2",
    warnings?: Array<Record<string, unknown>>,
    model?: Record<string, unknown> | null,
    sessionId?: string,
  ) => void;
}

type PersistableState = Pick<
  ChatState,
  "storageOwnerId" | "history" | "activeSessionId" | "messagesBySession"
>;

const browserStorage = typeof window === "undefined" ? null : window.localStorage;
const storedModel = browserStorage?.getItem("reguaz-llm-model") as LLMModelId | null;
const initialModel =
  storedModel && SELECTABLE_MODELS.includes(storedModel) ? storedModel : "gemma";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isHistoryItem(value: unknown): value is UIHistoryItem {
  return (
    isRecord(value) &&
    typeof value.sessionId === "string" &&
    typeof value.title === "string" &&
    typeof value.updatedAt === "string"
  );
}

function isMessage(value: unknown): value is UIMessage {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    (value.role === "user" || value.role === "assistant") &&
    typeof value.content === "string"
  );
}

function storageKey(ownerId: string): string {
  return `${CHAT_STORAGE_KEY_PREFIX}:${ownerId}`;
}

function readPersistedChat(ownerId: string): RestoredChatState | null {
  if (!browserStorage) return null;
  const key = storageKey(ownerId);
  try {
    const raw = browserStorage.getItem(key);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (
      !isRecord(parsed) ||
      parsed.version !== CHAT_STORAGE_VERSION ||
      typeof parsed.expiresAt !== "number" ||
      parsed.expiresAt <= Date.now() ||
      !Array.isArray(parsed.history) ||
      !isRecord(parsed.messagesBySession)
    ) {
      browserStorage.removeItem(key);
      return null;
    }

    const history: UIHistoryItem[] = [];
    const messagesBySession: MessagesBySession = {};
    for (const item of parsed.history.filter(isHistoryItem)) {
      const rawMessages = parsed.messagesBySession[item.sessionId];
      const messages = Array.isArray(rawMessages)
        ? rawMessages
            .filter(isMessage)
            .filter((message) => !message.isStreaming)
            .slice(-MAX_MESSAGES_PER_SESSION)
        : [];
      // Old or interrupted clients may have persisted a draft session. Empty
      // sessions are deliberately omitted, matching ChatGPT/Claude behaviour.
      if (messages.length === 0) continue;
      history.push(item);
      messagesBySession[item.sessionId] = messages;
      if (history.length >= MAX_PERSISTED_SESSIONS) break;
    }
    const requestedActive =
      typeof parsed.activeSessionId === "string" ? parsed.activeSessionId : null;
    const activeSessionId = history.some(
      (item) => item.sessionId === requestedActive,
    )
      ? requestedActive
      : history[0]?.sessionId ?? null;

    return { history, activeSessionId, messagesBySession };
  } catch {
    browserStorage.removeItem(key);
    return null;
  }
}

function writePersistedChat(state: PersistableState): void {
  if (!browserStorage || !state.storageOwnerId) return;
  const history = state.history
    .filter(
      (item) =>
        (state.messagesBySession[item.sessionId] || []).filter(
          (message) => !message.isStreaming,
        ).length > 0,
    )
    .slice(0, MAX_PERSISTED_SESSIONS);
  const messagesBySession: MessagesBySession = {};
  for (const item of history) {
    messagesBySession[item.sessionId] = (state.messagesBySession[item.sessionId] || [])
      .filter((message) => !message.isStreaming)
      .slice(-MAX_MESSAGES_PER_SESSION);
  }
  const now = Date.now();
  const payload: PersistedChatState = {
    version: CHAT_STORAGE_VERSION,
    savedAt: now,
    expiresAt: now + CHAT_RETENTION_MS,
    history,
    activeSessionId: history.some(
      (item) => item.sessionId === state.activeSessionId,
    )
      ? state.activeSessionId
      : null,
    messagesBySession,
  };
  try {
    browserStorage.setItem(storageKey(state.storageOwnerId), JSON.stringify(payload));
  } catch {
    // localStorage may be disabled or full. Chat remains available in memory;
    // persistence must never prevent the user from continuing the session.
  }
}

function retainKnownSessions(
  messagesBySession: MessagesBySession,
  history: UIHistoryItem[],
): MessagesBySession {
  return Object.fromEntries(
    history.map((item) => [item.sessionId, messagesBySession[item.sessionId] || []]),
  );
}

function createId(prefix: string): string {
  const suffix = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  return `${prefix}-${suffix}`;
}

export const useChatStore = create<ChatState>((set) => ({
  storageOwnerId: null,
  history: [],
  activeSessionId: null,
  messages: [],
  messagesBySession: {},
  isGenerating: false,
  selectedModel: initialModel,

  setStorageOwner: (storageOwnerId) =>
    set((state) => {
      if (storageOwnerId === state.storageOwnerId) return state;
      const restored = storageOwnerId ? readPersistedChat(storageOwnerId) : null;
      const history = restored?.history ?? [];
      const activeSessionId = restored?.activeSessionId ?? null;
      const messagesBySession = restored?.messagesBySession ?? {};
      return {
        storageOwnerId,
        history,
        activeSessionId,
        messages: activeSessionId
          ? messagesBySession[activeSessionId] || []
          : [],
        messagesBySession,
      };
    }),

  setHistory: (history) =>
    set((state) => {
      const boundedHistory = history.slice(0, MAX_PERSISTED_SESSIONS);
      const activeSessionId = boundedHistory.some(
        (item) => item.sessionId === state.activeSessionId,
      )
        ? state.activeSessionId
        : boundedHistory[0]?.sessionId ?? null;
      const messagesBySession = retainKnownSessions(
        state.messagesBySession,
        boundedHistory,
      );
      const next = {
        history: boundedHistory,
        activeSessionId,
        messagesBySession,
        messages: activeSessionId ? messagesBySession[activeSessionId] || [] : [],
      };
      writePersistedChat({ ...state, ...next });
      return next;
    }),

  setActiveSessionId: (activeSessionId) =>
    set((state) => {
      const validSessionId = state.history.some(
        (item) => item.sessionId === activeSessionId,
      )
        ? activeSessionId
        : null;
      const next = {
        activeSessionId: validSessionId,
        messages: validSessionId
          ? state.messagesBySession[validSessionId] || []
          : [],
      };
      writePersistedChat({ ...state, ...next });
      return next;
    }),

  setMessages: (messages) =>
    set((state) => {
      if (!state.activeSessionId) return { messages };
      const messagesBySession = {
        ...state.messagesBySession,
        [state.activeSessionId]: messages,
      };
      const next = { messages, messagesBySession };
      writePersistedChat({ ...state, ...next });
      return next;
    }),

  setIsGenerating: (isGenerating) => set({ isGenerating }),

  setSelectedModel: (selectedModel) => {
    browserStorage?.setItem("reguaz-llm-model", selectedModel);
    set({ selectedModel });
  },

  createNewSession: () => {
    const newId = createId("session");
    set((state) => {
      const next = {
        activeSessionId: newId,
        messages: [],
      };
      // This is an in-memory draft. addUserMessage promotes it to history.
      // Repeated clicks simply replace the draft and cannot create ghost rows.
      writePersistedChat({ ...state, ...next });
      return next;
    });
    return newId;
  },

  deleteSession: (sessionId) =>
    set((state) => {
      const history = state.history.filter((item) => item.sessionId !== sessionId);
      const messagesBySession = { ...state.messagesBySession };
      delete messagesBySession[sessionId];
      const activeSessionId =
        state.activeSessionId === sessionId
          ? history[0]?.sessionId ?? null
          : state.activeSessionId;
      const next = {
        history,
        activeSessionId,
        messages: activeSessionId ? messagesBySession[activeSessionId] || [] : [],
        messagesBySession,
      };
      writePersistedChat({ ...state, ...next });
      return next;
    }),

  addUserMessage: (content) => {
    const message: UIMessage = {
      id: createId("msg-user"),
      role: "user",
      content,
    };
    let targetSessionId = "";
    set((state) => {
      const activeSessionId = state.activeSessionId || createId("session");
      targetSessionId = activeSessionId;
      const existingSession = state.history.find(
        (item) => item.sessionId === activeSessionId,
      );
      const activeHistory: UIHistoryItem = {
        sessionId: activeSessionId,
        title:
          !existingSession || existingSession.title === "Yeni Söhbət"
            ? content.length > 28
              ? `${content.slice(0, 28)}...`
              : content
            : existingSession.title,
        updatedAt: new Date().toISOString(),
      };
      const history = [
        activeHistory,
        ...state.history.filter((item) => item.sessionId !== activeSessionId),
      ].slice(0, MAX_PERSISTED_SESSIONS);
      const messages = [...(state.messagesBySession[activeSessionId] || []), message];
      const messagesBySession = retainKnownSessions(
        { ...state.messagesBySession, [activeSessionId]: messages },
        history,
      );
      const next = { history, activeSessionId, messages, messagesBySession };
      writePersistedChat({ ...state, ...next });
      return next;
    });
    return { message, sessionId: targetSessionId };
  },

  appendStreamChunk: (messageId, chunk, sessionId) =>
    set((state) => {
      const targetSessionId = sessionId || state.activeSessionId;
      if (!targetSessionId) return state;
      const sessionMessages = state.messagesBySession[targetSessionId] || [];
      const existingMessage = sessionMessages.find(
        (item) => item.id === messageId,
      );
      const nextSessionMessages = existingMessage
        ? sessionMessages.map((item) =>
            item.id === messageId
              ? { ...item, content: item.content + chunk }
              : item,
          )
        : [
            ...sessionMessages,
            {
              id: messageId,
              role: "assistant" as const,
              content: chunk,
              isStreaming: true,
            },
          ];
      const messagesBySession = {
        ...state.messagesBySession,
        [targetSessionId]: nextSessionMessages,
      };
      return {
        messages:
          state.activeSessionId === targetSessionId
            ? nextSessionMessages
            : state.messages,
        messagesBySession,
      };
    }),

  finalizeMessage: (
    messageId,
    finalContent,
    sources,
    metrics,
    citations,
    pipelineVersion,
    warnings,
    model,
    sessionId,
  ) =>
    set((state) => {
      const targetSessionId = sessionId || state.activeSessionId;
      if (!targetSessionId) return state;
      const sessionMessages = state.messagesBySession[targetSessionId] || [];
      const nextSessionMessages = sessionMessages.map((item) =>
        item.id === messageId
          ? {
              ...item,
              content: finalContent,
              sources,
              metrics,
              citations,
              pipelineVersion,
              warnings,
              model,
              isStreaming: false,
            }
          : item,
      );
      const messagesBySession = {
        ...state.messagesBySession,
        [targetSessionId]: nextSessionMessages,
      };
      const next = {
        messages:
          state.activeSessionId === targetSessionId
            ? nextSessionMessages
            : state.messages,
        messagesBySession,
      };
      writePersistedChat({ ...state, ...next });
      return next;
    }),
}));
