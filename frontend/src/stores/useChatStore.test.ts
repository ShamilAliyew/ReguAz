import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";


class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();

  get length(): number {
    return this.values.size;
  }

  clear(): void {
    this.values.clear();
  }

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  key(index: number): string | null {
    return [...this.values.keys()][index] ?? null;
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

const OWNER_ID = "user-1";
const STORAGE_KEY = `reguaz-chat-history-v2:${OWNER_ID}`;
let storage: MemoryStorage;

beforeEach(() => {
  storage = new MemoryStorage();
  vi.stubGlobal("window", { localStorage: storage });
  vi.resetModules();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

describe("useChatStore persistence", () => {
  it("restores a completed conversation after a page reload", async () => {
    const firstModule = await import("./useChatStore");
    const firstStore = firstModule.useChatStore;
    firstStore.getState().setStorageOwner(OWNER_ID);
    const sessionId = firstStore.getState().createNewSession();

    firstStore.getState().addUserMessage("Likvidlik normativi nə qədərdir?");
    firstStore.getState().appendStreamChunk("assistant-1", "");
    firstStore.getState().finalizeMessage("assistant-1", "Normativ 30 faizdir.");

    vi.resetModules();
    const secondModule = await import("./useChatStore");
    secondModule.useChatStore.getState().setStorageOwner(OWNER_ID);
    const restored = secondModule.useChatStore.getState();

    expect(restored.activeSessionId).toBe(sessionId);
    expect(restored.history[0].title).toBe("Likvidlik normativi nə qədər...");
    expect(restored.messages.map((message) => message.content)).toEqual([
      "Likvidlik normativi nə qədərdir?",
      "Normativ 30 faizdir.",
    ]);
  });

  it("keeps messages isolated when switching between conversations", async () => {
    const { useChatStore } = await import("./useChatStore");
    useChatStore.getState().setStorageOwner(OWNER_ID);
    const firstSession = useChatStore.getState().createNewSession();
    useChatStore.getState().addUserMessage("Birinci söhbət");
    const secondSession = useChatStore.getState().createNewSession();
    useChatStore.getState().addUserMessage("İkinci söhbət");

    useChatStore.getState().setActiveSessionId(firstSession);
    expect(useChatStore.getState().messages[0].content).toBe("Birinci söhbət");

    useChatStore.getState().setActiveSessionId(secondSession);
    expect(useChatStore.getState().messages[0].content).toBe("İkinci söhbət");
  });

  it("discards expired browser history", async () => {
    const firstModule = await import("./useChatStore");
    firstModule.useChatStore.getState().setStorageOwner(OWNER_ID);
    const expiredSession = firstModule.useChatStore.getState().createNewSession();
    firstModule.useChatStore.getState().addUserMessage("Müvəqqəti söhbət");

    const persisted = JSON.parse(storage.getItem(STORAGE_KEY) || "{}") as {
      expiresAt: number;
    };
    persisted.expiresAt = Date.now() - 1;
    storage.setItem(STORAGE_KEY, JSON.stringify(persisted));

    vi.resetModules();
    const secondModule = await import("./useChatStore");
    secondModule.useChatStore.getState().setStorageOwner(OWNER_ID);
    const restored = secondModule.useChatStore.getState();

    expect(restored.activeSessionId).toBeNull();
    expect(restored.history.some((item) => item.sessionId === expiredSession)).toBe(
      false,
    );
    expect(storage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("does not persist an unfinished assistant stream", async () => {
    const firstModule = await import("./useChatStore");
    firstModule.useChatStore.getState().setStorageOwner(OWNER_ID);
    const sessionId = firstModule.useChatStore.getState().createNewSession();
    firstModule.useChatStore.getState().addUserMessage("Tamamlanmamış sorğu");
    firstModule.useChatStore
      .getState()
      .appendStreamChunk("assistant-pending", "Yarımçıq cavab");

    vi.resetModules();
    const secondModule = await import("./useChatStore");
    secondModule.useChatStore.getState().setStorageOwner(OWNER_ID);
    const restored = secondModule.useChatStore.getState();

    expect(restored.activeSessionId).toBe(sessionId);
    expect(restored.messages).toHaveLength(1);
    expect(restored.messages[0].role).toBe("user");
  });

  it("keeps a streaming response in its original conversation", async () => {
    const { useChatStore } = await import("./useChatStore");
    useChatStore.getState().setStorageOwner(OWNER_ID);
    const originalSession = useChatStore.getState().createNewSession();
    const { sessionId } = useChatStore
      .getState()
      .addUserMessage("Birinci söhbətin sorğusu");
    useChatStore
      .getState()
      .appendStreamChunk("assistant-original", "", sessionId);

    const otherSession = useChatStore.getState().createNewSession();
    useChatStore.getState().appendStreamChunk(
      "assistant-original",
      "Birinci söhbətin cavabı",
      sessionId,
    );
    useChatStore.getState().finalizeMessage(
      "assistant-original",
      "Birinci söhbətin cavabı",
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      sessionId,
    );

    expect(useChatStore.getState().activeSessionId).toBe(otherSession);
    expect(useChatStore.getState().messages).toEqual([]);

    useChatStore.getState().setActiveSessionId(originalSession);
    expect(useChatStore.getState().messages.map((message) => message.content)).toEqual([
      "Birinci söhbətin sorğusu",
      "Birinci söhbətin cavabı",
    ]);
  });

  it("does not add repeated empty drafts to history or persistence", async () => {
    const { useChatStore } = await import("./useChatStore");
    useChatStore.getState().setStorageOwner(OWNER_ID);

    const firstDraft = useChatStore.getState().createNewSession();
    const secondDraft = useChatStore.getState().createNewSession();

    expect(secondDraft).not.toBe(firstDraft);
    expect(useChatStore.getState().activeSessionId).toBe(secondDraft);
    expect(useChatStore.getState().history).toEqual([]);

    const persisted = JSON.parse(storage.getItem(STORAGE_KEY) || "{}") as {
      history?: unknown[];
      activeSessionId?: string | null;
    };
    expect(persisted.history).toEqual([]);
    expect(persisted.activeSessionId).toBeNull();
  });

  it("keeps browser history isolated per authenticated user", async () => {
    const { useChatStore } = await import("./useChatStore");
    useChatStore.getState().setStorageOwner(OWNER_ID);
    useChatStore.getState().createNewSession();
    useChatStore.getState().addUserMessage("Birinci istifadəçinin söhbəti");

    useChatStore.getState().setStorageOwner("user-2");
    expect(useChatStore.getState().history).toEqual([]);
    expect(useChatStore.getState().messages).toEqual([]);

    useChatStore.getState().setStorageOwner(OWNER_ID);
    expect(useChatStore.getState().messages[0].content).toBe(
      "Birinci istifadəçinin söhbəti",
    );
  });
});
