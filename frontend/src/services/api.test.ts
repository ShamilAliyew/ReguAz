import { afterEach, describe, expect, it, vi } from "vitest";

import { apiService, FALLBACK_LLM_MODELS } from "./api";


const storage = {
  getItem: vi.fn(() => null),
  setItem: vi.fn(),
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("apiService model selection", () => {
  it("sends whitelisted model choice with chat request", async () => {
    vi.stubGlobal("localStorage", storage);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: "s1",
          question: "Sual",
          answer: "Cavab",
          sources: [],
          metrics: {
            retrieval_time: 0.1,
            generation_time: 0.2,
            total_time: 0.3,
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiService.postChat("Sual", "s1", "groq_gpt_oss_20b");

    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(body).toEqual({
      question: "Sual",
      session_id: "s1",
      llm_model: "groq_gpt_oss_20b",
    });
  });

  it("reads dynamic model availability from backend", async () => {
    vi.stubGlobal("localStorage", storage);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify([
            {
              id: "gemma",
              label: "Gemma",
              available: true,
              loaded: true,
              provider: "gemma",
              model_id: "local",
              device: "local",
              is_default: true,
            },
          ]),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );

    const models = await apiService.getLlmModels();
    expect(models[0].id).toBe("gemma");
    expect(models[0].available).toBe(true);
  });

  it("keeps model selection populated when the backend catalog is unavailable", async () => {
    vi.stubGlobal("localStorage", storage);
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    const models = await apiService.getLlmModels();

    expect(models).toEqual(FALLBACK_LLM_MODELS);
    expect(models.map((model) => model.id)).toContain("groq_gpt_oss_120b");
  });

  it("requests MP3 speech without exposing provider credentials", async () => {
    vi.stubGlobal("localStorage", storage);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(new Blob([new Uint8Array([73, 68, 51, 0])]), {
        status: 200,
        headers: {
          "Content-Type": "audio/mpeg",
          "X-Generation-Id": "gen-1",
          "X-TTS-Model": "fish-audio/s2.1-pro-free:free",
          "X-TTS-Latency-Ms": "125.5",
        },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await apiService.createSpeech("Cavab [1]");

    expect(JSON.parse(fetchMock.mock.calls[0][1].body as string)).toEqual({
      text: "Cavab [1]",
    });
    expect(fetchMock.mock.calls[0][1].headers).toEqual({
      "Content-Type": "application/json",
    });
    expect(result.audio.size).toBeGreaterThan(0);
    expect(result.generationId).toBe("gen-1");
    expect(result.latencyMs).toBe(125.5);
  });

  it("rejects a non-audio speech response", async () => {
    vi.stubGlobal("localStorage", storage);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("not audio", {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(apiService.createSpeech("Cavab")).rejects.toThrow(
      "Server etibarlı MP3 qaytarmadı.",
    );
  });
});
