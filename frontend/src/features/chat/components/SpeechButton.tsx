import React, { useEffect, useRef, useState } from "react";
import { Loader2, Pause, Volume2 } from "lucide-react";
import { toast } from "sonner";

import { apiService } from "@/services/api";


type PlaybackState = "idle" | "loading" | "playing" | "paused" | "ready" | "error";

interface SpeechButtonProps {
  text: string;
  available: boolean;
  unavailableReason?: string | null;
}

export const SpeechButton: React.FC<SpeechButtonProps> = ({
  text,
  available,
  unavailableReason,
}) => {
  const [playbackState, setPlaybackState] = useState<PlaybackState>("idle");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      audioRef.current?.pause();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      audioRef.current = null;
      objectUrlRef.current = null;
    };
  }, [text]);

  const playExistingAudio = async () => {
    const audio = audioRef.current;
    if (!audio) return;
    try {
      if (audio.ended) audio.currentTime = 0;
      await audio.play();
      setPlaybackState("playing");
    } catch {
      setPlaybackState("error");
      toast.error("Audio başladılmadı. Brauzerin səs icazəsini yoxlayın.");
    }
  };

  const handleClick = async () => {
    if (!available || playbackState === "loading") return;
    if (playbackState === "playing") {
      audioRef.current?.pause();
      setPlaybackState("paused");
      return;
    }
    if (audioRef.current) {
      await playExistingAudio();
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setPlaybackState("loading");
    try {
      const result = await apiService.createSpeech(text, controller.signal);
      const objectUrl = URL.createObjectURL(result.audio);
      const audio = new Audio(objectUrl);
      audio.preload = "auto";
      audio.onended = () => setPlaybackState("ready");
      audio.onerror = () => {
        audio.pause();
        audioRef.current = null;
        if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
        setPlaybackState("error");
        toast.error("Audio oxunarkən xəta baş verdi.");
      };
      objectUrlRef.current = objectUrl;
      audioRef.current = audio;
      await playExistingAudio();
    } catch (error) {
      if (!controller.signal.aborted) {
        setPlaybackState("error");
        toast.error(error instanceof Error ? error.message : "Səs yaradılmadı.");
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  };

  const isPlaying = playbackState === "playing";
  const isLoading = playbackState === "loading";
  const label = isPlaying ? "Səsləndirməni dayandır" : "Cavabı dinlə";
  const disabledTitle = unavailableReason === "api_key_not_configured"
    ? "Səsləndirmə API açarı konfiqurasiya edilməyib"
    : "Səsləndirmə hazırda əlçatan deyil";

  return (
    <button
      type="button"
      aria-label={label}
      title={available ? label : disabledTitle}
      disabled={!available || isLoading}
      onClick={handleClick}
      className="mt-2 inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
    >
      {isLoading ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
      ) : isPlaying ? (
        <Pause className="h-3.5 w-3.5" aria-hidden="true" />
      ) : (
        <Volume2 className="h-3.5 w-3.5" aria-hidden="true" />
      )}
    </button>
  );
};
