import React, { useRef, useEffect, useState } from "react";
import { 
  Send, 
  Paperclip, 
  HelpCircle, 
  Sparkles, 
  Loader2, 
  ArrowRight,
  Info,
  ChevronDown,
  BookOpen
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useChatStore } from "@/stores/useChatStore";
import { useUIStore } from "@/stores/useUIStore";
import { useChat } from "@/hooks/useChat";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { Citation, SourceDocument } from "@/types/api";
import type { LLMModelId } from "@/types/api";
import { apiService } from "@/services/api";
import { useQuery } from "@tanstack/react-query";
import { SpeechButton } from "./SpeechButton";

type DisplaySource = SourceDocument | Citation;

const MessageSources: React.FC<{ sources?: DisplaySource[] }> = ({ sources }) => {
  const [isOpen, setIsOpen] = useState(false);
  const { setSelectedCitation, setActiveDocument } = useUIStore();

  if (!sources || sources.length === 0) return null;

  const handleSourceClick = (source: DisplaySource) => {
    const isV2 = "evidence_id" in source;
    const chunkId = isV2
      ? source.chunk_id || source.parent_chunk_id || ""
      : source.chunk_id;
    setSelectedCitation({
      citationId: source.citation,
      chunkId,
      documentId: source.document_id || "",
      evidenceId: isV2 ? source.evidence_id : undefined,
    });
    setActiveDocument({
      documentId: source.document_id || "",
      activePage: (isV2 ? source.page_start : source.page) || 1,
      highlightText: isV2 ? source.exact_quotes[0] || null : source.chunk_preview,
      selectors: isV2 ? source.selectors : source.selectors || [],
      evidenceId: isV2 ? source.evidence_id : null,
    });
  };

  return (
    <div className="mt-3 pt-2 border-t border-border/40">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1 text-[11px] font-semibold text-gold-600 dark:text-gold-400 hover:text-gold-700 dark:hover:text-gold-300 transition-colors focus:outline-none cursor-pointer"
      >
        <BookOpen className="h-3 w-3" />
        <span>Mənbələr ({sources.length})</span>
        <ChevronDown className={`h-3 w-3 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`} />
      </button>

      {isOpen && (
        <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2 animate-in fade-in slide-in-from-top-1 duration-200">
          {sources.map((source, idx) => (
            <div
              key={idx}
              onClick={() => handleSourceClick(source)}
              className="p-2.5 border rounded-xl bg-card hover:bg-secondary/40 transition-all cursor-pointer text-left hover:border-gold-300 group relative shadow-sm"
              title={
                "evidence_id" in source
                  ? `${source.document_title} • ${source.canonical_locator} • Səhifə ${source.page_start || "—"} • ${source.role}`
                  : source.document_name
              }
            >
              <div className="flex justify-between items-start gap-1">
                <span className="text-[10px] bg-gold-100 text-gold-800 dark:bg-gold-950/40 dark:text-gold-300 font-bold px-1.5 py-0.5 rounded leading-none">
                  [{source.citation}]
                </span>
                {"rerank_score" in source && source.rerank_score !== undefined && source.rerank_score !== null && (
                  <span className="text-[9px] text-muted-foreground font-mono bg-secondary px-1 py-0.5 rounded">
                    Score: {source.rerank_score.toFixed(3)}
                  </span>
                )}
              </div>
              <h4 className="text-[11px] font-bold text-foreground mt-1.5 line-clamp-1 group-hover:text-gold-600 transition-colors">
                {"evidence_id" in source ? source.document_title : source.document_name}
              </h4>
              <p className="text-[10px] text-muted-foreground mt-0.5 font-light">
                {"evidence_id" in source ? (
                  <>{source.canonical_locator} • Səhifə {source.page_start || "—"} • {source.role === "seed" ? "əsas" : "əlavə context"}</>
                ) : (
                  <>{source.chapter && `${source.chapter} • `}{source.article && `${source.article}`}{source.page && ` • Səhifə ${source.page}`}</>
                )}
              </p>
              <p className="text-[10px] text-muted-foreground/80 mt-1.5 line-clamp-2 italic font-light border-l border-gold-400/30 pl-1.5">
                "{"evidence_id" in source ? source.exact_quotes[0] : source.chunk_preview}"
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export const ChatContainer: React.FC = () => {
  const { messages, isGenerating, selectedModel, setSelectedModel } = useChatStore();
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const { sendMessage } = useChat();
  const [inputValue, setInputValue] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const { data: modelOptions = [] } = useQuery({
    queryKey: ["llm-models"],
    queryFn: apiService.getLlmModels,
    staleTime: 30_000,
  });
  const { data: speechStatus } = useQuery({
    queryKey: ["speech-status"],
    queryFn: apiService.getSpeechStatus,
    staleTime: 60_000,
  });

  const suggestionPrompts = [
    "Bankın minimum nizamnamə kapitalı nə qədərdir?",
    "Kredit təşkilatlarında risklərin idarə olunması qaydaları hansılardır?",
    "Mərkəzi Bankın funksiyaları və kapitalı barədə məlumat verin."
  ];

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  useEffect(() => {
    if (!modelOptions.length) return;
    const selected = modelOptions.find((option) => option.id === selectedModel);
    if (selected?.available) return;
    const fallback =
      modelOptions.find((option) => option.is_default && option.available) ||
      modelOptions.find((option) => option.available);
    if (fallback) setSelectedModel(fallback.id);
  }, [modelOptions, selectedModel, setSelectedModel]);

  const handleSend = () => {
    if (!inputValue.trim() || isGenerating) return;
    sendMessage(inputValue.trim());
    setInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestionClick = (prompt: string) => {
    if (isGenerating) return;
    sendMessage(prompt);
  };

  return (
    <div className="flex-1 flex flex-col h-full min-w-0 bg-background">
      
      {/* Chat Area Header */}
      <header className="h-16 border-b flex items-center justify-between px-4 sm:px-6 shrink-0 bg-card">
        <div className="flex items-center gap-2">
          {!sidebarOpen && (
            <Button 
              variant="ghost" 
              size="sm" 
              className="md:hidden text-muted-foreground mr-1"
              onClick={toggleSidebar}
            >
              ☰
            </Button>
          )}
          <div className="text-left">
            <h2 className="text-sm font-bold flex items-center gap-1.5 text-foreground">
              <span>Tənzimləyici Köməkçi</span>
              <span className="text-[9px] bg-gold-100 text-gold-800 dark:bg-gold-950/30 dark:text-gold-400 font-bold px-1 py-0.5 rounded">AI</span>
            </h2>
            <p className="text-[10px] text-muted-foreground font-light">Azərbaycan Respublikası Mərkəzi Bankının tənzimləmələri</p>
          </div>
        </div>
        <label className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <span className="hidden sm:inline">Cavab modeli</span>
          <select
            aria-label="Cavab modeli"
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value as LLMModelId)}
            className="h-8 max-w-[220px] rounded-lg border border-border bg-background px-2 text-[11px] text-foreground focus:outline-none focus:ring-1 focus:ring-gold-500"
          >
            {modelOptions.map((option) => (
              <option
                key={option.id}
                value={option.id}
                disabled={!option.available}
              >
                {option.label}{option.available ? "" : " — əlçatan deyil"}
              </option>
            ))}
          </select>
        </label>
      </header>

      {/* Messages / Suggestions */}
      <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        {messages.length === 0 ? (
          /* Empty State / Suggested Prompts */
          <div className="max-w-2xl mx-auto h-full flex flex-col justify-center items-center text-center space-y-6">
            <div className="p-4 bg-gold-100/50 dark:bg-navy-900 rounded-2xl border border-gold-200/20">
              <Sparkles className="h-8 w-8 text-gold-500 animate-pulse" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-bold">Necə kömək edə bilərəm?</h3>
              <p className="text-xs text-muted-foreground font-light max-w-md">
                Mərkəzi Bankın normativ aktları, prudensial tələbləri, risk limitləri və daxili audit normaları barədə sual verin.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-3 w-full max-w-lg pt-4">
              {suggestionPrompts.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSuggestionClick(prompt)}
                  className="flex items-center justify-between p-3.5 border rounded-xl bg-card hover:bg-secondary text-xs text-left font-medium transition-colors group cursor-pointer"
                >
                  <span className="text-foreground/90">{prompt}</span>
                  <ArrowRight className="h-3.5 w-3.5 text-muted-foreground group-hover:text-gold-500 transition-transform group-hover:translate-x-1" />
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Chat History Thread */
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-4 ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                
                {/* Assistant icon */}
                {message.role === "assistant" && (
                  <div className="h-8 w-8 rounded-full border bg-navy-900 text-white dark:bg-gold-500 dark:text-navy-950 flex items-center justify-center text-xs font-bold shrink-0 shadow-sm">
                    R
                  </div>
                )}

                {/* Message Bubble Container */}
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm border text-left ${
                    message.role === "user"
                      ? "bg-navy-900 text-white dark:bg-white dark:text-navy-900 border-navy-950 dark:border-gray-200"
                      : "bg-card border-border/80 border-l-4 border-l-gold-500"
                  }`}
                >
                  {message.role === "user" ? (
                    <p className="text-sm whitespace-pre-wrap font-light">{message.content}</p>
                  ) : (
                    <div className="space-y-2">
                      {/* Check if generating empty token message */}
                      {message.content === "" && message.isStreaming ? (
                        <div className="flex items-center gap-2 py-2">
                          <Loader2 className="h-4 w-4 animate-spin text-gold-500" />
                          <span className="text-xs text-muted-foreground font-light">Araşdırılır...</span>
                        </div>
                      ) : (
                        <>
                          {message.pipelineVersion === "v2" && (
                            <span className="inline-flex text-[9px] font-bold px-1.5 py-0.5 rounded bg-gold-100 text-gold-800 dark:bg-gold-950/40 dark:text-gold-300">V2</span>
                          )}
                          {typeof message.model?.model_id === "string" && (
                            <span className="ml-1 inline-flex text-[9px] px-1.5 py-0.5 rounded bg-secondary text-muted-foreground">
                              {message.model.model_id}
                            </span>
                          )}
                          <MarkdownRenderer content={message.content} sources={message.citations?.length ? message.citations : message.sources} />
                          {!message.isStreaming && (
                            <SpeechButton
                              text={message.content}
                              available={speechStatus?.available ?? false}
                              unavailableReason={speechStatus?.reason}
                            />
                          )}
                          {!message.isStreaming && <MessageSources sources={message.citations?.length ? message.citations : message.sources} />}
                        </>
                      )}

                      {/* Display Metrics metadata */}
                      {message.metrics && !message.isStreaming && (
                        <div className="pt-2 mt-2 border-t border-border/40 flex items-center gap-1.5 text-[9px] text-muted-foreground font-light font-mono">
                          <Info className="h-3 w-3 text-gold-500" />
                          <span>Axtarış: {message.metrics.retrieval_time.toFixed(3)}s</span>
                          <span>•</span>
                          <span>Generasiya: {message.metrics.generation_time.toFixed(3)}s</span>
                          <span>•</span>
                          <span>Cəmi: {message.metrics.total_time.toFixed(3)}s</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* User avatar initial */}
                {message.role === "user" && (
                  <div className="h-8 w-8 rounded-full bg-gold-500 text-white font-bold flex items-center justify-center text-xs shrink-0 shadow-sm">
                    U
                  </div>
                )}

              </div>
            ))}
            
            {/* Scroll bottom placeholder */}
            <div ref={scrollRef} />
          </div>
        )}
      </div>

      {/* Input box section */}
      <div className="p-4 sm:p-6 bg-gradient-to-t from-background via-background/95 to-transparent shrink-0">
        <div className="max-w-3xl mx-auto space-y-2">
          
          <div className="relative border rounded-2xl bg-card shadow-sm focus-within:ring-1 focus-within:ring-gold-500 focus-within:border-gold-500 transition-shadow">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isGenerating}
              rows={1}
              placeholder="Bank normativ aktları barədə sual verin..."
              className="w-full pl-4 pr-24 py-3 bg-transparent text-foreground text-sm focus:outline-none resize-none min-h-[48px] max-h-[160px] font-light placeholder:text-muted-foreground/60"
              style={{ height: "auto" }}
            />
            
            <div className="absolute right-2 bottom-2.5 flex items-center gap-1.5">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-muted-foreground hover:text-foreground opacity-50 cursor-not-allowed"
                title="Sənəd yükləyin (Tezliklə)"
                disabled
              >
                <Paperclip className="h-4 w-4" />
              </Button>
              <Button
                onClick={handleSend}
                disabled={!inputValue.trim() || isGenerating}
                size="icon"
                className="h-8 w-8 bg-navy-900 text-white hover:bg-navy-800 dark:bg-gold-500 dark:text-navy-950 dark:hover:bg-gold-600 rounded-xl"
              >
                {isGenerating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
          
          {/* Disclaimer copy */}
          <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground font-light text-center">
            <HelpCircle className="h-3.5 w-3.5 text-gold-500 shrink-0" />
            <span>Süni intellekt xətalara yol verə bilər. Qərarların qəbulu üçün normativ aktların rəsmi mətninə istinad edin.</span>
          </div>

        </div>
      </div>

    </div>
  );
};
