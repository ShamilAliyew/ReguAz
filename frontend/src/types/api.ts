export type MessageRole = "user" | "assistant";
export type LLMModelId =
  | "gemma"
  | "nvidia_gpt_oss"
  | "groq_gpt_oss_20b"
  | "groq_gpt_oss_120b";

export interface LLMModelOption {
  id: LLMModelId;
  label: string;
  available: boolean;
  reason?: string | null;
  loaded: boolean;
  provider: string;
  model_id: string;
  device: string;
  is_default: boolean;
}

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface SourceDocument {
  citation: number;
  chunk_id: string;
  document_id: string | null;
  document_name: string;
  category: string;
  chapter?: string | null;
  article?: string | null;
  page?: number | null;
  chunk_preview: string;
  rerank_score?: number | null;
  rrf_score?: number | null;
  semantic_rank?: number | null;
  bm25_rank?: number | null;
  canonical_locator?: string | null;
  role?: "seed" | "expanded" | null;
  relation_type?: string | null;
  selectors?: SourceSelector[];
}

export interface SourceSelector {
  representation_id: string;
  representation_type: "cleaned_markdown";
  document_version_id: string;
  source_sha256: string;
  page?: number | null;
  position: { start: number; end: number };
  page_position?: { start: number; end: number } | null;
  quote: { exact: string; prefix: string; suffix: string };
  normalization_version: string;
  resolved_position?: { start: number; end: number };
  resolution_mode?: "position" | "quote_fallback";
}

export interface Citation {
  citation: number;
  evidence_id: string;
  role: "seed" | "expanded";
  relation_type?: string | null;
  chunk_id?: string | null;
  logical_chunk_id?: string | null;
  parent_chunk_id?: string | null;
  document_id: string;
  document_version_id: string;
  document_title: string;
  category: string;
  canonical_locator: string;
  hierarchy: Record<string, unknown>;
  page_start?: number | null;
  page_end?: number | null;
  exact_quotes: string[];
  selectors: SourceSelector[];
  content_sha256: string;
  source_sha256: string;
  source_validated: boolean;
  claim_support_status: "not_evaluated";
  provenance: Record<string, unknown>[];
}

export interface MetricsResponse {
  retrieval_time: number;
  generation_time: number;
  total_time: number;
  retrieval_ms?: number | null;
  relation_expansion_ms?: number | null;
  evidence_selection_ms?: number | null;
  context_budget_ms?: number | null;
  prompt_build_ms?: number | null;
  generation_ms?: number | null;
  provider_queue_ms?: number | null;
  provider_prompt_ms?: number | null;
  provider_completion_ms?: number | null;
  provider_total_ms?: number | null;
  citation_resolution_ms?: number | null;
  total_ms?: number | null;
}

export interface ChatRequest {
  question: string;
  session_id?: string | null;
  llm_model?: LLMModelId | null;
}

export interface ChatResponse {
  session_id: string;
  question: string;
  answer: string;
  sources: SourceDocument[];
  metrics: MetricsResponse;
  status?: "answered" | "insufficient_evidence" | "conflicting_evidence";
  answer_blocks?: Array<{
    text: string;
    evidence_ids: string[];
    citation_numbers: number[];
  }>;
  citations?: Citation[];
  warnings?: Array<Record<string, unknown>>;
  pipeline_version?: "v1" | "v2";
  model?: Record<string, unknown> | null;
}

export interface DocumentMetadataResponse {
  document_id: string;
  title: string;
  category: string;
  total_pages?: number | null;
  total_chunks?: number | null;
  language?: string | null;
  parser?: string | null;
  publication_date?: string | null;
  status: string; // "active" | "archived"
  related_articles: string[];
  document_metadata?: Record<string, any>;
  document_version_id?: string | null;
  source_representation?: string | null;
  source_sha256?: string | null;
}

export interface ArticleInfo {
  chapter?: string | null;
  article?: string | null;
  section?: string | null;
  chunk_id: string;
}

export interface DocumentPageResponse {
  document_id: string;
  page_number: number;
  page_content: string;
  article_information: ArticleInfo[];
  metadata?: Record<string, any>;
  document_version_id?: string | null;
  representation_id?: string | null;
  source_sha256?: string | null;
  page_start_offset?: number | null;
}

export interface DocumentHighlightResponse {
  document_id: string;
  page: number;
  article?: string | null;
  chunk_id: string;
  chunk_start?: number | null;
  chunk_end?: number | null;
  highlighted_text: string;
  offset_status: string; // "supported" | "future_enhancement"
  selectors?: SourceSelector[];
  warning?: string | null;
}
