import { 
  DocumentMetadataResponse, 
  DocumentPageResponse, 
  DocumentHighlightResponse, 
  ChatResponse, 
  SourceDocument,
  LLMModelId,
  LLMModelOption,
} from "../types/api";

// Minimal set of CBAR documents for UI display
const mockDocuments: DocumentMetadataResponse[] = [
  {
    document_id: "802-IIQ-Azərbaycan Respublikasının Mərkəzi Bankı haqqında",
    title: "802-IIQ - Azərbaycan Respublikasının Mərkəzi Bankı haqqında Qanun",
    category: "laws",
    total_pages: 54,
    total_chunks: 123,
    language: "az",
    parser: "pdfplumber",
    publication_date: "2004-12-10",
    status: "active",
    related_articles: ["Maddə 4", "Maddə 19", "Maddə 36"],
  },
  {
    document_id: "590-IIQ-Banklar haqqında",
    title: "590-IIQ - Banklar haqqında Azərbaycan Respublikasının Qanunu",
    category: "laws",
    total_pages: 72,
    total_chunks: 185,
    language: "az",
    parser: "pdfplumber",
    publication_date: "2004-01-16",
    status: "active",
    related_articles: ["Maddə 10", "Maddə 12", "Maddə 34"],
  },
  {
    document_id: "risk-management-rules",
    title: "Kredit təşkilatlarında risklərin idarə olunması Qaydaları",
    category: "risk_management",
    total_pages: 18,
    total_chunks: 40,
    language: "az",
    parser: "pdfplumber",
    publication_date: "2019-05-22",
    status: "active",
    related_articles: ["Maddə 3", "Maddə 7"],
  }
];

// Map of page contents
const mockPageContents: Record<string, Record<number, DocumentPageResponse>> = {
  "802-IIQ-Azərbaycan Respublikasının Mərkəzi Bankı haqqında": {
    12: {
      document_id: "802-IIQ-Azərbaycan Respublikasının Mərkəzi Bankı haqqında",
      page_number: 12,
      page_content: `Azərbaycan Respublikasının Mərkəzi Bankı haqqında Qanun.
Fəsil I. Ümumi müddəalar.
Maddə 4. Nizamnamə kapitalı.
4.1. Mərkəzi Bankın nizamnamə kapitalı dövlətə məxsusdur və 50,000,000 AZN (əlli milyon manat) təşkil edir.
4.2. Nizamnamə kapitalının məbləği yalnız Azərbaycan Respublikasının Qanunu ilə artırıla bilər.
4.3. Nizamnamə kapitalı Mərkəzi Bankın əmlakını və xüsusi vəsaitlərini formalaşdırmaq, öhdəliklərini ödəmək və fəaliyyətini sərbəst tənzimləmək üçün istifadə edilir.
Nizamnamə kapitalının ödənilməsi dövlət büdcəsinin vəsaitləri və Mərkəzi Bankın sərbəst mənfəəti hesabına həyata keçirilə bilər.`,
      article_information: [
        {
          chapter: "I Fəsil",
          article: "Maddə 4",
          section: "Nizamnamə kapitalı",
          chunk_id: "802-IIQ_Art4_p1",
        }
      ]
    }
  },
  "590-IIQ-Banklar haqqında": {
    8: {
      document_id: "590-IIQ-Banklar haqqında",
      page_number: 8,
      page_content: `Banklar haqqında Azərbaycan Respublikasının Qanunu.
Fəsil III. Bank kapitalı və ehtiyatları.
Maddə 12. Nizamnamə kapitalının ödənilməsi.
12.1. Bankın nizamnamə kapitalı yalnız pul vəsaitləri ilə ödənilməlidir. Borc və ya girov götürülmüş vəsaitlərdən nizamnamə kapitalının formalaşdırılmasına icazə verilmir.
12.2. Bank təsisçiləri nizamnamə kapitalındakı paylarını bankın dövlət qeydiyyatına alındığı tarixdən 3 aydan gec olmayaraq tam ödəməlidirlər.
12.3. Mərkəzi Bank bank fəaliyyəti üçün lisenziya verilməzdən əvvəl ödənilmiş nizamnamə kapitalının mövcudluğunu yoxlayır.`,
      article_information: [
        {
          chapter: "III Fəsil",
          article: "Maddə 12",
          section: "Nizamnamə kapitalının ödənilməsi",
          chunk_id: "590-IIQ_Art12_p2",
        }
      ]
    }
  },
  "risk-management-rules": {
    3: {
      document_id: "risk-management-rules",
      page_number: 3,
      page_content: `Kredit təşkilatlarında risklərin idarə olunması Qaydaları.
Maddə 3. Risklərin İdarə Edilməsi Sistemi.
3.1. Kredit təşkilatlarında risklərin idarə edilməsi sistemi bank fəaliyyətinin miqyasına, mürəkkəbliyinə və həyata keçirilən əməliyyatların xarakterinə mütənasib olmalıdır.
3.2. Müşahidə Şurası risklərin idarə olunması strategiyasını təsdiq edir və risk limitlərini müəyyən edir.
3.3. Risklərin idarə edilməsi departamenti İdarə Heyətindən müstəqil olmalı və birbaşa Müşahidə Şurasına hesabat verməlidir.`,
      article_information: [
        {
          chapter: "I Hissə",
          article: "Maddə 3",
          section: "Risklərin İdarə Edilməsi Sistemi",
          chunk_id: "risk_rules_art3",
        }
      ]
    }
  }
};

export const mockService = {
  // GET /health
  getHealth: async (): Promise<{ status: string; app: string }> => {
    await new Promise(r => setTimeout(r, 100));
    return { status: "healthy", app: "ReguAZ API Gateway" };
  },

  getLlmModels: async (): Promise<LLMModelOption[]> => [
    {
      id: "gemma",
      label: "Gemma 3 4B (lokal)",
      available: true,
      loaded: true,
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
  ],

  // GET /documents
  getDocuments: async (): Promise<DocumentMetadataResponse[]> => {
    await new Promise(r => setTimeout(r, 300));
    return mockDocuments;
  },

  // GET /documents/:id
  getDocumentMetadata: async (documentId: string): Promise<DocumentMetadataResponse | null> => {
    await new Promise(r => setTimeout(r, 200));
    const doc = mockDocuments.find(d => d.document_id === documentId);
    return doc || null;
  },

  // GET /documents/:id/page/:page_number
  getDocumentPage: async (documentId: string, pageNumber: number): Promise<DocumentPageResponse | null> => {
    await new Promise(r => setTimeout(r, 250));
    const docPages = mockPageContents[documentId];
    if (docPages && docPages[pageNumber]) {
      return docPages[pageNumber];
    }
    
    // Catch-all mock fallback page if requested page doesn't exist explicitly
    return {
      document_id: documentId,
      page_number: pageNumber,
      page_content: `[Səhifə ${pageNumber}] Azərbaycan Respublikası Mərkəzi Bankının normativ sənədinin ${pageNumber}-ci səhifəsindəki tənzimləyici mətn placeholderi. 
Bu hissə bank əməliyyatlarının aparılması, prudential hesabatlıq və ya risklərin idarə edilməsi üzrə ümumi prinsipləri ehtiva edir.`,
      article_information: [
        {
          chapter: "Ümumi Müddəalar",
          article: `Maddə ${Math.ceil(pageNumber / 2)}`,
          section: "Tənzimləyici Müddəa",
          chunk_id: `${documentId}_chunk_${pageNumber}`,
        }
      ]
    };
  },

  // GET /documents/highlight
  getHighlight: async (documentId: string, chunkId: string): Promise<DocumentHighlightResponse | null> => {
    await new Promise(r => setTimeout(r, 200));
    
    // Find page of the chunk
    let targetPage = 1;
    let text = "Müvafiq normativ aktın müddəası.";
    let article = "Maddə";

    if (chunkId === "802-IIQ_Art4_p1") {
      targetPage = 12;
      text = "Mərkəzi Bankın nizamnamə kapitalı dövlətə məxsusdur və 50,000,000 AZN (əlli milyon manat) təşkil edir.";
      article = "Maddə 4";
    } else if (chunkId === "590-IIQ_Art12_p2") {
      targetPage = 8;
      text = "Bankın nizamnamə kapitalı yalnız pul vəsaitləri ilə ödənilməlidir. Borc və ya girov götürülmüş vəsaitlərdən nizamnamə kapitalının formalaşdırılmasına icazə verilmir.";
      article = "Maddə 12";
    } else if (chunkId === "risk_rules_art3") {
      targetPage = 3;
      text = "Risklərin idarə edilməsi departamenti İdarə Heyətindən müstəqil olmalı və birbaşa Müşahidə Şurasına hesabat verməlidir.";
      article = "Maddə 3";
    } else {
      // Fallback parser based on chunk_id suffix
      const parts = chunkId.split("_chunk_");
      if (parts.length === 2) {
        targetPage = parseInt(parts[1]) || 1;
      }
      text = `Mərkəzi Bankın tənzimləyici normativ sənədindəki ${chunkId} identifikasiyalı mətn bloku.`;
    }

    return {
      document_id: documentId,
      page: targetPage,
      article,
      chunk_id: chunkId,
      highlighted_text: text,
      offset_status: "future_enhancement",
    };
  },

  // POST /chat
  postChat: async (
    question: string,
    sessionId?: string | null,
    llmModel?: LLMModelId | null,
  ): Promise<ChatResponse> => {
    await new Promise(r => setTimeout(r, 1200)); // Simulate LLM response time
    
    const qLower = question.toLowerCase();
    let answer = "";
    let sources: SourceDocument[] = [];

    if (qLower.includes("kapital") || qLower.includes("nizamnamə")) {
      answer = "Azərbaycan Respublikasının Mərkəzi Bankının normativ aktlarına uyğun olaraq, bankların minimum nizamnamə kapitalı **50,000,000 AZN** məbləğində müəyyən edilmişdir [1]. Nizamnamə kapitalı yalnız nağdsız pul vəsaitləri ilə formalaşdırılmalıdır və nizamnamə kapitalını borc və ya girov şəklində olan vəsaitlərlə ödəmək qadağandır [2].";
      sources = [
        {
          citation: 1,
          chunk_id: "802-IIQ_Art4_p1",
          document_id: "802-IIQ-Azərbaycan Respublikasının Mərkəzi Bankı haqqında",
          document_name: "802-IIQ - Azərbaycan Respublikasının Mərkəzi Bankı haqqında Qanun",
          category: "laws",
          chapter: "I Fəsil",
          article: "Maddə 4",
          page: 12,
          chunk_preview: "Mərkəzi Bankın nizamnamə kapitalı dövlətə məxsusdur və 50,000,000 AZN (əlli milyon manat) təşkil edir..."
        },
        {
          citation: 2,
          chunk_id: "590-IIQ_Art12_p2",
          document_id: "590-IIQ-Banklar haqqında",
          document_name: "590-IIQ - Banklar haqqında Azərbaycan Respublikasının Qanunu",
          category: "laws",
          chapter: "III Fəsil",
          article: "Maddə 12",
          page: 8,
          chunk_preview: "Bankın nizamnamə kapitalı yalnız pul vəsaitləri ilə ödənilməlidir. Borc və ya girov götürülmüş vəsaitlərdən nizamnamə kapitalının formalaşdırılmasına..."
        }
      ];
    } else if (qLower.includes("risk") || qLower.includes("idarə")) {
      answer = "Mərkəzi Bankın kredit təşkilatlarında risklərin idarə olunması üzrə qaydalarına əsasən, risklərin idarə edilməsi departamenti bank daxilində tam müstəqil funksiyadır [1]. Risk menecment departamenti İdarə Heyətinin tabeçiliyində olmamalı, birbaşa Müşahidə Şurasına hesabat verməlidir ki, maraqlar toqquşması riski minimuma endirilsin.";
      sources = [
        {
          citation: 1,
          chunk_id: "risk_rules_art3",
          document_id: "risk-management-rules",
          document_name: "Kredit təşkilatlarında risklərin idarə olunması Qaydaları",
          category: "risk_management",
          chapter: "I Hissə",
          article: "Maddə 3",
          page: 3,
          chunk_preview: "Risklərin idarə edilməsi departamenti İdarə Heyətindən müstəqil olmalı və birbaşa Müşahidə Şurasına hesabat verməlidir..."
        }
      ];
    } else {
      answer = "Verdiyiniz suala uyğun olaraq, Mərkəzi Bankın normativ aktlarında müvafiq tənzimləmə qaydası tapılmışdır [1]. Bu qaydaya əsasən, kredit təşkilatları bütün fəaliyyətlərində prudensial tənzimləmə normativlərinə əməl etməyə borcludurlar. Sualları daha spesifik etməklə dəqiq maddələrə keçid edə bilərsiniz.";
      sources = [
        {
          citation: 1,
          chunk_id: "802-IIQ_Art4_p1",
          document_id: "802-IIQ-Azərbaycan Respublikasının Mərkəzi Bankı haqqında",
          document_name: "802-IIQ - Azərbaycan Respublikasının Mərkəzi Bankı haqqında Qanun",
          category: "laws",
          chapter: "I Fəsil",
          article: "Maddə 4",
          page: 12,
          chunk_preview: "Mərkəzi Bank bank sisteminin dayanıqlığını təmin etmək məqsədilə kredit təşkilatları üçün məcburi normativləri müəyyən edir..."
        }
      ];
    }

    return {
      session_id: sessionId || "session-" + Math.random().toString(36).substr(2, 9),
      question,
      answer,
      sources,
      metrics: {
        retrieval_time: 0.085,
        generation_time: 1.115,
        total_time: 1.200,
      },
      model: {
        provider: llmModel?.startsWith("groq_") ? "groq" : "gemma",
        model_id: llmModel || "gemma",
      },
    };
  }
};
