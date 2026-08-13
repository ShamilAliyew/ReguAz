import React, { useState, useEffect } from "react";
import { 
  FileText, 
  Search, 
  ZoomIn, 
  ZoomOut, 
  Download, 
  Printer, 
  ChevronLeft, 
  ChevronRight, 
  X,
  Tag,
  Calendar,
  Layers,
  CheckCircle
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { useUIStore } from "@/stores/useUIStore";
import { useGetDocumentMetadata, useGetDocumentPage } from "@/hooks/useDocuments";
import { toast } from "sonner";
import { resolveHighlightRanges } from "../highlightSelectors";

export const DocumentContainer: React.FC = () => {
  const { activeDocument, setActiveDocument, setSelectedCitation } = useUIStore();
  
  const [searchQuery, setSearchQuery] = useState("");
  const [zoomLevel, setZoomLevel] = useState(100);
  const [activeTab, setActiveTab] = useState<"content" | "metadata">("content");


  const metadataQuery = useGetDocumentMetadata(activeDocument?.documentId || null);
  const pageQuery = useGetDocumentPage(
    activeDocument?.documentId || null, 
    activeDocument?.activePage || 1
  );

  const documentData = metadataQuery.data;
  const pageData = pageQuery.data;

  // Sync zoom level from store
  useEffect(() => {
    if (activeDocument?.zoomLevel) {
      setZoomLevel(activeDocument.zoomLevel);
    }
  }, [activeDocument?.zoomLevel]);

  // When a citation is clicked, switch to content tab and scroll to the highlighted text after render
  useEffect(() => {
    if (activeDocument?.highlightText) {
      setActiveTab("content");
      // Defer scroll until after React renders the highlight mark
      const timer = setTimeout(() => {
        const el = document.getElementById("reguaz-highlight");
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [activeDocument?.highlightText, activeDocument?.documentId, activeDocument?.activePage, pageQuery.data]);

  if (!activeDocument || !activeDocument.documentId) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-6 bg-card border-l">
        <div className="p-4 bg-secondary/50 dark:bg-navy-900 rounded-2xl mb-4 border">
          <FileText className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-sm font-bold text-foreground mb-1">Normativ Akt Oxuyucusu</h3>
        <p className="text-xs text-muted-foreground max-w-xs font-light leading-relaxed">
          Sualınıza aid istinad maddələrini dərhal araşdırmaq üçün chat-dakı citation badge-lərə klikləyin.
        </p>
      </div>
    );
  }

  const totalPages = documentData?.total_pages || 54;
  const currentPage = activeDocument.activePage;

  const handlePageChange = (page: number) => {
    if (page < 1 || page > totalPages) return;
    setActiveDocument({ activePage: page });
  };

  const handleZoom = (direction: "in" | "out") => {
    const nextZoom = direction === "in" ? zoomLevel + 10 : zoomLevel - 10;
    if (nextZoom < 50 || nextZoom > 200) return;
    setZoomLevel(nextZoom);
    setActiveDocument({ zoomLevel: nextZoom });
  };

  const handleDownload = () => {
    toast.success("Sənəd yüklənməyə hazırlanır...", {
      description: `${documentData?.title || "Normativ akt"} PDF formatında endirilir.`,
    });
  };

  const handlePrint = () => {
    toast.success("Çap pəncərəsi açılır...");
  };

  // Highlights citation snippet or term searches inside the raw content text
  const renderHighlightedContent = () => {
    const content = pageData?.page_content || "";
    if (!content) return <p className="text-xs text-muted-foreground italic font-light">Məzmun tapılmadı</p>;

    const highlightText = activeDocument.highlightText;

    // Priority 1: In-document search query highlight (yellow)
    if (searchQuery.trim().length >= 2) {
      const queryEscaped = searchQuery.replace(/[-\\^$*+?.()|[\]{}]/g, '\\$&');
      const regex = new RegExp(`(${queryEscaped})`, 'gi');
      const parts = content.split(regex);
      
      return (
        <p className="whitespace-pre-wrap leading-relaxed font-light break-words text-left text-xs">
          {parts.map((part, i) => 
            regex.test(part) ? (
              <mark key={i} className="bg-yellow-200 dark:bg-yellow-800 text-foreground px-0.5 rounded font-medium">
                {part}
              </mark>
            ) : part
          )}
        </p>
      );
    }

    // Priority 2: exact V2 source selectors. No loose/preview regex matching.
    if (pageData && activeDocument.selectors.length > 0) {
      const resolution = resolveHighlightRanges(pageData, activeDocument.selectors);
      if (resolution.warning) {
        return (
          <div className="space-y-3">
            <p role="alert" className="text-[11px] rounded border border-amber-400/60 bg-amber-50 dark:bg-amber-950/30 px-3 py-2 text-amber-900 dark:text-amber-200">
              {resolution.warning} Əlaqəsiz mətn highlight edilmədi.
            </p>
            <p className="whitespace-pre-wrap leading-relaxed font-light break-words text-left text-xs">{content}</p>
          </div>
        );
      }
      const nodes: React.ReactNode[] = [];
      let cursor = 0;
      resolution.ranges.forEach((range, index) => {
        nodes.push(content.slice(cursor, range.start));
        nodes.push(
          <mark
            key={`${range.start}-${range.end}`}
            id={index === 0 ? "reguaz-highlight" : undefined}
            className={`${highlightColor(activeDocument.evidenceId)} text-navy-950 dark:text-white border-b-2 font-semibold py-0.5 px-0.5 rounded-sm scroll-mt-4`}
          >
            {content.slice(range.start, range.end)}
          </mark>,
        );
        cursor = range.end;
      });
      nodes.push(content.slice(cursor));
      return <p className="whitespace-pre-wrap leading-relaxed font-light break-words text-left text-xs">{nodes}</p>;
    }

    // V1 fallback remains preview-based, explicitly non-authoritative.
    if (highlightText && highlightText.trim().length > 0) {
      const start = content.indexOf(highlightText);
      if (start >= 0 && content.indexOf(highlightText, start + 1) < 0) {
        const end = start + highlightText.length;
        return (
          <p className="whitespace-pre-wrap leading-relaxed font-light break-words text-left text-xs">
            {content.slice(0, start)}
            <mark id="reguaz-highlight" className="bg-gold-100 dark:bg-gold-950/60 text-navy-950 dark:text-gold-200 border-b-2 border-gold-500 font-semibold">{content.slice(start, end)}</mark>
            {content.slice(end)}
          </p>
        );
      }
    }
    return <p className="whitespace-pre-wrap leading-relaxed font-light break-words text-left text-xs">{content}</p>;
  };

  return (
    <div className="h-full flex flex-col bg-card text-card-foreground border-l relative z-10 w-full">
      
      {/* Header Panel */}
      <header className="h-16 px-4 border-b flex items-center justify-between bg-card shrink-0">
        <div className="flex items-center gap-2 truncate pr-4 text-left">
          <FileText className="h-4 w-4 text-gold-500 shrink-0" />
          <h3 
            className="text-xs font-bold truncate text-foreground"
            title={documentData?.title || activeDocument.documentId}
          >
            {documentData?.title || activeDocument.documentId}
          </h3>
        </div>
        <Button 
          variant="ghost" 
          size="icon" 
          className="h-8 w-8 text-muted-foreground hover:text-foreground shrink-0"
          onClick={() => {
            setActiveDocument(null);
            setSelectedCitation(null);
          }}
        >
          <X className="h-4 w-4" />
        </Button>
      </header>

      {/* Control Actions Bar */}
      <div className="px-4 py-2 border-b bg-secondary/30 flex flex-wrap items-center justify-between gap-2 shrink-0">
        
        {/* Pagination controls */}
        <div className="flex items-center gap-1 text-xs">
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-7 w-7"
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage <= 1 || pageQuery.isLoading}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="font-medium text-foreground">Səhifə {currentPage} / {totalPages}</span>
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-7 w-7"
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage >= totalPages || pageQuery.isLoading}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-1.5 text-xs">
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-7 w-7"
            onClick={() => handleZoom("out")}
            disabled={zoomLevel <= 50}
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </Button>
          <span className="font-mono w-9 text-center font-medium">{zoomLevel}%</span>
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-7 w-7"
            onClick={() => handleZoom("in")}
            disabled={zoomLevel >= 200}
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Print / Download actions */}
        <div className="flex items-center gap-1">
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-7 w-7"
            onClick={handlePrint}
            title="Sənədi çap et"
          >
            <Printer className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground" />
          </Button>
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-7 w-7"
            onClick={handleDownload}
            title="PDF sənədi endir"
          >
            <Download className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground" />
          </Button>
        </div>

      </div>

      {/* Tabs Menu */}
      <div className="flex px-4 border-b bg-card shrink-0">
        <button
          onClick={() => setActiveTab("content")}
          className={`py-2 px-3 text-xs font-semibold border-b-2 transition-colors ${
            activeTab === "content" 
              ? "border-gold-500 text-navy-900 dark:text-white" 
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          Sənəd Mətni
        </button>
        <button
          onClick={() => setActiveTab("metadata")}
          className={`py-2 px-3 text-xs font-semibold border-b-2 transition-colors ${
            activeTab === "metadata" 
              ? "border-gold-500 text-navy-900 dark:text-white" 
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          Metaməlumatlar
        </button>
      </div>

      {/* Main Tab Content Panel */}
      <div className="flex-1 overflow-hidden relative">
        {activeTab === "content" ? (
          
          /* CONTENT TAB VIEW */
          <div className="h-full flex flex-col">
            {/* Search inside Document Input */}
            <div className="p-3 border-b bg-secondary/10 shrink-0">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Sənəd daxilində axtar..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-8 pr-3 py-1.5 border rounded-lg bg-background text-foreground text-xs focus:outline-none focus:ring-1 focus:ring-gold-500 focus:border-gold-500"
                />
              </div>
            </div>

            {/* Document page text container */}
            <ScrollArea className="flex-1">
              <div 
                className="p-6 transition-all duration-150 origin-top-left"
                style={{ fontSize: `${(zoomLevel / 100) * 0.875}rem` }}
              >
                {pageQuery.isLoading ? (
                  <div className="space-y-3">
                    <div className="h-4 bg-muted rounded w-3/4 animate-pulse" />
                    <div className="h-4 bg-muted rounded w-5/6 animate-pulse" />
                    <div className="h-4 bg-muted rounded w-2/3 animate-pulse" />
                    <div className="h-4 bg-muted rounded w-full animate-pulse" />
                  </div>
                ) : (
                  renderHighlightedContent()
                )}
              </div>
            </ScrollArea>
          </div>

        ) : (

          /* METADATA TAB VIEW */
          <ScrollArea className="h-full p-4">
            <div className="space-y-4 text-xs text-left">
              <Card>
                <CardContent className="p-4 space-y-3.5">
                  <div className="flex justify-between items-center">
                    <span className="font-semibold text-muted-foreground flex items-center gap-1.5">
                      <Tag className="h-3.5 w-3.5 text-gold-500" />
                      Kateqoriya:
                    </span>
                    <Badge variant="navy" className="capitalize text-[10px]">
                      {documentData?.category.replace("_", " ") || "Qanun"}
                    </Badge>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="font-semibold text-muted-foreground flex items-center gap-1.5">
                      <Calendar className="h-3.5 w-3.5 text-gold-500" />
                      Dərc tarixi:
                    </span>
                    <span className="text-foreground">{documentData?.publication_date || "Məlum deyil"}</span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="font-semibold text-muted-foreground flex items-center gap-1.5">
                      <CheckCircle className="h-3.5 w-3.5 text-gold-500" />
                      Status:
                    </span>
                    <Badge variant="gold" className="text-[10px] uppercase font-bold">
                      {documentData?.status === "active" ? "Qüvvədədir" : "Arxiv"}
                    </Badge>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="font-semibold text-muted-foreground flex items-center gap-1.5">
                      <Layers className="h-3.5 w-3.5 text-gold-500" />
                      Ümumi səhifə:
                    </span>
                    <span className="text-foreground font-mono">{totalPages} səhifə</span>
                  </div>
                </CardContent>
              </Card>

              {/* Related articles */}
              {documentData?.related_articles && documentData.related_articles.length > 0 && (
                <div className="space-y-2">
                  <h4 className="font-bold text-foreground">Əlaqəli Maddələr:</h4>
                  <div className="flex flex-wrap gap-2">
                    {documentData.related_articles.map((article, index) => (
                      <Badge 
                        key={index} 
                        variant="secondary" 
                        className="cursor-pointer hover:bg-gold-50 hover:text-gold-950 transition-colors"
                        onClick={() => {
                          setActiveTab("content");
                          // Find matching page index if we click related articles (simulated pages)
                          const mockPage = index === 0 ? 12 : (index === 1 ? 8 : 3);
                          handlePageChange(mockPage);
                        }}
                      >
                        {article}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

        )}
      </div>

    </div>
  );
};

function highlightColor(evidenceId: string | null): string {
  const colors = [
    "bg-yellow-200 dark:bg-yellow-800/70 border-yellow-500",
    "bg-cyan-200 dark:bg-cyan-800/70 border-cyan-500",
    "bg-emerald-200 dark:bg-emerald-800/70 border-emerald-500",
    "bg-violet-200 dark:bg-violet-800/70 border-violet-500",
    "bg-rose-200 dark:bg-rose-800/70 border-rose-500",
  ];
  const hash = Array.from(evidenceId || "E1").reduce(
    (value, character) => value + character.charCodeAt(0),
    0,
  );
  return colors[hash % colors.length];
}
