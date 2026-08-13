import React, { useEffect, useState } from "react";
import { SidebarContainer } from "@/features/sidebar/components/SidebarContainer";
import { ChatContainer } from "@/features/chat/components/ChatContainer";
import { DocumentContainer } from "@/features/document-viewer/components/DocumentContainer";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { useUIStore } from "@/stores/useUIStore";
import { Button } from "@/components/ui/button";
import { FileText } from "lucide-react";

export const DashboardPage: React.FC = () => {
  const { sidebarOpen, setSidebarOpen, activeDocument, setActiveDocument } = useUIStore();
  const [isMobile, setIsMobile] = useState(window.innerWidth < 1024);

  // Track screen size reactively
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 1024;
      setIsMobile(mobile);
      // Auto-adjust sidebar visibility
      setSidebarOpen(!mobile);
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [setSidebarOpen]);

  return (
    <div className="h-screen w-screen flex bg-background text-foreground overflow-hidden font-sans">

      {/* 1. SIDEBAR COLUMN */}
      {/* Desktop view: Sidebar resides in normal flow */}
      <div className="hidden lg:block shrink-0">
        <SidebarContainer />
      </div>

      {/* Mobile/Tablet view: Sidebar inside a collapsible Sheet drawer */}
      <Sheet open={sidebarOpen && isMobile} onOpenChange={setSidebarOpen}>
        <SheetContent side="left" className="p-0 w-64 border-r bg-card">
          <SidebarContainer />
        </SheetContent>
      </Sheet>

      {/* 2. CHAT PANEL (Main center column) */}
      <div className="flex-1 h-full flex flex-col min-w-0 overflow-hidden relative">
        <ChatContainer />

        {/* Mobile floating button to open Document Viewer if a document is selected and viewer is hidden */}
        {activeDocument && isMobile && (
          <div className="absolute right-4 top-20 z-30">
            <Button
              size="sm"
              onClick={() => setActiveDocument({ ...activeDocument })}
              className="bg-gold-500 hover:bg-gold-600 text-white rounded-full p-2.5 shadow-lg flex items-center gap-1 text-xs"
            >
              <FileText className="h-4 w-4" />
              <span>Sənədə bax</span>
            </Button>
          </div>
        )}
      </div>

      {/* 3. DOCUMENT VIEWER COLUMN */}
      {/* Desktop view: Inline side-by-side panel — NO overlay/blur */}
      {activeDocument && !isMobile && (
        <div className="w-[450px] xl:w-[500px] shrink-0 h-full">
          <DocumentContainer />
        </div>
      )}

      {/* Mobile/Tablet view: Document Viewer slides in as a drawer */}
      {/* Only opens on mobile so the blur overlay never affects desktop */}
      <Sheet
        open={!!activeDocument && isMobile}
        onOpenChange={(open) => {
          if (!open) setActiveDocument(null);
        }}
      >
        <SheetContent
          side="right"
          className="p-0 w-full sm:max-w-lg md:max-w-xl border-l bg-card"
        >
          <DocumentContainer />
        </SheetContent>
      </Sheet>

    </div>
  );
};
