import { create } from "zustand";
import { SourceSelector } from "../types/api";

interface CitationSelection {
  citationId: number;
  chunkId: string;
  documentId: string;
  evidenceId?: string;
}

interface ActiveDocumentState {
  documentId: string;
  activePage: number;
  zoomLevel: number;
  highlightText: string | null;
  selectors: SourceSelector[];
  evidenceId: string | null;
}

interface UIState {
  theme: "light" | "dark";
  sidebarOpen: boolean;
  selectedCitation: CitationSelection | null;
  activeDocument: ActiveDocumentState | null;
  toggleTheme: () => void;
  setTheme: (theme: "light" | "dark") => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setSelectedCitation: (citation: CitationSelection | null) => void;
  setActiveDocument: (doc: Partial<ActiveDocumentState> | null) => void;
}

export const useUIStore = create<UIState>((set) => {
  // Read initial theme from localStorage or system preference
  const getInitialTheme = (): "light" | "dark" => {
    const saved = localStorage.getItem("reguaz-theme");
    if (saved === "light" || saved === "dark") return saved;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    return media.matches ? "dark" : "light";
  };

  const initialTheme = getInitialTheme();
  // Apply theme to document element immediately upon store creation
  if (initialTheme === "dark") {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }

  return {
    theme: initialTheme,
    sidebarOpen: true,
    selectedCitation: null,
    activeDocument: null,

    toggleTheme: () =>
      set((state) => {
        const next = state.theme === "light" ? "dark" : "light";
        localStorage.setItem("reguaz-theme", next);
        if (next === "dark") {
          document.documentElement.classList.add("dark");
        } else {
          document.documentElement.classList.remove("dark");
        }
        return { theme: next };
      }),

    setTheme: (theme) => {
      localStorage.setItem("reguaz-theme", theme);
      if (theme === "dark") {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
      set({ theme });
    },

    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
    setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
    
    setSelectedCitation: (selectedCitation) => set({ selectedCitation }),
    
    setActiveDocument: (doc) =>
      set((state) => {
        if (doc === null) return { activeDocument: null };
        const prev = state.activeDocument || {
          documentId: "",
          activePage: 1,
          zoomLevel: 100,
          highlightText: null,
          selectors: [],
          evidenceId: null,
        };
        return {
          activeDocument: {
            ...prev,
            ...doc,
          } as ActiveDocumentState,
        };
      }),
  };
});
