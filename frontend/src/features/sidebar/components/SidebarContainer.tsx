import React, { useState } from "react";
import { 
  MessageSquarePlus, 
  Search, 
  Settings, 
  Bookmark, 
  LogOut, 
  ChevronLeft, 
  ChevronRight, 
  FileText,
  Trash2,
  Sliders,
  Moon,
  Sun
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useUIStore } from "@/stores/useUIStore";
import { useChatStore } from "@/stores/useChatStore";
import { useAuth } from "@/app/providers/AuthProvider";
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from "@/components/ui/dialog";
import { apiService } from "@/services/api";

export const SidebarContainer: React.FC = () => {
  const { sidebarOpen, toggleSidebar, theme, toggleTheme } = useUIStore();
  const { history, activeSessionId, createNewSession, deleteSession, setActiveSessionId } = useChatStore();
  const { user, logout } = useAuth();
  const [searchTerm, setSearchTerm] = useState("");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [useMock, setUseMock] = useState(apiService.isMockEnabled());

  const filteredHistory = history.filter((item) =>
    item.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleToggleMock = (checked: boolean) => {
    setUseMock(checked);
    apiService.setMockEnabled(checked);
  };

  return (
    <div
      className={`h-screen flex flex-col border-r bg-card text-card-foreground transition-all duration-300 relative z-20 ${
        sidebarOpen ? "w-64" : "w-16"
      }`}
    >
      {/* Sidebar Header / Logo */}
      <div className="h-16 flex items-center justify-between px-4">
        {sidebarOpen ? (
          <div className="flex items-center gap-1.5">
            <span className="text-xl font-black tracking-tight">
              Regu<span className="text-gold-500">AZ</span>
            </span>
            <span className="text-[9px] border border-gold-500/40 rounded px-1 text-gold-500 font-semibold uppercase tracking-wider">
              CBAR
            </span>
          </div>
        ) : (
          <span className="text-lg font-black tracking-tight text-center w-full">R<span className="text-gold-500">A</span></span>
        )}

        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 text-muted-foreground hover:text-foreground hidden md:flex"
          onClick={toggleSidebar}
        >
          {sidebarOpen ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </Button>
      </div>

      <Separator />

      {/* Action: New Chat */}
      <div className="p-3">
        {sidebarOpen ? (
          <Button
            onClick={() => createNewSession()}
            className="w-full justify-start gap-2 bg-navy-900 text-white hover:bg-navy-800 dark:bg-gold-500 dark:text-navy-950 dark:hover:bg-gold-600 font-semibold"
          >
            <MessageSquarePlus className="h-4 w-4" />
            <span>Yeni söhbət</span>
          </Button>
        ) : (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => createNewSession()}
            className="w-full text-navy-900 dark:text-gold-500"
            title="Yeni söhbət"
          >
            <MessageSquarePlus className="h-5 w-5" />
          </Button>
        )}
      </div>

      {/* Action: Search History */}
      {sidebarOpen && (
        <div className="px-3 pb-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Söhbətlərdə axtar..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 border rounded-lg bg-background text-foreground text-xs focus:outline-none focus:ring-1 focus:ring-gold-500 focus:border-gold-500"
            />
          </div>
        </div>
      )}

      {/* History List */}
      <div className="flex-1 min-h-0">
        {sidebarOpen ? (
          <div className="h-full flex flex-col">
            <span className="px-4 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1 block">
              Tarixçə
            </span>
            <ScrollArea className="flex-1 px-2">
              <div className="space-y-1">
                {filteredHistory.length === 0 ? (
                  <p className="text-xs text-muted-foreground text-center py-4 font-light">Söhbət tapılmadı</p>
                ) : (
                  filteredHistory.map((item) => (
                    <div
                      key={item.sessionId}
                      className={`group flex items-center justify-between rounded-lg p-2 text-xs font-medium cursor-pointer transition-colors ${
                        activeSessionId === item.sessionId
                          ? "bg-navy-50 text-navy-900 dark:bg-navy-900/60 dark:text-gold-400"
                          : "hover:bg-secondary text-muted-foreground hover:text-foreground"
                      }`}
                      onClick={() => setActiveSessionId(item.sessionId)}
                    >
                      <div className="flex items-center gap-2 truncate">
                        <FileText className="h-3.5 w-3.5 shrink-0" />
                        <span className="truncate">{item.title}</span>
                      </div>
                      <button
                        className="opacity-0 group-hover:opacity-100 hover:text-red-500 p-0.5 rounded transition-opacity"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteSession(item.sessionId);
                        }}
                        title="Sil"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 py-4">
            <FileText className="h-4 w-4 text-muted-foreground" />
          </div>
        )}
      </div>

      <Separator />

      {/* Settings / Bookmarks Placeholder Footer */}
      <div className="p-2 space-y-1">
        <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
          <DialogTrigger asChild>
            {sidebarOpen ? (
              <Button
                variant="ghost"
                className="w-full justify-start gap-3 text-xs text-muted-foreground hover:text-foreground"
              >
                <Settings className="h-4 w-4" />
                <span>Nizamlamalar</span>
              </Button>
            ) : (
              <Button
                variant="ghost"
                size="icon"
                className="w-full text-muted-foreground"
                title="Nizamlamalar"
              >
                <Settings className="h-4 w-4" />
              </Button>
            )}
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Sistem Nizamlamaları</DialogTitle>
              <DialogDescription>
                ReguAZ interfeys parametrələrini və API konfiqurasiyalarını idarə edin.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4 text-sm">
              <div className="flex items-center justify-between border-b pb-3">
                <div className="space-y-0.5 text-left">
                  <p className="font-semibold">Qaranlıq rejim</p>
                  <p className="text-xs text-muted-foreground">Interfeysin mövzusunu dəyişin</p>
                </div>
                <Button variant="outline" size="sm" onClick={toggleTheme}>
                  {theme === "light" ? <Moon className="h-4 w-4 mr-2" /> : <Sun className="h-4 w-4 mr-2" />}
                  {theme === "light" ? "Qaranlıq rejim" : "Aydınlıq rejim"}
                </Button>
              </div>
              <div className="flex items-center justify-between pb-1">
                <div className="space-y-0.5 text-left">
                  <p className="font-semibold flex items-center gap-1.5">
                    <Sliders className="h-4 w-4 text-gold-500" />
                    Simulyasiya Rejimi (Mock API)
                  </p>
                  <p className="text-xs text-muted-foreground">Real backend yoxdursa, mock data istifadə edin</p>
                </div>
                <input
                  type="checkbox"
                  checked={useMock}
                  onChange={(e) => handleToggleMock(e.target.checked)}
                  className="w-9 h-5 bg-gray-200 rounded-full appearance-none checked:bg-gold-500 relative before:content-[''] before:absolute before:w-4 before:h-4 before:bg-white before:rounded-full before:top-[2px] before:left-[2px] before:transition-transform checked:before:translate-x-4 cursor-pointer"
                />
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {sidebarOpen ? (
          <Button
            variant="ghost"
            className="w-full justify-start gap-3 text-xs text-muted-foreground hover:text-foreground opacity-50 cursor-not-allowed"
            title="Gələcək modul"
          >
            <Bookmark className="h-4 w-4" />
            <span>Seçilmişlər (Tezliklə)</span>
          </Button>
        ) : (
          <Button
            variant="ghost"
            size="icon"
            className="w-full text-muted-foreground opacity-50 cursor-not-allowed"
            title="Seçilmişlər"
          >
            <Bookmark className="h-4 w-4" />
          </Button>
        )}
      </div>

      <Separator />

      {/* User Profile Info */}
      <div className="p-3">
        {sidebarOpen ? (
          <div className="flex items-center justify-between bg-secondary/50 dark:bg-navy-900/40 p-2 rounded-xl">
            <div className="flex items-center gap-2 truncate">
              <Avatar className="h-8 w-8">
                <AvatarFallback className="bg-navy-900 text-white dark:bg-gold-500 dark:text-navy-950 font-bold">
                  {user?.name?.slice(0, 2).toUpperCase() || "SA"}
                </AvatarFallback>
              </Avatar>
              <div className="text-left truncate">
                <p className="text-xs font-bold truncate">{user?.name || "Shamil Aliyev"}</p>
                <p className="text-[10px] text-muted-foreground truncate">{user?.email || "shamil@cbar.az"}</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="text-muted-foreground hover:text-red-500 p-1.5 rounded transition-colors"
              title="Çıxış"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <Avatar className="h-8 w-8 mb-2">
              <AvatarFallback className="bg-navy-900 text-white dark:bg-gold-500 dark:text-navy-950 font-bold">
                {user?.name?.slice(0, 2).toUpperCase() || "SA"}
              </AvatarFallback>
            </Avatar>
            <button
              onClick={logout}
              className="text-muted-foreground hover:text-red-500 transition-colors p-1"
              title="Çıxış"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
