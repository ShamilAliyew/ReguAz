import React, { useState } from "react";
import {
  Bookmark,
  ChevronUp,
  LogOut,
  Moon,
  Settings,
  Sliders,
  Sun,
} from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { apiService } from "@/services/api";

interface ProfileUser {
  name?: string;
  email?: string;
}

interface ProfileMenuProps {
  sidebarOpen: boolean;
  user: ProfileUser | null;
  theme: "light" | "dark";
  toggleTheme: () => void;
  logout: () => void | Promise<void>;
}

export const ProfileMenu: React.FC<ProfileMenuProps> = ({
  sidebarOpen,
  user,
  theme,
  toggleTheme,
  logout,
}) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [logoutConfirmOpen, setLogoutConfirmOpen] = useState(false);
  const [useMock, setUseMock] = useState(apiService.isMockEnabled());

  const displayName = user?.name || "Shamil Aliyev";
  const displayEmail = user?.email || "shamil@cbar.az";
  const initials = displayName.slice(0, 2).toUpperCase();

  const handleToggleMock = (checked: boolean) => {
    setUseMock(checked);
    apiService.setMockEnabled(checked);
  };

  const confirmLogout = () => {
    setLogoutConfirmOpen(false);
    void logout();
  };

  return (
    <>
      <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
        <DropdownMenuTrigger asChild>
          {sidebarOpen ? (
            <button
              type="button"
              aria-label="Profil menyusunu aç"
              className="flex w-full items-center justify-between rounded-xl bg-secondary/50 p-2 text-left outline-none transition-colors hover:bg-secondary focus-visible:ring-2 focus-visible:ring-gold-500 dark:bg-navy-900/40 dark:hover:bg-navy-900/70"
            >
              <span className="flex min-w-0 items-center gap-2">
                <Avatar className="h-8 w-8 shrink-0">
                  <AvatarFallback className="bg-navy-900 font-bold text-white dark:bg-gold-500 dark:text-navy-950">
                    {initials}
                  </AvatarFallback>
                </Avatar>
                <span className="min-w-0 text-left">
                  <span className="block truncate text-xs font-bold">{displayName}</span>
                  <span className="block truncate text-[10px] text-muted-foreground">
                    {displayEmail}
                  </span>
                </span>
              </span>
              <ChevronUp
                aria-hidden="true"
                className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200 ${
                  menuOpen ? "rotate-180" : ""
                }`}
              />
            </button>
          ) : (
            <button
              type="button"
              aria-label="Profil menyusunu aç"
              title="Profil"
              className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl outline-none transition-colors hover:bg-secondary focus-visible:ring-2 focus-visible:ring-gold-500"
            >
              <Avatar className="h-8 w-8">
                <AvatarFallback className="bg-navy-900 font-bold text-white dark:bg-gold-500 dark:text-navy-950">
                  {initials}
                </AvatarFallback>
              </Avatar>
            </button>
          )}
        </DropdownMenuTrigger>

        <DropdownMenuContent
          side="top"
          align={sidebarOpen ? "start" : "center"}
          sideOffset={10}
          collisionPadding={8}
          className="w-60 origin-bottom rounded-xl border-border/80 p-1.5 shadow-xl duration-200 data-[state=open]:slide-in-from-bottom-3"
        >
          <DropdownMenuLabel className="px-2 py-2 font-normal">
            <span className="block truncate text-xs font-semibold">{displayName}</span>
            <span className="block truncate text-[10px] text-muted-foreground">
              {displayEmail}
            </span>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="cursor-pointer gap-2 rounded-lg py-2 text-xs"
            onSelect={() => setSettingsOpen(true)}
          >
            <Settings className="h-4 w-4" />
            <span>Nizamlamalar</span>
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled
            className="gap-2 rounded-lg py-2 text-xs"
          >
            <Bookmark className="h-4 w-4" />
            <span>Seçilmişlər (tezliklə)</span>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="cursor-pointer gap-2 rounded-lg py-2 text-xs text-red-600 focus:bg-red-50 focus:text-red-700 dark:text-red-400 dark:focus:bg-red-950/30"
            onSelect={() => setLogoutConfirmOpen(true)}
          >
            <LogOut className="h-4 w-4" />
            <span>Çıxış</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Sistem nizamlamaları</DialogTitle>
            <DialogDescription>
              ReguAZ interfeys parametrlərini və API rejimini idarə edin.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4 text-sm">
            <div className="flex items-center justify-between border-b pb-3">
              <div className="space-y-0.5 text-left">
                <p className="font-semibold">Qaranlıq rejim</p>
                <p className="text-xs text-muted-foreground">
                  İnterfeysin mövzusunu dəyişin
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={toggleTheme}>
                {theme === "light" ? (
                  <Moon className="mr-2 h-4 w-4" />
                ) : (
                  <Sun className="mr-2 h-4 w-4" />
                )}
                {theme === "light" ? "Qaranlıq rejim" : "Aydınlıq rejim"}
              </Button>
            </div>
            <div className="flex items-center justify-between pb-1">
              <div className="space-y-0.5 text-left">
                <p className="flex items-center gap-1.5 font-semibold">
                  <Sliders className="h-4 w-4 text-gold-500" />
                  Simulyasiya rejimi (Mock API)
                </p>
                <p className="text-xs text-muted-foreground">
                  Real backend yoxdursa, mock data istifadə edin
                </p>
              </div>
              <input
                type="checkbox"
                aria-label="Simulyasiya rejimi"
                checked={useMock}
                onChange={(event) => handleToggleMock(event.target.checked)}
                className="relative h-5 w-9 cursor-pointer appearance-none rounded-full bg-gray-200 before:absolute before:left-[2px] before:top-[2px] before:h-4 before:w-4 before:rounded-full before:bg-white before:transition-transform before:content-[''] checked:bg-gold-500 checked:before:translate-x-4"
              />
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={logoutConfirmOpen} onOpenChange={setLogoutConfirmOpen}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Hesabdan çıxış</DialogTitle>
            <DialogDescription>
              Hesabdan çıxmaq istədiyinizə əminsiniz?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-2 gap-2 sm:space-x-0">
            <Button
              type="button"
              variant="outline"
              onClick={() => setLogoutConfirmOpen(false)}
            >
              Xeyr
            </Button>
            <Button type="button" variant="destructive" onClick={confirmLogout}>
              Bəli, çıxış et
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
