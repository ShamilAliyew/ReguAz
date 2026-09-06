import { create } from "zustand";

import { apiService } from "../services/api";
import type { AuthUser } from "../types/api";

type AuthStatus = "idle" | "loading" | "authenticated" | "anonymous";

interface AuthState {
  user: AuthUser | null;
  status: AuthStatus;
  isAuthenticated: boolean;
  initialize: () => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

let initializationPromise: Promise<void> | null = null;

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  status: "idle",
  isAuthenticated: false,

  initialize: async () => {
    if (get().status === "authenticated" || get().status === "anonymous") return;
    if (initializationPromise) return initializationPromise;

    set({ status: "loading" });
    initializationPromise = apiService
      .getCurrentUser()
      .then(({ user }) => {
        set({ user, status: "authenticated", isAuthenticated: true });
      })
      .catch(() => {
        set({ user: null, status: "anonymous", isAuthenticated: false });
      })
      .finally(() => {
        initializationPromise = null;
      });
    return initializationPromise;
  },

  login: async (email, password) => {
    const { user } = await apiService.login(email, password);
    set({ user, status: "authenticated", isAuthenticated: true });
  },

  register: async (name, email, password) => {
    const { user } = await apiService.register(name, email, password);
    set({ user, status: "authenticated", isAuthenticated: true });
  },

  logout: async () => {
    try {
      await apiService.logout();
    } finally {
      set({ user: null, status: "anonymous", isAuthenticated: false });
    }
  },
}));
