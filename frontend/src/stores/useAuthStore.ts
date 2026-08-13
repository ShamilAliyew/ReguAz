import { create } from "zustand";

interface User {
  id: string;
  name: string;
  email: string;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string) => Promise<boolean>;
  register: (name: string, email: string) => Promise<boolean>;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => {
  // Read initial session from localStorage
  const getInitialUser = (): User | null => {
    const saved = localStorage.getItem("reguaz-user");
    return saved ? JSON.parse(saved) : null;
  };

  const initialUser = getInitialUser();

  return {
    user: initialUser,
    isAuthenticated: !!initialUser,

    login: async (email: string) => {
      // Mock network latency
      await new Promise((resolve) => setTimeout(resolve, 500));
      
      const mockUser: User = {
        id: "usr-1",
        name: "Shamil Aliyev",
        email: email,
      };

      localStorage.setItem("reguaz-user", JSON.stringify(mockUser));
      set({ user: mockUser, isAuthenticated: true });
      return true;
    },

    register: async (name: string, email: string) => {
      await new Promise((resolve) => setTimeout(resolve, 500));
      
      const mockUser: User = {
        id: "usr-" + Math.random().toString(36).substr(2, 9),
        name,
        email,
      };

      localStorage.setItem("reguaz-user", JSON.stringify(mockUser));
      set({ user: mockUser, isAuthenticated: true });
      return true;
    },

    logout: () => {
      localStorage.removeItem("reguaz-user");
      set({ user: null, isAuthenticated: false });
    },
  };
});
