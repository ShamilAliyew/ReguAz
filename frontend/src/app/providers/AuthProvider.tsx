import React, { createContext, useContext, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuthStore } from "../../stores/useAuthStore";
import { useChatStore } from "../../stores/useChatStore";
import type { AuthUser } from "../../types/api";

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  user: AuthUser | null;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const auth = useAuthStore();

  useEffect(() => {
    void auth.initialize();
  }, [auth.initialize]);

  useEffect(() => {
    useChatStore.getState().setStorageOwner(auth.user?.id ?? null);
  }, [auth.user?.id]);

  return (
    <AuthContext.Provider value={{
      isAuthenticated: auth.isAuthenticated,
      isLoading: auth.status === "idle" || auth.status === "loading",
      login: auth.login,
      register: auth.register,
      logout: auth.logout,
      user: auth.user
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // For development convenience, we'll allow navigation or redirect if not authenticated.
    // If not authenticated, redirect to /login
    if (!isLoading && !isAuthenticated) {
      navigate("/login", { replace: true, state: { from: location } });
    }
  }, [isAuthenticated, isLoading, navigate, location]);

  // Avoid redirect flicker while the HttpOnly session cookie is checked.
  return !isLoading && isAuthenticated ? <>{children}</> : (
    <div className="flex h-screen w-screen items-center justify-center bg-background">
      <div className="flex flex-col items-center space-y-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-gold-500 border-t-transparent" />
        <p className="text-sm text-muted-foreground font-medium">Yönləndirilir...</p>
      </div>
    </div>
  );
};
