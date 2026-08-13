import React, { createContext, useContext, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuthStore } from "../../stores/useAuthStore";

interface AuthContextType {
  isAuthenticated: boolean;
  login: (email: string) => Promise<boolean>;
  register: (name: string, email: string) => Promise<boolean>;
  logout: () => void;
  user: any;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const auth = useAuthStore();

  return (
    <AuthContext.Provider value={{
      isAuthenticated: auth.isAuthenticated,
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
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // For development convenience, we'll allow navigation or redirect if not authenticated.
    // If not authenticated, redirect to /login
    if (!isAuthenticated) {
      navigate("/login", { replace: true, state: { from: location } });
    }
  }, [isAuthenticated, navigate, location]);

  // If authenticated, render children. Otherwise render placeholder while redirecting
  return isAuthenticated ? <>{children}</> : (
    <div className="flex h-screen w-screen items-center justify-center bg-background">
      <div className="flex flex-col items-center space-y-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-gold-500 border-t-transparent" />
        <p className="text-sm text-muted-foreground font-medium">Yönləndirilir...</p>
      </div>
    </div>
  );
};
