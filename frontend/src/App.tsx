import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider, ProtectedRoute } from "./app/providers/AuthProvider";
import { Toaster } from "./components/ui/sonner";
import { LandingPage } from "./pages/LandingPage";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import { DashboardPage } from "./pages/DashboardPage";

// Initialize TanStack Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            {/* Public landing page */}
            <Route path="/" element={<LandingPage />} />
            
            {/* Auth pages */}
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
            
            {/* Protected dashboard interface */}
            <Route 
              path="/dashboard" 
              element={
                <ProtectedRoute>
                  <DashboardPage />
                </ProtectedRoute>
              } 
            />
            
            {/* Fallback route redirection */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
          
          {/* Sonner toast alerts */}
          <Toaster position="top-right" closeButton richColors />
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
};
