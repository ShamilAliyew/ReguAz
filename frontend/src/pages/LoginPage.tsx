import React, { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { ShieldCheck, Mail, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/app/providers/AuthProvider";

export const LoginPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const from = (location.state as any)?.from?.pathname || "/dashboard";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) {
      setError("E-poçt ünvanınızı daxil edin.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      await login(email);
      navigate(from, { replace: true });
    } catch {
      setError("Giriş zamanı xəta baş verdi.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#faf8f5] dark:bg-navy-950 px-4 transition-colors duration-300">
      <Card className="w-full max-w-md shadow-lg border rounded-2xl bg-card">
        <CardHeader className="space-y-2 text-center">
          <div className="flex justify-center mb-2">
            <span className="text-3xl font-extrabold tracking-tight text-navy-900 dark:text-white">
              Regu<span className="text-gold-500">AZ</span>
            </span>
          </div>
          <CardTitle className="text-xl font-bold">Portala Giriş</CardTitle>
          <CardDescription className="text-sm font-light text-muted-foreground">
            Mərkəzi Bank normativ aktları üzrə intellektual axtarış
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="text-xs bg-red-50 text-red-600 dark:bg-red-950/20 dark:text-red-400 p-2.5 rounded-lg border border-red-200">
                {error}
              </div>
            )}
            
            <div className="space-y-1 text-left">
              <label className="text-xs font-semibold text-muted-foreground">E-poçt</label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@company.com"
                  className="w-full pl-10 pr-4 py-2 border rounded-lg bg-background text-foreground text-sm focus:outline-none focus:ring-1 focus:ring-gold-500 focus:border-gold-500"
                />
              </div>
            </div>

            <div className="space-y-1 text-left">
              <label className="text-xs font-semibold text-muted-foreground flex justify-between">
                Şifrə
                <span className="text-[10px] text-gold-500 hover:underline cursor-pointer">Unutmusunuz?</span>
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full pl-10 pr-4 py-2 border rounded-lg bg-background text-foreground text-sm focus:outline-none focus:ring-1 focus:ring-gold-500 focus:border-gold-500"
                />
              </div>
            </div>

            <Button
              type="submit"
              disabled={loading}
              className="w-full bg-navy-900 text-white hover:bg-navy-800 dark:bg-gold-500 dark:text-navy-950 dark:hover:bg-gold-600 font-semibold"
            >
              {loading ? "Giriş edilir..." : "Daxil ol"}
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex flex-col space-y-2 text-center text-xs">
          <p className="text-muted-foreground font-light">
            Hesabınız yoxdur?{" "}
            <Link to="/register" className="text-gold-500 hover:underline font-medium">
              Qeydiyyatdan keçin
            </Link>
          </p>
          <div className="pt-2 border-t w-full flex items-center justify-center gap-1 text-[10px] text-muted-foreground">
            <ShieldCheck className="h-3 w-3 text-gold-500" />
            <span>Məlumatlar CBAR təhlükəsizlik standartları ilə qorunur.</span>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
};
