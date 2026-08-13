import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { 
  Search, 
  FileText, 
  ShieldCheck, 
  BookOpen,
  ArrowRight,
  TrendingUp,
  Cpu
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useUIStore } from "@/stores/useUIStore";

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const { theme, toggleTheme } = useUIStore();

  const features = [
    {
      icon: <Search className="h-6 w-6 text-gold-500" />,
      title: "AI əsaslı semantik axtarış",
      description: "Mərkəzi Bankın qaydalarında axtardığınız tənzimləmələri açar sözlərlə deyil, təbii dildə suallarla tapın."
    },
    {
      icon: <ShieldCheck className="h-6 w-6 text-gold-500" />,
      title: "Normativ aktlara əsaslanan cavablar",
      description: "Hər bir cavab yalnız Mərkəzi Bankın təsdiq etdiyi rəsmi normativ sənədlərə əsaslanır, hallüsinasiyalara yol verilmir."
    },
    {
      icon: <BookOpen className="h-6 w-6 text-gold-500" />,
      title: "Mənbələrin avtomatik göstərilməsi",
      description: "Cavab daxilindəki istinadlar sizi birbaşa mənbə sənədinə, müvafiq fəsil, maddə və paraqrafa yönləndirir."
    },
    {
      icon: <TrendingUp className="h-6 w-6 text-gold-500" />,
      title: "Sürətli sənəd araşdırması",
      description: "Saatlarla uzun sənədləri oxumaq əvəzinə, saniyələr ərzində sizə lazım olan tənzimləyici normanı təhlil edin."
    },
    {
      icon: <Cpu className="h-6 w-6 text-gold-500" />,
      title: "RAG texnologiyası",
      description: "Müasir Retrieval-Augmented Generation texnologiyası tənzimləyici sənədlər toplusundan ən uyğun hissələri seçib gətirir."
    },
    {
      icon: <FileText className="h-6 w-6 text-gold-500" />,
      title: "İstinad olunan maddələrin göstərilməsi",
      description: "Normativ sənədin maddəsi üzərinə klikləməklə, yan paneldə həmin maddənin tam rəsmi mətnini dərhal oxuyun."
    }
  ];

  const steps = [
    {
      num: "01",
      title: "Sual ver",
      description: "Mərkəzi Bankın normativ aktları, nizamnamə kapitalı, risk limitləri və ya prudensial tələblər barədə sualınızı yazın."
    },
    {
      num: "02",
      title: "AI cavab hazırlayır",
      description: "ReguAZ RAG sistemi sualınıza ən uyğun normativ sənəd parçalarını tapıb, onları anlaşıqlı şəkildə izah edir."
    },
    {
      num: "03",
      title: "Normativ aktlara bax",
      description: "Təqdim olunan istinad düymələrinə klikləyərək interaktiv sənəd oxuyucuda rəsmi aktın həmin hissəsini araşdırın."
    }
  ];

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground transition-colors duration-300">
      {/* Header */}
      <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-2">
            <span className="text-2xl font-extrabold tracking-tight text-navy-900 dark:text-white">
              Regu<span className="text-gold-500">AZ</span>
            </span>
            <span className="hidden md:inline-block text-xs border border-gold-500/30 rounded px-1.5 py-0.5 text-gold-500 font-semibold uppercase tracking-wider">
              Mərkəzi Bank
            </span>
          </div>

          <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
            <a href="#features" className="transition-colors hover:text-gold-500">Özəlliklər</a>
            <a href="#how-it-works" className="transition-colors hover:text-gold-500">Necə işləyir?</a>
            <a href="https://github.com" target="_blank" rel="noreferrer" className="transition-colors hover:text-gold-500">GitHub</a>
          </nav>

          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              aria-label="Toggle Theme"
              className="text-foreground"
            >
              {theme === "light" ? "🌙" : "☀️"}
            </Button>
            <Link to="/login">
              <Button variant="outline" className="border-navy-900 text-navy-900 hover:bg-navy-50 dark:border-white dark:text-white dark:hover:bg-navy-900">
                Giriş
              </Button>
            </Link>
            <Link to="/dashboard">
              <Button className="bg-navy-900 text-white hover:bg-navy-800 dark:bg-gold-500 dark:text-navy-950 dark:hover:bg-gold-600 font-semibold">
                Axtarışa başla
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative py-20 md:py-28 overflow-hidden bg-gradient-to-br from-background via-[#faf8f5] to-gold-50/20 dark:from-background dark:via-navy-950 dark:to-navy-900">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            
            {/* Left Content */}
            <div className="lg:col-span-7 space-y-6 text-left">
              <div className="inline-flex items-center gap-1.5 rounded-full bg-gold-100 dark:bg-gold-950/30 px-3 py-1 text-xs font-semibold text-gold-700 dark:text-gold-400">
                <span>🏛️ Yalnız Azərbaycan Respublikası Mərkəzi Bankının Normativ Aktları</span>
              </div>
              <h1 className="text-4xl sm:text-5xl md:text-6xl font-black tracking-tight leading-tight">
                Mərkəzi Bank <br/>
                <span className="text-navy-900 dark:text-white">normativ aktlarını</span> <br/>
                saniyələr içində araşdırın.
              </h1>
              <p className="text-lg text-muted-foreground max-w-xl font-light">
                ReguAZ süni intellekt vasitəsilə Azərbaycan Respublikası Mərkəzi Bankının normativ sənədlərini axtarmağa, müqayisə etməyə və izah etməyə kömək edir.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 pt-4">
                <Button 
                  size="lg" 
                  onClick={() => navigate("/dashboard")} 
                  className="bg-navy-900 text-white hover:bg-navy-800 dark:bg-gold-500 dark:text-navy-950 dark:hover:bg-gold-600 text-base font-semibold group shadow-md"
                >
                  Axtarışa başla
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Button>
                <a href="#how-it-works">
                  <Button size="lg" variant="outline" className="text-base w-full">
                    Necə işləyir?
                  </Button>
                </a>
              </div>
            </div>

            {/* Right Dashboard Mock */}
            <div className="lg:col-span-5 relative">
              <div className="relative mx-auto w-full max-w-[480px] rounded-2xl border bg-card p-4 shadow-xl dark:bg-navy-900/60 dark:backdrop-blur">
                <div className="flex items-center justify-between pb-3 border-b mb-4">
                  <div className="flex space-x-1.5">
                    <span className="h-3 w-3 rounded-full bg-red-400 block" />
                    <span className="h-3 w-3 rounded-full bg-yellow-400 block" />
                    <span className="h-3 w-3 rounded-full bg-green-400 block" />
                  </div>
                  <span className="text-[10px] text-muted-foreground font-mono">reguaz.cbar.az</span>
                </div>
                
                {/* Simulated Interface Preview */}
                <div className="space-y-4 text-xs">
                  <div className="rounded-lg bg-secondary p-3 text-left">
                    <span className="font-bold text-navy-800 dark:text-gold-400">Sual: </span>
                    <span>Bankların minimum nizamnamə kapitalı tələbi nə qədərdir?</span>
                  </div>
                  <div className="space-y-2 border-l-2 border-gold-500 pl-3">
                    <div className="font-bold text-navy-900 dark:text-white flex items-center gap-1.5">
                      <span>💡 ReguAZ AI Köməkçisi</span>
                      <span className="text-[9px] bg-gold-100 text-gold-800 dark:bg-gold-950/20 dark:text-gold-400 px-1 rounded">RAG</span>
                    </div>
                    <p className="leading-relaxed">
                      Mərkəzi Bankın normativ aktlarına əsasən, bankların minimum nizamnamə kapitalının məbləği <strong>50,000,000 AZN</strong> (əlli milyon manat) müəyyən edilmişdir <span className="bg-gold-200 dark:bg-gold-950/50 dark:text-gold-300 font-bold px-1 rounded cursor-pointer text-gold-900">[1]</span>...
                    </p>
                  </div>
                  <div className="rounded border bg-background p-2.5 flex items-center justify-between text-[10px]">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-gold-500" />
                      <div>
                        <p className="font-bold">Mərkəzi Bank haqqında Qanun</p>
                        <p className="text-muted-foreground text-[9px]">Maddə 4 • Səhifə 12</p>
                      </div>
                    </div>
                    <span className="text-[9px] bg-green-100 text-green-800 dark:bg-green-950/40 dark:text-green-400 font-semibold px-1 rounded">Qüvvədədir</span>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-background border-t">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto space-y-4 mb-16">
            <h2 className="text-3xl font-bold tracking-tight">Əsas Üstünlüklər</h2>
            <p className="text-muted-foreground text-base">
              Bank sahəsində qərar qəbuletmə və tənzimləmə auditini sürətləndirmək üçün hazırlanmış ixtisaslaşmış funksionallıq.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, idx) => (
              <Card key={idx} className="hover:-translate-y-1 hover:shadow-md transition-all duration-300">
                <CardContent className="p-6 space-y-4 text-left">
                  <div className="p-3 bg-gold-100/50 dark:bg-navy-800 rounded-lg w-fit">
                    {feature.icon}
                  </div>
                  <h3 className="font-bold text-lg">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground font-light leading-relaxed">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section id="how-it-works" className="py-20 bg-secondary/30 dark:bg-navy-950/10 border-t">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto space-y-4 mb-16">
            <h2 className="text-3xl font-bold tracking-tight">ReguAZ Necə İşləyir?</h2>
            <p className="text-muted-foreground text-base">
              Mərkəzi Bankın qəliz hüquqi mətni saniyələr içində anlaşıqlı və tam istinadlı cavaba çevrilir.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {steps.map((step, idx) => (
              <div key={idx} className="relative p-6 bg-card border rounded-xl shadow-sm text-left">
                <span className="absolute top-4 right-4 text-4xl font-black text-gold-200 dark:text-navy-800">{step.num}</span>
                <h3 className="font-bold text-xl mb-3 pr-8">{step.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed font-light">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t bg-navy-950 text-white py-12">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="space-y-2 text-center md:text-left">
            <p className="text-xl font-bold">Regu<span className="text-gold-500">AZ</span></p>
            <p className="text-xs text-navy-200 font-light max-w-sm">
              Azərbaycan Respublikası Mərkəzi Bankının (CBAR) normativ aktları üzrə ixtisaslaşmış süni intellekt platforması.
            </p>
          </div>
          
          <div className="flex flex-wrap justify-center gap-6 text-sm text-navy-200">
            <a href="#features" className="hover:text-gold-500 transition-colors">Haqqında</a>
            <a href="mailto:info@reguaz.az" className="hover:text-gold-500 transition-colors">Əlaqə</a>
            <a href="https://github.com" target="_blank" rel="noreferrer" className="hover:text-gold-500 transition-colors">GitHub</a>
          </div>
        </div>
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 mt-8 pt-8 border-t border-navy-900 text-center text-xs text-navy-300">
          © {new Date().getFullYear()} ReguAZ. Bütün hüquqlar qorunur. Bu bir layihə nümayişidir və rəsmi Mərkəzi Bank xidməti deyil.
        </div>
      </footer>
    </div>
  );
};
