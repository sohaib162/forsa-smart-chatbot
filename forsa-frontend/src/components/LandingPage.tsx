import { Button } from "@/components/ui/button";
import { ArrowRight, Bot, Search, FileText, MessageSquare, Sparkles } from "lucide-react";
import { useNavigate } from "react-router-dom";
import algerieTelecomLogo from "@/assets/algerie-telecom-logo.png";
import forsaTicLogo from "@/assets/forsa-tic-logo.png";

export function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5 overflow-hidden">
      {/* Decorative Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary/5 rounded-full blur-3xl animate-pulse-subtle" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-accent/5 rounded-full blur-3xl animate-pulse-subtle" style={{ animationDelay: "1s" }} />
      </div>

      {/* Header */}
      <header className="relative z-10 p-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-8">
            <img
              src={algerieTelecomLogo}
              alt="Algérie Télécom"
              className="h-12 object-contain animate-fade-in"
            />
          </div>
          <img
            src={forsaTicLogo}
            alt="Forsa TIC"
            className="h-10 object-contain opacity-80 animate-fade-in"
            style={{ animationDelay: "0.2s" }}
          />
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-6xl mx-auto px-6 pt-20 pb-32">
        <div className="text-center space-y-8">
          {/* Hero Title */}
          <div className="space-y-4 animate-fade-in" style={{ animationDelay: "0.1s" }}>
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-foreground leading-tight">
              Assistant IA pour
              <br />
              <span className="bg-gradient-to-r from-primary via-primary to-accent bg-clip-text text-transparent">
                Algérie Télécom
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto">
              Accédez instantanément aux informations sur les offres, conventions et produits B2B
            </p>
          </div>

          {/* CTA Button */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in" style={{ animationDelay: "0.2s" }}>
            <Button
              size="lg"
              onClick={() => navigate("/app")}
              className="group bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-6 text-lg font-semibold shadow-xl hover:shadow-2xl transition-all hover:scale-105"
            >
              Commencer
              <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Button>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-20 animate-fade-in" style={{ animationDelay: "0.3s" }}>
            <FeatureCard
              icon={<MessageSquare className="h-6 w-6" />}
              title="Assistant IA Intelligent"
              description="Posez vos questions en langage naturel et obtenez des réponses précises"
            />
            <FeatureCard
              icon={<FileText className="h-6 w-6" />}
              title="Documentation Complète"
              description="Accédez aux guides, conventions et fiches produits"
            />
            <FeatureCard
              icon={<Search className="h-6 w-6" />}
              title="Recherche Instantanée"
              description="Trouvez rapidement les informations dont vous avez besoin"
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-border bg-background/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-sm text-muted-foreground">
              © 2025 Algérie Télécom. Tous droits réservés.
            </p>
            <p className="text-sm text-muted-foreground">
              Développé pour Forsa TIC Hackathon 2025
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="group bg-card border border-border rounded-2xl p-6 shadow-card hover:shadow-card-hover transition-all duration-300 hover:-translate-y-1 flex flex-col items-center justify-center">
      <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform text-primary">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-foreground mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );
}

function StatCard({ value, label }: { value: string; label: string }) {
  return (
    <div className="text-center">
      <div className="text-3xl md:text-4xl font-bold text-primary mb-2">{value}</div>
      <div className="text-sm text-muted-foreground">{label}</div>
    </div>
  );
}
