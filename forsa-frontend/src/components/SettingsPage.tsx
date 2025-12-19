import { useState } from "react";
import { Settings, Code, User, Bell, Shield, Palette } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

interface SettingsPageProps {
  debugMode: boolean;
  onDebugModeChange: (value: boolean) => void;
}

export function SettingsPage({ debugMode, onDebugModeChange }: SettingsPageProps) {
  const [notifications, setNotifications] = useState(true);

  return (
    <div className="p-6 max-w-2xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-foreground mb-2">Paramètres</h1>
        <p className="text-muted-foreground">Gérez vos préférences et configurations</p>
      </div>

      <div className="space-y-8">
        {/* Profile Section */}
        <section className="animate-fade-in">
          <div className="flex items-center gap-2 mb-4">
            <User className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Profil</h2>
          </div>
          <div className="bg-card border border-border rounded-lg p-4 space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-primary flex items-center justify-center">
                <User className="h-8 w-8 text-primary-foreground" />
              </div>
              <div>
                <p className="font-semibold text-foreground">Commercial B2B</p>
                <p className="text-sm text-muted-foreground">utilisateur@algerietelecom.dz</p>
              </div>
            </div>
            <Button variant="outline" className="w-full sm:w-auto">
              Modifier le profil
            </Button>
          </div>
        </section>

        <Separator />

        {/* Notifications */}
        <section className="animate-fade-in" style={{ animationDelay: "50ms" }}>
          <div className="flex items-center gap-2 mb-4">
            <Bell className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Notifications</h2>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="notifications" className="text-foreground font-medium">
                  Notifications par email
                </Label>
                <p className="text-sm text-muted-foreground">
                  Recevez des alertes pour les nouvelles offres et mises à jour
                </p>
              </div>
              <Switch
                id="notifications"
                checked={notifications}
                onCheckedChange={setNotifications}
              />
            </div>
          </div>
        </section>

        <Separator />

        {/* Developer Mode */}
        <section className="animate-fade-in" style={{ animationDelay: "100ms" }}>
          <div className="flex items-center gap-2 mb-4">
            <Code className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Mode Développeur</h2>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="debug-mode" className="text-foreground font-medium">
                  Afficher le JSON Debug
                </Label>
                <p className="text-sm text-muted-foreground">
                  Affiche les entrées/sorties JSON brutes dans les réponses de l'assistant
                </p>
              </div>
              <Switch
                id="debug-mode"
                checked={debugMode}
                onCheckedChange={onDebugModeChange}
              />
            </div>
          </div>
          {debugMode && (
            <div className="mt-4 p-4 bg-secondary rounded-lg border border-border animate-fade-in">
              <p className="text-sm font-mono text-muted-foreground">
                🔧 Mode développeur activé - Les réponses JSON seront visibles dans le chat
              </p>
            </div>
          )}
        </section>

        <Separator />

        {/* Security */}
        <section className="animate-fade-in" style={{ animationDelay: "150ms" }}>
          <div className="flex items-center gap-2 mb-4">
            <Shield className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Sécurité</h2>
          </div>
          <div className="bg-card border border-border rounded-lg p-4 space-y-3">
            <Button variant="outline" className="w-full justify-start">
              Changer le mot de passe
            </Button>
            <Button variant="outline" className="w-full justify-start">
              Activer l'authentification à deux facteurs
            </Button>
          </div>
        </section>

        <Separator />

        {/* Appearance */}
        <section className="animate-fade-in" style={{ animationDelay: "200ms" }}>
          <div className="flex items-center gap-2 mb-4">
            <Palette className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Apparence</h2>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <p className="text-sm text-muted-foreground">
              Thème clair activé par défaut. Les options de personnalisation seront disponibles prochainement.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
