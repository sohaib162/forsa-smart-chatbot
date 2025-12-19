import { useState } from "react";
import { History, Search, Trash2, MessageSquare, Calendar } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface ChatHistory {
  id: string;
  title: string;
  preview: string;
  date: string;
  messageCount: number;
}

const mockHistory: ChatHistory[] = [
  {
    id: "1",
    title: "Offres B2B pour entreprises",
    preview: "Quelles sont les offres B2B disponibles pour les grandes entreprises?",
    date: "Aujourd'hui",
    messageCount: 8
  },
  {
    id: "2",
    title: "Convention Sonatrach",
    preview: "Détails de la convention avec Sonatrach pour 2024",
    date: "Hier",
    messageCount: 12
  },
  {
    id: "3",
    title: "Procédure NGBSS",
    preview: "Comment fonctionne la procédure NGBSS pour les nouveaux clients?",
    date: "Il y a 3 jours",
    messageCount: 5
  },
  {
    id: "4",
    title: "Partenariat BNA",
    preview: "Informations sur le partenariat bancaire avec BNA",
    date: "Il y a 1 semaine",
    messageCount: 15
  },
  {
    id: "5",
    title: "Offres PME",
    preview: "Solutions internet pour les petites et moyennes entreprises",
    date: "Il y a 2 semaines",
    messageCount: 6
  },
];

export function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [history, setHistory] = useState(mockHistory);

  const filteredHistory = history.filter((item) =>
    item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.preview.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleDelete = (id: string) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground mb-2">Historique des conversations</h1>
        <p className="text-muted-foreground">Retrouvez vos échanges précédents avec l'assistant</p>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
        <Input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Rechercher dans l'historique..."
          className="pl-12 h-12 bg-secondary border-border"
        />
      </div>

      {/* History List */}
      <div className="space-y-3">
        {filteredHistory.map((item, index) => (
          <div
            key={item.id}
            className="bg-card border border-border rounded-lg p-4 hover:shadow-card-hover transition-all duration-200 cursor-pointer group animate-fade-in"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <MessageSquare className="h-4 w-4 text-primary flex-shrink-0" />
                  <h3 className="font-semibold text-foreground truncate">{item.title}</h3>
                </div>
                <p className="text-sm text-muted-foreground truncate mb-2">{item.preview}</p>
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Calendar className="h-3 w-3" />
                    {item.date}
                  </span>
                  <span>{item.messageCount} messages</span>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(item.id);
                }}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        ))}
      </div>

      {filteredHistory.length === 0 && (
        <div className="text-center py-12">
          <History className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium text-foreground mb-2">Aucun historique trouvé</h3>
          <p className="text-muted-foreground">
            {searchQuery ? "Essayez une autre recherche" : "Commencez une conversation avec l'assistant"}
          </p>
        </div>
      )}
    </div>
  );
}
