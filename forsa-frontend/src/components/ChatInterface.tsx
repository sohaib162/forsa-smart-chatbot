import { useState, useRef, useEffect } from "react";
import { Send, Paperclip, Bot, User, Download, Calendar, Loader2, FileText, BookOpen, Package, Tag, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { TypingMarkdownMessage } from "@/components/TypingMarkdownMessage";
import { processQuestion, type ChatResponse } from "@/lib/chatApi";
import { openDocument, downloadDocument } from "@/lib/api";
import { toast } from "sonner";

type CategoryType = "guides" | "conventions" | "produits" | "offres";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  offers?: Offer[];
  sources?: Array<{
    s3_key: string;
    filename: string;
    category: string;
    ext: string;
    lang: 'AR' | 'FR';
  }>;
  thinking?: boolean;
  isTyping?: boolean;
  category?: CategoryType;
  inputJson?: any;
}

interface Offer {
  id: string;
  title: string;
  validityDate: string;
  description: string;
  type: "B2B" | "B2C" | "Partenaire";
}

const suggestionChips = [
  { label: "Offres B2B", query: "Quelles sont les offres B2B disponibles pour les entreprises?" },
  { label: "Conventions Partenaires", query: "Montrez-moi les conventions partenaires actives" },
  { label: "Procédure NGBSS", query: "Quelle est la procédure NGBSS pour les nouveaux clients?" },
];

// Category mapping: frontend -> backend ID
const CATEGORY_MAPPING: Record<CategoryType, string> = {
  guides: "3",      // GUIDE
  conventions: "2", // CONVENTIONS
  produits: "4",    // DEPOT
  offres: "1",      // OFFERS
};

const categories: { id: CategoryType; label: string; icon: any }[] = [
  { id: "guides", label: "Guides", icon: BookOpen },
  { id: "conventions", label: "Conventions", icon: FileText },
  { id: "produits", label: "Produits", icon: Package },
  { id: "offres", label: "Offres", icon: Tag },
];

const mockOffers: Offer[] = [
  {
    id: "1",
    title: "Pack Entreprise Premium",
    validityDate: "31/12/2024",
    description: "Solution complète pour grandes entreprises avec fibre optique dédiée",
    type: "B2B"
  },
  {
    id: "2",
    title: "Offre PME Connect",
    validityDate: "30/06/2025",
    description: "Internet haut débit pour PME avec support prioritaire 24/7",
    type: "B2B"
  }
];

export function ChatInterface({ debugMode, sidebarCollapsed }: { debugMode: boolean; sidebarCollapsed?: boolean }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<CategoryType | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const userScrolledRef = useRef(false);

  const isNearBottom = () => {
    const container = messagesContainerRef.current;
    if (!container) return true;

    const threshold = 100; // pixels from bottom
    const position = container.scrollTop + container.clientHeight;
    const height = container.scrollHeight;

    return height - position < threshold;
  };

  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    if (shouldAutoScrollRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior });
    }
  };

  const handleScroll = () => {
    const nearBottom = isNearBottom();
    shouldAutoScrollRef.current = nearBottom;

    // If user manually scrolled away from bottom, mark it
    if (!nearBottom) {
      userScrolledRef.current = true;
    } else {
      userScrolledRef.current = false;
    }
  };

  useEffect(() => {
    // Auto-scroll on new messages
    scrollToBottom();
  }, [messages]);

  // Smooth scroll during typing animation
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage?.role === "assistant" && !lastMessage.thinking && lastMessage.isTyping) {
      const scrollInterval = setInterval(() => {
        if (shouldAutoScrollRef.current) {
          messagesEndRef.current?.scrollIntoView({ behavior: "auto" });
        }
      }, 100);
      return () => clearInterval(scrollInterval);
    }
  }, [messages]);

  const handleCategorySelect = (categoryId: CategoryType) => {
    setSelectedCategory(selectedCategory === categoryId ? null : categoryId);
  };

  const handleSend = async (query?: string) => {
    const messageContent = query || inputValue;
    if (!messageContent.trim()) return;

    // Build the backend payload
    const categorieId = selectedCategory ? CATEGORY_MAPPING[selectedCategory] : "1"; // Default to offers
    const payload = {
      equipe: "IA_Team",
      question: {
        categorie_id: {
          [categorieId]: messageContent,
        },
      },
    };

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: messageContent,
      category: selectedCategory || undefined,
      inputJson: payload,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setSelectedCategory(null); // Reset category after sending
    setIsLoading(true);
    shouldAutoScrollRef.current = true; // Always scroll when user sends a message

    // Add thinking message
    const thinkingMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "",
      thinking: true,
    };
    setMessages((prev) => [...prev, thinkingMessage]);

    try {
      // Call real API
      const response: ChatResponse = await processQuestion(payload);

      // Remove thinking and add response with typing animation
      setMessages((prev) => {
        const filtered = prev.filter((m) => !m.thinking);
        const assistantMessage: Message = {
          id: (Date.now() + 2).toString(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          isTyping: true,
        };
        return [...filtered, assistantMessage];
      });
    } catch (error) {
      console.error("Chat API error:", error);
      toast.error("Erreur de chargement de la réponse. Veuillez réessayer.");

      // Remove thinking and add error message
      setMessages((prev) => {
        const filtered = prev.filter((m) => !m.thinking);
        const errorMessage: Message = {
          id: (Date.now() + 2).toString(),
          role: "assistant",
          content: "Désolé, une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer.",
          isTyping: true,
        };
        return [...filtered, errorMessage];
      });
    }

    setIsLoading(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Messages Area */}
      <div
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-6 pb-52 scrollbar-thin"
      >
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto">
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-6">
              <Bot className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold text-foreground mb-3">
              Bonjour! Je suis l'assistant Algérie Télécom.
            </h2>
            <p className="text-muted-foreground mb-8">
              Que cherchez-vous aujourd'hui?
            </p>
            <div className="flex flex-wrap gap-3 justify-center">
              {suggestionChips.map((chip) => (
                <button
                  key={chip.label}
                  onClick={() => handleSend(chip.query)}
                  className="px-4 py-2 rounded-full border border-border bg-card hover:bg-secondary hover:border-primary/30 transition-all duration-200 text-sm font-medium text-foreground shadow-card hover:shadow-card-hover"
                >
                  {chip.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex gap-4 animate-fade-in",
                  message.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                {message.role === "assistant" && (
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                    <Bot className="h-4 w-4 text-primary-foreground" />
                  </div>
                )}
                <div
                  className={cn(
                    "max-w-[80%] rounded-lg p-4",
                    message.role === "user"
                      ? "bg-chat-user text-primary-foreground"
                      : "bg-chat-ai text-foreground"
                  )}
                >
                  {message.thinking ? (
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="animate-pulse-subtle">Consultation de la base de connaissances...</span>
                    </div>
                  ) : (
                    <>
                      {message.role === "assistant" ? (
                        <TypingMarkdownMessage content={message.content} isTyping={message.isTyping} />
                      ) : (
                        <>
                          <p className="whitespace-pre-wrap">{message.content}</p>
                          {message.category && (
                            <div className="mt-2 inline-flex">
                              <span className="px-2.5 py-1 text-xs rounded-full bg-primary-foreground/20 text-primary-foreground font-medium flex items-center gap-1.5">
                                {(() => {
                                  const Icon = categories.find((c) => c.id === message.category)?.icon;
                                  return Icon ? <Icon className="h-3 w-3" /> : null;
                                })()}
                                {categories.find((c) => c.id === message.category)?.label}
                              </span>
                            </div>
                          )}
                        </>
                      )}
                       {message.offers && (
                         <div className="mt-4 space-y-3">
                           {message.offers.map((offer) => (
                             <OfferCard key={offer.id} offer={offer} />
                           ))}
                         </div>
                       )}
                       {message.sources && message.sources.length > 0 && (
                         <div className="mt-4">
                           <p className="text-sm font-medium text-foreground mb-2">Documents référencés:</p>
                           <div className="space-y-2">
                             {message.sources.map((source, index) => (
                               <div key={index} className="flex items-center gap-3 p-2 bg-muted/50 rounded-md">
                                 <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                                 <div className="flex-1 min-w-0">
                                   <p className="text-sm font-medium text-foreground truncate">
                                     {source.filename}
                                   </p>
                                   <p className="text-xs text-muted-foreground">
                                     {source.category} • {source.lang}
                                   </p>
                                 </div>
                                 <div className="flex gap-1">
                                   <Button
                                     variant="ghost"
                                     size="sm"
                                     onClick={() => openDocument(source.s3_key)}
                                     className="h-8 px-2"
                                   >
                                     <ExternalLink className="h-3 w-3" />
                                   </Button>
                                   <Button
                                     variant="ghost"
                                     size="sm"
                                     onClick={() => downloadDocument(source.s3_key, source.filename)}
                                     className="h-8 px-2"
                                   >
                                     <Download className="h-3 w-3" />
                                   </Button>
                                 </div>
                               </div>
                             ))}
                           </div>
                         </div>
                       )}
                      {debugMode && message.inputJson && (
                        <div className="mt-4 p-3 bg-foreground/5 rounded-md">
                          <p className="text-xs font-mono text-muted-foreground mb-2">Input JSON:</p>
                          <pre className="text-xs font-mono text-muted-foreground overflow-x-auto">
                            {JSON.stringify(message.inputJson, null, 2)}
                          </pre>
                        </div>
                      )}
                       {debugMode && message.role === "assistant" && (
                         <div className="mt-4 p-3 bg-foreground/5 rounded-md">
                           <p className="text-xs font-mono text-muted-foreground mb-2">Debug JSON:</p>
                           <pre className="text-xs font-mono text-muted-foreground overflow-x-auto">
                             {JSON.stringify({ role: message.role, content: message.content, offers: message.offers, sources: message.sources }, null, 2)}
                           </pre>
                         </div>
                       )}
                    </>
                  )}
                </div>
                {message.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
                    <User className="h-4 w-4 text-secondary-foreground" />
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area*/}
      <div className={cn(
        "fixed bottom-0 right-0 border-t border-border bg-background/95 backdrop-blur-sm p-4 z-10 transition-all duration-300",
        sidebarCollapsed ? "left-16" : "left-64"
      )}>
        <div className="max-w-3xl mx-auto space-y-4">
          {/* Category Pills */}
          <div className="flex items-center gap-3 px-1">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Catégorie
            </span>
            <div className="flex flex-wrap gap-2">
              {categories.map((category) => {
                const Icon = category.icon;
                const isSelected = selectedCategory === category.id;
                return (
                  <button
                    key={category.id}
                    onClick={() => handleCategorySelect(category.id)}
                    className={cn(
                      "group relative px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 flex items-center gap-2",
                      isSelected
                        ? "bg-primary text-primary-foreground shadow-lg shadow-primary/25 scale-105"
                        : "bg-card border border-border text-foreground hover:border-primary/50 hover:shadow-md hover:scale-[1.02] shadow-sm"
                    )}
                  >
                    <Icon className={cn(
                      "h-4 w-4 transition-transform",
                      isSelected && "scale-110"
                    )} />
                    <span>{category.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Input Bar */}
          <div className="flex items-center gap-3 bg-card border border-border rounded-xl p-3 shadow-lg">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Posez votre question..."
              className="flex-1 bg-transparent border-0 resize-none focus:outline-none focus:ring-0 text-foreground placeholder:text-muted-foreground py-1 leading-6"
              rows={1}
              style={{ minHeight: '24px', maxHeight: '120px' }}
            />
            <Button
              onClick={() => handleSend()}
              disabled={!inputValue.trim() || isLoading}
              size="icon"
              className="bg-primary hover:bg-primary/90 text-primary-foreground flex-shrink-0 shadow-lg hover:shadow-xl transition-all hover:scale-105"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// TypingMessage has been replaced by TypingMarkdownMessage for proper markdown rendering

function OfferCard({ offer }: { offer: Offer }) {
  return (
    <div className="bg-card border border-border rounded-lg p-4 shadow-card hover:shadow-card-hover transition-shadow">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <h4 className="font-semibold text-foreground">{offer.title}</h4>
            <span className={cn(
              "px-2 py-0.5 text-xs rounded-full font-medium",
              offer.type === "B2B" ? "bg-primary/10 text-primary" :
              offer.type === "B2C" ? "bg-success/10 text-success" :
              "bg-accent/10 text-accent"
            )}>
              {offer.type}
            </span>
          </div>
          <p className="text-sm text-muted-foreground mb-3">{offer.description}</p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Calendar className="h-3 w-3" />
            <span>Valide jusqu'au {offer.validityDate}</span>
          </div>
        </div>
        <Button variant="outline" size="sm" className="flex-shrink-0">
          <Download className="h-4 w-4 mr-2" />
          PDF
        </Button>
      </div>
    </div>
  );
}
