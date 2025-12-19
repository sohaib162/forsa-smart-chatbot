import { useState } from "react";
import { Search, Filter, Building2, FileText, Download, BookOpen, Package, Tag } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type CategoryType = "guides" | "conventions" | "produits" | "offres";

interface SearchResult {
  id: string;
  title: string;
  partner: string;
  category: CategoryType;
  date: string;
  snippet: string;
}

const mockResults: SearchResult[] = [
  {
    id: "1",
    title: "Convention Cadre - Sonatrach",
    partner: "Sonatrach",
    category: "conventions",
    date: "15/03/2024",
    snippet: "Accord de partenariat pour la fourniture de services de télécommunications et solutions IoT pour les sites de production..."
  },
  {
    id: "2",
    title: "Offre Spéciale Banques",
    partner: "BNA",
    category: "offres",
    date: "01/02/2024",
    snippet: "Package complet incluant connectivité haut débit, services cloud sécurisés et support dédié 24/7..."
  },
  {
    id: "3",
    title: "Partenariat Éducation Nationale",
    partner: "Ministère de l'Éducation",
    category: "conventions",
    date: "20/01/2024",
    snippet: "Mise en place d'une infrastructure réseau pour 500 établissements scolaires dans le cadre de la digitalisation..."
  },
  {
    id: "4",
    title: "Guide NGBSS - Procédures Techniques",
    partner: "Algérie Télécom",
    category: "guides",
    date: "10/12/2023",
    snippet: "Documentation complète des procédures NGBSS pour l'interconnexion des systèmes de facturation..."
  },
  {
    id: "5",
    title: "Offre PME Connect+",
    partner: "CACI",
    category: "offres",
    date: "05/11/2023",
    snippet: "Solution intégrée pour les PME membres incluant internet fibre, téléphonie IP et services de visioconférence..."
  },
  {
    id: "6",
    title: "Produit Fibre Optique 10Gbps",
    partner: "Algérie Télécom",
    category: "produits",
    date: "28/10/2023",
    snippet: "Solution de connectivité ultra-rapide pour les grandes entreprises avec SLA garanti..."
  },
  {
    id: "7",
    title: "Guide d'Installation Réseau",
    partner: "Algérie Télécom",
    category: "guides",
    date: "15/09/2023",
    snippet: "Documentation technique pour l'installation et la configuration des équipements réseau..."
  },
  {
    id: "8",
    title: "Produit Cloud Computing AT",
    partner: "Algérie Télécom",
    category: "produits",
    date: "01/09/2023",
    snippet: "Infrastructure cloud sécurisée avec stockage et calcul distribué pour les entreprises..."
  },
];

const categories: { id: CategoryType | "tous"; label: string; icon: any }[] = [
  { id: "tous", label: "Tous", icon: Filter },
  { id: "guides", label: "Guides", icon: BookOpen },
  { id: "conventions", label: "Conventions", icon: FileText },
  { id: "produits", label: "Produits", icon: Package },
  { id: "offres", label: "Offres", icon: Tag },
];

export function SearchDashboard() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<CategoryType | "tous">("tous");
  const [results] = useState<SearchResult[]>(mockResults);

  const filteredResults = results.filter((result) => {
    const matchesSearch = result.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      result.snippet.toLowerCase().includes(searchQuery.toLowerCase()) ||
      result.partner.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === "tous" || result.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground mb-2">Recherche Avancée</h1>
        <p className="text-muted-foreground">Explorez les conventions et offres partenaires</p>
      </div>

      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
        <Input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Rechercher par titre, partenaire ou contenu..."
          className="pl-12 h-12 text-base bg-secondary border-border"
        />
      </div>

      {/* Category Filter Pills */}
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
                onClick={() => setSelectedCategory(category.id)}
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

      {/* Results Count */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {filteredResults.length} résultat{filteredResults.length !== 1 ? "s" : ""} trouvé{filteredResults.length !== 1 ? "s" : ""}
        </p>
      </div>

      {/* Results Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredResults.map((result, index) => (
          <ResultCard key={result.id} result={result} index={index} />
        ))}
      </div>

      {filteredResults.length === 0 && (
        <div className="text-center py-12">
          <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium text-foreground mb-2">Aucun résultat trouvé</h3>
          <p className="text-muted-foreground">Essayez de modifier vos critères de recherche</p>
        </div>
      )}
    </div>
  );
}

function ResultCard({ result, index }: { result: SearchResult; index: number }) {
  const categoryInfo = categories.find((c) => c.id === result.category);
  const CategoryIcon = categoryInfo?.icon || FileText;

  return (
    <div
      className="bg-card border border-border rounded-lg p-5 shadow-card hover:shadow-card-hover transition-all duration-200 hover:-translate-y-0.5 animate-fade-in"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* Partner Logo Placeholder */}
      <div className="w-12 h-12 rounded-lg bg-secondary flex items-center justify-center mb-4">
        <Building2 className="h-6 w-6 text-primary" />
      </div>

      {/* Category Badge */}
      <div className="flex items-center gap-2 mb-3">
        <span className="px-2.5 py-1 text-xs rounded-full font-medium bg-primary/10 text-primary flex items-center gap-1.5">
          <CategoryIcon className="h-3 w-3" />
          {categoryInfo?.label}
        </span>
      </div>

      {/* Title */}
      <h3 className="font-semibold text-foreground mb-2 line-clamp-2">{result.title}</h3>

      {/* Partner */}
      <p className="text-sm text-primary font-medium mb-2">{result.partner}</p>

      {/* Snippet */}
      <p className="text-sm text-muted-foreground mb-4 line-clamp-3">{result.snippet}</p>

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-border">
        <span className="text-xs text-muted-foreground">{result.date}</span>
        <Button variant="ghost" size="sm" className="text-primary hover:text-primary/80 hover:bg-primary/5">
          <Download className="h-4 w-4 mr-2" />
          PDF
        </Button>
      </div>
    </div>
  );
}