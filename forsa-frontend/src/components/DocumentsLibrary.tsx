import { useState, useEffect } from "react";
import { FileText, Search, Download, Eye, Filter, Globe, ExternalLink, BookOpen, Package, Tag } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { fetchDocuments, getDocumentUrl, openDocument, downloadDocument, Document } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";

const categories = [
  { id: "all", label: "Toutes catégories", icon: Filter },
  { id: "Guides", label: "Guides", icon: BookOpen },
  { id: "Offres", label: "Offres", icon: Tag },
  { id: "Conventions", label: "Conventions", icon: FileText },
  { id: "Produits", label: "Produits", icon: Package },
];

const languages = [
  { id: "all", label: "Toutes langues", icon: Globe },
  { id: "FR", label: "Français", icon: Globe },
  { id: "AR", label: "Arabe", icon: Globe },
];

export function DocumentsLibrary() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedLang, setSelectedLang] = useState<string>("all");
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 7;

  // Fetch all documents from API
  const { data, isLoading, error } = useQuery({
    queryKey: ['documents'],
    queryFn: () => fetchDocuments({}),
  });

  const allDocuments = data?.documents || [];

  // Filter documents on client side
  const filteredDocuments = allDocuments.filter((doc) => {
    const matchesCategory = selectedCategory === 'all' || doc.category === selectedCategory;
    const matchesLang = selectedLang === 'all' || doc.lang === selectedLang;
    const matchesSearch = !searchQuery || doc.filename.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesLang && matchesSearch;
  });

  const totalDocuments = filteredDocuments.length;
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = startIndex + pageSize;
  const documents = filteredDocuments.slice(startIndex, endIndex);
  const totalPages = Math.ceil(totalDocuments / pageSize);

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [selectedCategory, selectedLang, searchQuery]);

  // Handle document preview
  const handlePreview = (doc: Document) => {
    setSelectedDocument(doc);
    setPreviewOpen(true);
  };

  // Handle document download
  const handleDownload = async (doc: Document) => {
    try {
      await downloadDocument(doc.s3_key, doc.filename);
      toast.success(`Téléchargement de ${doc.filename} démarré`);
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Échec du téléchargement');
    }
  };

  // Handle document open in new tab
  const handleOpen = (doc: Document) => {
    openDocument(doc.s3_key);
  };

  // Get icon color based on extension
  const getIconColor = (ext: string) => {
    if (ext === '.pdf') return 'text-red-500';
    if (ext === '.docx' || ext === '.doc') return 'text-blue-500';
    if (ext === '.odt') return 'text-orange-500';
    return 'text-gray-500';
  };

  // Get category badge color
  const getCategoryColor = (category: string) => {
    switch (category.toLowerCase()) {
      case 'guides':
        return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'offres':
        return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'conventions':
        return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400';
      case 'produits':
        return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400';
      default:
        return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
    }
  };

  // Document Card Component
  function DocumentCard({ document, index, onPreview, onDownload, onOpen }: { document: Document; index: number; onPreview: (doc: Document) => void; onDownload: (doc: Document) => void; onOpen: (doc: Document) => void }) {
    const categoryInfo = categories.find((c) => c.id === document.category);
    const CategoryIcon = categoryInfo?.icon || FileText;

    return (
      <div
        className="group bg-card border border-border rounded-lg p-4 shadow-sm hover:shadow-md transition-all duration-200 hover:-translate-y-0.5 animate-fade-in flex items-center gap-4"
        style={{ animationDelay: `${index * 50}ms` }}
      >
        {/* File Icon */}
        <div className="w-12 h-12 rounded bg-white flex items-center justify-center flex-shrink-0">
          <FileText className={cn("h-6 w-6", getIconColor(document.ext))} />
        </div>

        {/* Title and Category */}
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-foreground line-clamp-1">{document.filename}</h3>
          <div className="flex items-center gap-2 mt-1">
            <span className="px-2 py-0.5 text-xs rounded-full font-medium bg-white text-primary border border-primary/20 flex items-center gap-1">
              <CategoryIcon className="h-3 w-3" />
              {categoryInfo?.label}
            </span>
          </div>
        </div>

        {/* Language */}
        <div className="flex items-center gap-1 text-sm text-primary font-medium">
          <Globe className="h-4 w-4" />
          {document.lang}
        </div>

        {/* Type */}
        <span className="px-2 py-1 text-xs font-medium bg-muted text-muted-foreground rounded uppercase">
          {document.ext.replace('.', '')}
        </span>

        {/* Action Buttons */}
        <div className="flex items-center gap-1">
          {document.ext === '.pdf' && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onPreview(document)}
              className="text-primary hover:text-primary hover:bg-primary/10 transition-colors"
              title="Aperçu"
            >
              <Eye className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onOpen(document)}
            className="text-primary hover:text-primary hover:bg-primary/10 transition-colors"
            title="Ouvrir"
          >
            <ExternalLink className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onDownload(document)}
            title="Télécharger"
          >
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground mb-2">Bibliothèque de Documents</h1>
        <p className="text-muted-foreground">
          {isLoading ? 'Chargement...' : `${totalDocuments} document(s) disponible(s)`}
        </p>
      </div>

      {/* Search and Filters */}
      <div className="relative mb-6">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
        <Input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Rechercher un document..."
          className="pl-12 h-12 bg-secondary border-border"
        />
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-6">
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
          <div className="w-6"></div>
          {languages.map((language) => {
            const Icon = language.icon;
            const isSelected = selectedLang === language.id;
            return (
              <button
                key={language.id}
                onClick={() => setSelectedLang(language.id)}
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
                <span>{language.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 mb-6">
          <p className="text-destructive font-medium">Erreur de chargement</p>
          <p className="text-sm text-muted-foreground mt-1">
            Impossible de charger les documents. Vérifiez que l'API est démarrée.
          </p>
        </div>
      )}

      {/* Documents Grid */}
      {isLoading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Chargement des documents...</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="text-center py-12">
          <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium text-foreground mb-2">Aucun document trouvé</h3>
          <p className="text-muted-foreground">Essayez de modifier vos critères de recherche</p>
        </div>
      ) : (
        <>
          <div className="flex flex-col gap-4">
            {documents.map((doc, index) => (
              <DocumentCard key={doc.s3_key} document={doc} index={index} onPreview={handlePreview} onDownload={handleDownload} onOpen={handleOpen} />
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 mt-8">
              <Button
                variant="outline"
                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                disabled={currentPage === 1}
              >
                Précédent
              </Button>
              <span className="text-sm text-muted-foreground">
                Page {currentPage} sur {totalPages}
              </span>
              <Button
                variant="outline"
                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                disabled={currentPage === totalPages}
              >
                Suivant
              </Button>
            </div>
          )}
        </>
      )}

      {/* PDF Preview Dialog */}
      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="max-w-4xl h-[90vh] bg-background">
          <DialogHeader>
            <DialogTitle className="text-foreground">{selectedDocument?.filename}</DialogTitle>
          </DialogHeader>
          <div className="flex-1 overflow-hidden">
            {selectedDocument && (
              <iframe
                src={getDocumentUrl(selectedDocument.s3_key)}
                className="w-full h-full rounded-lg border border-border"
                title={selectedDocument.filename}
              />
            )}
          </div>
          <div className="flex gap-2 justify-end">
            <Button
              variant="outline"
              onClick={() => selectedDocument && handleOpen(selectedDocument)}
            >
              <ExternalLink className="h-4 w-4 mr-2" />
              Ouvrir dans un nouvel onglet
            </Button>
            <Button
              onClick={() => selectedDocument && handleDownload(selectedDocument)}
            >
              <Download className="h-4 w-4 mr-2" />
              Télécharger
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function DocumentCard({ document, index, onPreview, onDownload, onOpen }: { document: Document; index: number; onPreview: (doc: Document) => void; onDownload: (doc: Document) => void; onOpen: (doc: Document) => void }) {
  const categoryInfo = categories.find((c) => c.id === document.category);
  const CategoryIcon = categoryInfo?.icon || FileText;

  return (
    <div
      className="bg-card border border-border rounded-lg p-5 shadow-card hover:shadow-card-hover transition-all duration-200 hover:-translate-y-0.5 animate-fade-in"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* File Icon */}
      <div className="w-12 h-12 rounded-lg bg-secondary flex items-center justify-center mb-4">
        <FileText className={cn("h-6 w-6", getIconColor(document.ext))} />
      </div>

      {/* Category Badge */}
      <div className="flex items-center gap-2 mb-3">
        <span className="px-2.5 py-1 text-xs rounded-full font-medium bg-primary/10 text-primary flex items-center gap-1.5">
          <CategoryIcon className="h-3 w-3" />
          {categoryInfo?.label}
        </span>
      </div>

      {/* Title */}
      <h3 className="font-semibold text-foreground mb-2 line-clamp-2">{document.filename}</h3>

      {/* Language */}
      <p className="text-sm text-primary font-medium mb-2 flex items-center gap-1">
        <Globe className="h-3 w-3" />
        {document.lang}
      </p>

      {/* Type */}
      <p className="text-sm text-muted-foreground mb-4 uppercase">{document.ext.replace('.', '')}</p>

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-border">
        <div className="flex gap-1">
          {document.ext === '.pdf' && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onPreview(document)}
              className="text-primary hover:text-primary/80 hover:bg-primary/5"
              title="Aperçu"
            >
              <Eye className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onOpen(document)}
            className="text-primary hover:text-primary/80 hover:bg-primary/5"
            title="Ouvrir"
          >
            <ExternalLink className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onDownload(document)}
            className="text-primary hover:text-primary/80 hover:bg-primary/5"
            title="Télécharger"
          >
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
