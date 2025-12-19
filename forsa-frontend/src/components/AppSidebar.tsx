import { useState } from "react";
import { cn } from "@/lib/utils";
import { NavLink } from "@/components/NavLink";
import { 
  MessageSquare, 
  Search, 
  History, 
  FileText, 
  Settings,
  ChevronLeft,
  ChevronRight,
  User
} from "lucide-react";
import algerieTelecomLogo from "@/assets/algerie-telecom-logo.png";
import forsaTicLogo from "@/assets/forsa-tic-logo.png";

interface AppSidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

const navItems = [
  { title: "Assistant IA", url: "/app", icon: MessageSquare },
  { title: "Historique", url: "/app/history", icon: History },
  { title: "Bibliothèque de Documents", url: "/app/documents", icon: FileText },
];

export function AppSidebar({ collapsed, onToggle }: AppSidebarProps) {
  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen bg-sidebar border-r border-sidebar-border transition-all duration-300 flex flex-col",
        collapsed ? "w-16" : "w-64"
      )}
    >
      {/* Logo Section */}
      <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
        {!collapsed && (
          <img 
            src={algerieTelecomLogo} 
            alt="Algérie Télécom" 
            className="h-10 object-contain animate-fade-in"
          />
        )}
        <button
          onClick={onToggle}
          className="p-2 rounded-md hover:bg-sidebar-accent transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4 text-sidebar-foreground" />
          ) : (
            <ChevronLeft className="h-4 w-4 text-sidebar-foreground" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.url}
            to={item.url}
            end={item.url === "/app"}
            className={cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-md text-sidebar-foreground hover:bg-sidebar-accent transition-all duration-200",
              collapsed && "justify-center px-2"
            )}
            activeClassName="bg-primary text-primary-foreground hover:bg-primary/90"
          >
            <item.icon className="h-5 w-5 flex-shrink-0" />
            {!collapsed && (
              <span className="text-sm font-medium animate-fade-in">{item.title}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Settings */}
      <div className="p-3 border-t border-sidebar-border">
        <NavLink
          to="/app/settings"
          className={cn(
            "flex items-center gap-3 px-3 py-2.5 rounded-md text-sidebar-foreground hover:bg-sidebar-accent transition-all duration-200",
            collapsed && "justify-center px-2"
          )}
          activeClassName="bg-primary text-primary-foreground hover:bg-primary/90"
        >
          <Settings className="h-5 w-5 flex-shrink-0" />
          {!collapsed && <span className="text-sm font-medium">Paramètres</span>}
        </NavLink>
      </div>

      {/* User Profile */}
      <div className="p-3 border-t border-sidebar-border">
        <div className={cn(
          "flex items-center gap-3 px-3 py-2.5",
          collapsed && "justify-center px-2"
        )}>
          <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
            <User className="h-4 w-4 text-primary-foreground" />
          </div>
          {!collapsed && (
            <div className="animate-fade-in">
              <p className="text-sm font-medium text-sidebar-foreground">Utilisateur</p>
              <p className="text-xs text-muted-foreground">Commercial B2B</p>
            </div>
          )}
        </div>
      </div>

      {/* Forsa TIC Logo */}
      {!collapsed && (
        <div className="p-4 border-t border-sidebar-border">
          <img 
            src={forsaTicLogo} 
            alt="Forsa TIC" 
            className="h-8 object-contain opacity-70 mx-auto animate-fade-in"
          />
        </div>
      )}
    </aside>
  );
}
