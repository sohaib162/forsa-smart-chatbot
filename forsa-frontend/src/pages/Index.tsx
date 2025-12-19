import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import { LandingPage } from "@/components/LandingPage";
import { AppSidebar } from "@/components/AppSidebar";
import { ChatInterface } from "@/components/ChatInterface";
import { HistoryPage } from "@/components/HistoryPage";
import { DocumentsLibrary } from "@/components/DocumentsLibrary";
import { SettingsPage } from "@/components/SettingsPage";
import { cn } from "@/lib/utils";

const AppLayout = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [debugMode, setDebugMode] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      <AppSidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <main
        className={cn(
          "min-h-screen transition-all duration-300",
          sidebarCollapsed ? "ml-16" : "ml-64"
        )}
      >
        <Routes>
           <Route path="/" element={<ChatInterface debugMode={debugMode} sidebarCollapsed={sidebarCollapsed} />} />
           <Route path="/history" element={<HistoryPage />} />
           <Route path="/documents" element={<DocumentsLibrary />} />
           <Route
             path="/settings"
             element={
               <SettingsPage
                 debugMode={debugMode}
                 onDebugModeChange={setDebugMode}
               />
             }
           />
         </Routes>
      </main>
    </div>
  );
};

const Index = () => {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/app/*" element={<AppLayout />} />
    </Routes>
  );
};

export default Index;
