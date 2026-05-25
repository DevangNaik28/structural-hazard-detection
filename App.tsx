import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import StructureMap from "@/pages/StructureMap";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppLayout } from "@/components/AppLayout";
import { GlobalDataProvider } from "@/api/GlobalDataContext";
import Landing from "@/pages/Landing";
import Dashboard from "@/pages/Dashboard";
import Structures from "@/pages/Structures";
import Anomalies from "@/pages/Anomalies";
import RiskAssessment from "@/pages/RiskAssessment";
import AIAssistant from "@/pages/AIAssistant";
import StructureDetail from "@/pages/StructureDetail";
import DigitalTwin from "@/pages/DigitalTwin";
import NotFound from "./pages/NotFound.tsx";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <GlobalDataProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route element={<AppLayout />}>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/structures" element={<Structures />} />
              <Route path="/map" element={<StructureMap />} />
              <Route path="/structures/:id" element={<StructureDetail />} />
              <Route path="/anomalies" element={<Anomalies />} />
              <Route path="/risk" element={<RiskAssessment />} />
              <Route path="/digital-twin" element={<DigitalTwin />} />
              <Route path="/assistant" element={<AIAssistant />} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </GlobalDataProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
