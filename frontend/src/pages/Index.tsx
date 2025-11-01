import { useState, useRef, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Hero } from "@/components/Hero";
import { Navigation } from "@/components/Navigation";
import { InputPanel } from "@/components/InputPanel";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { Footer } from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useLibrary } from "@/contexts/LibraryContext";
import { toast } from "sonner";
import { Save, AlertCircle } from "lucide-react";
import { apiService } from "@/services/api";
import { BudgetTier, GenerateArchitectureResponse } from "@/config/api";

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [generatedInput, setGeneratedInput] = useState<string | null>(null);
  const [architectureData, setArchitectureData] = useState<GenerateArchitectureResponse | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string>();
  const [budgetTier, setBudgetTier] = useState<BudgetTier>("medium");
  const inputRef = useRef<HTMLDivElement>(null);
  const { isAuthenticated } = useAuth();
  const { saveArchitecture, getArchitecture } = useLibrary();
  const [searchParams] = useSearchParams();

  // Check backend health on mount
  useEffect(() => {
    const checkHealth = async () => {
      const isHealthy = await apiService.healthCheck();
      if (!isHealthy) {
        toast.error("Backend service is unavailable. Using mock data.", {
          description: "Please ensure the FastAPI backend is running on http://localhost:8000",
          icon: <AlertCircle className="w-4 h-4" />,
          duration: 5000,
        });
      }
    };
    checkHealth();
  }, []);

  // Handle regenerate from dashboard
  useEffect(() => {
    const regenerateId = searchParams.get("regenerate");
    if (regenerateId) {
      const architecture = getArchitecture(regenerateId);
      if (architecture) {
        setGeneratedInput(architecture.input);
        setUploadedFileName(architecture.uploadedFile);
        inputRef.current?.scrollIntoView({ behavior: "smooth" });
        toast.info("Architecture loaded for regeneration");
      }
    }
  }, [searchParams, getArchitecture]);

  const handleGetStarted = () => {
    inputRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleGenerate = async (input: string, file?: File, budgetTierParam?: BudgetTier) => {
    setIsLoading(true);
    setUploadedFileName(file?.name);
    if (budgetTierParam) setBudgetTier(budgetTierParam);
    
    try {
      setLoadingMessage("Parsing requirements...");
      await new Promise((resolve) => setTimeout(resolve, 800));
      
      setLoadingMessage("Analyzing with Gemini and ML model...");
      
      // Try to call the real backend API
      const response = await apiService.generateArchitecture({
        requirements: file || input,
        budget_tier: budgetTierParam || budgetTier,
      });
      
      setArchitectureData(response);
      setGeneratedInput(input);
      toast.success("Architecture generated successfully!");
      
    } catch (error) {
      console.error("API call failed:", error);
      toast.warning("Using mock data (backend unavailable)", {
        description: "Connect to FastAPI backend for real AI recommendations",
      });
      
      // Fallback to mock data
      setLoadingMessage("Generating architecture...");
      await new Promise((resolve) => setTimeout(resolve, 1500));
      
      setArchitectureData(null); // Will use mock data in ResultsDisplay
      setGeneratedInput(input);
      toast.success("Architecture generated with mock data");
    } finally {
      setIsLoading(false);
      setLoadingMessage("");
    }
  };

  const handleSaveArchitecture = () => {
    if (!isAuthenticated) {
      toast.error("Please sign in to save architectures");
      return;
    }

    if (!generatedInput) return;

    const services = architectureData?.recommended_services.map(s => s.name) || 
                    ["EC2", "S3", "RDS", "Lambda", "API Gateway"];
    
    const costEstimate = architectureData?.total_monthly_cost 
      ? `$${architectureData.total_monthly_cost.toFixed(2)}/month`
      : "$250/month";

    saveArchitecture({
      name: `Architecture ${new Date().toLocaleDateString()}`,
      description: generatedInput.slice(0, 100) + "...",
      input: generatedInput,
      uploadedFile: uploadedFileName,
      services,
      diagram: JSON.stringify(architectureData?.architecture_graph || {}),
      costEstimate,
      budgetTier,
    });

    toast.success("Architecture saved to your library!");
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      
      <main className="flex-1">
        <Hero onGetStarted={handleGetStarted} />

        <section className="py-16 bg-gradient-to-b from-background to-muted/20" ref={inputRef}>
          <div className="container mx-auto px-4">
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
              <div className="lg:col-span-2">
                <InputPanel 
                  onGenerate={handleGenerate} 
                  isLoading={isLoading}
                  initialInput={generatedInput || ""}
                  initialFileName={uploadedFileName}
                />
              </div>
              <div className="lg:col-span-3">
                {isLoading ? (
                  <div className="glass-card rounded-2xl p-12 h-full flex items-center justify-center text-center">
                    <div>
                      <div className="mb-6">
                        <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                      </div>
                      <h3 className="text-2xl font-bold mb-2">
                        {loadingMessage || "Processing..."}
                      </h3>
                      <p className="text-muted-foreground">
                        Powered by FastAPI, Gemini 2.0 & 90% F1 Transformer
                      </p>
                    </div>
                  </div>
                ) : generatedInput ? (
                  <div className="space-y-4">
                    <div className="flex justify-end">
                      <Button
                        onClick={handleSaveArchitecture}
                        disabled={!isAuthenticated}
                        className="gap-2"
                      >
                        <Save className="w-4 h-4" />
                        Save to Library
                      </Button>
                    </div>
                    <ResultsDisplay 
                      input={generatedInput} 
                      architectureData={architectureData}
                    />
                  </div>
                ) : (
                  <div className="glass-card rounded-2xl p-12 h-full flex items-center justify-center text-center">
                    <div>
                      <div className="text-6xl mb-4">🚀</div>
                      <h3 className="text-2xl font-bold mb-2">
                        Ready to Generate
                      </h3>
                      <p className="text-muted-foreground">
                        Describe your application to see AI-powered architecture recommendations
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      </main>

      {generatedInput && <Footer />}
    </div>
  );
};

export default Index;
