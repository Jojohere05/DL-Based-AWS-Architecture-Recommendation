import { useEffect } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Navigation } from "@/components/Navigation";
import { Footer } from "@/components/Footer";
import { useAuth } from "@/contexts/AuthContext";
import { useLibrary } from "@/contexts/LibraryContext";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Trash2, RefreshCw, Calendar, Download, FileJson, FileText } from "lucide-react";
import { toast } from "sonner";
import { exportToPDF, exportToJSON } from "@/utils/export";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const Dashboard = () => {
  const { user, isAuthenticated } = useAuth();
  const { architectures, deleteArchitecture } = useLibrary();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isAuthenticated) {
      navigate("/auth");
    }
  }, [isAuthenticated, navigate]);

  const handleDelete = (id: string, name: string) => {
    deleteArchitecture(id);
    toast.success(`Deleted "${name}"`);
  };

  const handleRegenerate = (id: string) => {
    navigate(`/?regenerate=${id}`);
  };

  const handleExportPDF = async (architecture: any) => {
    try {
      await exportToPDF(architecture);
      toast.success("Exported to PDF successfully!");
    } catch (error) {
      toast.error("Failed to export PDF");
    }
  };

  const handleExportJSON = (architecture: any) => {
    try {
      exportToJSON(architecture);
      toast.success("Exported to JSON successfully!");
    } catch (error) {
      toast.error("Failed to export JSON");
    }
  };

  if (!user) return null;

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      
      <main className="flex-1 py-12 px-4">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="mb-8">
              <h1 className="text-4xl font-bold mb-2">Welcome back, {user.name}!</h1>
              <p className="text-muted-foreground">
                Manage your saved cloud architectures
              </p>
            </div>

            {architectures.length === 0 ? (
              <Card className="glass-card border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-16">
                  <div className="text-6xl mb-4">📦</div>
                  <h3 className="text-2xl font-bold mb-2">No Architectures Yet</h3>
                  <p className="text-muted-foreground mb-6 text-center max-w-md">
                    Start by generating your first cloud architecture from the home page
                  </p>
                  <Button onClick={() => navigate("/")}>
                    Generate Architecture
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {architectures.map((architecture, index) => (
                  <motion.div
                    key={architecture.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                  >
                    <Card className="glass-card hover:shadow-elegant transition-all h-full flex flex-col">
                      <CardHeader>
                        <CardTitle className="text-xl">{architecture.name}</CardTitle>
                        <CardDescription className="line-clamp-2">
                          {architecture.description}
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="flex-1 flex flex-col">
                        <div className="flex items-center gap-2 text-sm text-muted-foreground mb-4">
                          <Calendar className="w-4 h-4" />
                          {new Date(architecture.timestamp).toLocaleDateString()}
                        </div>
                        
                        {architecture.costEstimate && (
                          <div className="mb-4 p-3 bg-accent/10 rounded-lg">
                            <p className="text-sm text-muted-foreground">Estimated Cost</p>
                            <p className="text-xl font-bold text-accent">{architecture.costEstimate}</p>
                          </div>
                        )}
                        
                        <div className="mb-4">
                          <p className="text-sm font-medium mb-2">AWS Services:</p>
                          <div className="flex flex-wrap gap-2">
                            {architecture.services.slice(0, 3).map((service, idx) => (
                              <Badge key={idx} variant="secondary">
                                {service}
                              </Badge>
                            ))}
                            {architecture.services.length > 3 && (
                              <Badge variant="outline">
                                +{architecture.services.length - 3} more
                              </Badge>
                            )}
                          </div>
                        </div>

                        <div className="flex gap-2 mt-auto">
                          <Button
                            variant="outline"
                            size="sm"
                            className="flex-1"
                            onClick={() => handleRegenerate(architecture.id)}
                          >
                            <RefreshCw className="w-4 h-4 mr-1" />
                            Regenerate
                          </Button>
                          
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="outline" size="sm">
                                <Download className="w-4 h-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent>
                              <DropdownMenuItem onClick={() => handleExportPDF(architecture)}>
                                <FileText className="w-4 h-4 mr-2" />
                                Export PDF
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => handleExportJSON(architecture)}>
                                <FileJson className="w-4 h-4 mr-2" />
                                Export JSON
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                          
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => handleDelete(architecture.id, architecture.name)}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Dashboard;
