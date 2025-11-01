import { motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Network, Server, BrainCircuit, DollarSign } from "lucide-react";
import { ArchitectureDiagram } from "./ArchitectureDiagram";
import { ServiceCard } from "./ServiceCard";
import { ExplainabilityView } from "./ExplainabilityView";
import { GenerateArchitectureResponse } from "@/config/api";

interface ResultsDisplayProps {
  input: string;
  architectureData?: GenerateArchitectureResponse | null;
}

// Mock data - in production, this would come from AI backend
const mockServices = [
  {
    name: "Amazon EC2",
    description: "Scalable virtual servers in the cloud",
    confidence: 87,
    category: "Compute",
    reasoning: [
      "Application requires web server hosting",
      "Need for scalable compute resources",
      "Mentioned server infrastructure requirements",
    ],
  },
  {
    name: "Amazon S3",
    description: "Object storage for files and images",
    confidence: 95,
    category: "Storage",
    reasoning: [
      "Photo storage requirement identified",
      "Need for scalable file management",
      "Large media handling mentioned",
    ],
  },
  {
    name: "Amazon RDS",
    description: "Managed relational database service",
    confidence: 82,
    category: "Database",
    reasoning: [
      "Structured data storage needed",
      "User profiles and relationships",
      "ACID compliance requirements",
    ],
  },
  {
    name: "API Gateway",
    description: "RESTful API management",
    confidence: 91,
    category: "Networking",
    reasoning: [
      "Mobile app backend API needed",
      "Request routing and throttling",
      "Authentication integration point",
    ],
  },
  {
    name: "Amazon Cognito",
    description: "User authentication and authorization",
    confidence: 94,
    category: "Security",
    reasoning: [
      "User authentication explicitly mentioned",
      "Social login requirements",
      "Secure user management needed",
    ],
  },
  {
    name: "Amazon CloudFront",
    description: "Content delivery network (CDN)",
    confidence: 78,
    category: "Networking",
    reasoning: [
      "Global content distribution",
      "Low-latency image delivery",
      "Static asset caching",
    ],
  },
];

const mockNodes = [
  { id: "1", name: "CloudFront", category: "Networking", x: 50, y: 10 },
  { id: "2", name: "API Gateway", category: "Networking", x: 50, y: 30 },
  { id: "3", name: "Lambda", category: "Compute", x: 30, y: 50 },
  { id: "4", name: "EC2", category: "Compute", x: 70, y: 50 },
  { id: "5", name: "RDS", category: "Database", x: 30, y: 70 },
  { id: "6", name: "S3", category: "Storage", x: 70, y: 70 },
  { id: "7", name: "Cognito", category: "Security", x: 10, y: 50 },
];

const mockConnections = [
  { from: "1", to: "2" },
  { from: "2", to: "3" },
  { from: "2", to: "4" },
  { from: "3", to: "5" },
  { from: "4", to: "5" },
  { from: "4", to: "6" },
  { from: "7", to: "2" },
];

const mockHighlights = [
  { text: "photo-sharing", category: "Storage" },
  { text: "user authentication", category: "Security" },
  { text: "mobile app", category: "Compute" },
  { text: "real-time notifications", category: "Networking" },
];

const mockExplanations = [
  {
    service: "Amazon S3",
    icon: "S3",
    category: "Storage",
    confidence: 95,
    highlights: ["photo-sharing", "image storage"],
    reasons: [
      "Optimized for large file storage",
      "Automatic scaling and durability",
      "Cost-effective for media files",
    ],
  },
  {
    service: "Amazon Cognito",
    icon: "CG",
    category: "Security",
    confidence: 94,
    highlights: ["user authentication", "social features"],
    reasons: [
      "Built-in OAuth and social login",
      "User pool management included",
      "Secure token-based authentication",
    ],
  },
  {
    service: "API Gateway",
    icon: "AG",
    category: "Networking",
    confidence: 91,
    highlights: ["mobile app", "real-time"],
    reasons: [
      "WebSocket support for real-time features",
      "RESTful API management",
      "Request throttling and caching",
    ],
  },
];

export const ResultsDisplay = ({ input, architectureData }: ResultsDisplayProps) => {
  // Use API data if available, otherwise fall back to mock data
  const apiNodes = architectureData?.architecture_graph.nodes.map((node, idx) => ({
    ...node,
    x: (idx % 3) * 33 + 17,
    y: Math.floor(idx / 3) * 25 + 15,
  })) || mockNodes;
  const nodes = apiNodes;
  const connections = architectureData?.architecture_graph.edges || mockConnections;
  const services = architectureData?.recommended_services.map(s => ({
    name: s.name,
    description: s.description,
    confidence: Math.round(s.confidence * 100),
    category: s.category,
    reasoning: [s.description],
  })) || mockServices;
  
  const totalCost = architectureData?.total_monthly_cost;
  const costBreakdown = architectureData?.cost_breakdown;

  return (
    <motion.div
      className="mt-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Tabs defaultValue="diagram" className="w-full">
        <TabsList className="grid w-full grid-cols-4 mb-6 glass-card p-1">
          <TabsTrigger value="diagram" className="flex items-center gap-2">
            <Network size={16} />
            <span className="hidden sm:inline">Architecture</span>
          </TabsTrigger>
          <TabsTrigger value="services" className="flex items-center gap-2">
            <Server size={16} />
            <span className="hidden sm:inline">Services</span>
          </TabsTrigger>
          <TabsTrigger value="explainability" className="flex items-center gap-2">
            <BrainCircuit size={16} />
            <span className="hidden sm:inline">Explainability</span>
          </TabsTrigger>
          <TabsTrigger value="cost" className="flex items-center gap-2">
            <DollarSign size={16} />
            <span className="hidden sm:inline">Cost</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="diagram">
          <ArchitectureDiagram nodes={nodes} connections={connections} />
        </TabsContent>

        <TabsContent value="services">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {services.map((service, i) => (
              <ServiceCard key={i} {...service} index={i} />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="explainability">
          <ExplainabilityView
            originalInput={input}
            highlights={mockHighlights}
            explanations={mockExplanations}
          />
        </TabsContent>

        <TabsContent value="cost">
          <div className="glass-card rounded-2xl p-8">
            <div className="text-center mb-6">
              <DollarSign className="mx-auto mb-4 text-accent" size={48} />
              <h3 className="text-2xl font-bold mb-2">Cost Estimation</h3>
              {totalCost ? (
                <div className="glass-card inline-block px-8 py-4 rounded-lg">
                  <p className="text-sm text-muted-foreground">Estimated Monthly Cost</p>
                  <p className="text-4xl font-bold text-accent">${totalCost.toFixed(2)}</p>
                </div>
              ) : (
                <div className="glass-card inline-block px-6 py-3 rounded-lg">
                  <p className="text-sm text-muted-foreground">Estimated Monthly Cost</p>
                  <p className="text-3xl font-bold text-accent">$127 - $340</p>
                </div>
              )}
            </div>
            
            {costBreakdown && costBreakdown.length > 0 && (
              <div className="mt-6">
                <h4 className="font-semibold mb-4">Cost Breakdown</h4>
                <div className="space-y-2">
                  {costBreakdown.map((item, i) => (
                    <div key={i} className="flex justify-between items-center p-3 rounded-lg bg-muted/20">
                      <span className="font-medium">{item.service}</span>
                      <span className="text-accent font-semibold">${item.monthly_cost.toFixed(2)}/mo</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </motion.div>
  );
};
