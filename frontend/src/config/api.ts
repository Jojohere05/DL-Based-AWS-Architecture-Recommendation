// API Configuration
export const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  endpoints: {
    generateArchitecture: '/api/generate-architecture',
    health: '/api/health',
  },
};

export type BudgetTier = 'low' | 'medium' | 'high';

export interface GenerateArchitectureRequest {
  requirements: string | File;
  budget_tier: BudgetTier;
}

export interface ArchitectureNode {
  id: string;
  name: string;
  category: string;
  confidence?: number;
  description?: string;
}

export interface ArchitectureEdge {
  from: string;
  to: string;
  label?: string;
}

export interface ServiceRecommendation {
  name: string;
  category: string;
  confidence: number;
  description: string;
  icon?: string;
}

export interface CostBreakdown {
  service: string;
  monthly_cost: number;
}

export interface GenerateArchitectureResponse {
  architecture_graph: {
    nodes: ArchitectureNode[];
    edges: ArchitectureEdge[];
  };
  recommended_services: ServiceRecommendation[];
  cost_breakdown: CostBreakdown[];
  total_monthly_cost: number;
  deployment_guide: string;
}
