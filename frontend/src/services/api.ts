import axios from 'axios';
import { API_CONFIG, GenerateArchitectureRequest, GenerateArchitectureResponse } from '@/config/api';

const api = axios.create({
  baseURL: API_CONFIG.baseURL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const apiService = {
  async healthCheck(): Promise<boolean> {
    try {
      const response = await api.get(API_CONFIG.endpoints.health);
      return response.status === 200;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  },

  async generateArchitecture(
    request: GenerateArchitectureRequest
  ): Promise<GenerateArchitectureResponse> {
    const formData = new FormData();
    
    if (typeof request.requirements === 'string') {
      formData.append('text', request.requirements);
    } else {
      formData.append('file', request.requirements);
    }
    
    formData.append('budget_tier', request.budget_tier);

    const response = await api.post<GenerateArchitectureResponse>(
      API_CONFIG.endpoints.generateArchitecture,
      formData
    );

    return response.data;
  },
};
