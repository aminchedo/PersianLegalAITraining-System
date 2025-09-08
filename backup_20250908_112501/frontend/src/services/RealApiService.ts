import axios, { AxiosResponse } from 'axios';
import { 
  RealTeamMember, 
  RealModelTraining, 
  RealSystemMetrics, 
  RealAnalyticsData,
  RealLegalDocument,
  RealTrainingJob,
  RealUser,
  ApiResponse,
  PaginatedResponse,
  ApiError
} from '../types/realData';

const API_BASE = import.meta.env.VITE_REAL_API_URL || 'http://localhost:8000/api/real';

class RealApiService {
  private client = axios.create({
    baseURL: API_BASE,
    timeout: 15000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor for logging
  private setupInterceptors(): void {
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    this.client.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error);
        return Promise.reject(this.handleError(error));
      }
    );
  }

  constructor() {
    this.setupInterceptors();
  }

  // REAL TEAM DATA
  async getTeamMembers(params?: {
    skip?: number;
    limit?: number;
    activeOnly?: boolean;
    department?: string;
  }): Promise<RealTeamMember[]> {
    const response = await this.client.get('/team/members', { params });
    return response.data;
  }

  async getTeamMember(id: number): Promise<RealTeamMember> {
    const response = await this.client.get(`/team/members/${id}`);
    return response.data;
  }

  async createTeamMember(member: Omit<RealTeamMember, 'id'>): Promise<RealTeamMember> {
    const response = await this.client.post('/team/members', member);
    return response.data;
  }

  async updateTeamMember(id: number, member: Partial<RealTeamMember>): Promise<RealTeamMember> {
    const response = await this.client.put(`/team/members/${id}`, member);
    return response.data;
  }

  async deleteTeamMember(id: number): Promise<{ message: string }> {
    const response = await this.client.delete(`/team/members/${id}`);
    return response.data;
  }

  async getTeamStats(): Promise<{
    totalMembers: number;
    activeMembers: number;
    onlineMembers: number;
    departments: string[];
  }> {
    const response = await this.client.get('/team/stats');
    return response.data;
  }

  // REAL MODEL DATA
  async getTrainingJobs(params?: {
    skip?: number;
    limit?: number;
    status?: string;
    framework?: string;
  }): Promise<RealModelTraining[]> {
    const response = await this.client.get('/models/training', { params });
    return response.data;
  }

  async getTrainingJob(id: number): Promise<RealModelTraining> {
    const response = await this.client.get(`/models/training/${id}`);
    return response.data;
  }

  async createTrainingJob(job: Omit<RealModelTraining, 'id'>): Promise<RealModelTraining> {
    const response = await this.client.post('/models/training', job);
    return response.data;
  }

  async updateTrainingJob(id: number, job: Partial<RealModelTraining>): Promise<RealModelTraining> {
    const response = await this.client.put(`/models/training/${id}`, job);
    return response.data;
  }

  async startTrainingJob(id: number): Promise<{ message: string; job: RealModelTraining }> {
    const response = await this.client.post(`/models/training/${id}/start`);
    return response.data;
  }

  async pauseTrainingJob(id: number): Promise<{ message: string; job: RealModelTraining }> {
    const response = await this.client.post(`/models/training/${id}/pause`);
    return response.data;
  }

  async stopTrainingJob(id: number): Promise<{ message: string; job: RealModelTraining }> {
    const response = await this.client.post(`/models/training/${id}/stop`);
    return response.data;
  }

  async getModelStats(): Promise<{
    totalJobs: number;
    activeJobs: number;
    completedJobs: number;
    failedJobs: number;
    averageAccuracy: number;
  }> {
    const response = await this.client.get('/models/stats');
    return response.data;
  }

  // REAL SYSTEM METRICS
  async getCurrentSystemMetrics(): Promise<RealSystemMetrics> {
    const response = await this.client.get('/monitoring/system-metrics');
    return response.data;
  }

  async getSystemMetricsHistory(hours: number = 24): Promise<RealSystemMetrics[]> {
    const response = await this.client.get('/monitoring/system-metrics/history', {
      params: { hours }
    });
    return response.data;
  }

  async storeSystemMetrics(metrics: RealSystemMetrics): Promise<{ message: string; id: number }> {
    const response = await this.client.post('/monitoring/system-metrics', metrics);
    return response.data;
  }

  async getHealthCheck(): Promise<{
    status: string;
    timestamp: string;
    checks?: Record<string, any>;
    error?: string;
  }> {
    const response = await this.client.get('/monitoring/health');
    return response.data;
  }

  // REAL ANALYTICS DATA
  async getAnalyticsData(params?: {
    metricName?: string;
    hours?: number;
    limit?: number;
  }): Promise<RealAnalyticsData[]> {
    const response = await this.client.get('/analytics/data', { params });
    return response.data;
  }

  // REAL LEGAL DOCUMENTS
  async getLegalDocuments(params?: {
    skip?: number;
    limit?: number;
    documentType?: string;
    category?: string;
    language?: string;
  }): Promise<RealLegalDocument[]> {
    const response = await this.client.get('/documents', { params });
    return response.data;
  }

  async getLegalDocument(id: number): Promise<RealLegalDocument> {
    const response = await this.client.get(`/documents/${id}`);
    return response.data;
  }

  // REAL TRAINING JOBS
  async getTrainingJobsList(params?: {
    skip?: number;
    limit?: number;
    status?: string;
    assignedTo?: number;
  }): Promise<RealTrainingJob[]> {
    const response = await this.client.get('/training/jobs', { params });
    return response.data;
  }

  // SYSTEM ENDPOINTS
  async getSystemStats(): Promise<{
    teamMembers: number;
    totalModels: number;
    activeModels: number;
    timestamp: string;
  }> {
    const response = await this.client.get('/stats');
    return response.data;
  }

  async getSystemHealth(): Promise<{
    status: string;
    timestamp: string;
    system?: Record<string, any>;
    error?: string;
  }> {
    const response = await this.client.get('/health');
    return response.data;
  }

  // ERROR HANDLING - Real error messages
  private handleError(error: any): ApiError {
    if (error.response) {
      // Server responded with error status
      return {
        detail: error.response.data?.detail || error.response.data?.message || 'Server error',
        status_code: error.response.status,
        timestamp: new Date().toISOString()
      };
    } else if (error.request) {
      // Request was made but no response received
      return {
        detail: 'Network error - no response from server',
        status_code: 0,
        timestamp: new Date().toISOString()
      };
    } else {
      // Something else happened
      return {
        detail: error.message || 'Unknown error occurred',
        status_code: 0,
        timestamp: new Date().toISOString()
      };
    }
  }

  // UTILITY METHODS
  async testConnection(): Promise<boolean> {
    try {
      await this.client.get('/health');
      return true;
    } catch (error) {
      console.error('Connection test failed:', error);
      return false;
    }
  }

  getBaseUrl(): string {
    return API_BASE;
  }
}

export const realApiService = new RealApiService();