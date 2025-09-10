import axios, { AxiosResponse, AxiosError } from 'axios';

interface BoltApiConfig {
  baseURL: string;
  timeout: number;
  retryAttempts: number;
}

class BoltApiService {
  private axiosInstance;
  private config: BoltApiConfig;

  constructor() {
    this.config = {
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api',
      timeout: 15000,
      retryAttempts: 3
    };

    this.axiosInstance = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.axiosInstance.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        config.headers['X-Request-ID'] = this.generateRequestId();
        return config;
      },
      (error) => {
        console.error('Request interceptor error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor with retry logic
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as any;

        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
          return Promise.reject(error);
        }

        // Retry logic for server errors
        if (error.response?.status >= 500 && originalRequest._retryCount < this.config.retryAttempts) {
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;
          await this.delay(1000 * originalRequest._retryCount);
          return this.axiosInstance(originalRequest);
        }

        return Promise.reject(error);
      }
    );
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Health check method
  async healthCheck(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/bolt/health');
  }

  // Training endpoints
  async startTraining(modelId: string, config: any): Promise<AxiosResponse> {
    return this.axiosInstance.post('/training/start', { 
      model_id: modelId, 
      config 
    });
  }

  async stopTraining(sessionId: string): Promise<AxiosResponse> {
    return this.axiosInstance.post('/training/stop', { 
      session_id: sessionId 
    });
  }

  async getTrainingStatus(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/training/status');
  }

  async getTrainingLogs(sessionId: string): Promise<AxiosResponse> {
    return this.axiosInstance.get(`/training/logs/${sessionId}`);
  }

  // Models
  async getModels(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/models');
  }

  async createModel(modelData: any): Promise<AxiosResponse> {
    return this.axiosInstance.post('/models', modelData);
  }

  // Data Sources
  async getDataSources(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/data/sources');
  }

  async startDataCollection(sourceId: string): Promise<AxiosResponse> {
    return this.axiosInstance.post('/data/collect', { 
      source_id: sourceId 
    });
  }

  async getDataStatus(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/data/status');
  }

  async getDataQuality(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/data/quality');
  }

  // System
  async getSystemMetrics(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/metrics/system');
  }

  async getTrainingMetrics(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/metrics/training');
  }

  async getSystemHealth(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/system/health');
  }

  // Analytics
  async getAnalytics(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/analytics');
  }

  async getPerformanceMetrics(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/analytics/performance');
  }

  // Documents
  async uploadDocument(file: File): Promise<AxiosResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.axiosInstance.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
  }

  async getDocuments(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/documents');
  }

  async processDocument(documentId: string): Promise<AxiosResponse> {
    return this.axiosInstance.post(`/documents/${documentId}/process`);
  }

  async deleteDocument(documentId: string): Promise<AxiosResponse> {
    return this.axiosInstance.delete(`/documents/${documentId}`);
  }
}

export const boltApi = new BoltApiService();
export default boltApi;