/**
 * Persian Legal AI API Service
 * سرویس API برای سیستم هوش مصنوعی حقوقی فارسی
 */

import axios, { AxiosResponse } from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export interface PersianDocument {
  id: number;
  title: string;
  content: string;
  category: string;
  document_type: string;
  persian_date: string;
  quality_score: number;
  summary: string;
  snippet?: string;
  rank?: number;
}

export interface DocumentSearchRequest {
  query: string;
  category?: string;
  document_type?: string;
  limit?: number;
  offset?: number;
}

export interface DocumentSearchResponse {
  documents: PersianDocument[];
  total: number;
  query: string;
  filters: {
    category: string | null;
    document_type: string | null;
  };
  pagination: {
    limit: number;
    offset: number;
  };
  timestamp: string;
}

export interface ClassificationRequest {
  text: string;
  return_probabilities?: boolean;
}

export interface ClassificationResponse {
  classification: {
    category_fa: string;
    category_en: string;
    confidence: number;
    predicted_class: number;
    timestamp: string;
    model_info: {
      name: string;
      device: string;
      dora_enabled: boolean;
    };
    all_probabilities?: Record<string, number>;
  };
  document_type: {
    document_type_fa: string;
    document_type_en: string;
    confidence: number;
    scores: Record<string, number>;
    timestamp: string;
  };
  keywords: string[];
  summary: string;
  text_stats: {
    word_count: number;
    char_count: number;
  };
  timestamp: string;
}

export interface SystemHealth {
  status: string;
  timestamp: string;
  components: {
    database: string;
    ai_models: string;
    gpu: string;
    memory: string;
  };
  system_info: {
    python_version: string;
    torch_version: string;
    device: string;
    cpu_count: number;
    memory_total: number;
    memory_available: number;
    memory_usage_percent: number;
  };
}

export interface SystemStatus {
  system_status: {
    database: { status: string; last_check: string };
    ai_models: { status: string; last_check: string };
    training: { active_sessions: number; total_sessions: number };
    scraping: { is_running: boolean; documents_scraped: number };
  };
  database_stats: {
    total_documents: number;
    by_category: Record<string, number>;
    by_type: Record<string, number>;
    by_status: Record<string, number>;
    timestamp: string;
  };
  model_info: {
    model_name: string;
    device: string;
    is_initialized: boolean;
    dora_enabled: boolean;
    num_categories: number;
    categories: Record<string, string>;
    torch_version: string;
    cuda_available: boolean;
  };
  training_sessions: {
    active: number;
    total: number;
    sessions: string[];
  };
  timestamp: string;
}

export interface TrainingStartRequest {
  model_type?: string;
  epochs?: number;
  learning_rate?: number;
  batch_size?: number;
  use_dora?: boolean;
  notes?: string;
}

export interface TrainingSession {
  id: string;
  status: string;
  config: TrainingStartRequest;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress: {
    current_epoch: number;
    total_epochs: number;
    current_step: number;
    total_steps: number;
    loss: number | null;
    accuracy: number | null;
  };
  logs: Array<{
    timestamp: string;
    message: string;
    level: string;
  }>;
  error: string | null;
}

class PersianApiService {
  private client = axios.create({
    baseURL: API_BASE,
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  constructor() {
    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    this.client.interceptors.request.use(
      (config) => {
        console.log(`🔄 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('❌ API Request Error:', error);
        return Promise.reject(error);
      }
    );

    this.client.interceptors.response.use(
      (response) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('❌ API Response Error:', error);
        return Promise.reject(this.handleError(error));
      }
    );
  }

  // System endpoints
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await this.client.get('/system/health');
    return response.data;
  }

  async getSystemStatus(): Promise<SystemStatus> {
    const response = await this.client.get('/system/status');
    return response.data;
  }

  // Document endpoints
  async searchDocuments(request: DocumentSearchRequest): Promise<DocumentSearchResponse> {
    const response = await this.client.post('/documents/search', request);
    return response.data;
  }

  async insertDocument(document: {
    title: string;
    content: string;
    source_url?: string;
    document_type?: string;
    category?: string;
    subcategory?: string;
    persian_date?: string;
  }): Promise<{ success: boolean; document_id: number; message: string; timestamp: string }> {
    const response = await this.client.post('/documents/insert', document);
    return response.data;
  }

  async getDocumentStats(): Promise<{
    total_documents: number;
    by_category: Record<string, number>;
    by_type: Record<string, number>;
    by_status: Record<string, number>;
    timestamp: string;
  }> {
    const response = await this.client.get('/documents/stats');
    return response.data;
  }

  // AI endpoints
  async classifyDocument(request: ClassificationRequest): Promise<ClassificationResponse> {
    const response = await this.client.post('/ai/classify', request);
    return response.data;
  }

  async getModelInfo(): Promise<{
    model_name: string;
    device: string;
    is_initialized: boolean;
    dora_enabled: boolean;
    num_categories: number;
    categories: Record<string, string>;
    torch_version: string;
    cuda_available: boolean;
  }> {
    const response = await this.client.get('/ai/model-info');
    return response.data;
  }

  // Training endpoints
  async startTraining(request: TrainingStartRequest): Promise<{
    session_id: string;
    status: string;
    config: TrainingStartRequest;
    estimated_duration: string;
    timestamp: string;
  }> {
    const response = await this.client.post('/training/start', request);
    return response.data;
  }

  async getTrainingStatus(sessionId: string): Promise<TrainingSession> {
    const response = await this.client.get(`/training/status/${sessionId}`);
    return response.data;
  }

  async listTrainingSessions(): Promise<{
    sessions: TrainingSession[];
    total: number;
    active: number;
    timestamp: string;
  }> {
    const response = await this.client.get('/training/sessions');
    return response.data;
  }

  // Utility methods
  async testConnection(): Promise<boolean> {
    try {
      await this.client.get('/system/health');
      return true;
    } catch (error) {
      console.error('Connection test failed:', error);
      return false;
    }
  }

  getBaseUrl(): string {
    return API_BASE;
  }

  private handleError(error: any): Error {
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.message || 'Server error';
      return new Error(`${error.response.status}: ${message}`);
    } else if (error.request) {
      // Request was made but no response received
      return new Error('Network error - no response from server');
    } else {
      // Something else happened
      return new Error(error.message || 'Unknown error occurred');
    }
  }
}

export const persianApiService = new PersianApiService();
export default persianApiService;