const API_BASE = 'http://localhost:8000/api';

export class PersianAIAPI {
  private static async request(endpoint: string, options: RequestInit = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    return response.json();
  }

  // Training endpoints (Real API integration)
  static async startTraining(modelId: string, config: any) {
    return this.request('/training/start', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, config }),
    });
  }

  static async stopTraining(sessionId: string) {
    return this.request('/training/stop', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId }),
    });
  }

  static async getTrainingStatus() {
    return this.request('/training/status');
  }

  static async getTrainingLogs(sessionId: string) {
    return this.request(`/training/logs/${sessionId}`);
  }

  // Models
  static async getModels() {
    return this.request('/models');
  }

  static async createModel(modelData: any) {
    return this.request('/models', {
      method: 'POST',
      body: JSON.stringify(modelData),
    });
  }


  // Data Sources
  static async getDataSources() {
    return this.request('/data/sources');
  }

  static async startDataCollection(sourceId: string) {
    return this.request('/data/collect', {
      method: 'POST',
      body: JSON.stringify({ source_id: sourceId }),
    });
  }

  static async getDataStatus() {
    return this.request('/data/status');
  }

  static async getDataQuality() {
    return this.request('/data/quality');
  }

  // System
  static async getSystemMetrics() {
    return this.request('/metrics/system');
  }

  static async getTrainingMetrics() {
    return this.request('/metrics/training');
  }

  static async getSystemHealth() {
    return this.request('/system/health');
  }

  static async getSystemLogs() {
    return this.request('/system/logs');
  }
}